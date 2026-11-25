"""Train wider-coverage MACE models, then repeat a guarded improvement."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from model_growth_data import DATA, ROOT, freeze, now, prepare
from model_study_data import save_json

OUTPUT = ROOT / "results/v0.10.0"


def valid_metrics(metrics):
    required = ("barrier_mae_eV", "force_component_rmse_eV_per_A",
                "ts_region_force_component_rmse_eV_per_A")
    if any(not math.isfinite(metrics[key]) or metrics[key] < 0 for key in required):
        raise ValueError("Non-finite or negative model-selection metric")


def wait_for_cuda(protocol, update):
    """Use the same kernel locks as the existing CUDA queue, in its lock order."""
    locks = []
    start = time.monotonic()
    try:
        for relative in ("results/overnight-20261124/omen/owner.lock",
                         "results/v0.9.0/runner.lock", "results/v0.7.0/runner.lock"):
            lock = (ROOT / relative).open("a")
            locks.append(lock)
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()-start > protocol["predecessor_wait_hours"]*3600:
                        raise TimeoutError("Existing CUDA queue did not release its locks")
                    update("waiting_for_existing_cuda_owner", waiting_for=relative)
                    time.sleep(30)
        predecessor = json.loads((ROOT / "results/overnight-20261124/omen/state.json").read_text())
        if predecessor["status"] not in ("completed", "completed_with_failures"):
            raise RuntimeError("Existing queue is not in a documented terminal state")
        return locks
    except BaseException:
        for lock in locks:
            lock.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/v0.10.0.json").read_text())
    if args.prepare_only:
        DATA.mkdir(parents=True, exist_ok=True)
        lock = (DATA / "preparation.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(freeze(protocol))
        print("PASS - nested growth labels prepared, prospective holdouts unopened", flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "state.json").exists():
        raise FileExistsError("Existing growth owner/evidence; no implicit restart")
    state = dict(owner_pid=os.getpid(), host="OMEN", started_at_utc=now(),
                 holdout_labels_evaluated=False, automatic_training_retries=0, attempts={})

    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save_json(OUTPUT / "state.json", state)

    try:
        manifest = json.loads((DATA / "manifest.json").read_text())
        if manifest["partition"] != freeze(protocol):
            raise ValueError("Training data does not match the frozen growth partition")
        for size in protocol["reaction_counts"]:
            if len(manifest["datasets"][f"coverage{size}"]) != size*8:
                raise ValueError("Prepared growth labels have the wrong count")
        update("ready_waiting_for_cuda")
        predecessor_locks = wait_for_cuda(protocol, update)
        import torch
        from run_model_study import train, benchmark
        from run_optimizer_study import choose_guarded
        from ase.io import read
        if not torch.cuda.is_available():
            raise RuntimeError("Growth training requires the accepted OMEN CUDA environment")
        torch.set_num_threads(4)
        baseline = protocol["baseline"]["name"]
        metrics = {baseline: protocol["baseline"]["validation_metrics"]}
        valid_metrics(metrics[baseline])

        def attempt(size, seed, name):
            experiment = dict(name=name, dataset=f"coverage{size}", train_size=size*8,
                train_file=f"data/v0.10.0/coverage{size}.extxyz", seed=seed,
                model_file=f"models/v0.10.0/{name}.model")
            runs = []
            record = dict(status="running", experiment=experiment, started_at_utc=now())
            state["attempts"][name] = record
            update("starting_model", experiment=name)
            try:
                atoms = read(ROOT / experiment["train_file"], index=":")
                if len(atoms) != experiment["train_size"]:
                    raise ValueError("Actual training data count changed")
                del atoms
                train(experiment, protocol, runs, output_root=OUTPUT,
                      manifest_path=OUTPUT / f"training/{name}/manifest.json", state_writer=update)
                if runs[0]["trained_epochs"] != protocol["training"]["epochs"]:
                    raise RuntimeError("Logged epochs differ from the frozen training budget")
                record["run"] = runs[0]
                for role, count in (("validation", 60), ("meta_development", 128)):
                    update(role, experiment=name)
                    result = benchmark(f"{role}/{name}", [name],
                        {name: str(ROOT / experiment["model_file"])}, role=role,
                        count=count, output_root=OUTPUT)["models"][name]["aggregate"]
                    valid_metrics(result)
                    record[role] = result
                record.update(status="completed", finished_at_utc=now())
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
                print(f"Retained failed attempt {name}: {error}", flush=True)
            update("model_terminal", experiment=name)
            return record

        for size in protocol["reaction_counts"]:
            name = f"mh1_coverage{size}_s0"
            record = attempt(size, protocol["training_seed"], name)
            if record["status"] == "completed":
                metrics[name] = record["validation"]
        selected = choose_guarded(metrics, baseline)
        selection = dict(name=selected, frozen_at_utc=now(),
                         validation_metrics=metrics, criterion=protocol["selection"])
        save_json(OUTPUT / "selection.json", selection, immutable=True)
        update("selection_frozen", selection=selection)
        repeat_names = []
        if selected != baseline:
            size = state["attempts"][selected]["experiment"]["train_size"] // 8
            for index, seed in enumerate(protocol["additional_seeds"], 1):
                name = f"mh1_coverage{size}_s{index}"
                repeat_names.append(name)
                attempt(size, seed, name)
        completed = [state["attempts"][name] for name in ([selected]+repeat_names)
                     if name in state["attempts"] and state["attempts"][name]["status"] == "completed"]
        aggregate = {}
        if completed:
            for key in completed[0]["validation"]:
                values = [x["validation"][key] for x in completed]
                aggregate[key] = dict(mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values) if len(values)>1 else None, values=values)
        status = ("no_guarded_improvement" if selected == baseline else
                  "growth_training_and_repeats_complete" if len(completed) == 3 else "incomplete_seed_repeats")
        summary = dict(status=status, protocol=protocol, selection=selection,
            attempts=state["attempts"], selected_seed_validation_aggregate=aggregate,
            selected_seed_names=[selected]+repeat_names, finished_at_utc=now(),
            holdout_labels_evaluated=False,
            remaining=["freeze final model and prospective evaluation budgets", "independent static/search/DFT acceptance"])
        save_json(OUTPUT / "screening_summary.json", summary)
        update(status, finished_at_utc=now())
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        raise


if __name__ == "__main__":
    main()
