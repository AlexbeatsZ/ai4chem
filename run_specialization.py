"""Finite equal-update specialization comparison; no prospective holdout access."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from ase.io import read
from model_study_data import save_json
from run_model_study import benchmark, train

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/v0.8.0"


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_budgets(protocol):
    updates = Counter()
    for experiment in protocol["experiments"]:
        settings = {**protocol["training"], **experiment.get("training_overrides", {})}
        if experiment["train_size"] % settings["batch_size"]:
            raise ValueError("Budget accounting requires complete batches")
        updates[experiment["system"]] += experiment["train_size"] // settings["batch_size"] * settings["epochs"]
    if set(updates.values()) != {40960} or len(updates) != 4:
        raise ValueError(f"Systems do not have the frozen equal-update budget: {dict(updates)}")
    return dict(updates)


def worker(experiment, protocol):
    name = experiment["name"]
    directory = OUTPUT / "training" / name
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    state = {"host": "Mac", "owner_pid": os.getpid(), "experiment": name,
             "started_at_utc": now(), "holdout_labels_evaluated": False}

    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save_json(directory / "state.json", state)

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    try:
        update("checking_inputs")
        if len(read(ROOT / experiment["train_file"], index=":")) != experiment["train_size"]:
            raise ValueError("Training dataset differs from frozen frame budget")
        train(experiment, protocol, manifest, output_root=OUTPUT,
              manifest_path=manifest_path, state_writer=update)
        checkpoint = str(ROOT / experiment["model_file"])
        metrics = {}
        for role, count in (("validation", 60), ("meta_development", 128)):
            update(role)
            metrics[role] = benchmark(f"{role}/{name}", [name], {name: checkpoint}, role=role,
                                      count=count, output_root=OUTPUT, device="cpu")["models"][name]["aggregate"]
        update("completed", metrics=metrics, finished_at_utc=now())
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        raise


def coordinator(protocol):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"host": "Mac", "owner_pid": os.getpid(), "started_at_utc": now(),
             "holdout_labels_evaluated": False, "update_budgets": validate_budgets(protocol)}

    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save_json(OUTPUT / "state.json", state)

    try:
        partition = json.loads((ROOT / "configs/v0.8.0_partitions.json").read_text())
        if partition["protocol"] != protocol:
            raise ValueError("Prepared specialization protocol differs")
        deadline = time.monotonic()+12*3600
        while True:
            search = ROOT / "results/v0.7.0/diversity_search"
            statuses = {}
            for name in ("legacy_mh1_256", "aimnet2-rxn", "orbmol-conservative"):
                path = search / name / "state.json"
                statuses[name] = json.loads(path.read_text())["stage"] if path.exists() else "missing"
            if all(stage == "completed" for stage in statuses.values()):
                break
            if any(stage == "failed" for stage in statuses.values()):
                raise RuntimeError(f"Inspect failed search coordinator before spending more CPU: {statuses}")
            if time.monotonic() > deadline:
                raise TimeoutError("Search pilot did not finish within twelve hours")
            update("waiting_for_search_pilot", search_status=statuses)
            time.sleep(60)
        pending = list(protocol["experiments"])
        active, completed, failed = {}, [], []
        while pending or active:
            while pending and len(active) < 3 and not failed:
                experiment = pending.pop(0)
                directory = OUTPUT / "training" / experiment["name"]
                directory.mkdir(parents=True, exist_ok=True)
                existing = directory / "state.json"
                if existing.exists() and json.loads(existing.read_text())["stage"] == "completed":
                    completed.append(experiment["name"])
                    continue
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH=str(ROOT))
                with (directory / "owner_console.log").open("ab") as log:
                    process = subprocess.Popen([sys.executable, "-u", "run_specialization.py", "--arm",
                        experiment["name"]], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                active[experiment["name"]] = process
            update("running", active={name:process.pid for name,process in active.items()},
                   completed=completed, failed=failed, pending=[x["name"] for x in pending])
            if active:
                try:
                    next(iter(active.values())).wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            for name,process in list(active.items()):
                code = process.poll()
                if code is not None:
                    (completed if code == 0 else failed).append(name)
                    del active[name]
            if failed and not active:
                raise RuntimeError(f"Failed workers retained: {failed}; pending work was not overwritten")
        summary = {"status":"specialization_training_and_development_benchmarks_complete",
            "protocol":protocol, "runs":[json.loads((OUTPUT/"training"/x["name"]/"manifest.json").read_text())[0]
                for x in protocol["experiments"]], "finished_at_utc":now(), "holdout_labels_evaluated":False,
            "remaining":["fixed/learned router development comparison", "independent repeated seeds and optimizer controls",
                         "freeze final candidates", "prospective static/search/DFT evaluation"]}
        save_json(OUTPUT / "screening_summary.json", summary)
        update("specialization_training_and_development_benchmarks_complete", finished_at_utc=now())
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/v0.8.0.json").read_text())
    validate_budgets(protocol)
    if args.arm:
        experiment = next(x for x in protocol["experiments"] if x["name"] == args.arm)
        worker(experiment, protocol)
    else:
        coordinator(protocol)
