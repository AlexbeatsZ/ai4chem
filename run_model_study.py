"""Finite first-stage model screening; never opens prospective holdout labels."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import yaml

from model_study_data import (DATA, ROOT, freeze_roles, prepare_meta, prepare_training,
                              prepare_validation, save_json)

OUTPUT = ROOT / "results/v0.7.0"
STATE = OUTPUT / "state.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def state(stage, **extra):
    save_json(STATE, {"host": "OMEN", "updated_at_utc": now(), "stage": stage,
                      "holdout_labels_evaluated": False, **extra})


def benchmark(name, models, checkpoints, role="validation", count=60, *, output_root=None, device="cuda"):
    directory = (output_root or OUTPUT) / name
    summary_path = directory / "benchmark_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if set(summary["models"]) != set(models) or summary["reactions"] != count:
            raise ValueError(f"Cached benchmark identity mismatch: {name}")
        return summary
    directory.mkdir(parents=True, exist_ok=True)
    split = "val" if role == "validation" else "train"
    command = [sys.executable, "-u", "benchmark_transition1x.py", "--input",
        str(DATA / f"{role}.h5"), "--output", str(directory), "--split", split,
        "--count", str(count), "--models", *models, "--plot-limit", "2", "--device", device]
    for key, value in checkpoints.items():
        command += ["--checkpoint", f"{key}={value}"]
    with (directory / "console.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(summary_path.read_text())


def train(experiment, protocol, manifest, *, output_root=None, manifest_path=None, state_writer=state):
    output_root = output_root or OUTPUT
    settings = {**protocol["training"], **experiment.get("training_overrides", {})}
    name = experiment["name"]
    directory = output_root / "training" / name
    if "train_file" in experiment:
        dataset = ROOT / experiment["train_file"]
    else:
        dataset = ROOT / "data/v0.3.0/train_256.extxyz" if experiment["dataset"] == "anchor256" else DATA / f"{experiment['dataset']}.extxyz"
    model_path = ROOT / experiment.get("model_file", f"models/v0.7.0/{name}.model")
    seed = experiment.get("seed", protocol["training_seed"])
    config = {
        "name": name, "seed": seed, "device": settings.get("device", "cuda"),
        "default_dtype": settings["dtype"],
        "foundation_model": str(ROOT / experiment.get("foundation_model", "models/mh1-omol.model")),
        "foundation_model_readout": True, "multiheads_finetuning": False,
        "train_file": str(dataset), "valid_file": str(ROOT / experiment.get("valid_file", "data/v0.7.0/validation.extxyz")),
        "energy_key": "REF_energy", "forces_key": "REF_forces", "E0s": settings["E0s"],
        "model": "ScaleShiftMACE", "loss": "weighted", "scaling": "no_scaling",
        "energy_weight": settings["energy_weight"], "forces_weight": settings["forces_weight"],
        "batch_size": settings["batch_size"], "valid_batch_size": settings["batch_size"],
        "max_num_epochs": settings["epochs"], "patience": settings["patience"],
        "lr": settings["learning_rate"], "freeze": 0,
        "num_workers": 0, "eval_interval": 1, "save_cpu": True,
        "model_dir": str(model_path.parent), "checkpoints_dir": str(directory / "checkpoints"),
        "log_dir": str(directory / "logs"), "results_dir": str(directory / "metrics"),
    }
    completed = next((entry for entry in manifest if entry["name"] == name), None)
    config_path = directory / "config.yaml"
    if completed:
        if not model_path.exists() or yaml.safe_load(config_path.read_text()) != config:
            raise ValueError(f"Completed training changed: {name}")
        return
    if (directory / "console.log").exists():
        raise RuntimeError(f"Interrupted {name} has evidence; inspect it before an explicit retry")
    directory.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    started_at, start = now(), time.perf_counter()
    state_writer("training", experiment=name, started_at_utc=started_at, config=str(config_path.relative_to(ROOT)))
    print(f"Starting {name}, {experiment['train_size']} frames, up to {settings['epochs']} epochs", flush=True)
    with (directory / "console.log").open("w") as log:
        result = subprocess.run([str(Path(sys.executable).with_name("mace_run_train")), "--config", str(config_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Training {name} failed ({result.returncode}); inspect {directory / 'console.log'}")
    if not model_path.exists():
        raise RuntimeError(f"No saved standard model: {name}")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError(f"Non-finite trained weights: {name}")
    console = (directory / "console.log").read_text()
    best = re.search(r"Loaded Stage one model from epoch (\d+)", console)
    epochs = re.findall(r"INFO: Epoch (\d+):", console)
    trained_epochs = max(map(int, epochs)) + 1 if epochs else None
    manifest.append({**experiment, "seed": seed,
        "model": str(model_path.relative_to(ROOT)), "config": str(config_path.relative_to(ROOT)),
        "best_epoch": int(best.group(1)) if best else None, "trained_epochs": trained_epochs,
        "gradient_steps_estimated_from_logged_epochs": trained_epochs * math.ceil(experiment["train_size"] / settings["batch_size"]) if trained_epochs else None,
        "started_at_utc": started_at, "finished_at_utc": now(),
        "elapsed_seconds": time.perf_counter() - start, "heads": list(model.heads),
        "optional_compiled_export_warning": "Failed to compile" in console or "Failed to save compiled" in console})
    save_json(manifest_path or output_root / "training/manifest.json", manifest)
    print(f"Completed {name}; validation-loss-selected epoch {manifest[-1]['best_epoch']}", flush=True)


def wait_for_downloads(timeout_seconds=4*3600):
    start = time.monotonic()
    while True:
        path = OUTPUT / "download.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get("status") == "completed":
            return
        if record.get("status") == "failed":
            raise RuntimeError(f"Pinned data download failed: {record.get('error')}")
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("Training source files did not finish within four hours")
        state("waiting_for_training_shards", completed=record.get("completed", []))
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "runner.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((ROOT / "configs/v0.7.0.json").read_text())
    partition = freeze_roles(protocol)
    prepare_validation(partition)
    if args.freeze_only:
        print("PASS - prospective roles and validation frozen before new training", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The training study requires the OMEN CUDA environment")
    manifest_path = OUTPUT / "training/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    summaries = {}
    for index, experiment in enumerate(protocol["experiments"]):
        if index:
            wait_for_downloads()
            prepare_training(partition)
        train(experiment, protocol, manifest)
        name = experiment["name"]
        state("validation", experiment=name)
        model = ROOT / next(x["model"] for x in manifest if x["name"] == name)
        summaries[name] = benchmark(f"validation/{name}", [name], {name: str(model)})["models"][name]["aggregate"]
        save_json(OUTPUT / "validation_progress.json", {"status": "provisional", "models": summaries,
            "holdout_labels_evaluated": False})
        if index == 1:
            prepare_meta(partition)
            state("frozen_expert_meta_development")
            benchmark("meta_development", ["off23-small", "mh1-omol", "mh0-rgd1", "legacy_mh1_256"],
                {"legacy_mh1_256": str(ROOT / "models/v0.3.0/mh1_full_256.model")},
                role="meta_development", count=protocol["meta_development_count"])
    state("baseline_validation")
    legacy = benchmark("validation/legacy", ["mh1-omol", "legacy_mh1_256"],
        {"legacy_mh1_256": str(ROOT / "models/v0.3.0/mh1_full_256.model")})
    summaries.update({name: value["aggregate"] for name, value in legacy["models"].items()})
    ranked = sorted(summaries, key=lambda name: (summaries[name]["barrier_mae_eV"],
        summaries[name]["ts_region_force_component_rmse_eV_per_A"], name))
    summary = {"status": "first_stage_screening_complete", "protocol": protocol,
        "selected_for_next_stage": ranked[0], "validation_ranking": ranked, "validation_metrics": summaries,
        "runs": manifest, "holdout_labels_evaluated": False, "finished_at_utc": now(),
        "remaining": ["optimizer/replay controls", "three-seed repetition", "freeze models/router", "prospective static/search/DFT evaluation"]}
    save_json(OUTPUT / "screening_summary.json", summary)
    state("first_stage_screening_complete", selected_for_next_stage=ranked[0])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        # A second invocation must not overwrite the live owner's scientific state.
        print("Another model-study runner owns this output; inspect its existing state", file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        state("failed", error=f"{type(error).__name__}: {error}")
        raise
