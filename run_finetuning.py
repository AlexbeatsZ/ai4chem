"""Run the fixed v0.3.0 experiments through the installed official MACE trainer."""

import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from mace.cli.select_head import remove_pt_head


ROOT = Path(__file__).resolve().parent


def main():
    protocol = json.loads((ROOT / "configs/v0.3.0.json").read_text())
    settings = protocol["training"]
    source_models = {"mh1-omol": ("mace-mh-1.model", "omol"),
                     "mh0-rgd1": ("mace-mh-0.model", "rgd1_b3lyp")}
    output = ROOT / "results/v0.3.0/training"
    output.mkdir(parents=True, exist_ok=True)
    for foundation, (filename, head) in source_models.items():
        model = torch.load(ROOT / "models" / filename, map_location="cpu", weights_only=False)
        torch.set_default_dtype(next(model.parameters()).dtype)
        selected = remove_pt_head(model, head)
        selected = selected.to(dtype=getattr(torch, settings["dtype"]))
        if len(selected.heads) != 1:
            raise ValueError("E0 estimation requires the selected single-head model")
        torch.save(selected, ROOT / "models" / f"{foundation}.model")
        print(f"Extracted {foundation}: {selected.heads}", flush=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    completed = {entry["name"] for entry in manifest}
    for experiment in protocol["experiments"]:
        name = experiment["name"]
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "name": name, "seed": protocol["seed"], "device": "cuda",
            "default_dtype": settings["dtype"],
            "foundation_model": str(ROOT / "models" / f"{experiment['foundation']}.model"),
            "foundation_model_readout": True, "multiheads_finetuning": False,
            "train_file": str(ROOT / f"data/v0.3.0/train_{experiment['train_size']}.extxyz"),
            "valid_file": str(ROOT / "data/v0.3.0/validation.extxyz"),
            "energy_key": "REF_energy", "forces_key": "REF_forces", "E0s": settings["E0s"],
            "model": "ScaleShiftMACE", "loss": "weighted", "scaling": "no_scaling",
            "energy_weight": settings["energy_weight"], "forces_weight": settings["forces_weight"],
            "batch_size": settings["batch_size"], "valid_batch_size": settings["batch_size"],
            "max_num_epochs": settings["epochs"], "patience": settings["epochs"],
            "lr": settings["learning_rate"], "freeze": experiment["freeze"],
            "num_workers": 0, "eval_interval": 1,
            "model_dir": str(ROOT / "models/v0.3.0"),
            "checkpoints_dir": str(directory / "checkpoints"),
            "log_dir": str(directory / "logs"), "results_dir": str(directory / "metrics"),
            "save_cpu": True,
        }
        config_path = directory / "config.yaml"
        model_path = ROOT / "models/v0.3.0" / f"{name}.model"
        if name in completed and model_path.exists() and config_path.exists():
            if yaml.safe_load(config_path.read_text()) != config:
                raise ValueError(f"Completed experiment {name} has a different configuration")
            print(f"Using completed {name}", flush=True)
            continue
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Starting {name}", flush=True)
        with (directory / "console.log").open("w") as log:
            result = subprocess.run(
                [str(Path(sys.executable).with_name("mace_run_train")), "--config", str(config_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print("\n".join((directory / "console.log").read_text().splitlines()[-45:]), flush=True)
            raise RuntimeError(f"MACE training failed for {name}, exit {result.returncode}")
        if not model_path.exists():
            raise RuntimeError(f"Trainer did not produce {model_path}")
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError(f"Non-finite trained parameters in {name}")
        selected_epoch = re.search(r"Loaded Stage one model from epoch (\d+)", (directory / "console.log").read_text())
        manifest.append({**experiment, "model": str(model_path.relative_to(ROOT)),
                         "best_epoch": int(selected_epoch.group(1)) if selected_epoch else None,
                         "heads": list(model.heads), "config": str(config_path.relative_to(ROOT))})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"PASS - {name} saved and parameters checked", flush=True)


if __name__ == "__main__":
    main()
