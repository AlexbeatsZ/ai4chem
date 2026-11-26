"""Run exactly two bounded, matched adapter trainings on the frozen MACE trunk."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from residual_moe import SharedResidualMoE, ResidualMoECalculator
from transition1x_data import load_paths


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluate(model, provider, paths, deadline, progress):
    calc = ResidualMoECalculator(model, provider)
    records, force_sse, ts_sse, force_count, ts_count = [], 0., 0., 0, 0
    for index, path in enumerate(paths):
        if time.monotonic() > deadline:
            raise TimeoutError("Residual pilot exhausted its wall budget")
        energy, forces = [], []
        for atoms in path.atoms():
            atoms.calc = calc
            energy.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
        relative = np.asarray(energy)-energy[0]
        reference = path.energies-path.energies[0]
        errors = np.asarray(forces)-path.forces
        ts = path.reference_ts_image
        ts_errors = errors[max(0, ts-1):min(10, ts+2)]
        force_sse += np.square(errors).sum(); force_count += errors.size
        ts_sse += np.square(ts_errors).sum(); ts_count += ts_errors.size
        records.append({"reaction": path.reaction, "formula": path.formula,
            "barrier_error_eV": float(relative.max()-reference.max()),
            "force_component_RMSE_eV_per_A": float(np.sqrt(np.mean(errors**2))),
            "ts_force_component_RMSE_eV_per_A": float(np.sqrt(np.mean(ts_errors**2)))})
        if (index+1) % 10 == 0:
            progress(validation_reactions=index+1)
    return {"barrier_MAE_eV": float(np.mean([abs(r["barrier_error_eV"]) for r in records])),
        "force_component_RMSE_eV_per_A": float(np.sqrt(force_sse/force_count)),
        "ts_force_component_RMSE_eV_per_A": float(np.sqrt(ts_sse/ts_count)), "reaction_metrics": records}


def guard(candidate, baseline):
    return bool(candidate["barrier_MAE_eV"] < baseline["barrier_MAE_eV"] and
        all(candidate[key] <= baseline[key]+1e-10 for key in (
            "force_component_RMSE_eV_per_A", "ts_force_component_RMSE_eV_per_A")))


def run(protocol, output, update):
    start = time.monotonic(); deadline = start+protocol["budget"]["max_wall_seconds"]
    acceptance = json.loads((ROOT/"results/v0.11.0/residual_interface/acceptance.json").read_text())
    if acceptance["status"] != "passed":
        raise RuntimeError("Real force-gradient acceptance is required")
    queue = ROOT/"results/overnight-20261124/mac"
    if json.loads((queue/"state.json").read_text())["status"] != "completed":
        raise RuntimeError("Do not compete with the existing Mac paired queue")
    with (queue/"owner.lock").open("a") as predecessor:
        fcntl.flock(predecessor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_partition = json.loads((ROOT/"configs/v0.7.0_partitions.json").read_text())
    partition = json.loads((ROOT/"configs/v0.8.0_partitions.json").read_text())
    rng = np.random.default_rng(protocol["seed"])
    chosen = sorted(int(sid) for region in ("0", "1") for sid in
                    rng.choice(sorted(partition["regions"][region]), protocol["training_reactions"]//2, replace=False))
    forbidden = set().union(*(set(old_partition[key]) for key in ("validation", "meta_development", "holdout_train", "holdout_val")))
    if len(set(chosen)) != protocol["training_reactions"] or set(chosen) & forbidden or not set(chosen) <= set(partition["training_ids"]):
        raise ValueError("Residual training identity violates frozen roles")
    save(output/"frozen_protocol.json", {"frozen_at_utc": now(), "protocol": protocol,
        "training_ids": chosen, "validation_ids": old_partition["validation"], "labels_used_for_selection": False})
    all_frames = read(ROOT/protocol["train_file"], index=":")
    chosen_names = {f"rxn{sid:04d}" for sid in chosen}
    frames = [frame for frame in all_frames if frame.info["reaction"] in chosen_names]
    from collections import Counter
    if set(Counter(f.info["reaction"] for f in frames).values()) != {protocol["frames_per_reaction"]} or len(frames) != len(chosen)*protocol["frames_per_reaction"]:
        raise ValueError("Training labels differ from the frozen frame budget")
    base = ROOT/protocol["base_model"]
    if base.stat().st_size != protocol["base_model_bytes"]:
        raise ValueError("Residual baseline weight changed")
    provider = MACECalculator(model_paths=str(base), device="cpu", default_dtype="float64")
    validation = load_paths(ROOT/"data/v0.7.0/validation.h5", "val", 60)
    baseline = json.loads((ROOT/"results/v0.11.0/summary.json").read_text())["validation"]["single_general"]
    results = {}
    train_settings, architecture = protocol["training"], protocol["architecture"]
    for kind in protocol["experiments"]:
        torch.manual_seed(protocol["seed"])
        model = SharedResidualMoE(provider.models[0], kind, architecture["width_per_branch"], architecture["routed_experts"])
        optimizer = torch.optim.Adam(model.adapter_parameters(), lr=train_settings["learning_rate"])
        history, step_count = [], 0
        order_rng = np.random.default_rng(protocol["seed"])
        train_start = time.monotonic()
        for epoch in range(train_settings["epochs"]):
            order = order_rng.permutation(len(frames))
            for begin in range(0, len(order), train_settings["batch_size"]):
                if time.monotonic() > deadline:
                    raise TimeoutError("Residual pilot exhausted its wall budget")
                batch_indices = order[begin:begin+train_settings["batch_size"]]
                optimizer.zero_grad()
                batch_loss = 0.
                for frame_index in batch_indices:
                    frame = frames[frame_index]
                    prediction = model(provider._atoms_to_batch(frame).to_dict(), training=True)
                    reference_force = torch.as_tensor(frame.arrays["REF_forces"])
                    loss = train_settings["energy_per_atom_weight"]*((prediction["energy"].sum()-float(frame.info["REF_energy"]))/len(frame))**2
                    loss = loss+train_settings["force_component_weight"]*torch.mean((prediction["forces"]-reference_force)**2)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite residual training loss")
                    (loss/len(batch_indices)).backward()
                    batch_loss += float(loss.detach())/len(batch_indices)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.adapter_parameters()):
                    raise ValueError("Nonfinite residual gradient")
                optimizer.step(); step_count += 1
                history.append({"step": step_count, "epoch": epoch+1, "loss": batch_loss})
                if step_count % 16 == 0:
                    save(output/f"{kind}-training.json", history)
                    update("training", experiment=kind, epoch=epoch+1, updates=step_count,
                           elapsed_seconds=time.monotonic()-start)
        if step_count != protocol["budget"]["updates_per_model"]:
            raise ValueError("Residual updates differ from the fixed budget")
        model.save_adapter(output/f"{kind}.pt", protocol["base_model"])
        save(output/f"{kind}-training.json", history)
        train_seconds = time.monotonic()-train_start
        restored = SharedResidualMoE.load_adapter(output/f"{kind}.pt", provider.models[0])
        update("validation", experiment=kind, updates=step_count)
        metrics = evaluate(restored, provider, validation, deadline, lambda **extra: update("validation", experiment=kind, **extra))
        results[kind] = {"metrics": metrics, "updates": step_count, "epochs": train_settings["epochs"],
            "training_seconds": train_seconds, "adapter_parameters": sum(p.numel() for p in model.adapter_parameters()),
            "adapter_file": str((output/f"{kind}.pt").relative_to(ROOT)), "adapter_bytes": (output/f"{kind}.pt").stat().st_size,
            "baseline_multi_metric_pass": guard(metrics, baseline)}
        save(output/f"{kind}-result.json", results[kind])
    expert, single = results["shared_experts"], results["single_head"]
    result = {"status": "completed", "finished_at_utc": now(), "protocol": protocol,
        "scope": "Two-epoch small-subset adapter pilot on already trained global512; validation60 is development, not an independent test",
        "experiments": results, "baseline": {key: baseline[key] for key in (
            "barrier_MAE_eV", "force_component_RMSE_eV_per_A", "ts_force_component_RMSE_eV_per_A")},
        "eligible_for_larger_matched_training": expert["baseline_multi_metric_pass"] and guard(expert["metrics"], single["metrics"]),
        "wall_seconds": time.monotonic()-start, "new_GPU_training_hours": 0, "new_DFT_labels": 0,
        "holdout_labels_evaluated": False, "backbone_updated": False, "automatic_retries": 0}
    save(output/"summary.json", result)
    update("completed", eligible_for_larger_matched_training=result["eligible_for_larger_matched_training"],
           elapsed_seconds=result["wall_seconds"])
    print(json.dumps({"status": "completed", "wall_seconds": result["wall_seconds"],
        "eligible_for_larger_matched_training": result["eligible_for_larger_matched_training"]}), flush=True)


def main():
    protocol = json.loads((ROOT/"configs/residual-moe-pilot-20261125.json").read_text())
    output = ROOT/"results/v0.11.0/residual_pilot"
    output.mkdir(parents=True, exist_ok=True)
    with (output/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/"state.json").exists():
            raise FileExistsError("Existing training evidence must not be overwritten or automatically retried")
        torch.set_num_threads(protocol["budget"]["threads"])
        torch.set_default_dtype(torch.float64)
        os.nice(protocol["budget"]["nice"])
        state = {"owner_pid": os.getpid(), "started_at_utc": now(), "holdout_labels_evaluated": False}
        def update(stage, **extra):
            state.update(stage=stage, updated_at_utc=now(), **extra)
            save(output/"state.json", state)
        update("started")
        try:
            run(protocol, output, update)
        except Exception as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
