"""Fit the frozen v0.11 external-mixture pilot on existing paired predictions."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
from sklearn.model_selection import GroupKFold

from expert_routing import grouped_delta_interval
from soft_mixture import SCHEMA, FEATURE_NAMES, composition_features, fit_gate, gate_weights
from transition1x_data import load_paths


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def role_records(partition, role):
    if role not in ("meta_development", "validation"):
        raise ValueError("Pilot refuses to open any other data role")
    if role == "validation":
        # v0.7 froze only validation IDs here; endpoint records were stored for
        # meta and unopened prospective roles. Geometry is checked against HDF5.
        records = [{"sid": sid, "reaction": f"rxn{sid:04d}"} for sid in sorted(partition[role])]
    else:
        records = sorted(partition["endpoint_records"][role], key=lambda row: row["sid"])
    if [row["sid"] for row in records] != sorted(partition[role]):
        raise ValueError("Endpoint identity differs from the frozen role")
    forbidden = set(partition["holdout_train"]) | set(partition["holdout_val"])
    if any(row["sid"] in forbidden for row in records):
        raise ValueError("Prospective reaction leaked into the mixture pilot")
    return records


@dataclass
class CachedPaths:
    records: list
    numbers: list
    features: np.ndarray
    energy: np.ndarray
    reference_energy: np.ndarray
    force_gram: np.ndarray
    ts_force_gram: np.ndarray
    force_count: np.ndarray
    ts_force_count: np.ndarray
    fit_gram: np.ndarray
    sources: list


def load_cached_paths(root, protocol, partition, role, progress=None):
    records = role_records(partition, role)
    expected_count = protocol["data"]["gate_fit_reactions" if role == "meta_development" else "development_check_reactions"]
    if len(records) != expected_count:
        raise ValueError("Role count differs from the frozen pilot")
    paths = load_paths(root/f"data/v0.7.0/{role}.h5", "train" if role == "meta_development" else "val", len(records))
    lookup = {path.reaction: path for path in paths}
    if set(lookup) != {entry["reaction"] for entry in records}:
        raise ValueError("Role HDF5 reaction identities differ")
    members, frames = protocol["members"], protocol["data"]["frames_per_reaction"]
    count, models = len(records), len(members)
    energy = np.empty((count, frames, models))
    reference_energy = np.empty((count, frames))
    force_gram = np.empty((count, models, models))
    ts_gram = np.empty_like(force_gram)
    force_count, ts_count = np.empty(count), np.empty(count)
    sources, all_numbers = [], []
    for index, entry in enumerate(records):
        path = lookup[entry["reaction"]]
        if path.formula != entry.get("formula", path.formula) or not np.array_equal(path.numbers, entry.get("atomic_numbers", path.numbers)):
            raise ValueError("Role endpoint formula or atomic ordering differs")
        reference_energy[index] = path.energies
        errors = []
        for model_index, member in enumerate(members):
            name = member["name"]
            prediction = root/f"results/v0.8.0/{role}/{name}/{name}_{path.reaction}.extxyz"
            images = read(prediction, index=":")
            if len(images) != frames or len(path.energies) != frames:
                raise ValueError("Cached path must contain the entire frozen ten-image band")
            for image_index, atoms in enumerate(images):
                if atoms.info.get("reaction") != path.reaction or atoms.info.get("formula") != path.formula:
                    raise ValueError("Cached reaction/formula identity differs")
                np.testing.assert_array_equal(atoms.numbers, path.numbers)
                np.testing.assert_allclose(atoms.positions, path.positions[image_index], atol=1e-8, rtol=0)
                np.testing.assert_allclose(atoms.info["reference_energy_eV"], path.energies[image_index], atol=1e-8, rtol=0)
                np.testing.assert_allclose(atoms.arrays["reference_forces"], path.forces[image_index], atol=1e-8, rtol=0)
            energies = np.array([atoms.get_potential_energy() for atoms in images])
            forces = np.array([atoms.get_forces() for atoms in images])
            if forces.shape != path.forces.shape or not np.isfinite(energies).all() or not np.isfinite(forces).all():
                raise ValueError("Nonfinite or incomplete cached energy/force predictions")
            energy[index, :, model_index] = energies
            errors.append(forces-path.forces)
            sources.append({"path": str(prediction.relative_to(root)), "bytes": prediction.stat().st_size})
        errors = np.moveaxis(np.array(errors), 0, -1)
        flat = errors.reshape(-1, models)
        ts = path.reference_ts_image
        ts_flat = errors[max(0, ts-1):min(frames, ts+2)].reshape(-1, models)
        force_gram[index], ts_gram[index] = flat.T @ flat, ts_flat.T @ ts_flat
        force_count[index], ts_count[index] = len(flat), len(ts_flat)
        all_numbers.append(path.numbers)
        entry.update(formula=path.formula)
        if progress and (index+1) % 32 == 0:
            progress(role=role, reactions_loaded=index+1)
    atoms = np.array([len(numbers) for numbers in all_numbers])
    energy_error = (energy-reference_energy[:, :, None])/atoms[:, None, None]
    fit_gram = protocol["objective"]["energy_per_atom_weight"]*np.einsum("rtm,rtn->rmn", energy_error, energy_error)/frames
    fit_gram += protocol["objective"]["force_component_weight"]*force_gram/force_count[:, None, None]
    features = np.array([composition_features(numbers) for numbers in all_numbers])
    return CachedPaths(records, all_numbers, features, energy, reference_energy, force_gram, ts_gram,
                       force_count, ts_count, fit_gram, sources)


def mixture_metrics(data, weights):
    weights = np.asarray(weights, float)
    if weights.shape != (len(data.records), data.energy.shape[-1]) or not np.isfinite(weights).all():
        raise ValueError("Mixture weights differ from paired path dimensions")
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1., atol=1e-10, rtol=0):
        raise ValueError("Each reaction needs one convex mixture for the entire path")
    prediction = np.einsum("rtm,rm->rt", data.energy, weights)
    relative = prediction-prediction[:, :1]
    reference = data.reference_energy-data.reference_energy[:, :1]
    barrier_errors = relative.max(axis=1)-reference.max(axis=1)
    force_sse = np.einsum("ri,rij,rj->r", weights, data.force_gram, weights)
    ts_sse = np.einsum("ri,rij,rj->r", weights, data.ts_force_gram, weights)
    force_sse, ts_sse = np.maximum(force_sse, 0), np.maximum(ts_sse, 0)
    return {"barrier_MAE_eV": float(np.abs(barrier_errors).mean()),
        "barrier_absolute_error_p90_eV": float(np.quantile(np.abs(barrier_errors), .9)),
        "force_component_RMSE_eV_per_A": float(np.sqrt(force_sse.sum()/data.force_count.sum())),
        "ts_force_component_RMSE_eV_per_A": float(np.sqrt(ts_sse.sum()/data.ts_force_count.sum())),
        "relative_energy_RMSE_eV": float(np.sqrt(np.mean((relative-reference)**2))),
        "reaction_energy_MAE_eV": float(np.abs(relative[:, -1]-reference[:, -1]).mean()),
        "mean_weights": weights.mean(axis=0).tolist(),
        "reaction_weights": weights.tolist(), "reaction_barrier_errors_eV": barrier_errors.tolist()}


def constant_gate(members, weights):
    return {"schema": SCHEMA, "kind": "constant", "feature_names": FEATURE_NAMES,
            "members": members, "weights": list(weights)}


def static_guard(candidate, baseline):
    return bool(candidate["barrier_MAE_eV"] < baseline["barrier_MAE_eV"] and all(
        candidate[key] <= baseline[key]+1e-10 for key in (
            "force_component_RMSE_eV_per_A", "ts_force_component_RMSE_eV_per_A")))


def run(protocol, output, update):
    started = time.monotonic()
    partition = json.loads((ROOT/"configs/v0.7.0_partitions.json").read_text())
    records = role_records(partition, "meta_development")
    groups = np.array([entry["formula"] for entry in records])
    splitter = GroupKFold(n_splits=protocol["folds"], shuffle=True, random_state=protocol["seed"])
    folds = list(splitter.split(np.zeros(len(records)), groups=groups))
    frozen = {"frozen_at_utc": now(), "protocol": protocol,
        "fit_reactions": [r["reaction"] for r in records],
        "validation_reactions": [r["reaction"] for r in role_records(partition, "validation")],
        "folds": [{"train": train.tolist(), "test": test.tolist()} for train, test in folds],
        "holdout_labels_evaluated": False}
    save(output/"frozen_protocol.json", frozen)
    for member in protocol["members"]:
        if (ROOT/member["model_file"]).stat().st_size != member["size_bytes"]:
            raise ValueError(f"Frozen model size differs: {member['name']}")
    fit = load_cached_paths(ROOT, protocol, partition, "meta_development", lambda **extra: update("loading_cache", **extra))
    validation = load_cached_paths(ROOT, protocol, partition, "validation", lambda **extra: update("loading_cache", **extra))
    count, members = len(records), protocol["members"]
    baseline = np.zeros((count, len(members))); baseline[:, 0] = 1
    oof_weights = {"single_general": baseline, "uniform": np.full_like(baseline, 1/len(members))}
    optimization = []
    fitted = {}
    for kind in ("constant", "composition_softmax"):
        oof_weights[kind] = np.empty_like(baseline)
        for fold_index, (train, test) in enumerate(folds):
            if set(groups[train]) & set(groups[test]):
                raise ValueError("Formula leakage across gate folds")
            if time.monotonic()-started > protocol["budget"]["max_wall_seconds"]:
                raise TimeoutError("Mixture pilot exceeded its frozen wall budget")
            update("fitting", gate=kind, fold=fold_index)
            gate = fit_gate(kind, fit.features[train], fit.fit_gram[train], members, protocol["optimizer"])
            oof_weights[kind][test] = [gate_weights(gate, fit.numbers[i]) for i in test]
            optimization.append({"kind": kind, "fold": fold_index, **gate["optimization"]})
            save(output/f"folds/{kind}-{fold_index}.json", gate)
        gate = fit_gate(kind, fit.features, fit.fit_gram, members, protocol["optimizer"])
        gate.update(fitted_at_utc=now(), fit_role="meta_development", fit_reactions=count,
            prospective_labels_used=False, interpretation="Frozen external mixture; all three MACE members are evaluated")
        fitted[kind] = gate
        optimization.append({"kind": kind, "fold": "all_meta", **gate["optimization"]})
        save(output/f"models/{kind}.json", gate)
    fitted["single_general"] = constant_gate(members, [1., 0., 0.])
    fitted["uniform"] = constant_gate(members, [1/len(members)]*len(members))
    result = {"version": protocol["version"], "status": "completed", "finished_at_utc": now(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": protocol["data"]["roles"], "holdout_labels_evaluated": False, "protocol": protocol,
        "optimization": optimization, "oof": {}, "validation": {}, "promotion": {},
        "new_GPU_training_hours": 0, "new_DFT_labels": 0,
        "member_bytes": sum(member["size_bytes"] for member in members),
        "inference_member_evaluations": 3, "wall_seconds": time.monotonic()-started,
        "runtime_acceptance": "pending real MACE calculator verification"}
    for name in protocol["controls"]:
        result["oof"][name] = mixture_metrics(fit, oof_weights[name])
        valid_weights = np.array([gate_weights(fitted[name], numbers) for numbers in validation.numbers])
        result["validation"][name] = mixture_metrics(validation, valid_weights)
    # Reconstruct published single-model metrics before interpreting any gain.
    for role, metrics in (("meta_development", result["oof"]["single_general"]),
                          ("validation", result["validation"]["single_general"])):
        old = json.loads((ROOT/f"results/v0.8.0/{role}/{members[0]['name']}/benchmark_summary.json").read_text())
        published = old["models"][members[0]["name"]]["aggregate"]
        for key, old_key in (("barrier_MAE_eV", "barrier_mae_eV"),
                             ("force_component_RMSE_eV_per_A", "force_component_rmse_eV_per_A"),
                             ("ts_force_component_RMSE_eV_per_A", "ts_region_force_component_rmse_eV_per_A")):
            if not np.isclose(metrics[key], published[old_key], atol=2e-7, rtol=0):
                raise ValueError(f"Cached baseline failed to reproduce {role}/{key}")
    base_errors = np.abs(result["oof"]["single_general"]["reaction_barrier_errors_eV"])
    for name in protocol["controls"]:
        metrics = result["oof"][name]
        interval = grouped_delta_interval(np.abs(metrics["reaction_barrier_errors_eV"])-base_errors, groups,
                                          seed=protocol["seed"])
        metrics["paired_barrier_delta_vs_general"] = interval
        oof_pass = static_guard(metrics, result["oof"]["single_general"])
        val_pass = static_guard(result["validation"][name], result["validation"]["single_general"])
        result["promotion"][name] = {"oof_multi_metric_pass": oof_pass, "validation_multi_metric_pass": val_pass,
            "oof_ci_upper_below_zero": interval["formula_cluster_ci95_eV"][1] < 0,
            "eligible_for_cost_search_assessment": bool(oof_pass and val_pass and interval["formula_cluster_ci95_eV"][1] < 0)}
    save(output/"cache_sources.json", {"fit": fit.sources, "validation": validation.sources})
    save(output/"summary.json", result)
    update("completed", elapsed_seconds=result["wall_seconds"], promotion=result["promotion"])
    print(json.dumps({"status": "completed", "wall_seconds": result["wall_seconds"], "promotion": result["promotion"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/v0.11.0.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = ROOT/"results/v0.11.0"
    output.mkdir(parents=True, exist_ok=True)
    with (output/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/"state.json").exists():
            raise FileExistsError("Pilot already has evidence; inspect it instead of overwriting or retrying")
        if hasattr(os, "nice"):
            os.nice(10)
        state = {"owner_pid": os.getpid(), "host": "Mac", "started_at_utc": now(), "holdout_labels_evaluated": False}
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
