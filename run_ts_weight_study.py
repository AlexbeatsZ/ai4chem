"""One bounded TS-weight pilot pair, then a guarded complete MACE model."""

from collections import defaultdict
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from model_study_data import ROOT, save_json
from run_model_growth import now, valid_metrics

OUTPUT = ROOT / "results/ts-weight-20261126"


def normalized_weights(frame_ids, ts_ids, multiplier=2.0):
    raw = [multiplier if fid in ts_ids else 1.0 for fid in frame_ids]
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]


def promotes(weighted, control, baseline):
    for metrics in (weighted, control, baseline):
        valid_metrics(metrics)
    return (weighted["barrier_mae_eV"] < min(control["barrier_mae_eV"], baseline["barrier_mae_eV"])
            and all(weighted[key] <= min(control[key], baseline[key]) for key in
                    ("force_component_rmse_eV_per_A", "ts_region_force_component_rmse_eV_per_A")))


def prepare_dataset(size):
    """Read only frozen train reactions; preserve labels and weight each reaction."""
    import numpy as np
    import pyarrow.parquet as pq
    from ase.io import read, write
    from transition1x_data import final_band_indices

    directory = ROOT / "data/ts-weight-20261126"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"weighted{size}.extxyz"
    manifest_path = directory / f"weighted{size}.json"
    reuse = target.exists() and manifest_path.exists()
    if target.exists() != manifest_path.exists():
        raise FileExistsError("Incomplete weighted preparation; inspect without retry")
    if size == 512:
        source = ROOT / "data/v0.8.0/general4096.extxyz"
        manifest = json.loads((ROOT / "data/v0.8.0/manifest.json").read_text())
        identities = manifest["datasets"]["general4096"]
    else:
        source = ROOT / "data/v0.10.0/coverage2048.extxyz"
        manifest = json.loads((ROOT / "data/v0.10.0/manifest.json").read_text())
        identities = manifest["datasets"]["coverage2048"]
    expected_ids = set(manifest["partition"]["training_ids"])
    roles = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    forbidden = set().union(*(set(roles[key]) for key in
                              ("validation", "meta_development", "holdout_train", "holdout_val")))
    if expected_ids & forbidden or len(expected_ids) != size:
        raise ValueError("Weighted training source violates frozen role isolation")
    atoms = read(source, index=":")
    if len(atoms) != size * 8 or len(identities) != len(atoms):
        raise ValueError("Wrong weighted source frame count")
    grouped = defaultdict(list)
    for index, (structure, identity) in enumerate(zip(atoms, identities, strict=True)):
        sid, fid = int(identity["sid"]), int(identity["fid"])
        if structure.info["reaction"] != f"rxn{sid:04d}" or structure.info["source_frame"] != fid:
            raise ValueError("Weighted source frame identity differs")
        grouped[sid].append((index, fid))
    if set(grouped) != expected_ids or any(len(items) != 8 for items in grouped.values()):
        raise ValueError("Weighted dataset must contain eight frames per frozen reaction")
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    # Scalar columns only: identify complete final bands without decoding all geometries.
    table = pq.read_table(sources, columns=["sid", "fid", "y"],
                          filters=[("sid", "in", sorted(expected_ids))], use_threads=False)
    histories = defaultdict(dict)
    for sid, fid, energy in zip(table["sid"].to_pylist(), table["fid"].to_pylist(), table["y"].to_pylist(), strict=True):
        if fid in histories[sid] or not math.isfinite(energy):
            raise ValueError("Duplicated/nonfinite training reference history")
        histories[sid][fid] = energy
    records = []
    for sid in sorted(grouped):
        history = histories[sid]
        if set(history) != set(range(len(history))):
            raise ValueError("Incomplete training reference history")
        final = final_band_indices(len(history)).tolist()
        candidates = final if len(history) == 10 else final[1:-1]
        peak = max(candidates, key=lambda fid: history[fid])
        offset = final.index(peak)
        ts = set(final[max(0, offset-1):offset+2])
        weights = normalized_weights([fid for _, fid in grouped[sid]], ts)
        for (index, fid), weight in zip(grouped[sid], weights, strict=True):
            atoms[index].info["config_weight"] = weight
            records.append(dict(sid=sid, fid=fid, ts_region=fid in ts, weight=weight))
    evidence = dict(source=str(source.relative_to(ROOT)), frames=records,
                    holdout_labels_evaluated=False, maximum_multiplier=2, mean_weight_per_reaction=1)
    if reuse:
        if json.loads(manifest_path.read_text()) != evidence:
            raise ValueError("Prepared weighted source/identity changed")
    else:
        write(target, atoms)
    restored = read(target, index=":")
    if len(restored) != len(atoms):
        raise ValueError("Weighted readback frame count differs")
    from mace.data.utils import config_from_atoms, KeySpecification
    keys = KeySpecification(info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"})
    for original, saved in zip(atoms, restored, strict=True):
        np.testing.assert_allclose(saved.positions, original.positions, atol=1e-8, rtol=0)
        np.testing.assert_allclose(saved.info["REF_energy"], original.info["REF_energy"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(saved.arrays["REF_forces"], original.arrays["REF_forces"], atol=1e-8, rtol=0)
        config = config_from_atoms(saved, key_specification=keys)
        if not math.isclose(config.weight, original.info["config_weight"], abs_tol=1e-12):
            raise ValueError("Official MACE loader did not preserve frame weights")
    save_json(manifest_path, evidence, immutable=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        files = [str(prepare_dataset(size).relative_to(ROOT)) for size in (512, 2048)]
        save_json(OUTPUT / "preparation.json", dict(status="completed", files=files,
                  finished_at_utc=now(), holdout_labels_evaluated=False))
        print("PASS - weighted train labels read back through official MACE loader", flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "state.json").exists():
        raise FileExistsError("TS study already submitted; no implicit retry")
    protocol = json.loads((ROOT / "configs/compute-20261126.json").read_text())
    state = dict(owner_pid=os.getpid(), status="waiting_for_growth", attempts={},
                 started_at_utc=now(), holdout_labels_evaluated=False, automatic_retries=0)

    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save_json(OUTPUT / "state.json", state)

    update("waiting_for_growth")
    predecessor = (ROOT / "results/v0.10.0/runner.lock").open("a")
    deadline = time.monotonic() + protocol["omen"]["wait_hours"] * 3600
    while True:
        try:
            fcntl.flock(predecessor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                update("failed", status="failed", error="Growth predecessor wait expired")
                return
            time.sleep(30)
    try:
        growth = json.loads((ROOT / "results/v0.10.0/state.json").read_text())
        if growth["stage"] not in ("growth_training_and_repeats_complete", "no_guarded_improvement", "incomplete_seed_repeats"):
            raise RuntimeError("Growth predecessor has no scientific terminal state")
        from run_model_growth import wait_for_cuda
        cuda_locks = wait_for_cuda({"predecessor_wait_hours": 72}, update)
        import torch
        from run_model_study import train, benchmark
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("Accepted OMEN CUDA environment unavailable")
        base = json.loads((ROOT / "configs/v0.10.0.json").read_text())
        weighted_file = prepare_dataset(512)

        def attempt(name, dataset, size, epochs):
            record = dict(status="running", started_at_utc=now())
            state["attempts"][name] = record
            update("starting_model", status="running", experiment=name)
            experiment = dict(name=name, dataset="ts-weight", train_size=size*8,
                              train_file=str(dataset.relative_to(ROOT)), seed=20261116,
                              model_file=f"models/ts-weight-20261126/{name}.model")
            settings = {**base, "training": {**base["training"], "epochs": epochs}}
            runs = []
            try:
                train(experiment, settings, runs, output_root=OUTPUT,
                      manifest_path=OUTPUT / f"training/{name}/manifest.json", state_writer=update)
                if runs[0]["trained_epochs"] != epochs:
                    raise RuntimeError("Actual training epochs differ from frozen budget")
                record["run"] = runs[0]
                record["validation"] = benchmark(f"validation/{name}", [name],
                    {name: str(ROOT / experiment["model_file"])}, output_root=OUTPUT)["models"][name]["aggregate"]
                valid_metrics(record["validation"])
                record.update(status="completed", finished_at_utc=now())
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
            update("model_terminal")
            return record

        control = attempt("uniform512_e8", ROOT / "data/v0.8.0/general4096.extxyz", 512, 8)
        weighted = attempt("tsweighted512_e8", weighted_file, 512, 8)
        eligible = all(x["status"] == "completed" for x in (control, weighted)) and promotes(
            weighted["validation"], control["validation"], base["baseline"]["validation_metrics"])
        decision = dict(promoted=eligible, frozen_at_utc=now(), criterion=protocol["omen"]["promotion"])
        save_json(OUTPUT / "pilot-decision.json", decision, immutable=True)
        if eligible:
            full_file = prepare_dataset(2048)
            full = attempt("tsweighted2048_e20", full_file, 2048, 20)
            decision["full_model_passes_growth_guard"] = full["status"] == "completed" and promotes(
                full["validation"], growth["attempts"]["mh1_coverage2048_s0"]["validation"],
                base["baseline"]["validation_metrics"])
        state["decision"] = decision
        update("completed", status="completed", finished_at_utc=now())
        save_json(OUTPUT / "summary.json", state)
    except Exception as error:
        update("failed", status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        raise


if __name__ == "__main__":
    main()
