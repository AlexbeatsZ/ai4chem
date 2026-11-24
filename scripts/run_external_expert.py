"""Accept and benchmark one additional family without opening prospective holdouts."""

import argparse
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
import torch
from external_experts import EXTERNAL_MODELS, external_calculator, external_metadata
from model_study_data import save_json
from transition1x_data import load_paths


def now():
    return datetime.now(timezone.utc).isoformat()


def acceptance(name, directory):
    path = load_paths(ROOT / "data/v0.7.0/meta_development.h5", "train", 128)[0]
    started = time.perf_counter()
    calc = external_calculator(name, "cpu", "float64")
    metadata = external_metadata(name, calc, "cpu")
    records = []
    rng = np.random.default_rng(20261123)
    for image in (0, path.reference_ts_image):
        atoms = path.atoms()[image]
        atoms.info.update(charge=0, spin=1)
        atoms.calc = calc
        energy, forces = float(atoms.get_potential_energy()), atoms.get_forces()
        if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError("Non-finite energy/forces or wrong force dimensions")
        direction = rng.normal(size=atoms.positions.shape)
        direction /= np.linalg.norm(direction)
        analytic = -float(np.sum(forces * direction))
        differences = []
        for step in (.001, .003):
            values = []
            for sign in (1, -1):
                displaced = atoms.copy()
                displaced.positions += sign * step * direction
                displaced.calc = calc
                values.append(float(displaced.get_potential_energy()))
            numeric = (values[0] - values[1]) / (2 * step)
            differences.append({"step_A": step, "directional_dE_dR_eV_per_A": numeric,
                "absolute_gradient_difference_eV_per_A": abs(numeric-analytic),
                "passed": abs(numeric-analytic) <= .02 + .01*abs(analytic)})
        records.append({"reaction": path.reaction, "image": image, "energy_eV": energy,
            "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
            "negative_force_direction_projection_eV_per_A": analytic,
            "net_force_norm_eV_per_A": float(np.linalg.norm(forces.sum(axis=0))),
            "finite_difference": differences})
    result = {"model": name, "host": os.uname().nodename, "status": "passed" if all(
        row["passed"] for probe in records for row in probe["finite_difference"]) else "gradient_check_failed",
        "finished_at_utc": now(), "elapsed_seconds": time.perf_counter()-started,
        "scope": "native CPU real weights, one meta-development reactant and TS; not chemistry accuracy",
        "dtype_request": "float64; AIMNet2 keeps native precision", "metadata": metadata,
        "gradient_tolerance": "absolute 0.02 eV/A + 1% directional derivative, at both displacements",
        "probes": records, "holdout_labels_evaluated": False}
    save_json(directory / "acceptance.json", result)
    if result["status"] != "passed":
        raise RuntimeError("Energy-force consistency probe failed; preserve evidence before using in search")
    del calc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(EXTERNAL_MODELS))
    parser.add_argument("--acceptance-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    directory = ROOT / "results/v0.7.0/expert_diversity" / args.model
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / "state.json"
    record = {"model": args.model, "host": os.uname().nodename, "owner_pid": os.getpid(),
        "started_at_utc": now(), "stage": "acceptance", "holdout_labels_evaluated": False}
    save_json(state_path, record)
    try:
        accepted = directory / "acceptance.json"
        if not accepted.exists():
            acceptance(args.model, directory)
        elif json.loads(accepted.read_text())["status"] != "passed":
            raise RuntimeError("Existing failed acceptance needs explicit investigation")
        if not args.acceptance_only:
            for role, split, count in (("meta_development", "train", 128), ("validation", "val", 60)):
                output = directory / role
                record.update(stage=role, updated_at_utc=now())
                save_json(state_path, record)
                if (output / "benchmark_summary.json").exists():
                    cached = json.loads((output / "benchmark_summary.json").read_text())
                    if cached["reactions"] != count or set(cached["models"]) != {args.model}:
                        raise ValueError("Existing external benchmark identity mismatch")
                    continue
                output.mkdir(parents=True, exist_ok=True)
                with (output / "console.log").open("w") as log:
                    subprocess.run([sys.executable, "-u", "benchmark_transition1x.py", "--input",
                        str(ROOT / "data/v0.7.0" / f"{role}.h5"), "--split", split,
                        "--count", str(count), "--output", str(output), "--models", args.model,
                        "--device", "cpu", "--dtype", "float64", "--plot-limit", "2"], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
        record.update(stage="acceptance_complete" if args.acceptance_only else "completed", finished_at_utc=now())
        save_json(state_path, record)
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(stage="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        save_json(state_path, record)
        raise


if __name__ == "__main__":
    main()
