"""Endpoint-only, fixed-budget search across three different potential families."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from external_experts import external_calculator
from model_study_data import save_json
from ts_search import (numerical_hessian, projected_frequencies, relax_endpoint,
                       search_one)

PROTOCOL = ROOT / "configs/v0.7.0_diversity_search.json"
OUTPUT = ROOT / "results/v0.7.0/diversity_search"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=[
        "legacy_mh1_256", "aimnet2-rxn", "orbmol-conservative", "global512", "coverage2048s0"])
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--max-hours', type=float)
    args = parser.parse_args()
    began_worker = time.monotonic()
    if args.max_hours is not None and not 0 < args.max_hours <= 3:
        parser.error('Search admission budget must be within3hours')
    torch.set_num_threads(3)
    protocol = json.loads(args.protocol.read_text())
    if args.expert not in protocol['experts']:
        raise ValueError('Expert not in the frozen search protocol')
    config = protocol["search_settings"]
    source = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    allowed = set(source["meta_development"])
    if any(row["sid"] not in allowed for row in protocol["endpoint_records"]):
        raise ValueError("Search inputs must belong to the frozen meta-development pool")
    directory = args.output / args.expert
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"expert": args.expert, "host": os.uname().nodename,
        "device": args.device,
        "owner_pid": os.getpid(), "started_at_utc": now(), "stage": "loading_model",
        "protocol": protocol, "holdout_labels_evaluated": False,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    save_json(directory / "state.json", state)
    try:
        if args.expert in ("legacy_mh1_256", "global512", "coverage2048s0"):
            from mace.calculators import MACECalculator
            checkpoints = {'legacy_mh1_256':'models/v0.3.0/mh1_full_256.model',
                           'global512':'models/v0.8.0/mh1_global512_e40.model',
                           'coverage2048s0':'models/v0.10.0/mh1_coverage2048_s0.model'}
            calc = MACECalculator(model_paths=str(ROOT / checkpoints[args.expert]),
                                  device=args.device, default_dtype="float64", head="Default")
        else:
            calc = external_calculator(args.expert, args.device, "float64")
        completed = []
        for entry in protocol["endpoint_records"]:
            if ((args.deadline_utc and datetime.now(timezone.utc) >= datetime.fromisoformat(args.deadline_utc)) or
                (args.max_hours is not None and time.monotonic()-began_worker >= args.max_hours*3600)):
                state.update(stage='admission_deadline_reached', completed=completed, finished_at_utc=now())
                save_json(directory/'state.json',state)
                return
            folder = directory / entry["reaction"]
            state.update(stage="searching", reaction=entry["reaction"], updated_at_utc=now(), completed=completed)
            save_json(directory / "state.json", state)
            if (folder / "pipeline.json").exists():
                cached = json.loads((folder / "pipeline.json").read_text())
                if cached["protocol"] != protocol or cached["expert"] != args.expert or cached.get('device','cpu') != args.device:
                    raise ValueError("Cached search identity differs")
                completed.append(entry["reaction"])
                continue
            if folder.exists():
                raise RuntimeError(f"Interrupted search has evidence; inspect before retry: {folder}")
            folder.mkdir()
            began = time.perf_counter()
            try:
                item = {**entry, "endpoints": [Atoms(entry["atomic_numbers"], positions=entry[key])
                    for key in ("reactant_positions", "product_positions")]}
                relaxed, checks = [], []
                for label, atoms in zip(("reactant", "product"), item["endpoints"]):
                    atoms.info.update(charge=0, spin=1)
                    write(folder / f"{label}_input.xyz", atoms)
                    endpoint, check = relax_endpoint(atoms, calc, folder, label, config)
                    relaxed.append(endpoint)
                    checks.append(check)
                gate = {"passed": all(x["converged"] and x["input_bonds_preserved"] for x in checks),
                        "endpoints": checks, "elapsed_seconds": time.perf_counter()-began}
                save_json(folder / "endpoint_gate.json", gate)
                if gate["passed"]:
                    result = search_one(item, calc, folder, config, prepared_endpoints=(relaxed, checks))
                else:
                    result = {"status": "endpoint_gate_rejected", "mlip_verified": False,
                              "saddle_irc_verified": False, "endpoints": checks}
                minima = []
                if result.get("saddle_irc_verified", False):
                    for direction in ("forward", "reverse"):
                        atoms = read(folder / f"irc_{direction}_minimum.extxyz")
                        atoms.calc = calc
                        hessian = numerical_hessian(atoms, config["hessian_delta_A"])
                        frequencies, _ = projected_frequencies(atoms, hessian)
                        np.savez(folder / f"irc_{direction}_minimum_hessian.npz", hessian=hessian,
                                 frequencies_cm1=frequencies)
                        minima.append({"direction": direction, "frequencies_cm1": frequencies.tolist(),
                            "significant_negative_modes": int(np.sum(frequencies < -config["imaginary_threshold_cm1"]))})
                pipeline = {"expert": args.expert, "reaction": entry["reaction"], "formula": entry["formula"],
                    "device": args.device,
                    "protocol": protocol, "result": result, "endpoint_gate": gate, "minima": minima,
                    "strict_mlip_success": bool(result.get("mlip_verified", False) and len(minima) == 2
                        and all(x["significant_negative_modes"] == 0 for x in minima)),
                    "saddle_irc_and_stable_minima": bool(result.get("saddle_irc_verified", False) and len(minima) == 2
                        and all(x["significant_negative_modes"] == 0 for x in minima)),
                    "pipeline_elapsed_seconds": time.perf_counter()-began, "finished_at_utc": now(),
                    "scope": "MLIP-only development; no DFT verification or prospective holdout claim"}
            except Exception as error:
                (folder / "pipeline_error.txt").write_text(traceback.format_exc())
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                pipeline = {"expert": args.expert, "reaction": entry["reaction"], "formula": entry["formula"],
                    "device": args.device, "protocol": protocol, "result": result, "strict_mlip_success": False,
                    "saddle_irc_and_stable_minima": False, "pipeline_elapsed_seconds": time.perf_counter()-began,
                    "finished_at_utc": now(), "scope": "failed MLIP-only development unit; retained in denominator"}
            save_json(folder / "pipeline.json", pipeline)
            completed.append(entry["reaction"])
            print(args.expert, entry["reaction"], result["status"], pipeline["strict_mlip_success"], flush=True)
        state.update(stage="completed", completed=completed, finished_at_utc=now())
        save_json(directory / "state.json", state)
    except Exception as error:
        state.update(stage="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        save_json(directory / "state.json", state)
        raise


if __name__ == "__main__":
    main()
