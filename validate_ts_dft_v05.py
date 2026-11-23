"""Evaluate all nine preselected strategy candidates, reusing identical TS jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from ase.io import read

from adaptive_ts_search import digest
from prepare_v05_reactions import ROOT
from ts_search import save_json
from validate_ts_dft import validate_one


def candidates(output, manifest, config):
    if not config.get("electronic_state_verified"):
        raise ValueError("Verify the source electronic-state convention before Gaussian")
    selected = []
    jobs = {}
    entries = {item["reaction"]: item for item in manifest["reactions"]}
    for reaction in manifest["dft_reactions"]:
        item = entries[reaction]
        if (sum(item["atomic_numbers"]) - config["charge"]) % 2:
            raise ValueError(f"Singlet electron-count inconsistency for {reaction}")
        for strategy in manifest["protocol"]["strategies"]:
            path = output / "holdout/mh1-finetuned" / reaction / strategy / "result.json"
            result = json.loads(path.read_text())
            source = Path(result["selected_folder"]) / "ts_candidate.extxyz"
            entry = {"reaction": reaction, "strategy": strategy, "model": "mh1-finetuned",
                     "source_attempt": result["selected_attempt"], "source": str(source),
                     "mlip_verified": result["mlip_verified"],
                     "saddle_irc_verified": result["saddle_irc_verified"]}
            if not source.exists():
                entry["status"] = "no_candidate"
                selected.append(entry)
                continue
            atoms = read(source)
            np.testing.assert_array_equal(atoms.numbers, item["atomic_numbers"])
            if not np.isfinite(atoms.positions).all():
                raise ValueError("Non-finite DFT candidate")
            # Exact saved mapped coordinates identify identical candidates across strategies.
            key = digest({"reaction": reaction, "numbers": atoms.numbers.tolist(),
                          "positions": atoms.positions.tolist(), "dft_config": config})
            entry.update(status="candidate", dft_job=key)
            jobs.setdefault(key, {"reaction": reaction, "formula": item["formula"],
                                  "candidate_path": str(source),
                                  "input_folder": str(output / "inputs" / reaction),
                                  "output_folder": str(output / "dft" / reaction / key[:16])})
            selected.append(entry)
    record = {"preselected_reactions": manifest["dft_reactions"], "configuration": config,
              "strategy_units": selected, "unique_jobs": jobs,
              "deduplication": "identical mapped coordinates and DFT configuration"}
    path = output / "dft/selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("DFT candidate selection changed after jobs were prepared")
    save_json(path, record)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reactions", nargs="+")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Jobs must be positive")
    manifest = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    config = json.loads((ROOT / "configs/v0.5.0_dft.json").read_text())
    if args.reactions and set(args.reactions) - set(manifest["dft_reactions"]):
        parser.error("Reaction is outside the preselected DFT subset")
    jobs = candidates(args.output, manifest, config)
    if args.prepare_only:
        print(json.dumps(jobs, indent=2))
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {}
        for key, item in jobs.items():
            if args.reactions and item["reaction"] not in args.reactions:
                continue
            path = Path(item["output_folder"]) / "result.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["status"] == "completed" or not args.retry_failed:
                    print(f"Keeping recorded DFT outcome {item['reaction']} {key[:16]}", flush=True)
                    continue
            pending[executor.submit(validate_one, args.output, item, config)] = (key, item)
        for future in as_completed(pending):
            key, item = pending[future]
            result = future.result()
            print(f"DFT {item['reaction']} {key[:16]} {result['status']} verified={result['dft_verified']}", flush=True)


if __name__ == "__main__":
    main()
