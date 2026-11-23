"""Evaluate all nine preselected strategy candidates, reusing identical TS jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from ase.io import read

from adaptive_ts_search import digest
from prepare_v05_reactions import ROOT
from ts_search import save_json
from validate_ts_dft import validate_one


def candidates(output, manifest, config, save_selection=True):
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
    if save_selection:
        path = output / "dft/selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("DFT candidate selection changed after jobs were prepared")
        save_json(path, record)
    return jobs


def ready_reactions(output, manifest):
    ready = []
    for reaction in manifest["dft_reactions"]:
        for strategy in manifest["protocol"]["strategies"]:
            path = output / "holdout/mh1-finetuned" / reaction / strategy / "result.json"
            if not path.exists() or json.loads(path.read_text()).get("status") != "completed":
                break
        else:
            ready.append(reaction)
    return ready


def archive_previous_attempt(folder):
    previous = folder / "previous_attempts"
    previous.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
    archive = Path(tempfile.mkdtemp(prefix=stamp, dir=previous))
    for name in ("result.json", "error.txt"):
        source = folder / name
        if source.exists():
            shutil.copy2(source, archive / name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ready-only", action="store_true",
                        help="Run preselected reactions whose three strategy searches are completed")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reactions", nargs="+")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Jobs must be positive")
    manifest = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    config = json.loads((ROOT / "configs/v0.5.0_dft.json").read_text())
    if args.reactions and set(args.reactions) - set(manifest["dft_reactions"]):
        parser.error("Reaction is outside the preselected DFT subset")
    if args.ready_only:
        ready = ready_reactions(args.output, manifest)
        all_ready = len(ready) == len(manifest["dft_reactions"])
        jobs = candidates(args.output, {**manifest, "dft_reactions": ready}, config,
                          save_selection=all_ready)
        record = {"scope": "scheduling-only; the canonical DFT selection retains all nine units",
                  "provisional": True, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "preselected_reactions": manifest["dft_reactions"],
                  "ready_reactions": ready,
                  "pending_reactions": [reaction for reaction in manifest["dft_reactions"] if reaction not in ready],
                  "ready_search_strategy_units": len(ready) * len(manifest["protocol"]["strategies"]),
                  "canonical_selection_written": all_ready,
                  "configuration": config, "unique_jobs": jobs}
        path = args.output.parent / "coordination/dft-ready-selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, record)
    else:
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
                archive = archive_previous_attempt(path.parent)
                print(f"Archived prior DFT outcome at {archive}", flush=True)
            pending[executor.submit(validate_one, args.output, item, config)] = (key, item)
        for future in as_completed(pending):
            key, item = pending[future]
            result = future.result()
            print(f"DFT {item['reaction']} {key[:16]} {result['status']} verified={result['dft_verified']}", flush=True)


if __name__ == "__main__":
    main()
