"""Refine five endpoint-derived MLIP candidates in existing Windows Gaussian."""

import argparse
import fcntl
import json
import re
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from ts_search import aligned_rmsd, bond_set, fragment_rmsds, save_json


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def parse_output(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    frequencies = []
    frequency_section = text.rsplit("Harmonic frequencies", 1)[-1]
    for line in re.findall(r"Frequencies\s+--([^\n]+)", frequency_section):
        frequencies.extend(float(value) for value in line.split())
    energies = re.findall(r"SCF Done:.*?=\s*([-\d.]+)", text)
    elapsed = re.findall(r"Elapsed time:\s+(\d+) days\s+(\d+) hours\s+(\d+) minutes\s+([\d.]+) seconds", text)
    seconds = sum(int(days) * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(remainder)
                  for days, hours, minutes, remainder in elapsed)
    normal_termination = bool(re.search(r"Normal termination of Gaussian[^\n]*\s*\Z", text))
    minimum_detected = "PES minimum detected" in text
    direction_complete = bool(re.search(r"Calculation of (?:FORWARD|REVERSE) path complete", text))
    minimum_position = text.rfind("PES minimum detected")
    completion_position = text.rfind("Reaction path calculation complete.")
    index_error_position = text.rfind("PESDtA: Invalid IPoint.")
    postprocessing_error = ("PESDtA: Invalid IPoint." in text and minimum_detected and direction_complete
                            and minimum_position < completion_position < index_error_position
                            and text.count("Error termination") == 1)
    return {"normal_termination": normal_termination,
            "optimization_completed": "Optimization completed." in text,
            "frequencies_cm1": frequencies,
            "imaginary_modes": sum(value < 0 for value in frequencies),
            "imaginary_modes_above_50_cm1": sum(value < -50 for value in frequencies),
            "energy_hartree": float(energies[-1]) if energies else None,
            "gaussian_elapsed_seconds": seconds,
            "irc_minimum_detected": minimum_detected,
            "irc_direction_complete": direction_complete,
            "irc_postprocessing_error": postprocessing_error,
            "maximum_steps_reached": "Maximum number of steps reached." in text,
            "irc_path_complete": bool((normal_termination or postprocessing_error)
                                      and minimum_detected and direction_complete),
            "irc_points": len(re.findall(r"NET REACTION COORDINATE UP TO THIS POINT", text))}


def read_geometry(path, expected_numbers, energy_hartree):
    # ASE's Gaussian reader can attach incomplete forces from frequency logs.
    # Geometry and the explicitly parsed SCF energy are sufficient for this stage.
    atoms = read(path, format="gaussian-out", index=-1).copy()
    np.testing.assert_array_equal(atoms.numbers, expected_numbers)
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Non-finite Gaussian geometry")
    atoms.calc = SinglePointCalculator(atoms, energy=energy_hartree * units.Hartree)
    return atoms


def write_input(path, route, title, config, atoms=None, old_checkpoint=None):
    lines = []
    if old_checkpoint is not None:
        lines.append(f"%oldchk={old_checkpoint}")
    lines += [f"%chk={path.stem}.chk", f"%mem={config['dft_mem']}",
              f"%nprocshared={config['dft_nproc']}", f"#p {config['dft_method']} {route}", ""]
    if atoms is not None:
        lines += [title, "", "0 1"]
        lines += [f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
                  for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)]
    path.write_text("\n".join(lines) + "\n\n", encoding="ascii")


def run_job(folder, name, config, atoms=None, old_checkpoint=None, route=""):
    input_path = folder / f"{name}.gjf"
    output_path = input_path.with_suffix(".log")
    if output_path.exists() and parse_output(output_path)["normal_termination"]:
        print(f"Keeping Gaussian {folder.name} {name}", flush=True)
        return parse_output(output_path)
    if output_path.exists():
        raise RuntimeError(f"Earlier incomplete Gaussian job needs inspection: {output_path}")
    write_input(input_path, route, f"ai4chem v0.4.0 {folder.name} {name}", config,
                atoms=atoms, old_checkpoint=old_checkpoint)
    print(f"Starting Gaussian {folder.name} {name}", flush=True)
    powershell = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    command = [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               windows_path(Path(__file__).parent / "run_gaussian_job.ps1"),
               "-InputFile", windows_path(input_path)]
    subprocess.run(command, check=True)
    return parse_output(output_path)


def run_irc_job(folder, direction, config, common):
    attempts = []
    settings = [(f"irc_{direction}", 80, 10, 20, ""),
                (f"irc_{direction}_smallstep", 160, 5, 80, ""),
                (f"irc_{direction}_lqa", 160, 5, 80, ",lqa")]
    policy = json.loads(Path("configs/v0.4.0_dft_selection.json").read_text())
    if folder.name in policy.get("lqa_reactions", []):
        for name, _, _, _, _ in settings[:-1]:
            path = folder / f"{name}.log"
            if path.exists():
                interrupted = folder / f"{name}.interrupted.json"
                outcome = "interrupted for integrator change" if interrupted.exists() else "previous HPC attempt"
                attempts.append({"job": name, "outcome": outcome})
        settings = settings[-1:]
    for name, points, step, cycles, algorithm in settings:
        path = folder / f"{name}.log"
        if path.exists() and not parse_output(path)["normal_termination"]:
            failure = path.read_text(encoding="utf-8", errors="replace")
            if "Maximum number of corrector steps" not in failure:
                raise RuntimeError(f"Inspect failed Gaussian IRC before retrying: {path}")
            attempts.append({"job": name, "outcome": "corrector iteration limit"})
            continue
        try:
            result = run_job(folder, name, config, old_checkpoint="ts.chk",
                             route=f"irc=(rcfc,{direction},maxpoints={points},stepsize={step},"
                                   f"maxcycle={cycles}{algorithm}) geom=allcheck guess=read {common}")
        except subprocess.CalledProcessError:
            failure = path.read_text(encoding="utf-8", errors="replace")
            if "Maximum number of corrector steps" not in failure:
                raise
            attempts.append({"job": name, "outcome": "corrector iteration limit"})
            continue
        result.update(job=name, step_size=step, max_points=points, max_corrector_cycles=cycles,
                      algorithm="LQA" if algorithm else "HPC", previous_attempts=attempts)
        if result.get("maximum_steps_reached", False) and not result.get("irc_minimum_detected", False):
            extended_name = name + "_extended"
            extension_points = policy["irc_restart_max_points"]
            extended = run_irc_extension(folder, extended_name, name, extension_points, config, common)
            extended.update(job=extended_name, step_size=step, max_points=extension_points,
                            max_corrector_cycles=cycles, algorithm=result["algorithm"],
                            resumed_from=name,
                            previous_attempts=attempts + [{"job": name, "outcome": "point budget reached, resumed from checkpoint"}])
            return extended
        return result
    raise RuntimeError(f"All bounded IRC integration attempts failed for {direction}")


def run_irc_extension(folder, name, previous, points, config, common):
    log_path = folder / f"{name}.log"
    if log_path.exists():
        saved = parse_output(log_path)
        if saved["irc_postprocessing_error"] and (folder / f"{name}.chk").exists():
            saved["recovered_postprocessing_error"] = True
            return saved
    try:
        return run_job(folder, name, config, old_checkpoint=f"{previous}.chk",
                       route=f"irc=(restart,maxpoints={points}) geom=allcheck guess=read {common}")
    except subprocess.CalledProcessError:
        saved = parse_output(log_path)
        if not (saved["irc_postprocessing_error"] and (folder / f"{name}.chk").exists()):
            raise
        saved["recovered_postprocessing_error"] = True
        return saved


def prepare_candidates(root, config):
    selection_path = root / "dft" / "selection.json"
    fixed = json.loads(Path("configs/v0.4.0_dft_selection.json").read_text())
    search_selection = json.loads((root / "selection.json").read_text())
    if search_selection["config"] != config:
        raise ValueError("DFT protocol must match the saved search configuration")
    if selection_path.exists():
        manifest = json.loads(selection_path.read_text())
        manifest["fixed_protocol"] = fixed
        save_json(selection_path, manifest)
        return manifest["candidates"]
    selection = search_selection["reactions"]
    candidates = []
    for item in selection:
        folder = root / "mh1-finetuned" / item["reaction"]
        result_path = folder / "result.json"
        if not result_path.exists():
            raise RuntimeError("Finish all fine-tuned searches before selecting DFT candidates")
        result = json.loads(result_path.read_text())
        if (folder / "ts_candidate.extxyz").exists():
            candidates.append({"reaction": item["reaction"], "formula": item["formula"], "atoms": item["atoms"],
                               "mlip_verified": result["mlip_verified"],
                               "single_imaginary": result.get("vibrations", {}).get("imaginary_modes_above_threshold") == 1,
                               "stationary": result.get("saddle", {}).get("fmax_eV_A", 1e10) <= config["ts_fmax_eV_A"]})
    lookup = {item["reaction"]: item for item in candidates}
    selected = [lookup[name] for name in fixed["reactions"]]
    if len(selected) != config["dft_count"]:
        raise RuntimeError("Fixed DFT selection size differs from the configured count")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(selection_path, {"candidates": selected,
                              "ranking": "frozen five-candidate selection in configs/v0.4.0_dft_selection.json",
                              "selection_note": fixed["selection_note"],
                              "fixed_protocol": fixed,
                              "charge": 0, "multiplicity": 1,
                              "scope": "neutral closed-shell CHNO Transition1x development subset"})
    return selected


def validate_one(root, item, config):
    folder = root / "dft" / item["reaction"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return validate_locked(root, item, config)


def validate_locked(root, item, config):
    reaction = item["reaction"]
    folder = root / "dft" / reaction
    folder.mkdir(parents=True, exist_ok=True)
    result = {"reaction": reaction, "formula": item["formula"], "method": config["dft_method"],
              "status": "running", "dft_verified": False, "stage": "ts"}
    save_json(folder / "result.json", result)
    start = time.monotonic()
    common = "scf=(xqc,maxcycle=128) int=ultrafine nosymm"
    try:
        candidate = read(root / "mh1-finetuned" / reaction / "ts_candidate.extxyz")
        result["ts"] = run_job(folder, "ts", config, atoms=candidate,
                               route=f"opt=(ts,calcfc,noeigentest,maxcycles=80) freq {common}")
        ts = read_geometry(folder / "ts.log", candidate.numbers, result["ts"]["energy_hartree"])
        write(folder / "ts_dft.extxyz", ts)
        result["mlip_to_dft_ts_rmsd_A"] = aligned_rmsd(candidate, ts)
        if not (result["ts"]["optimization_completed"] and result["ts"]["imaginary_modes"] == 1
                and len(result["ts"]["frequencies_cm1"]) >= 3 * len(candidate) - 6
                and result["ts"]["imaginary_modes_above_50_cm1"] == 1):
            result["status"] = "completed"
            result["stage"] = "ts_failed_validation"
            save_json(folder / "result.json", result)
            return result
        result["stage"] = "irc"
        save_json(folder / "result.json", result)
        expected = [read(root / "inputs" / reaction / f"{name}.xyz") for name in ("reactant", "product")]
        branches = []
        result["branches"] = branches
        for direction in ("forward", "reverse"):
            result["irc_current_direction"] = direction
            save_json(folder / "result.json", result)
            irc = run_irc_job(folder, direction, config, common)
            irc_name = irc["job"]
            minimum_name = f"minimum_{direction}" + ("_extended" if "resumed_from" in irc else "")
            minimum = run_job(folder, minimum_name, config, old_checkpoint=f"{irc_name}.chk",
                              route=f"opt=(calcfc,maxcycles=100) freq geom=allcheck guess=read {common}")
            atoms = read_geometry(folder / f"{minimum_name}.log", candidate.numbers, minimum["energy_hartree"])
            write(folder / f"{minimum_name}.extxyz", atoms)
            bonds_match = [bond_set(atoms) == bond_set(endpoint) for endpoint in expected]
            fragment_errors = [fragment_rmsds(endpoint, atoms) for endpoint in expected]
            branches.append({"direction": direction, "irc": irc, "minimum": minimum, "minimum_job": minimum_name,
                             "matches_input_bonds": bonds_match,
                             "endpoint_rmsd_A": [aligned_rmsd(endpoint, atoms) for endpoint in expected],
                             "endpoint_fragment_rmsd_A": fragment_errors,
                             "matches_input_geometry": [bonds_match[index] and max(fragment_errors[index]) <= config["irc_endpoint_rmsd_A"]
                                                        for index in range(2)]})
            save_json(folder / "result.json", result)
        result.pop("irc_current_direction", None)
        bonds = [branch["matches_input_bonds"] for branch in branches]
        result["expected_bond_connectivity"] = bool((bonds[0][0] and bonds[1][1]) or (bonds[0][1] and bonds[1][0]))
        matches = [branch["matches_input_geometry"] for branch in branches]
        connected = (matches[0][0] and matches[1][1]) or (matches[0][1] and matches[1][0])
        stable = all(branch["minimum"]["optimization_completed"] and branch["minimum"]["imaginary_modes"] == 0
                     and len(branch["minimum"]["frequencies_cm1"]) >= 3 * len(candidate) - 6
                     for branch in branches)
        complete = all(branch["irc"]["irc_path_complete"] for branch in branches)
        result["expected_connectivity"] = bool(connected)
        result["stable_minima"] = bool(stable)
        result["irc_minima_detected"] = bool(complete)
        result["recovered_irc_postprocessing_error"] = any(
            branch["irc"].get("recovered_postprocessing_error", False) for branch in branches)
        result["dft_verified"] = bool(connected and stable and complete)
        if connected:
            reactant_branch = branches[0] if matches[0][0] else branches[1]
            result["electronic_barrier_eV"] = (result["ts"]["energy_hartree"]
                                                - reactant_branch["minimum"]["energy_hartree"]) * units.Hartree
        result["status"] = "completed"
        result["stage"] = "finished"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result["elapsed_seconds"] = time.monotonic() - start
    result["gaussian_reported_elapsed_seconds"] = sum(
        parse_output(path)["gaussian_elapsed_seconds"] for path in folder.glob("*.log")
        if not path.name.endswith(".launcher.log"))
    result["interrupted_irc_records"] = [path.name for path in sorted(folder.glob("*.interrupted.json"))]
    save_json(folder / "result.json", result)
    error_path = folder / "error.txt"
    if result["status"] == "completed" and error_path.exists():
        error_path.replace(folder / "recovered_error.txt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.4.0"))
    parser.add_argument("--config", type=Path, default=Path("configs/v0.4.0.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--reactions", nargs="+")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = prepare_candidates(args.output, config)
    identifiers = [item["reaction"] for item in selected]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != config["dft_count"]:
        raise ValueError("DFT selection must have the configured number of unique reactions")
    if args.jobs < 1:
        parser.error("Jobs must be positive")
    if args.reactions and set(args.reactions) - {item["reaction"] for item in selected}:
        parser.error("Requested reaction is outside the fixed DFT selection")
    if args.prepare_only:
        print(json.dumps(selected, indent=2), flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {}
        for item in selected:
            if args.reactions and item["reaction"] not in args.reactions:
                continue
            result_path = args.output / "dft" / item["reaction"] / "result.json"
            previous = None
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if not args.recheck and (previous["status"] not in {"failed", "running"} or not args.retry_failed):
                    print(f"Keeping recorded DFT outcome {item['reaction']}", flush=True)
                    continue
            future = executor.submit(validate_one, args.output, item, config)
            pending[future] = (item, result_path, previous)
        for future in as_completed(pending):
            item, result_path, previous = pending[future]
            result = future.result()
            if previous is not None and not args.recheck:
                result["previous_attempt"] = previous
                save_json(result_path, result)
            print(f"DFT {item['reaction']} {result['status']} verified={result['dft_verified']}", flush=True)


if __name__ == "__main__":
    main()
