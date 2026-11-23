"""Assess separately frozen native Mac or portable Gaussian full-chain evidence.

Run in the existing analysis environment (ASE/numpy/scipy), without MACE.
Mapped-geometry helpers reproduce ts_search's frozen definitions explicitly so
this independent reader never imports the CUDA inference stack.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms, units
from ase.build.rotate import minimize_rotation_and_translation
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def bonds(atoms):
    left, right = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
    return {tuple(sorted((int(i), int(j)))) for i, j in zip(left, right) if i != j}


def rmsd(reference, candidate):
    aligned = candidate.copy()
    minimize_rotation_and_translation(reference, aligned)
    return float(np.sqrt(np.mean(np.sum((reference.positions-aligned.positions)**2, axis=1))))


def fragment_errors(reference, candidate):
    adjacency = np.zeros((len(reference), len(reference)), dtype=bool)
    for left, right in bonds(reference):
        adjacency[left, right] = adjacency[right, left] = True
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    return [rmsd(reference[labels == i], candidate[labels == i]) for i in range(count)]


def atoms_from_record(record, expected_numbers):
    atoms = Atoms(record["symbols"], positions=record["positions_A"])
    np.testing.assert_array_equal(atoms.numbers, expected_numbers)
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Nonfinite or incorrectly mapped geometry")
    return atoms


def parse_gaussian(path, expected_numbers):
    text = path.read_text(errors="replace")
    section = text.rsplit("Harmonic frequencies", 1)[-1]
    frequencies = [float(v) for line in re.findall(r"Frequencies\s+--([^\n]+)", section) for v in line.split()]
    energies = re.findall(r"SCF Done:.*?=\s*([-\d.]+)", text)
    if not energies:
        raise ValueError(f"No SCF energy: {path}")
    normal = bool(re.search(r"Normal termination of Gaussian[^\n]*\s*\Z", text))
    minimum = "PES minimum detected" in text
    complete = bool(re.search(r"Calculation of (?:FORWARD|REVERSE) path complete", text))
    posterror = (minimum and complete and "PESDtA: Invalid IPoint." in text and
                 text.rfind("PES minimum detected") < text.rfind("Reaction path calculation complete.") <
                 text.rfind("PESDtA: Invalid IPoint.") and text.count("Error termination") == 1)
    record = {"normal_termination": normal, "optimization_completed": "Optimization completed." in text,
              "frequencies_cm1": frequencies, "energy_hartree": float(energies[-1]),
              "irc_path_complete": bool((normal or posterror) and minimum and complete),
              "postprocessing_error": bool(posterror), "source_log": path.name}
    # Discard ASE's attached Gaussian force array: its size is not reliable for
    # these frequency logs. Only mapped geometry and explicit SCF energy are used.
    atoms = read(path, format="gaussian-out", index=-1).copy()
    np.testing.assert_array_equal(atoms.numbers, expected_numbers)
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Nonfinite Gaussian geometry")
    return record, atoms


def assess(protocol, reaction, folder, engine):
    job = next(j for j in protocol["jobs"] if j["reaction"] == reaction and
               ((j["engine"] == "PySCF-geomeTRIC") == (engine == "mac")))
    endpoints = [atoms_from_record(job["endpoints"][name], job["numbers"]) for name in ("reactant", "product")]
    expected_modes = job["expected_vibrational_modes"]
    if engine == "mac":
        source = json.loads((folder/"summary.json").read_text())
        ts = source.get("ts")
        if not source["raw_chain_completed"] or ts is None:
            raise ValueError("Raw Mac chain is incomplete; no final assessment")
        ts_atoms = atoms_from_record(ts, job["numbers"])
        converged_ts = json.loads((folder/"ts/result.json").read_text())["converged"]
        analyses = []
        for branch in source["branches"]:
            root = folder/branch["direction"]
            analysis = branch["analysis"]
            complete_irc = json.loads((root/"irc/result.json").read_text())["converged"]
            complete_minimum = json.loads((root/"minimum/result.json").read_text())["converged"]
            analyses.append((branch["direction"], analysis, atoms_from_record(analysis, job["numbers"]),
                             complete_irc, complete_minimum))
    else:
        source = json.loads((folder/"worker-state.json").read_text(encoding="utf-8-sig"))
        if not source["raw_chain_completed"]:
            raise ValueError("Raw Gaussian chain is incomplete; no final assessment")
        ts, ts_atoms = parse_gaussian(folder/"ts.log", job["numbers"])
        converged_ts = ts["normal_termination"] and ts["optimization_completed"]
        analyses = []
        for branch in source["branches"]:
            irc, _ = parse_gaussian(folder/(branch["irc_job"]+".log"), job["numbers"])
            minimum, atoms = parse_gaussian(folder/(branch["minimum_job"]+".log"), job["numbers"])
            analyses.append((branch["direction"], minimum, atoms, irc["irc_path_complete"],
                             minimum["normal_termination"] and minimum["optimization_completed"]))
    ts_frequencies = np.asarray(ts["frequencies_cm1"])
    ts_valid = bool(converged_ts and len(ts_frequencies) == expected_modes and
                    np.isfinite(ts_frequencies).all() and (ts_frequencies < 0).sum() == 1 and
                    (ts_frequencies < -50).sum() == 1)
    branches = []
    for direction, analysis, atoms, irc_complete, minimum_complete in analyses:
        frequencies = np.asarray(analysis["frequencies_cm1"])
        stable = bool(minimum_complete and len(frequencies) == expected_modes and
                      np.isfinite(frequencies).all() and (frequencies >= 0).all())
        matches_bonds = [bonds(atoms) == bonds(endpoint) for endpoint in endpoints]
        fragment_rmsds = [fragment_errors(endpoint, atoms) for endpoint in endpoints]
        matches = [matches_bonds[i] and max(fragment_rmsds[i]) <= protocol["endpoint_tolerance_A"] for i in range(2)]
        branches.append({"direction": direction, "irc_complete": bool(irc_complete), "stable_minimum": stable,
                         "frequencies_cm1": analysis["frequencies_cm1"], "energy_hartree": analysis["energy_hartree"],
                         "matches_input_bonds": matches_bonds, "endpoint_fragment_rmsd_A": fragment_rmsds,
                         "endpoint_rmsd_A": [rmsd(endpoint, atoms) for endpoint in endpoints],
                         "matches_input_geometry": matches})
    matches = [b["matches_input_geometry"] for b in branches]
    connected = bool(len(matches) == 2 and ((matches[0][0] and matches[1][1]) or (matches[0][1] and matches[1][0])))
    valid = bool(ts_valid and connected and all(b["irc_complete"] and b["stable_minimum"] for b in branches))
    result = {"job_id": job["job_id"], "reaction": reaction, "engine": engine,
              "assessed_at_utc": datetime.now(timezone.utc).isoformat(), "study_type": protocol["study_type"],
              "changes_v05_results": False, "ts_valid": ts_valid, "ts": ts, "branches": branches,
              "expected_connectivity": connected, "dft_verified": valid,
              "source_started_at_utc": source["started_at_utc"], "source_finished_at_utc": source["finished_at_utc"],
              "source_elapsed_seconds": source["elapsed_seconds"]}
    if connected:
        reactant = next(b for b in branches if b["matches_input_geometry"][0])
        result["electronic_barrier_eV"] = (ts["energy_hartree"]-reactant["energy_hartree"])*units.Hartree
    candidate = atoms_from_record(job, job["numbers"])
    result["mlip_to_dft_ts_rmsd_A"] = rmsd(candidate, ts_atoms)
    if engine == "mac" and job["purpose"] == "known_positive_full_DFT_engine_calibration":
        reference = json.loads((Path(__file__).resolve().parents[1]/"configs/v0.6.0-pilot"/reaction/"gaussian_reference.json").read_text())
        saved = json.loads((Path(__file__).resolve().parents[1]/"configs/v0.6.0-pilot"/reaction/"gaussian_ts_geometry.json").read_text())
        rmsd_reference = rmsd(atoms_from_record(saved, job["numbers"]), ts_atoms)
        comparison = {"ts_energy_difference_hartree": ts["energy_hartree"]-reference["ts"]["energy_hartree"],
                      "ts_rmsd_A": rmsd_reference,
                      "imaginary_frequency_difference_cm1": float(min(ts_frequencies))-min(reference["ts"]["frequencies_cm1"])}
        checks = [abs(comparison["ts_energy_difference_hartree"]) <= protocol["mac"]["comparison_energy_tolerance_hartree"],
                  comparison["ts_rmsd_A"] <= protocol["mac"]["comparison_ts_rmsd_A"],
                  abs(comparison["imaginary_frequency_difference_cm1"]) <= protocol["mac"]["comparison_imaginary_frequency_tolerance_cm1"]]
        if connected:
            comparison["electronic_barrier_difference_eV"] = result["electronic_barrier_eV"]-reference["electronic_barrier_eV"]
            checks.append(abs(comparison["electronic_barrier_difference_eV"]) <= protocol["mac"]["comparison_barrier_tolerance_eV"])
        else:
            checks.append(False)
        comparison["within_predefined_tolerances"] = bool(all(checks))
        result["gaussian_comparison"] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reaction", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--engine", choices=("mac", "gaussian"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(json.loads(args.protocol.read_text()), args.reaction, args.folder, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("reaction", "engine", "dft_verified", "expected_connectivity")}))
    return 0 if result["dft_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
