"""Explain saved search failures using endpoint, NEB, saddle and IRC evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ts_search import save_json


def force_history(path):
    values = []
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "FIRE:":
                values.append(float(fields[-1]))
    return values


def diagnose(result, folder, policy):
    folder = Path(folder)
    history = force_history(folder / "neb_climb.log")
    window = int(np.ceil(len(history) * 0.2))
    ratio = None
    if len(history) >= 10:
        previous = float(np.median(history[-2 * window:-window]))
        if previous > 0:
            ratio = float(np.median(history[-window:]) / previous)
    endpoints = result.get("endpoints", [])
    neb = result.get("neb", {})
    saddle = result.get("saddle", {})
    vibrations = result.get("vibrations", {})
    irc = result.get("irc", {})
    evidence = {"stage": result.get("stage"), "error": result.get("error"),
                "endpoints": endpoints, "neb": neb, "neb_force_history": history,
                "force_trend_ratio": ratio, "saddle": saddle, "vibrations": vibrations, "irc": irc}
    flags = []
    for name, passed in (("neb_converged", neb.get("converged")),
                         ("saddle_converged", saddle.get("converged")),
                         ("single_significant_imaginary", vibrations.get("imaginary_modes_above_threshold") == 1),
                         ("irc_connected", irc.get("converged_and_connected"))):
        if passed:
            flags.append(name)
    error = result.get("error", "").lower()
    numerical = any(token in error for token in (
        "non-finite", "nonfinite", "nan", "cuda", "cublas", "cudnn", "out of memory",
        "oserror", "ioerror", "filenotfounderror", "assertionerror"))
    endpoint_failure = (len(endpoints) != 2 or any(
        not entry["converged"] or not entry["input_bonds_preserved"] for entry in endpoints))
    if numerical:
        failure, action = "infrastructure_failure", "stop"
    elif endpoint_failure:
        failure, action = "endpoint_failure", "stop"
    elif result.get("saddle_irc_verified", False):
        failure, action = "none", "stop"
        if not neb.get("converged"):
            flags.append("verified_saddle_without_neb_convergence")
    elif not neb.get("converged", False):
        improving = (ratio is not None and ratio <= policy["improving_ratio"]
                     and neb.get("fmax_eV_A", 1e10) <= policy["improving_final_fmax_eV_A"])
        failure = "neb_improving" if improving else "neb_stagnation"
        action = "continuation" if improving else "perturbed_idpp"
    elif not saddle.get("converged", False) or saddle.get("fmax_eV_A", 1e10) > 0.01:
        failure, action = "saddle_failure", "high_resolution_idpp"
    elif vibrations.get("imaginary_modes_above_threshold") != 1:
        failure, action = "frequency_failure", "high_resolution_idpp"
    else:
        complete = all(branch.get("converged") and branch.get("minimum_converged")
                       for branch in irc.get("branches", [])) and len(irc.get("branches", [])) == 2
        failure = "irc_wrong_endpoint" if complete else "irc_unconverged"
        action = "perturbed_idpp"
    # Image spacing is descriptive only. It never introduces an acceptance threshold.
    band_path = folder / "final_band.extxyz"
    if band_path.exists():
        from ase.io import read
        images = read(band_path, index=":")
        spacings = [float(np.sqrt(np.mean(np.sum((right.positions - left.positions) ** 2, axis=1))))
                    for left, right in zip(images[:-1], images[1:])]
        evidence["adjacent_image_rms_displacements_A"] = spacings
    return {"primary_failure": failure, "diagnostics": flags,
            "recommended_action": action, "retry_allowed": action != "stop",
            "rule_confidence": "deterministic stage checks; NEB trend is a development heuristic",
            "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/v0.4.0"))
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0/legacy_diagnosis.json"))
    args = parser.parse_args()
    policy = json.loads(Path("configs/v0.5.0.json").read_text())
    entries = []
    for path in sorted(args.input.glob("mh1-*/rxn*/result.json")):
        result = json.loads(path.read_text())
        entries.append({"model": path.parent.parent.name, "reaction": result["reaction"],
                        **diagnose(result, path.parent, policy)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, entries)
    print(json.dumps(dict(Counter(item["primary_failure"] for item in entries)), indent=2))


if __name__ == "__main__":
    main()
