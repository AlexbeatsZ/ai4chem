"""Freeze a small development DFT cohort, then package eligible MLIP candidates.

Run with the existing analysis .venv. Each candidate gets its own immutable
one-job protocol because the Gaussian preparer/assessor selects by reaction.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from ase.io import read

from prepare_gaussian_chain import prepare

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "configs/v0.7.0_diversity_dft.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def freeze():
    search = json.loads((ROOT / "configs/v0.7.0_diversity_search.json").read_text())
    smallest = sorted(search["endpoint_records"], key=lambda r: (r["atoms"], r["reaction"]))[:2]
    base = json.loads((ROOT / "configs/v0.6.0-pilot/protocol-r2.json").read_text())
    cohort = {key: base[key] for key in ("dft_method", "dft_nproc", "dft_mem", "charge",
        "multiplicity", "irc_endpoint_rmsd_A", "endpoint_tolerance_A")}
    cohort.update(version="v0.7.0-diversity-dft", frozen_at_utc=now(),
        study_type="post_hoc_development_diagnostic", changes_v05_results=False,
        search_protocol_frozen_at_utc=search["frozen_at_utc"],
        selection="Two smallest atom counts in the frozen eight-reaction search cohort, ties by reaction ID. Defined after some MLIP results were observed; this is exploratory development, not a prospective evaluation.",
        experts=search["experts"], endpoint_records=smallest,
        eligibility="strict_mlip_success and saddle_irc_and_stable_minima; failed or unavailable MLIP units remain in the six-unit accounting denominator, without submitting an invalid TS",
        candidate_units=[{"expert": expert, "reaction": row["reaction"]}
            for expert in search["experts"] for row in smallest],
        dft_policy={"irc_algorithm": "LQA", "irc_initial_max_points": 320,
            "irc_restart_max_points": 320,
            "decision_basis": "Reuse the bounded v0.6 r2 setting: initial 320 points per direction, no extension beyond 320, step 5. Preserve failures; no automatic retries."},
        acceptance="Converged TS, complete 3N-6 modes, exactly one negative mode below -50 cm^-1; both complete IRC branches, optimized minima with all frequencies nonnegative, mapped bonds and fragment RMSD <=0.5 A. Independent assessor required.")
    write_new(COHORT, cohort)
    return cohort


def package(expert, reaction):
    cohort = json.loads(COHORT.read_text())
    if {"expert": expert, "reaction": reaction} not in cohort["candidate_units"]:
        raise ValueError("Candidate is outside the frozen six-unit cohort")
    folder = ROOT / "results/v0.7.0/diversity_search" / expert / reaction
    pipeline = json.loads((folder / "pipeline.json").read_text())
    if not (pipeline["strict_mlip_success"] and pipeline["saddle_irc_and_stable_minima"]):
        raise ValueError("Candidate failed the frozen MLIP eligibility gate; do not submit")
    record = next(row for row in cohort["endpoint_records"] if row["reaction"] == reaction)
    atoms = read(folder / "ts_candidate.extxyz")
    if atoms.numbers.tolist() != record["atomic_numbers"] or not np.isfinite(atoms.positions).all():
        raise ValueError("Candidate mapping or coordinates are invalid")
    if (int(atoms.numbers.sum())-cohort["charge"]) % 2:
        raise ValueError("Odd electron count is incompatible with the frozen singlet")
    symbols = [chemical_symbols[z] for z in record["atomic_numbers"]]
    job_id = f"{reaction}-{expert}-gaussian16w-r1"
    job = {"job_id": job_id, "reaction": reaction, "expert": expert,
        "purpose": "heterogeneous_expert_development_DFT_diagnostic",
        "engine": "Gaussian16W-A.03", "formula": record["formula"],
        "source_candidate": str((folder / "ts_candidate.extxyz").relative_to(ROOT)),
        "symbols": symbols, "numbers": atoms.numbers.tolist(), "positions_A": atoms.positions.tolist(),
        "expected_vibrational_modes": 3*len(atoms)-6,
        "mlip_barrier_eV": pipeline["result"]["saddle"]["barrier_eV"],
        "endpoints": {label: {"symbols": symbols, "positions_A": record[key]}
            for label, key in (("reactant", "reactant_positions"), ("product", "product_positions"))}}
    protocol = {key: cohort[key] for key in ("version", "dft_method", "dft_nproc", "dft_mem",
        "charge", "multiplicity", "dft_policy", "irc_endpoint_rmsd_A", "endpoint_tolerance_A",
        "study_type", "changes_v05_results")}
    protocol.update(frozen_at_utc=now(), cohort_frozen_at_utc=cohort["frozen_at_utc"], jobs=[job])
    output = ROOT / "results/v0.7.0/diversity_dft/prepared" / job_id
    path = ROOT / "configs/v0.7.0-diversity-dft" / f"{expert}-{reaction}.json"
    if output.exists() or path.exists():
        raise FileExistsError("Candidate package already exists; inspect without overwriting")
    write_new(path, protocol)
    prepare(protocol, reaction, output)
    return {"job_id": job_id, "protocol": str(path), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-cohort", action="store_true")
    parser.add_argument("--expert")
    parser.add_argument("--reaction")
    args = parser.parse_args()
    if args.freeze_cohort:
        cohort = freeze()
        print(json.dumps({"frozen_at_utc": cohort["frozen_at_utc"], "units": cohort["candidate_units"]}))
    elif args.expert and args.reaction:
        print(json.dumps(package(args.expert, args.reaction)))
    else:
        parser.error("Use --freeze-cohort or both --expert and --reaction")


if __name__ == "__main__":
    main()
