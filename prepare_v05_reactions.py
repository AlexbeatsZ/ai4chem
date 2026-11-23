"""Select endpoint-only project holdouts, excluding entire previously used reactions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import read, write

from analyze_benchmark import endpoint_change
from download_file import download_file
from prepare_transition1x import REVISION
from ts_search import save_json


ROOT = Path(__file__).resolve().parent


def excluded_ids(partitions):
    used = {int(frame["sid"]) for frame in partitions["training_frames"]}
    for role in ("development", "validation", "final"):
        used.update(int(sid) for sid in partitions[role])
    return used


def endpoint_candidates(source, excluded, allowed_formulas):
    # Projection excludes labels. Predicate excludes every interior/TS configuration.
    table = pq.read_table(source, columns=["sid", "fid", "atomic_numbers", "pos"],
                          filters=[("fid", "in", [0, 9])])
    grouped = {}
    for row in table.to_pylist():
        if int(row["sid"]) not in excluded:
            grouped.setdefault(int(row["sid"]), []).append(row)
    pool = []
    incomplete = []
    for sid, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["fid"])
        if len(rows) == 1:
            incomplete.append(sid)
            continue
        if [row["fid"] for row in rows] != [0, 9]:
            raise ValueError(f"Duplicate mapped endpoints for {sid}")
        numbers = np.asarray(rows[0]["atomic_numbers"], dtype=int)
        np.testing.assert_array_equal(numbers, rows[1]["atomic_numbers"])
        endpoints = [Atoms(numbers, positions=row["pos"]) for row in rows]
        if not all(np.isfinite(atoms.positions).all() for atoms in endpoints):
            raise ValueError(f"Non-finite endpoints for {sid}")
        formula = endpoints[0].get_chemical_formula()
        if formula not in allowed_formulas:
            raise ValueError(f"Reaction {sid} is outside the source train formula split")
        kind, formed, broken = endpoint_change(*endpoints)
        pool.append({"sid": sid, "reaction": f"rxn{sid:04d}", "formula": formula,
                     "atoms": len(numbers), "kind": kind, "formed": formed, "broken": broken,
                     "endpoints": endpoints})
    return pool, incomplete


def diverse_selection(pool, count, seed):
    if count < 3 or count > len(pool):
        raise ValueError(f"Need {count} reactions, only {len(pool)} available")
    rng = np.random.default_rng(seed)
    remaining = [pool[index] for index in rng.permutation(len(pool))]
    selected = []
    required_kinds = {"formation", "cleavage", "rearrangement"}
    if not required_kinds.issubset({item["kind"] for item in pool}):
        raise ValueError("Candidate shard lacks a required endpoint-change class")
    for kind in sorted(required_kinds):
        chosen = next(item for item in remaining if item["kind"] == kind)
        selected.append(chosen)
        remaining.remove(chosen)
    while len(selected) < count:
        formulas = {item["formula"] for item in selected}
        sizes = {item["atoms"] // 4 for item in selected}
        signatures = {(item["formed"], item["broken"]) for item in selected}
        chosen = min(remaining, key=lambda item: (
            item["formula"] in formulas, item["atoms"] // 4 in sizes,
            (item["formed"], item["broken"]) in signatures))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def load_inputs(manifest, output, role):
    items = []
    for entry in manifest["reactions"]:
        if entry["role"] != role:
            continue
        folder = output / "inputs" / entry["reaction"]
        endpoints = [read(folder / f"{name}.xyz") for name in ("reactant", "product")]
        np.testing.assert_array_equal(endpoints[0].numbers, endpoints[1].numbers)
        np.testing.assert_allclose(endpoints[0].positions, entry["reactant_positions"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(endpoints[1].positions, entry["product_positions"], atol=1e-8, rtol=0)
        items.append({**entry, "endpoints": endpoints})
    return items


def prepare(protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "configs/v0.5.0_reactions.json"
    partitions = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    excluded = excluded_ids(partitions)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol"] != protocol or any(item["sid"] in excluded for item in manifest["reactions"]):
            raise ValueError("Existing frozen manifest conflicts with protocol or exclusions")
        for item in manifest["reactions"]:
            folder = output / "inputs" / item["reaction"]
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("reactant", "product"):
                atoms = Atoms(item["atomic_numbers"], positions=item[f"{name}_positions"])
                path = folder / f"{name}.xyz"
                if not path.exists():
                    write(path, atoms)
        return manifest
    if protocol["source_revision"] != REVISION or partitions["source_revision"] != REVISION:
        raise ValueError("Source revisions disagree")
    splits = json.loads((ROOT / "data/raw/official_formula_splits.json").read_text())
    for index, source in enumerate(protocol["sources"]):
        url = (f"https://huggingface.co/datasets/nimashoghi/transition1x/resolve/{REVISION}/data/"
               f"train-{index:05d}-of-00013.parquet")
        download_file(url, ROOT / source)
    pool, incomplete = endpoint_candidates([ROOT / source for source in protocol["sources"]], excluded, splits["train"])
    development = diverse_selection(pool, protocol["development_count"], protocol["seed"])
    development_ids = {item["sid"] for item in development}
    holdout = diverse_selection([item for item in pool if item["sid"] not in development_ids],
                               protocol["holdout_count"], protocol["seed"] + 1)
    entries = []
    for role, selection in (("development", development), ("holdout", holdout)):
        for item in selection:
            entry = {key: value for key, value in item.items() if key != "endpoints"}
            entry.update(role=role, atomic_numbers=item["endpoints"][0].numbers.tolist(),
                         reactant_positions=item["endpoints"][0].positions.tolist(),
                         product_positions=item["endpoints"][1].positions.tolist())
            entries.append(entry)
    dft = []
    for kind in ("formation", "cleavage", "rearrangement"):
        candidates = [item for item in holdout if item["kind"] == kind]
        dft.append(min(candidates, key=lambda item: (item["atoms"], item["sid"]))["reaction"])
    manifest = {"protocol": protocol, "source_revision": REVISION,
                "source_description": "JMP public Parquet export of Transition1x official train shards 0 and 1 of 13",
                "evaluation_role": "project-held-out reactions, pretraining overlap unverified",
                "selection_inputs": ["sid", "fid 0/9", "atomic_numbers", "endpoint positions"],
                "selection_rule": "seeded class coverage, then unseen formula, atom-count bin and bond-change signature",
                "candidate_count": len(pool), "candidate_kinds": dict(Counter(item["kind"] for item in pool)),
                "incomplete_endpoint_pairs_in_shard": len(incomplete),
                "shard_coverage": "frame-sharded export; only reactions with both fid 0 and 9 across the selected shards are eligible",
                "excluded_prior_reactions": sorted(excluded), "reactions": entries,
                "dft_reactions": dft, "dft_selection": "smallest atom count then ID within each holdout bond-change class, before search"}
    save_json(manifest_path, manifest)
    # The frozen manifest is also the recovery source for endpoint-only input files.
    return prepare(protocol, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0"))
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/v0.5.0.json").read_text())
    manifest = prepare(protocol, args.output)
    print(json.dumps({"candidates": manifest["candidate_count"], "kinds": manifest["candidate_kinds"],
                      "reactions": [{key: item[key] for key in ("reaction", "role", "formula", "kind", "atoms")}
                                    for item in manifest["reactions"]], "dft_reactions": manifest["dft_reactions"]}, indent=2))


if __name__ == "__main__":
    main()
