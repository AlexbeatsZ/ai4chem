"""Prospective roles and complete-path sampling for the model-improvement study."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import read, write

from analyze_benchmark import endpoint_change
from prepare_research_data import formula_splits, training_atoms, write_path_dataset
from prepare_transition1x import REVISION
from transition1x_data import final_band_indices

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/v0.7.0"
CONFIG = ROOT / "configs/v0.7.0_partitions.json"
COLUMNS = ["sid", "fid", "atomic_numbers", "pos", "y", "force"]


def save_json(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen scientific configuration differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def endpoint_pool(sources, excluded, allowed):
    table = pq.read_table(sources, columns=["sid", "fid", "atomic_numbers", "pos"],
                          filters=[("fid", "in", [0, 9])])
    groups = {}
    for row in table.to_pylist():
        if row["sid"] not in excluded:
            groups.setdefault(int(row["sid"]), []).append(row)
    pool = []
    for sid, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x["fid"])
        if len(rows) == 1:
            continue
        if [x["fid"] for x in rows] != [0, 9]:
            raise ValueError(f"Duplicate endpoint IDs for {sid}")
        np.testing.assert_array_equal(rows[0]["atomic_numbers"], rows[1]["atomic_numbers"])
        atoms = [Atoms(x["atomic_numbers"], positions=x["pos"]) for x in rows]
        if not all(np.isfinite(x.positions).all() for x in atoms):
            raise ValueError(f"Non-finite endpoints for {sid}")
        formula = atoms[0].get_chemical_formula()
        if formula not in allowed:
            raise ValueError(f"Wrong official formula split for {sid}")
        kind, formed, broken = endpoint_change(*atoms)
        pool.append({"sid": sid, "reaction": f"rxn{sid:04d}", "formula": formula,
                     "atoms": len(atoms[0]), "kind": kind, "formed": formed, "broken": broken,
                     "atomic_numbers": atoms[0].numbers.tolist(),
                     "reactant_positions": atoms[0].positions.tolist(),
                     "product_positions": atoms[1].positions.tolist()})
    return pool


def diverse_subset(pool, count, seed):
    if not 0 < count <= len(pool):
        raise ValueError(f"Need {count} complete endpoint pairs; available {len(pool)}")
    remaining = [pool[i] for i in np.random.default_rng(seed).permutation(len(pool))]
    selected = []
    while len(selected) < count:
        kinds = {x["kind"] for x in selected}
        formulas = {x["formula"] for x in selected}
        sizes = {x["atoms"] // 4 for x in selected}
        signatures = {(x["formed"], x["broken"]) for x in selected}
        chosen = min(remaining, key=lambda x: (x["kind"] in kinds,
            x["formula"] in formulas, x["atoms"] // 4 in sizes,
            (x["formed"], x["broken"]) in signatures))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def validate_roles(partition, old, prior_search_ids):
    train = {int(x["sid"]) for x in old["training_frames"]}
    roles = {role: set(partition[role]) for role in (
        "validation", "meta_development", "holdout_train", "holdout_val")}
    if any(len(ids) != len(partition[role]) for role, ids in roles.items()):
        raise ValueError("Duplicated reaction within a role")
    for left, ids in roles.items():
        if ids & train:
            raise ValueError(f"MACE train leakage into {left}")
        for right, other in roles.items():
            if left != right and ids & other:
                raise ValueError(f"Reaction leakage: {left}/{right}")
    historic = set(old["development"]) | set(old["final"]) | set(old["validation"]) | set(prior_search_ids)
    for role in ("meta_development", "holdout_train", "holdout_val"):
        if roles[role] & historic:
            raise ValueError(f"Previously used project reaction in {role}")


def freeze_roles(protocol):
    old = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    search = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    prior = {int(x["sid"]) for x in search["reactions"]}
    if protocol["source_revision"] != REVISION or old["source_revision"] != REVISION:
        raise ValueError("Source revision mismatch")
    if CONFIG.exists():
        partition = json.loads(CONFIG.read_text())
        if partition["protocol"] != protocol:
            raise ValueError("Frozen role protocol changed")
        validate_roles(partition, old, prior)
        return partition
    splits = formula_splits()
    excluded = {x["sid"] for x in old["training_frames"]} | prior
    for role in ("development", "final", "validation"):
        excluded.update(old[role])
    pool = endpoint_pool([ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in (0, 1)],
                         excluded, splits["train"])
    train_holdout = diverse_subset(pool, protocol["prospective_train_holdout_count"], protocol["seed"])
    taken = {x["sid"] for x in train_holdout}
    meta = diverse_subset([x for x in pool if x["sid"] not in taken],
                          protocol["meta_development_count"], protocol["seed"] + 1)
    val_pool = endpoint_pool(ROOT / "data/raw/transition1x_val.parquet", excluded, splits["val"])
    new_val = diverse_subset(val_pool, protocol["validation_count"] - len(old["validation"]), protocol["seed"] + 2)
    val_taken = {x["sid"] for x in new_val}
    val_holdout = diverse_subset([x for x in val_pool if x["sid"] not in val_taken],
                                protocol["prospective_val_holdout_count"], protocol["seed"] + 3)
    partition = {"protocol": protocol, "source_revision": REVISION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_inputs": ["sid", "endpoint atomic_numbers", "fid 0/9 positions"],
        "validation": sorted(old["validation"] + [x["sid"] for x in new_val]),
        "validation_previously_seen": old["validation"],
        "meta_development": sorted(x["sid"] for x in meta),
        "holdout_train": sorted(x["sid"] for x in train_holdout),
        "holdout_val": sorted(x["sid"] for x in val_holdout),
        "endpoint_records": {role: values for role, values in (("meta_development", meta),
            ("holdout_train", train_holdout), ("holdout_val", val_holdout))},
        "holdout_reference_labels_evaluated": False,
        "scope": "prospective project holdouts; val-formula holdout differs in composition from MACE train; foundation pretraining overlap unverified"}
    validate_roles(partition, old, prior)
    save_json(CONFIG, partition, immutable=True)
    print("Frozen roles: train=256, validation=60, meta development=128, holdouts=64+64", flush=True)
    return partition


def prepare_validation(partition):
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "validation.extxyz").exists() and (DATA / "validation.h5").exists():
        return
    table = pq.read_table(ROOT / "data/raw/transition1x_val.parquet", columns=COLUMNS,
                          filters=[("sid", "in", partition["validation"])])
    paths = write_path_dataset(table, partition["validation"], DATA / "validation.h5", "val",
                               formula_splits()["val"], "model_selection_validation")
    atoms_list = []
    for path in paths:
        for atoms in path.atoms():
            atoms.info.update(REF_energy=atoms.get_potential_energy(), config_type="Default")
            atoms.arrays["REF_forces"] = atoms.get_forces().copy()
            atoms.calc = None
            atoms_list.append(atoms)
    write(DATA / "validation.extxyz", atoms_list)
    if len(read(DATA / "validation.extxyz", index=":")) != 10 * len(partition["validation"]):
        raise ValueError("Validation readback count differs")


def sample_reaction(rows, anchor_fid, seed):
    """Nested unique anchor/random/path samples; paths include TS-adjacent coverage."""
    rows = sorted(rows, key=lambda x: x["fid"])
    if [x["fid"] for x in rows] != list(range(len(rows))):
        raise ValueError("Incomplete or duplicated history; never infer bands within a frame shard")
    final = final_band_indices(len(rows)).tolist()
    if len(rows) < 16 or anchor_fid not in range(len(rows)):
        raise ValueError("Insufficient unique frames or missing old anchor")
    energies = np.asarray([x["y"] for x in rows])
    ts_candidates = final if len(rows) == 10 else final[1:-1]
    peak = max(ts_candidates, key=lambda i: energies[i])
    peak_position = final.index(peak)
    selected = list(dict.fromkeys([anchor_fid, 0, 9] + final[max(0, peak_position-1):peak_position+2]))
    positions = np.asarray([x["pos"] for x in rows], dtype=float)
    pairs = np.triu_indices(positions.shape[1], k=1)
    descriptors = np.linalg.norm(positions[:, pairs[0]] - positions[:, pairs[1]], axis=-1)
    scores = np.full(len(rows), np.inf)
    for i in selected:
        scores = np.minimum(scores, np.mean((descriptors - descriptors[i])**2, axis=1))

    def fill(count):
        while len(selected) < count:
            scores[selected] = -np.inf
            chosen = int(np.argmax(scores))
            selected.append(chosen)
            scores[:] = np.minimum(scores, np.mean((descriptors - descriptors[chosen])**2, axis=1))

    fill(8)
    path8 = selected.copy()
    for i in final:
        if i not in selected:
            selected.append(i)
            scores = np.minimum(scores, np.mean((descriptors - descriptors[i])**2, axis=1))
    fill(16)
    others = np.asarray([i for i in range(len(rows)) if i != anchor_fid])
    random16 = [anchor_fid] + np.random.default_rng(seed).permutation(others)[:15].tolist()
    return {"path2048": path8, "path4096": selected, "random4096": random16}


def prepare_training(partition):
    output = DATA / "training_manifest.json"
    if output.exists():
        manifest = json.loads(output.read_text())
        if manifest["protocol"] != partition["protocol"]:
            raise ValueError("Training manifest protocol mismatch")
        for name, expected in (("path2048", 2048), ("path4096", 4096), ("random4096", 4096)):
            if len(read(DATA / f"{name}.extxyz", index=":")) != expected:
                raise ValueError(f"Wrong saved frame count: {name}")
        return manifest
    old = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    ids = sorted(x["sid"] for x in old["training_frames"])
    anchors = {x["sid"]: x["fid"] for x in old["training_frames"]}
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    if not all(x.exists() for x in sources):
        raise ValueError("All thirteen frame shards must be present before training-path preparation")
    table = pq.read_table(sources, columns=COLUMNS, filters=[("sid", "in", ids)]).sort_by([("sid", "ascending"), ("fid", "ascending")])
    counts = Counter(table["sid"].to_pylist())
    structures = {name: [] for name in ("path2048", "path4096", "random4096")}
    records = {name: [] for name in structures}
    splits = formula_splits()
    offset = 0
    for sid in ids:
        rows = table.slice(offset, counts[sid]).to_pylist()
        offset += counts[sid]
        indices = sample_reaction(rows, anchors[sid], partition["protocol"]["seed"] + sid)
        for name, fids in indices.items():
            for fid in fids:
                atoms = training_atoms(rows[fid], "train", splits["train"])
                structures[name].append(atoms)
                records[name].append({"sid": sid, "fid": int(fid)})
    for name, values in structures.items():
        write(DATA / f"{name}.extxyz", values)
        restored = read(DATA / f"{name}.extxyz", index=":")
        for original, saved in zip(values, restored, strict=True):
            np.testing.assert_allclose(original.positions, saved.positions, atol=1e-8, rtol=0)
            np.testing.assert_allclose(original.info["REF_energy"], saved.info["REF_energy"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(original.arrays["REF_forces"], saved.arrays["REF_forces"], atol=1e-8, rtol=0)
        print(f"Prepared {name}: {len(values)} frames from {len(ids)} complete histories", flush=True)
    manifest = {"protocol": partition["protocol"], "source_revision": REVISION,
        "training_reactions": ids, "source_history_frame_counts": dict(sorted(counts.items())),
        "datasets": records, "anchor_dataset": "data/v0.3.0/train_256.extxyz",
        "sampling": "all arms retain the original anchor; random16 is uniform without replacement; path8 covers endpoints, TS +/-1 and pair-distance FPS; path16 extends path8 with the entire final band and FPS"}
    save_json(output, manifest, immutable=True)
    return manifest


def prepare_meta(partition):
    """Only meta-development labels are opened; prospective holdouts remain untouched."""
    output = DATA / "meta_development.h5"
    if output.exists():
        return
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    table = pq.read_table(sources, columns=COLUMNS,
                          filters=[("sid", "in", partition["meta_development"])])
    write_path_dataset(table, partition["meta_development"], output, "train",
                       formula_splits()["train"], "decision_layer_development_only")
