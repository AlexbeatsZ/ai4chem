"""Freeze disjoint training regions, shared labels, validation gates and rehearsal."""

from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from ase.io import read, write
from model_study_data import (COLUMNS, diverse_subset, endpoint_pool, sample_reaction, save_json)
from prepare_research_data import formula_splits, training_atoms
from specialization_gate import fit_gate, predict_gate

DATA = ROOT / "data/v0.8.0"
CONFIG = ROOT / "configs/v0.8.0_partitions.json"


def freeze(protocol):
    previous = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    old = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    historic = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    old_ids = {int(x["sid"]) for x in old["training_frames"]}
    excluded = set(old_ids)
    for key in ("validation", "meta_development", "holdout_train", "holdout_val"):
        excluded.update(previous[key])
    for key in ("development", "final", "validation"):
        excluded.update(old[key])
    excluded.update(int(x["sid"]) for x in historic["reactions"])
    if CONFIG.exists():
        partition = json.loads(CONFIG.read_text())
        if partition["protocol"] != protocol:
            raise ValueError("Frozen specialization protocol changed")
        if set(partition["additional_training_ids"]) & excluded:
            raise ValueError("Additional training reaction leaks across a scientific role")
        if set(partition["training_ids"]) != old_ids | set(partition["additional_training_ids"]):
            raise ValueError("Old anchors/additional training identity changed")
        return partition
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    if not all(path.exists() for path in sources):
        raise ValueError("All thirteen source shards are required")
    counts_table = pq.read_table(sources, columns=["sid", "fid"]).group_by("sid").aggregate([("fid", "count")])
    counts = dict(zip(counts_table["sid"].to_pylist(), counts_table["fid_count"].to_pylist()))
    splits = formula_splits()
    pool = endpoint_pool(sources, set(), splits["train"])
    candidates = [row for row in pool if row["sid"] not in excluded and counts[row["sid"]] >= 18]
    added = diverse_subset(candidates, protocol["new_reactions"], protocol["seed"])
    train_ids = sorted(old_ids | {row["sid"] for row in added})
    by_id = {row["sid"]: row for row in pool}
    records = [by_id[sid] for sid in train_ids]
    gate, labels = fit_gate(records, protocol["seed"])
    val_pool = endpoint_pool(ROOT / "data/raw/transition1x_val.parquet", set(), splits["val"])
    val = sorted((row for row in val_pool if row["sid"] in previous["validation"]), key=lambda x:x["sid"])
    val_labels, _ = predict_gate(val, gate)
    counts = np.bincount(val_labels, minlength=2)
    if min(counts) < protocol["minimum_validation_reactions_per_region"]:
        raise ValueError(f"Too few geometry-assigned validation reactions: {counts.tolist()}")
    regions = {str(k): [row["sid"] for row,label in zip(records,labels) if label == k] for k in (0, 1)}
    replay = {str(k): sorted(row["sid"] for row in diverse_subset(
        [row for row,label in zip(records,labels) if label != k],
        protocol["replay_reactions_per_region"], protocol["seed"]+100+k)) for k in (0, 1)}
    partition = {"protocol": protocol, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_ids": train_ids, "old_anchor_reaction_ids": sorted(old_ids),
        "additional_training_ids": sorted(row["sid"] for row in added), "regions": regions,
        "target_replay_ids": replay, "endpoint_gate": gate,
        "regional_validation": {str(k): [row["sid"] for row,label in zip(val,val_labels) if label == k] for k in (0,1)},
        "training_composition": dict(Counter(row["kind"] for row in records)),
        "regional_composition": {str(k):dict(Counter(row["kind"] for row,label in zip(records,labels) if label == k)) for k in (0,1)},
        "training_formulas": len({row["formula"] for row in records}),
        "labels_used_for_partition": False, "holdout_labels_evaluated": False,
        "selection_inputs": ["endpoint elements/coordinates", "old anchor IDs", "source fid counts", "excluded reaction roles"]}
    save_json(CONFIG, partition, immutable=True)
    DATA.mkdir(parents=True, exist_ok=True)
    save_json(DATA / "training_endpoint_records.json", records, immutable=True)
    print("Frozen specialization roles", len(train_ids), "regional validation", counts.tolist(), flush=True)
    return partition


def prepare(partition):
    if (DATA / "manifest.json").exists():
        manifest = json.loads((DATA / "manifest.json").read_text())
        if manifest["partition"] != partition:
            raise ValueError("Existing specialization data manifest changed")
        for name, rows in manifest["datasets"].items():
            if len(read(DATA / f"{name}.extxyz", index=":")) != len(rows):
                raise ValueError(f"Cached dataset count differs: {name}")
        return
    protocol = partition["protocol"]
    old = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    anchors = {row["sid"]:row["fid"] for row in old["training_frames"]}
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    table = pq.read_table(sources, columns=COLUMNS, filters=[("sid","in",partition["training_ids"])]).sort_by([("sid","ascending"),("fid","ascending")])
    counts = Counter(table["sid"].to_pylist())
    structures, identities = {}, {}
    allowed_formulas = formula_splits()["train"]
    offset = 0
    for sid in partition["training_ids"]:
        rows = table.slice(offset, counts[sid]).to_pylist()
        offset += counts[sid]
        seed = protocol["seed"]+sid
        anchor = anchors.get(sid, int(np.random.default_rng(seed).integers(len(rows))))
        fids = sample_reaction(rows, anchor, seed)["path2048"]
        structures[sid] = [training_atoms(rows[fid], "train", allowed_formulas) for fid in fids]
        identities[sid] = [{"sid":sid,"fid":int(fid)} for fid in fids]
    datasets = {"general4096": partition["training_ids"]}
    for region in ("0","1"):
        datasets[f"region{region}_2048"] = partition["regions"][region]
        datasets[f"replay{region}_2560"] = sorted(partition["regions"][region]+partition["target_replay_ids"][region])
    manifest = {"partition": partition, "datasets": {}, "holdout_labels_evaluated":False}
    for name, ids in datasets.items():
        atoms = [item for sid in ids for item in structures[sid]]
        manifest["datasets"][name] = [item for sid in ids for item in identities[sid]]
        write(DATA / f"{name}.extxyz", atoms)
        restored = read(DATA / f"{name}.extxyz", index=":")
        for original,saved in zip(atoms,restored,strict=True):
            np.testing.assert_allclose(original.positions,saved.positions,atol=1e-8,rtol=0)
            np.testing.assert_allclose(original.info["REF_energy"],saved.info["REF_energy"],atol=1e-8,rtol=0)
            np.testing.assert_allclose(original.arrays["REF_forces"],saved.arrays["REF_forces"],atol=1e-8,rtol=0)
        print("Prepared",name,len(atoms),flush=True)
    validation = read(ROOT / "data/v0.7.0/validation.extxyz", index=":")
    for region in ("0","1"):
        ids = {f"rxn{sid:04d}" for sid in partition["regional_validation"][region]}
        selected = [atoms for atoms in validation if atoms.info["reaction"] in ids]
        if len(selected) != len(ids)*10:
            raise ValueError("Wrong regional validation image count")
        write(DATA / f"validation_region{region}.extxyz",selected)
    save_json(DATA / "manifest.json", manifest, immutable=True)
    print("PASS - shared labels and regional validation prepared, no prospective holdout labels",flush=True)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    lock = (DATA / "preparation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((ROOT / "configs/v0.8.0.json").read_text())
    prepare(freeze(protocol))
