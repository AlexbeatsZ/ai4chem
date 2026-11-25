"""Nested reaction coverage expansion with immutable source identities."""

from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase.io import read, write

from model_study_data import (COLUMNS, ROOT, diverse_subset, endpoint_pool,
                              sample_reaction, save_json)
from prepare_research_data import formula_splits, training_atoms
from prepare_transition1x import REVISION

DATA = ROOT / "data/v0.10.0"
PARTITION = ROOT / "configs/v0.10.0_partitions.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def excluded_roles(previous, old, historic):
    roles = {role: set(previous[role]) for role in
             ("validation", "meta_development", "holdout_train", "holdout_val")}
    roles["historic_evaluation"] = set().union(*(set(old[key]) for key in
        ("development", "final", "validation"))) | {int(x["sid"]) for x in historic["reactions"]}
    return roles


def validate_partition(partition, protocol, base, roles):
    if partition["protocol"] != protocol or protocol["source_revision"] != REVISION:
        raise ValueError("Frozen growth protocol or source revision changed")
    baseline = set(base["training_ids"])
    additions = partition["additional_training_ids"]
    if len(additions) != len(set(additions)) or set(additions) & baseline:
        raise ValueError("Duplicated or reused additional training reaction")
    blocked = set().union(*roles.values())
    if set(partition["training_ids"]) & blocked:
        raise ValueError("Growth training reaction leaks into an evaluation role")
    if set(partition["training_ids"]) != baseline | set(additions):
        raise ValueError("Growth training identity changed")
    if partition["baseline_partition_frozen_at_utc"] != base["frozen_at_utc"]:
        raise ValueError("Baseline partition changed")
    if sorted(x["sid"] for x in partition["endpoint_records"]) != sorted(partition["training_ids"]):
        raise ValueError("Endpoint records differ from training identities")
    prior = set()
    for size in protocol["reaction_counts"]:
        ids = partition["datasets"][str(size)]
        expected = baseline | set(additions[:size-len(baseline)])
        if len(ids) != size or set(ids) != expected or not prior <= set(ids):
            raise ValueError("Growth datasets must retain the baseline and be nested")
        prior = set(ids)


def freeze(protocol):
    previous = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    base = json.loads((ROOT / "configs/v0.8.0_partitions.json").read_text())
    old = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    historic = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    roles = excluded_roles(previous, old, historic)
    if PARTITION.exists():
        partition = json.loads(PARTITION.read_text())
        validate_partition(partition, protocol, base, roles)
        return partition
    sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
    if not all(x.exists() for x in sources):
        raise ValueError("All thirteen pinned shards are required")
    counts_table = pq.read_table(sources, columns=["sid", "fid"]).group_by("sid").aggregate([("fid", "count")])
    counts = dict(zip(counts_table["sid"].to_pylist(), counts_table["fid_count"].to_pylist()))
    pool = endpoint_pool(sources, set(), formula_splits()["train"])
    baseline = set(base["training_ids"])
    excluded = baseline | set().union(*roles.values())
    candidates = [x for x in pool if x["sid"] not in excluded and counts[x["sid"]] >= 16]
    added = diverse_subset(candidates, max(protocol["reaction_counts"])-len(baseline), protocol["seed"])
    added_ids = [x["sid"] for x in added]
    ids = sorted(baseline | set(added_ids))
    by_id = {x["sid"]: x for x in pool}
    records = [by_id[sid] for sid in ids]
    partition = dict(protocol=protocol, frozen_at_utc=now(), training_ids=ids,
        baseline_partition_frozen_at_utc=base["frozen_at_utc"],
        prior_roles_frozen_at_utc=previous["frozen_at_utc"],
        baseline_training_ids=sorted(baseline), additional_training_ids=added_ids,
        datasets={str(n): sorted(baseline | set(added_ids[:n-len(baseline)]))
                  for n in protocol["reaction_counts"]},
        endpoint_records=records, training_formulas=len({x["formula"] for x in records}),
        training_composition=dict(Counter(x["kind"] for x in records)),
        source_history_frame_counts={str(sid): counts[sid] for sid in ids},
        selection_inputs=["endpoint elements/coordinates", "frame counts", "excluded reaction IDs"],
        reference_labels_used_for_reaction_selection=False, holdout_labels_evaluated=False)
    validate_partition(partition, protocol, base, roles)
    save_json(PARTITION, partition, immutable=True)
    print(f"Frozen growth roles: {protocol['reaction_counts']}, {len(records)} reactions", flush=True)
    return partition


def prepare(partition):
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "manifest.json"
    protocol = partition["protocol"]
    if target.exists():
        manifest = json.loads(target.read_text())
        if manifest["partition"] != partition:
            raise ValueError("Prepared growth partition changed")
        for size in protocol["reaction_counts"]:
            if len(read(DATA / f"coverage{size}.extxyz", index=":")) != size*8:
                raise ValueError("Cached growth dataset count differs")
        return manifest
    # A failed preparation retains its evidence rather than overwriting partial data.
    state_path = DATA / "preparation_state.json"
    if state_path.exists():
        raise RuntimeError("Incomplete preparation has evidence; inspect before an explicit retry")
    state = dict(stage="preparing", started_at_utc=now(), completed_reactions=512,
                 holdout_labels_evaluated=False)
    save_json(state_path, state)
    try:
        baseline_manifest = json.loads((ROOT / "data/v0.8.0/manifest.json").read_text())
        baseline_atoms = read(ROOT / "data/v0.8.0/general4096.extxyz", index=":")
        identities = baseline_manifest["datasets"]["general4096"]
        if baseline_manifest["partition"]["frozen_at_utc"] != partition["baseline_partition_frozen_at_utc"]:
            raise ValueError("Baseline labels have the wrong partition")
        if len(baseline_atoms) != 4096 or len(identities) != 4096:
            raise ValueError("Wrong baseline label count")
        structures, records = {}, {}
        for atoms, identity in zip(baseline_atoms, identities, strict=True):
            sid, fid = int(identity["sid"]), int(identity["fid"])
            if atoms.info["reaction"] != f"rxn{sid:04d}" or atoms.info["source_frame"] != fid:
                raise ValueError("Baseline frame identity mismatch")
            structures.setdefault(sid, []).append(atoms)
            records.setdefault(sid, []).append(identity)
        if set(structures) != set(partition["baseline_training_ids"]) or any(len(x) != 8 for x in structures.values()):
            raise ValueError("Every baseline reaction must retain exactly eight original labels")
        sources = [ROOT / f"data/raw/transition1x_train_{i:02d}.parquet" for i in range(13)]
        allowed = formula_splits()["train"]
        added = partition["additional_training_ids"]
        batch_size = protocol["preparation_batch_reactions"]
        for first in range(0, len(added), batch_size):
            batch_ids = sorted(added[first:first+batch_size])
            table = pq.read_table(sources, columns=COLUMNS, filters=[("sid", "in", batch_ids)],
                                  use_threads=False).sort_by([("sid", "ascending"), ("fid", "ascending")])
            counts = Counter(table["sid"].to_pylist())
            offset = 0
            for sid in batch_ids:
                rows = table.slice(offset, counts[sid]).to_pylist()
                offset += counts[sid]
                if counts[sid] != partition["source_history_frame_counts"][str(sid)]:
                    raise ValueError("Complete source history count changed")
                if any(row["atomic_numbers"] != rows[0]["atomic_numbers"] for row in rows):
                    raise ValueError("Atomic mapping changed within a reaction")
                seed = protocol["seed"]+sid
                anchor = int(np.random.default_rng(seed).integers(len(rows)))
                fids = sample_reaction(rows, anchor, seed)["path2048"]
                structures[sid] = [training_atoms(rows[fid], "train", allowed) for fid in fids]
                records[sid] = [dict(sid=sid, fid=int(fid)) for fid in fids]
                del rows
            del table
            gc.collect()
            state.update(completed_reactions=len(structures), updated_at_utc=now())
            save_json(state_path, state)
            print(f"Prepared complete histories {len(structures)}/{len(partition['training_ids'])}", flush=True)
        manifest = dict(partition=partition, datasets={}, holdout_labels_evaluated=False,
                        baseline_frames_retained=4096, completed_at_utc=now())
        for size in protocol["reaction_counts"]:
            ids = partition["datasets"][str(size)]
            atoms = [x for sid in ids for x in structures[sid]]
            selected = [x for sid in ids for x in records[sid]]
            if len(atoms) != size*8 or len({(x['sid'], x['fid']) for x in selected}) != size*8:
                raise ValueError("Growth frame budget or uniqueness differs")
            output = DATA / f"coverage{size}.extxyz"
            write(output, atoms)
            restored = read(output, index=":")
            for original, saved in zip(atoms, restored, strict=True):
                np.testing.assert_allclose(original.positions, saved.positions, atol=1e-8, rtol=0)
                np.testing.assert_allclose(original.info["REF_energy"], saved.info["REF_energy"], atol=1e-8, rtol=0)
                np.testing.assert_allclose(original.arrays["REF_forces"], saved.arrays["REF_forces"], atol=1e-8, rtol=0)
            manifest["datasets"][f"coverage{size}"] = selected
        save_json(target, manifest, immutable=True)
        state.update(stage="completed", finished_at_utc=now())
        save_json(state_path, state)
        return manifest
    except Exception as error:
        state.update(stage="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        save_json(state_path, state)
        raise
