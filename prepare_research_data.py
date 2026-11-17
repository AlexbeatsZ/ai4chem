"""Fix development/final roles and prepare disjoint small training experiments."""

import ast
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import read, write

from download_file import download_file
from prepare_transition1x import REVISION, write_reaction
from transition1x_data import load_paths


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "v0.3.0"
OFFICIAL_SPLITS_URL = ("https://gitlab.com/matschreiner/Transition1x/-/raw/"
                       "08f728925d76e22ce04ec1ab3b75770d32e617ca/scripts/combine_dbs.py")


def formula_splits():
    path = ROOT / "data/raw/official_formula_splits.json"
    if path.exists():
        return json.loads(path.read_text())
    source = urllib.request.urlopen(OFFICIAL_SPLITS_URL, timeout=30).read().decode()
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("train_mols", "val_mols", "test_mols"):
                    result[target.id.replace("_mols", "")] = ast.literal_eval(node.value)
    if set(result) != {"train", "val", "test"}:
        raise ValueError("Could not extract official formula splits")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(result[left]) & set(result[right]):
            raise ValueError("Official formula splits overlap")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_path_dataset(table, ids, output, split, allowed_formulas, role):
    temporary = output.with_suffix(".h5.part")
    manifest = {"revision": REVISION, "source": "nimashoghi/transition1x Parquet export",
                "split": split, "role": role, "positions_forces_dtype": "float32", "reactions": []}
    with h5py.File(temporary, "w") as data:
        data.create_group("data")
        data.create_group(split)
        for sid in sorted(ids):
            rows = table.filter(pc.equal(table["sid"], sid)).to_pylist()
            manifest["reactions"].append(write_reaction(data, rows, sid, split, allowed_formulas))
        data.attrs["provenance"] = json.dumps(manifest)
    paths = load_paths(temporary, split=split, count=len(ids))
    temporary.replace(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def training_atoms(row, split, allowed_formulas):
    numbers = np.asarray(row["atomic_numbers"], dtype=int)
    np.testing.assert_array_equal(numbers, row["atomic_numbers"])
    atoms = Atoms(numbers=numbers, positions=row["pos"])
    formula = atoms.get_chemical_formula()
    forces = np.asarray(row["force"])
    if formula not in allowed_formulas or forces.shape != (len(atoms), 3):
        raise ValueError(f"Invalid {split} frame for reaction {row['sid']}")
    if not np.isfinite(row["y"]) or not np.isfinite(forces).all() or not np.isfinite(atoms.positions).all():
        raise ValueError("Non-finite training reference")
    atoms.info.update(REF_energy=float(row["y"]), config_type="Default", split=split,
                      reaction=f"rxn{row['sid']:04d}", source_frame=int(row["fid"]), formula=formula)
    atoms.arrays["REF_forces"] = forces
    return atoms


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "configs/v0.3.0.json").read_text())
    base = f"https://huggingface.co/datasets/nimashoghi/transition1x/resolve/{REVISION}/data/"
    for filename, local_name, size in (
        ("test-00000-of-00001.parquet", "transition1x_test.parquet", 123797333),
        ("val-00000-of-00001.parquet", "transition1x_val.parquet", 145224394),
        ("train-00000-of-00013.parquet", "transition1x_train_00.parquet", 378367220),
    ):
        download_file(base + filename, ROOT / "data/raw" / local_name, size)
    splits = formula_splits()
    columns = ["sid", "fid", "atomic_numbers", "pos", "y", "force"]
    test = pq.read_table(ROOT / "data/raw/transition1x_test.parquet", columns=columns)
    available = sorted(int(value) for value in pc.unique(test["sid"]).to_pylist())
    legacy = protocol["legacy_development_reactions"]
    remaining = sorted(set(available) - set(legacy))
    rng = np.random.default_rng(protocol["seed"])
    additional = rng.choice(remaining, protocol["development_reactions"] - len(legacy), replace=False).tolist()
    development = sorted(legacy + additional)
    final = sorted(set(available) - set(development))
    validation = pq.read_table(ROOT / "data/raw/transition1x_val.parquet", columns=columns)
    validation_ids = sorted(int(value) for value in pc.unique(validation["sid"]).to_pylist())
    selected_val = sorted(rng.choice(validation_ids, protocol["validation_reactions"], replace=False).tolist())
    train = pq.read_table(ROOT / "data/raw/transition1x_train_00.parquet", columns=columns)
    train_ids = sorted(int(value) for value in pc.unique(train["sid"]).to_pylist())
    selected_train = rng.choice(train_ids, max(protocol["train_sizes"]), replace=False).tolist()
    train_rows = []
    for sid in selected_train:
        candidates = train.filter(pc.equal(train["sid"], sid)).to_pylist()
        train_rows.append(candidates[int(rng.integers(len(candidates)))])
    training_ids = [int(row["sid"]) for row in train_rows]
    if (set(development) & set(final) or set(training_ids) & set(validation_ids)
            or set(training_ids) & set(available) or set(validation_ids) & set(available)):
        raise ValueError("Reaction IDs cross experiment roles")
    partition = {"source_revision": REVISION, "seed": protocol["seed"],
                 "development": development, "final": final, "validation": selected_val,
                 "training_frames": [{"sid": int(row["sid"]), "fid": int(row["fid"])} for row in train_rows],
                 "training_source": "train-00000-of-00013.parquet; one random configuration per selected reaction",
                 "train_sizes_are_nested": True}
    partition_path = ROOT / "configs/v0.3.0_partitions.json"
    if partition_path.exists() and json.loads(partition_path.read_text()) != partition:
        raise ValueError("Existing fixed experiment partition differs from the requested protocol")
    partition_path.write_text(json.dumps(partition, indent=2) + "\n")
    write_path_dataset(test, development, DATA / "development.h5", "test", splits["test"], "development")
    write_path_dataset(test, final, DATA / "final.h5", "test", splits["test"], "reserved_final")
    val_paths = write_path_dataset(validation, selected_val, DATA / "validation.h5", "val", splits["val"], "validation")
    for size in protocol["train_sizes"]:
        structures = [training_atoms(row, "train", splits["train"]) for row in train_rows[:size]]
        write(DATA / f"train_{size}.extxyz", structures)
        restored = read(DATA / f"train_{size}.extxyz", index=":")
        for original, saved in zip(structures, restored, strict=True):
            np.testing.assert_allclose(original.info["REF_energy"], saved.info["REF_energy"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(original.arrays["REF_forces"], saved.arrays["REF_forces"], atol=1e-8, rtol=0)
    val_structures = []
    for path in val_paths:
        for atoms in path.atoms():
            atoms.info["REF_energy"] = atoms.get_potential_energy()
            atoms.arrays["REF_forces"] = atoms.get_forces().copy()
            atoms.calc = None
            val_structures.append(atoms)
    write(DATA / "validation.extxyz", val_structures)
    print(f"PASS - fixed {len(development)} development, {len(final)} final, "
          f"{len(selected_val)} validation reactions; nested training sizes {protocol['train_sizes']}", flush=True)


if __name__ == "__main__":
    main()
