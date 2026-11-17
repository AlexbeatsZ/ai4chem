"""Create a traceable ten-reaction original-schema HDF5 from the JMP public export."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ase import Atoms

from download_file import download_file
from transition1x_data import ENERGY_KEY, FORCES_KEY, final_band_indices, load_paths


REVISION = "fac02fc7019ae95a6957b0723b467f6e56745fd2"
URL = (f"https://huggingface.co/datasets/nimashoghi/transition1x/resolve/{REVISION}/"
       "data/test-00000-of-00001.parquet")
OFFICIAL_TEST_FORMULAS = {"C3H5NO2", "C2H5NO2", "C2H3N3O2", "C2HNO3", "C3H8N2O", "C3HNO2", "C5H5NO", "C2HNO"}


def write_group(group, numbers, positions, energies, forces):
    group.create_dataset("atomic_numbers", data=numbers)
    group.create_dataset("positions", data=positions, compression="gzip")
    group.create_dataset(ENERGY_KEY, data=energies, compression="gzip")
    group.create_dataset(FORCES_KEY, data=forces, compression="gzip")


def write_reaction(output, rows, sid, split, allowed_formulas):
    rows.sort(key=lambda row: row["fid"])
    if [row["fid"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Missing or duplicated frame IDs for reaction {sid}")
    numbers = np.asarray(rows[0]["atomic_numbers"], dtype=np.int64)
    for row in rows:
        np.testing.assert_array_equal(row["atomic_numbers"], numbers)
    positions = np.asarray([row["pos"] for row in rows], dtype=np.float32)
    energies = np.asarray([row["y"] for row in rows], dtype=np.float64)
    forces = np.asarray([row["force"] for row in rows], dtype=np.float32)
    formula = Atoms(numbers=numbers).get_chemical_formula()
    if formula not in allowed_formulas:
        raise ValueError(f"Formula {formula} is outside the original {split} split")
    indices = final_band_indices(len(rows))
    if positions.shape != (len(rows), len(numbers), 3) or forces.shape != positions.shape:
        raise ValueError(f"Invalid dimensions for reaction {sid}")
    if not all(np.isfinite(values).all() for values in (positions, energies, forces)):
        raise ValueError(f"Non-finite values for reaction {sid}")
    reaction = f"rxn{int(sid):04d}"
    group = output["data"].require_group(formula).create_group(reaction)
    write_group(group, numbers, positions, energies, forces)
    ts_candidates = indices if len(rows) == 10 else indices[1:-1]
    ts = ts_candidates[np.argmax(energies[ts_candidates])]
    for name, index in (("reactant", 0), ("product", 9), ("transition_state", ts)):
        write_group(group.create_group(name), numbers, positions[index:index + 1],
                    energies[index:index + 1], forces[index:index + 1])
    if formula not in output[split]:
        output[split][formula] = h5py.SoftLink(f"/data/{formula}")
    return {"reaction": reaction, "formula": formula, "source_frames": len(rows), "atoms": len(numbers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"data/transition1x_test_{args.count}.h5")
    parquet = download_file(URL, "data/raw/transition1x_test.parquet", 123797333)
    table = pq.read_table(parquet, columns=["sid", "fid", "atomic_numbers", "pos", "y", "force"])
    available = np.sort(pc.unique(table["sid"]).to_numpy())
    if args.count < 1 or args.count > len(available):
        raise ValueError(f"Reaction count must be in [1, {len(available)}]")
    selected = np.sort(np.random.default_rng(args.seed).choice(available, args.count, replace=False))
    table = table.filter(pc.is_in(table["sid"], value_set=pa.array(selected)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".h5.part")
    manifest = {
        "source_type": "JMP-author Hugging Face Parquet export",
        "source_url": URL,
        "source_revision": REVISION,
        "original_dataset_doi": "10.6084/m9.figshare.19614657.v4",
        "original_hdf5_download_status": "HTTP 403 during 2026-11-16 access attempts",
        "split": "test", "seed": args.seed, "available_reactions": len(available),
        "selection": "uniform random reaction IDs without replacement, independent of energies",
        "positions_and_forces_precision": "float32 in the public export",
        "energies_precision": "float64 in the public export",
        "endpoint_source": "root frames 0 and 9, as defined by the original HDF5 writer",
        "transition_state_source": "original writer argmax on the final saved band; all ten images for a single band, eight interiors otherwise",
        "reactions": [],
    }
    with h5py.File(temporary, "w") as output:
        output.create_group("data")
        for split in ("train", "val", "test"):
            output.create_group(split)
        for sid in selected:
            rows = table.filter(pc.equal(table["sid"], int(sid))).to_pylist()
            entry = write_reaction(output, rows, sid, "test", OFFICIAL_TEST_FORMULAS)
            manifest["reactions"].append(entry)
            print(f"Prepared test/{entry['formula']}/{entry['reaction']}, {len(rows)} source frames", flush=True)
        output.attrs["provenance"] = json.dumps(manifest)
    load_paths(temporary, count=args.count, seed=args.seed)
    temporary.replace(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS - {args.count} reaction groups and final NEB bands validated", flush=True)


if __name__ == "__main__":
    main()
