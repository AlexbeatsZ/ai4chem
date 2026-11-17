"""Read original-schema Transition1x HDF5 and reconstruct final NEB bands."""

from dataclasses import dataclass

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"


def final_band_indices(frame_count):
    """The writer stores ten first-band images, then eight interior images per band."""
    if frame_count < 10 or (frame_count - 10) % 8:
        raise ValueError(f"Unsupported NEB frame count {frame_count}")
    interior = np.arange(1, 9) if frame_count == 10 else np.arange(frame_count - 8, frame_count)
    return np.concatenate(([0], interior, [9]))


@dataclass
class ReactionPath:
    split: str
    formula: str
    reaction: str
    numbers: np.ndarray
    positions: np.ndarray
    energies: np.ndarray
    forces: np.ndarray
    source_indices: np.ndarray
    source_frame_count: int
    reference_ts_image: int

    def atoms(self):
        structures = []
        for positions, energy, forces in zip(self.positions, self.energies, self.forces):
            atoms = Atoms(numbers=self.numbers, positions=positions)
            atoms.info.update(split=self.split, formula=self.formula, reaction=self.reaction)
            atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces)
            structures.append(atoms)
        return structures


def read_reaction(group, split, formula, reaction):
    numbers = group["atomic_numbers"][()]
    count = group["positions"].shape[0]
    indices = final_band_indices(count)
    # HDF5 indexing requires increasing indices, so read interiors and endpoints separately.
    interior = indices[1:-1]
    positions = np.concatenate((group["positions"][0:1], group["positions"][interior],
                                group["positions"][9:10]))
    energies = np.concatenate((group[ENERGY_KEY][0:1], group[ENERGY_KEY][interior],
                               group[ENERGY_KEY][9:10])).reshape(-1)
    forces = np.concatenate((group[FORCES_KEY][0:1], group[FORCES_KEY][interior],
                             group[FORCES_KEY][9:10]))
    expected_shape = (10, len(numbers), 3)
    if positions.shape != expected_shape or forces.shape != expected_shape or energies.shape != (10,):
        raise ValueError(f"Invalid reaction shapes in {group.name}")
    if not all(np.isfinite(array).all() for array in (positions, energies, forces)):
        raise ValueError(f"Non-finite reference data in {group.name}")
    if not np.isin(numbers, [1, 6, 7, 8]).all():
        raise ValueError(f"Unexpected elements in {group.name}")
    for endpoint, index in (("reactant", 0), ("product", -1)):
        np.testing.assert_array_equal(group[endpoint]["atomic_numbers"][()], numbers)
        np.testing.assert_allclose(group[endpoint]["positions"][0], positions[index], atol=1e-6, rtol=0)
        np.testing.assert_allclose(group[endpoint][ENERGY_KEY][0], energies[index], atol=1e-7, rtol=0)
        np.testing.assert_allclose(group[endpoint][FORCES_KEY][0], forces[index], atol=1e-6, rtol=0)
    ts = int(np.argmax(energies)) if count == 10 else 1 + int(np.argmax(energies[1:-1]))
    np.testing.assert_array_equal(group["transition_state"]["atomic_numbers"][()], numbers)
    np.testing.assert_allclose(group["transition_state"]["positions"][0], positions[ts], atol=1e-6, rtol=0)
    np.testing.assert_allclose(group["transition_state"][ENERGY_KEY][0], energies[ts], atol=1e-7, rtol=0)
    np.testing.assert_allclose(group["transition_state"][FORCES_KEY][0], forces[ts], atol=1e-6, rtol=0)
    return ReactionPath(split, formula, reaction, numbers, positions, energies, forces, indices, count, ts)


def load_paths(path, split="test", count=10, seed=42):
    if count < 1:
        raise ValueError("Reaction count must be positive")
    with h5py.File(path, "r") as data:
        available = [(formula, reaction) for formula in sorted(data[split])
                     for reaction in sorted(data[split][formula])]
        if len(available) < count:
            raise ValueError(f"Requested {count} reactions, found {len(available)} in {split}")
        selected = np.random.default_rng(seed).choice(len(available), size=count, replace=False)
        return [read_reaction(data[split][formula][reaction], split, formula, reaction)
                for formula, reaction in sorted(available[index] for index in selected)]
