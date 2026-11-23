"""Endpoint-only invariant features and grouped evaluation utilities."""

from itertools import combinations_with_replacement

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import natural_cutoffs, neighbor_list

EXPERTS = ["off23-small", "mh1-omol", "mh0-rgd1", "legacy_mh1_256"]


def endpoint_features(entry):
    """Strict geometry allowlist: no ID, reference label, force or TS features."""
    numbers = np.asarray(entry["atomic_numbers"], dtype=int)
    if len(numbers) < 2 or entry.get("charge", 0) != 0 or entry.get("multiplicity", 1) != 1 or np.any(entry.get("pbc", False)):
        raise ValueError("First router requires a nonperiodic neutral-singlet molecular reaction")
    if not np.isin(numbers, [1, 6, 7, 8]).all():
        raise ValueError("First router only covers neutral-singlet CHNO reaction data")
    structures = [Atoms(numbers, positions=entry[key]) for key in ("reactant_positions", "product_positions")]
    if not all(np.isfinite(atoms.positions).all() for atoms in structures):
        raise ValueError("Non-finite router input")
    result = {"atoms": float(len(numbers))}
    for z, symbol in ((1, "H"), (6, "C"), (7, "N"), (8, "O")):
        result[f"count_{symbol}"] = float(np.sum(numbers == z))
    pairs = np.triu_indices(len(numbers), k=1)
    pair_numbers = np.sort(np.column_stack((numbers[pairs[0]], numbers[pairs[1]])), axis=1)
    radii = covalent_radii[numbers[pairs[0]]] + covalent_radii[numbers[pairs[1]]]
    bond_sets, distances = [], []
    for label, atoms in zip(("reactant", "product"), structures):
        left, right = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
        bonds = {tuple(sorted((int(i), int(j)))) for i, j in zip(left, right) if i != j}
        bond_sets.append(bonds)
        adjacency = [set() for _ in numbers]
        for i, j in bonds:
            adjacency[i].add(j)
            adjacency[j].add(i)
        unseen, components = set(range(len(numbers))), 0
        while unseen:
            stack = [unseen.pop()]
            components += 1
            while stack:
                for j in adjacency[stack.pop()] & unseen:
                    unseen.remove(j)
                    stack.append(j)
        result[f"{label}_components"] = float(components)
        d = np.linalg.norm(atoms.positions[pairs[0]] - atoms.positions[pairs[1]], axis=1)
        distances.append(d)
        ratio = d / radii
        for a, b in combinations_with_replacement([1, 6, 7, 8], 2):
            values = ratio[(pair_numbers[:, 0] == a) & (pair_numbers[:, 1] == b)]
            result[f"{label}_min_radius_ratio_{a}_{b}"] = float(np.min(values)) if len(values) else np.nan
        for q in (.1, .5, .9):
            result[f"{label}_distance_q{int(q*100)}"] = float(np.quantile(d, q))
    result["formed_bonds"] = float(len(bond_sets[1] - bond_sets[0]))
    result["broken_bonds"] = float(len(bond_sets[0] - bond_sets[1]))
    delta = np.abs(distances[1] - distances[0])
    result["mean_pair_distance_change"] = float(np.mean(delta))
    result["max_pair_distance_change"] = float(np.max(delta))
    result["rms_pair_distance_change"] = float(np.sqrt(np.mean(delta**2)))
    return result


def grouped_delta_interval(delta, groups, seed=20261123, repetitions=2000):
    """Formula-cluster bootstrap, retaining reaction weighting within each draw."""
    delta, groups = np.asarray(delta, float), np.asarray(groups)
    unique = np.unique(groups)
    totals = np.array([delta[groups == key].sum() for key in unique])
    counts = np.array([np.sum(groups == key) for key in unique])
    draws = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1/len(unique)), repetitions)
    samples = (draws @ totals) / (draws @ counts)
    return {"mean_eV": float(np.mean(delta)), "formula_cluster_ci95_eV": np.quantile(samples, [.025, .975]).tolist(),
            "clusters": len(unique), "replicates": repetitions}
