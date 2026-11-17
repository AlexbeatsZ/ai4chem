"""Prevent reference leakage and incorrect saddle/endpoint diagnostics."""

import tempfile
import unittest
import json
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io import write

from ts_search import (
    aligned_rmsd,
    assess_irc_branches,
    endpoint_pool,
    fragment_rmsds,
    prepare,
    projected_frequencies,
    select_reactions,
    verified_result,
    verified_saddle,
    vibrational_subspace,
)


class TransitionStateSearchTests(unittest.TestCase):
    def test_reader_needs_only_endpoint_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoints.h5"
            with h5py.File(path, "w") as source:
                reaction = source.create_group("test/H2/rxn1")
                reaction["atomic_numbers"] = [1, 1]
                reaction.create_group("reactant")["positions"] = [[[0, 0, 0], [0.7, 0, 0]]]
                reaction.create_group("product")["positions"] = [[[0, 0, 0], [2, 0, 0]]]
            pool = endpoint_pool({"input": path, "split": "test"})
            self.assertEqual(pool[0]["kind"], "cleavage")
            self.assertEqual(pool[0]["broken"], 1)

    def test_saved_endpoint_inputs_run_without_the_reference_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"input": str(root / "unavailable-reference.h5"), "split": "test", "count": 1}
            item = {"formula": "H2", "reaction": "rxn1", "atoms": 2, "kind": "cleavage", "formed": 0, "broken": 1}
            (root / "selection.json").write_text(json.dumps({"config": config, "reactions": [item]}))
            folder = root / "inputs" / "rxn1"
            folder.mkdir(parents=True)
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
            write(folder / "reactant.xyz", atoms)
            atoms.positions[1, 0] = 2
            write(folder / "product.xyz", atoms)
            selected = prepare(config, root)
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(selected[0]["endpoints"][1].get_distance(0, 1), 2)

    def test_linear_molecule_removes_five_rigid_modes(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        self.assertEqual(vibrational_subspace(atoms).shape, (6, 1))

    def test_bent_molecule_removes_six_rigid_modes(self):
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        self.assertEqual(vibrational_subspace(atoms).shape, (9, 3))
        frequencies, _ = projected_frequencies(atoms, np.eye(9))
        self.assertTrue(np.all(frequencies > 0))

    def test_projected_hessian_keeps_a_single_unstable_vibration(self):
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        basis = vibrational_subspace(atoms)
        weights = np.repeat(np.sqrt(atoms.get_masses()), 3)
        hessian = weights[:, None] * (basis @ np.diag([-1, 2, 3]) @ basis.T) * weights[None, :]
        frequencies, _ = projected_frequencies(atoms, hessian)
        self.assertEqual(np.count_nonzero(frequencies < -50), 1)

    def test_alignment_preserves_atom_mapping_and_proper_rotation(self):
        initial = Atoms("CHNO", positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        rotated = initial.copy()
        rotated.rotate(80, "z")
        rotated.translate([3, 4, 5])
        self.assertLess(aligned_rmsd(initial, rotated), 1e-10)
        reflected = initial.copy()
        reflected.positions[:, 0] *= -1
        self.assertGreater(aligned_rmsd(initial, reflected), 0.1)

    def test_selection_covers_formulas_and_missing_bond_change_kind(self):
        pool = [{"formula": formula, "reaction": name, "atoms": 5, "kind": kind,
                 "formed": formed, "broken": broken}
                for formula, name, kind, formed, broken in [
                    ("a", "rxn1", "rearrangement", 1, 1),
                    ("b", "rxn2", "cleavage", 0, 1),
                    ("a", "rxn3", "formation", 1, 0),
                    ("a", "rxn4", "rearrangement", 1, 2)]]
        selected = select_reactions(pool, 3)
        self.assertEqual({item["formula"] for item in selected}, {"a", "b"})
        self.assertEqual({item["kind"] for item in selected}, {"formation", "cleavage", "rearrangement"})

    def test_separated_product_fragments_may_translate_and_rotate_independently(self):
        reactant = Atoms("HHHH", positions=[[0, 0, 0], [0.7, 0, 0], [4, 0, 0], [4.7, 0, 0]])
        product = Atoms("HHHH", positions=[[0, 0, 0], [4, 0, 0], [0.7, 0, 0], [4.7, 0, 0]])
        shifted = product.copy()
        shifted.positions[[1, 3]] += [0, 4, 0]
        self.assertGreater(aligned_rmsd(product, shifted), 0.5)
        self.assertLess(max(fragment_rmsds(product, shifted)), 1e-10)
        branches = [{"converged": True, "minimum_converged": True} for _ in range(2)]
        result = assess_irc_branches(branches, [reactant.copy(), shifted], [reactant, product], 0.5)
        self.assertTrue(result["expected_bond_connectivity"])
        self.assertTrue(result["converged_and_connected"])

    def test_valid_refined_saddle_survives_neb_iteration_limit(self):
        result = {"endpoints": [{"converged": True, "input_bonds_preserved": True}] * 2,
                  "neb": {"converged": False}, "saddle": {"converged": True, "fmax_eV_A": 0.001},
                  "vibrations": {"imaginary_modes_above_threshold": 1},
                  "irc": {"converged_and_connected": True}}
        config = {"ts_fmax_eV_A": 0.01}
        self.assertTrue(verified_saddle(result, config))
        self.assertFalse(verified_result(result, config))


if __name__ == "__main__":
    unittest.main()
