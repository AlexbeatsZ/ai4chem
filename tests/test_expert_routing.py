"""Input symmetry, reference-label exclusion and supported-state boundaries."""
import unittest
import numpy as np
from expert_routing import endpoint_features


class RouterFeatureBoundaries(unittest.TestCase):
    def entry(self):
        return {"atomic_numbers": [6, 8, 1, 1],
            "reactant_positions": [[0.,0.,0.],[1.2,0.,0.],[-.5,.9,0.],[-.5,-.9,0.]],
            "product_positions": [[0.,0.,0.],[1.13,0.,0.],[5.,0.,0.],[5.74,0.,0.]]}

    def test_rigid_transform_and_consistent_atom_permutation_invariance(self):
        entry = self.entry()
        original = endpoint_features(entry)
        permutation = np.array([2, 0, 3, 1])
        rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        moved = {"atomic_numbers": np.asarray(entry["atomic_numbers"])[permutation]}
        for key in ("reactant_positions", "product_positions"):
            moved[key] = np.asarray(entry[key])[permutation] @ rotation + [3.,7.,-2.]
        actual = endpoint_features(moved)
        self.assertEqual(original.keys(), actual.keys())
        np.testing.assert_allclose(list(original.values()), list(actual.values()), atol=1e-10, equal_nan=True)

    def test_reference_and_identity_fields_are_not_features(self):
        entry = self.entry()
        poisoned = {**entry, "reaction": "rxn9999", "sid": 9999, "formula": "WRONG",
            "REF_energy": -999999., "reference_barrier_eV": 123456., "best_expert": "oracle"}
        np.testing.assert_allclose(list(endpoint_features(entry).values()),
            list(endpoint_features(poisoned).values()), equal_nan=True)

    def test_unsupported_state_and_elements_are_rejected(self):
        for changes in ({"charge": 1}, {"multiplicity": 3}, {"pbc": True}, {"atomic_numbers": [26,8,1,1]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                endpoint_features({**self.entry(), **changes})


if __name__ == "__main__":
    unittest.main()
