"""Scientific boundaries: source completeness, nested budgets and role leakage."""

import unittest
import numpy as np

from model_study_data import sample_reaction, validate_roles
from transition1x_data import final_band_indices


class SamplingBoundaries(unittest.TestCase):
    def rows(self):
        rows = []
        for fid in range(42):
            x = fid / 41
            rows.append({"sid": 7, "fid": fid, "atomic_numbers": [1, 6, 8],
                         "pos": [[0, 0, 0], [1+x, .2*x, 0], [0, 1+x*x, .3*x]],
                         "y": -(fid-38)**2})
        return rows

    def test_unique_nested_exact_budget_and_final_band_coverage(self):
        sampled = sample_reaction(self.rows(), 20, 23)
        for name, count in (("path2048", 8), ("path4096", 16), ("random4096", 16)):
            self.assertEqual(len(sampled[name]), count)
            self.assertEqual(len(set(sampled[name])), count)
            self.assertIn(20, sampled[name])
        self.assertEqual(sampled["path4096"][:8], sampled["path2048"])
        self.assertTrue(set(final_band_indices(42)).issubset(sampled["path4096"]))
        self.assertTrue({0, 9, 37, 38, 39}.issubset(sampled["path2048"]))

    def test_missing_frame_shard_is_rejected(self):
        rows = self.rows()
        del rows[17]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            sample_reaction(rows, 20, 23)

    def test_pair_distance_sampling_is_rotation_and_translation_invariant(self):
        rows = self.rows()
        original = sample_reaction(rows, 20, 23)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = [{**row, "pos": (np.asarray(row["pos"]) @ rotation + [4., 8., -2.]).tolist()} for row in rows]
        self.assertEqual(original, sample_reaction(moved, 20, 23))

    def test_reference_and_prior_project_reactions_cannot_enter_new_holdouts(self):
        old = {"training_frames": [{"sid": 1, "fid": 4}], "development": [2], "final": [3], "validation": [4]}
        partition = {"validation": [4, 5], "meta_development": [6], "holdout_train": [7], "holdout_val": [8]}
        validate_roles(partition, old, {9})
        for role, sid in (("holdout_train", 1), ("meta_development", 3), ("holdout_val", 9), ("holdout_train", 6)):
            broken = {key: list(values) for key, values in partition.items()}
            broken[role] = [sid]
            with self.subTest(role=role, sid=sid), self.assertRaises(ValueError):
                validate_roles(broken, old, {9})


if __name__ == "__main__":
    unittest.main()
