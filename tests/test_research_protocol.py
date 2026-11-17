"""Regression checks for experiment independence and clustered uncertainty."""

import json
import unittest
from pathlib import Path

import numpy as np

from analyze_benchmark import bootstrap_mean


ROOT = Path(__file__).resolve().parents[1]


class ResearchProtocolTests(unittest.TestCase):
    def test_fixed_roles_are_disjoint_and_cover_the_original_test_pool(self):
        partitions = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
        development, final = set(partitions["development"]), set(partitions["final"])
        training = {entry["sid"] for entry in partitions["training_frames"]}
        validation = set(partitions["validation"])
        self.assertFalse(development & final)
        self.assertEqual(len(development | final), 287)
        self.assertFalse(training & validation)
        self.assertFalse((training | validation) & (development | final))
        legacy = json.loads((ROOT / "configs/v0.3.0.json").read_text())["legacy_development_reactions"]
        self.assertTrue(set(legacy) <= development)

    def test_training_frames_are_unique_and_reaction_balanced(self):
        frames = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())["training_frames"]
        self.assertEqual(len(frames), 256)
        self.assertEqual(len({frame["sid"] for frame in frames}), 256)

    def test_bootstrap_resamples_whole_clusters(self):
        result = bootstrap_mean([0, 0, 10, 10], ["a", "a", "b", "b"], replicates=2000)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["mean"], 5)
        self.assertEqual(result["lower_95"], 0)
        self.assertEqual(result["upper_95"], 10)

    def test_zero_errors_have_zero_uncertainty(self):
        result = bootstrap_mean(np.zeros(5), np.arange(5))
        self.assertEqual(result["lower_95"], 0)
        self.assertEqual(result["upper_95"], 0)

    def test_force_bootstrap_keeps_component_weights(self):
        result = bootstrap_mean([0, 4], ["a", "b"], observation_weights=[1, 3], root=True)
        self.assertAlmostEqual(result["mean"], np.sqrt(3))


if __name__ == "__main__":
    unittest.main()
