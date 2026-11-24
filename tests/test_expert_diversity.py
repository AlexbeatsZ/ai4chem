"""Distinguish per-reaction guards from genuine aggregate complementary choices."""

import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_expert_diversity import analyze


class OracleGuardTests(unittest.TestCase):
    def table(self, alternate_forces):
        rows = []
        for reaction, alternate_force, alternate_barrier in zip(
                ("r1", "r2"), alternate_forces, (.5, 2.1)):
            for model, barrier, force in (("base", 2., 1.), ("other", alternate_barrier, alternate_force)):
                rows.append({"reaction": reaction, "model": model, "atoms": 5,
                    "barrier_error_eV": barrier, "force_component_rmse_eV_per_A": force,
                    "ts_region_force_component_rmse_eV_per_A": 1. if model == "base" else .8,
                    "reaction_energy_error_eV": .2 if model == "base" else .1})
        return pd.DataFrame(rows)

    def test_aggregate_tradeoff_can_exist_without_joint_per_reaction_winner(self):
        report = analyze(self.table([1.2, .5]), ["base", "other"], baseline="base")
        self.assertEqual(report["joint_barrier_force_improvement_cases"]["other"], [])
        self.assertAlmostEqual(report["force_guarded_oracle_unattainable"]["barrier_MAE_eV"], 2.)
        oracle = report["aggregate_guarded_oracle_unattainable"]
        self.assertTrue(oracle["optimality_proved"])
        self.assertAlmostEqual(oracle["metrics"]["barrier_MAE_eV"], 1.3)
        self.assertLess(oracle["metrics"]["force_global_component_RMSE_eV_per_A"], 1.)

    def test_no_force_budget_can_be_borrowed_when_baseline_dominates_everywhere(self):
        report = analyze(self.table([1.2, 1.1]), ["base", "other"], baseline="base")
        self.assertEqual(report["alternative_force_improvement_reactions"], 0)
        self.assertEqual(report["aggregate_guarded_oracle_unattainable"]["metrics"]["choices"],
                         {"base": 2, "other": 0})


if __name__ == "__main__":
    unittest.main()
