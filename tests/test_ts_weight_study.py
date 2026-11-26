"""TS objective normalization and guarded promotion boundaries."""
import math
import unittest

from run_ts_weight_study import normalized_weights, promotes


class TSWeightBoundaries(unittest.TestCase):
    def test_normalization_retains_endpoints_and_only_doubles_the_relative_ts_weight(self):
        weights = normalized_weights(list(range(8)), {3, 4, 5})
        self.assertAlmostEqual(sum(weights), 8)
        self.assertAlmostEqual(weights[4] / weights[0], 2)
        self.assertGreater(weights[0], 0)
        self.assertEqual(normalized_weights([0, 9], set()), [1, 1])

    def test_barrier_improvement_cannot_hide_force_regression_or_nonfinite_metrics(self):
        base = dict(barrier_mae_eV=.11, force_component_rmse_eV_per_A=.09,
                    ts_region_force_component_rmse_eV_per_A=.13)
        winner = dict(barrier_mae_eV=.10, force_component_rmse_eV_per_A=.08,
                      ts_region_force_component_rmse_eV_per_A=.12)
        self.assertTrue(promotes(winner, base, base))
        self.assertFalse(promotes({**winner, "ts_region_force_component_rmse_eV_per_A": .14}, base, base))
        self.assertFalse(promotes(base, base, base))
        with self.assertRaises(ValueError):
            promotes({**winner, "barrier_mae_eV": math.nan}, base, base)
