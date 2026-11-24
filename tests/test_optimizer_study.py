"""Validation selection cannot buy a lower barrier error with degraded forces."""
import unittest
from run_optimizer_study import choose_guarded


class OptimizerSelectionTests(unittest.TestCase):
    def test_better_barrier_with_worse_force_is_not_selected(self):
        def metrics(barrier,force,ts):
            return {'barrier_mae_eV':barrier,'force_component_rmse_eV_per_A':force,
                    'ts_region_force_component_rmse_eV_per_A':ts}
        rows = {'base':metrics(.13,.09,.13),'barrier_only':metrics(.05,.2,.3),
                'joint':metrics(.11,.08,.12),'ts_worse':metrics(.09,.07,.2)}
        self.assertEqual(choose_guarded(rows,'base'),'joint')

    def test_baseline_remains_valid_when_no_trial_passes_both_guards(self):
        rows = {'base':{'barrier_mae_eV':.13,'force_component_rmse_eV_per_A':.09,
                         'ts_region_force_component_rmse_eV_per_A':.13},
                'trial':{'barrier_mae_eV':.05,'force_component_rmse_eV_per_A':.1,
                         'ts_region_force_component_rmse_eV_per_A':.12}}
        self.assertEqual(choose_guarded(rows,'base'),'base')


if __name__=='__main__':
    unittest.main()
