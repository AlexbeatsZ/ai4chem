"""Check metric weighting and the permitted deployment expert pools."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_specialization import fixed_choices, metric_summary, FALLBACK


class SpecializationAnalysisTests(unittest.TestCase):
    def test_force_error_uses_atom_weighted_squared_components(self):
        table = pd.DataFrame({'reaction':['a','b'],'model':['expert','expert'],
            'atoms':[1,3],'barrier_error_eV':[.1,-.3],
            'force_component_rmse_eV_per_A':[0.,4.],
            'ts_region_force_component_rmse_eV_per_A':[.2,.4],
            'reaction_energy_error_eV':[-.2,.4]})
        result = metric_summary(table,['a','b'],['expert','expert'])
        self.assertAlmostEqual(result['force_global_component_RMSE_eV_per_A'],np.sqrt(12))
        self.assertAlmostEqual(result['barrier_MAE_eV'],.2)

    def test_specialists_never_receive_new_global_as_free_fallback(self):
        self.assertEqual(fixed_choices({'0':'specialist0','1':'specialist1'},[0,1,1],[True,True,False]),
                         ['specialist0','specialist1',FALLBACK])
        self.assertEqual(fixed_choices({'global':'new_global'},[0,1],[True,False]),
                         ['new_global',FALLBACK])


if __name__=='__main__':
    unittest.main()
