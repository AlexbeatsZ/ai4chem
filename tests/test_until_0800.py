"""Deadline admission and separate label-extension/model-promotion decisions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_mac_offpath_study import before_deadline, extension_allowed
from run_ts_weight_study import promotes


class OvernightBoundaries(unittest.TestCase):
    def test_two_search_cohorts_are_disjoint_frozen_meta_and_have_two_models(self):
        root=Path(__file__).resolve().parents[1]
        roles=json.loads((root/'configs/v0.7.0_partitions.json').read_text())
        cohorts=[]
        for prefix in ('growth','omen-growth'):
            protocol=json.loads((root/f'configs/{prefix}-search-until-0800-20261126.json').read_text())
            self.assertEqual(protocol['experts'],['global512','coverage2048s0'])
            ids={row['sid'] for row in protocol['endpoint_records']}
            self.assertEqual(len(ids),8)
            self.assertTrue(ids<=set(roles['meta_development']))
            self.assertFalse(ids & (set(roles['holdout_train'])|set(roles['holdout_val'])))
            cohorts.append(ids)
        self.assertFalse(cohorts[0]&cohorts[1])

    def test_absolute_utc8_boundary(self):
        protocol={'deadline_utc':'2026-11-27T00:00:00+00:00'}
        with patch('run_mac_offpath_study.datetime') as clock:
            clock.fromisoformat.side_effect=datetime.fromisoformat
            clock.now.return_value=datetime(2026,11,26,23,59,59,tzinfo=timezone.utc)
            self.assertTrue(before_deadline(protocol))
            clock.now.return_value=datetime(2026,11,27,0,0,tzinfo=timezone.utc)
            self.assertFalse(before_deadline(protocol))

    def test_data_extension_is_not_model_promotion_and_cannot_hide_bad_fit(self):
        base=dict(barrier_mae_eV=.1,force_component_rmse_eV_per_A=.08,ts_region_force_component_rmse_eV_per_A=.12)
        mild={**base,'barrier_mae_eV':.102}
        self.assertTrue(extension_allowed(base,mild,base,.2,.1))
        self.assertFalse(promotes(mild,base,base))
        self.assertFalse(extension_allowed(base,mild,base,.1,.2))
        self.assertFalse(extension_allowed(base,{**mild,'barrier_mae_eV':.12},base,.2,.1))
        self.assertFalse(extension_allowed(base,mild,base,.2,float('nan')))
