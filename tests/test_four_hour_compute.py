"""Protect continuation admission against the actual two predecessor formats."""
from copy import deepcopy
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from run_four_hour_compute import open_window, predecessors_complete


class FourHourAdmission(unittest.TestCase):
    def test_growth_terminal_format_has_no_status_field(self):
        growth=dict(stage='growth_training_and_repeats_complete',finished_at_utc='2026-11-26T21:35:48Z',
            attempts={name:dict(status='completed') for name in ('1024s0','2048s0','2048s1','2048s2')})
        self.assertTrue(predecessors_complete({'status':'completed'},growth))
        failed=deepcopy(growth);failed['attempts']['2048s2']['status']='running'
        self.assertFalse(predecessors_complete({'status':'completed'},failed))
        self.assertFalse(predecessors_complete({'status':'failed'},growth))
        partial=deepcopy(growth);del partial['attempts']['2048s2']
        self.assertFalse(predecessors_complete({'status':'completed'},partial))

    def test_new_window_does_not_reuse_old_or_tomorrow_deadline(self):
        protocol={'deadline_utc':'2026-11-27T04:20:00+00:00'}
        with patch('run_four_hour_compute.datetime') as clock:
            clock.fromisoformat.side_effect=datetime.fromisoformat
            clock.now.return_value=datetime(2026,11,27,4,19,59,tzinfo=timezone.utc)
            self.assertTrue(open_window(protocol))
            clock.now.return_value=datetime(2026,11,27,4,20,tzinfo=timezone.utc)
            self.assertFalse(open_window(protocol))
