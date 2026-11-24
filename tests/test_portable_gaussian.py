import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('portable_gaussian',Path(__file__).resolve().parents[1]/'scripts/run_gaussian_portable.py')
portable=importlib.util.module_from_spec(spec); spec.loader.exec_module(portable)


class PortableGaussianTests(unittest.TestCase):
    def test_windows_bridge_does_not_require_appending_windows_to_linux_path(self):
        with patch.object(portable.shutil,'which',return_value=None),patch.object(Path,'is_file',return_value=True):
            self.assertEqual(portable.windows_powershell(),'/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')

    def test_earlier_normal_termination_does_not_mask_final_failure(self):
        record=portable.parse_stage('SCF Done: E(RHF) = -2.1\nNormal termination of Gaussian\nError termination\n')
        self.assertFalse(record['normal_termination'])

    def test_only_last_harmonic_section_supplies_frequency_acceptance(self):
        text='Harmonic frequencies\nFrequencies -- -500 100 200\nHarmonic frequencies\nFrequencies -- -10 120 220\nNormal termination of Gaussian\n'
        record=portable.parse_stage(text)
        self.assertEqual(record['frequencies_cm1'],[-10,120,220])
        self.assertEqual(record['imaginary_modes'],1)
        self.assertEqual(record['significant_imaginary_modes'],0)

    def test_only_exact_completed_irc_postprocessing_error_is_recoverable(self):
        text='PES minimum detected\nCalculation of FORWARD path complete\nReaction path calculation complete.\nPESDtA: Invalid IPoint.\nError termination\n'
        self.assertTrue(portable.parse_stage(text)['postprocessing_error'])
        self.assertFalse(portable.parse_stage(text.replace('PES minimum detected','not a minimum'))['postprocessing_error'])


if __name__=='__main__': unittest.main()
