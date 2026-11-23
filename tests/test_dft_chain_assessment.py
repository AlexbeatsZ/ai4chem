"""Regression for real parser/geometry failure boundaries of the portable pilot."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dft_assessment", ROOT/"scripts/assess_dft_chain.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)
task_spec = importlib.util.spec_from_file_location("gaussian_task", ROOT/"scripts/prepare_gaussian_chain.py")
gaussian_task = importlib.util.module_from_spec(task_spec)
task_spec.loader.exec_module(gaussian_task)


class AssessmentBoundaries(unittest.TestCase):
    def test_gaussian_initial_budget_defaults_and_explicit_replay(self):
        protocol = json.loads((ROOT/"configs/v0.6.0-pilot/protocol.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary)/"original"
            gaussian_task.prepare(protocol, "rxn9456", original)
            self.assertIn("maxpoints=160", (original/"irc_forward.gjf").read_text())
            self.assertIn("maxpoints=320", (original/"irc_forward_extended.gjf").read_text())
            protocol["dft_policy"]["irc_initial_max_points"] = 320
            replay = Path(temporary)/"replay"
            gaussian_task.prepare(protocol, "rxn9456", replay)
            self.assertIn("rcfc,forward,lqa,maxpoints=320", (replay/"irc_forward.gjf").read_text())
            protocol["dft_policy"]["irc_initial_max_points"] = 321
            with self.assertRaises(ValueError):
                gaussian_task.prepare(protocol, "rxn9456", Path(temporary)/"invalid")

    def test_separated_fragments_preserve_mapping_without_global_alignment(self):
        expected = Atoms("COH2", positions=[[0,0,0],[1.13,0,0],[5,0,0],[5.74,0,0]])
        moved = expected.copy()
        moved.positions[:2] += [10,3,-4]
        moved.positions[2:] += [-7,5,2]
        self.assertEqual(assessment.bonds(expected), assessment.bonds(moved))
        self.assertLess(max(assessment.fragment_errors(expected, moved)), 1e-12)
        self.assertGreater(assessment.rmsd(expected, moved), 1)
        changed = moved.copy()
        changed.positions[1] = changed.positions[0] + [3,0,0]
        self.assertNotEqual(assessment.bonds(expected), assessment.bonds(changed))

    def test_final_frequency_section_and_final_termination_are_required(self):
        # Combined Opt/Freq may have a normal intermediate termination, and
        # earlier Harmonic sections must never contaminate the final mode count.
        text = ("SCF Done: E(RHF) = -1.0 A.U.\nHarmonic frequencies\n"
                " Frequencies -- -900.0 100.0 200.0\nNormal termination of Gaussian 16\n"
                "SCF Done: E(RHF) = -1.1 A.U.\nOptimization completed.\nHarmonic frequencies\n"
                " Frequencies -- 10.0 20.0 30.0\nNormal termination of Gaussian 16\n")
        atoms = Atoms("H2O", positions=[[0,0,0],[1,0,0],[0,1,0]])
        with tempfile.TemporaryDirectory() as temporary, patch.object(assessment, "read", return_value=atoms):
            path = Path(temporary)/"combined.log"
            path.write_text(text)
            result, _ = assessment.parse_gaussian(path, atoms.numbers)
            self.assertEqual(result["energy_hartree"], -1.1)
            self.assertEqual(result["frequencies_cm1"], [10,20,30])
            self.assertTrue(result["normal_termination"])
            path.write_text(text+"Error termination\n")
            result, _ = assessment.parse_gaussian(path, atoms.numbers)
            self.assertFalse(result["normal_termination"])

    def test_pesdta_error_requires_both_minimum_and_completed_path(self):
        atoms = Atoms("H2", positions=[[0,0,0],[.74,0,0]])
        base = "SCF Done: E(RHF) = -1.0 A.U.\n"
        success = ("PES minimum detected on this side of the pathway.\nCalculation of FORWARD path complete.\n"
                   "Reaction path calculation complete.\nPESDtA: Invalid IPoint.\nError termination\n")
        with tempfile.TemporaryDirectory() as temporary, patch.object(assessment, "read", return_value=atoms):
            path = Path(temporary)/"irc.log"
            path.write_text(base+success)
            result, _ = assessment.parse_gaussian(path, atoms.numbers)
            self.assertTrue(result["irc_path_complete"])
            self.assertFalse(result["normal_termination"])
            path.write_text(base+success.replace("PES minimum detected", "No minimum found"))
            result, _ = assessment.parse_gaussian(path, atoms.numbers)
            self.assertFalse(result["irc_path_complete"])
            with self.assertRaises(AssertionError):
                assessment.parse_gaussian(path, np.array([8,1]))


if __name__ == "__main__":
    unittest.main()
