"""Regression checks for Gaussian stage parsing and geometry export."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from validate_ts_dft import parse_output, read_geometry, run_irc_job, write_input


class GaussianValidationTests(unittest.TestCase):
    def test_single_negative_mode_and_irc_completion_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ts.log"
            path.write_text("SCF Done: E(RwB97X) = -100.5 A.U.\nOptimization completed.\n"
                            " Frequencies -- -400.0 120.0 250.0\nNormal termination of Gaussian 16\n")
            result = parse_output(path)
            self.assertEqual(result["imaginary_modes"], 1)
            self.assertEqual(result["energy_hartree"], -100.5)
            self.assertFalse(result["irc_path_complete"])

    def test_irc_maximum_points_cannot_claim_connectivity_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irc.log"
            path.write_text("NET REACTION COORDINATE UP TO THIS POINT\nMaximum number of steps reached.\n")
            self.assertFalse(parse_output(path)["irc_path_complete"])
            self.assertTrue(parse_output(path)["maximum_steps_reached"])
            path.write_text(path.read_text() + "Calculation of REVERSE path complete.\n")
            self.assertFalse(parse_output(path)["irc_path_complete"])
            path.write_text(path.read_text() + "PES minimum detected on this side of the pathway.\n"
                            "Normal termination of Gaussian 16\n")
            self.assertTrue(parse_output(path)["irc_path_complete"])

    def test_intermediate_normal_termination_does_not_complete_the_whole_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ts.log"
            path.write_text("Optimization completed.\nNormal termination of Gaussian 16\n"
                            "Entering frequency calculation\n")
            self.assertFalse(parse_output(path)["normal_termination"])
            path.write_text(path.read_text() + "Error termination via Lnk1e\n")
            self.assertFalse(parse_output(path)["normal_termination"])

    def test_only_last_frequency_analysis_defines_saddle_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ts.log"
            path.write_text("Harmonic frequencies\nFrequencies -- -100 -50 200\n"
                            "Harmonic frequencies\nFrequencies -- -300 100 200\n"
                            "Normal termination of Gaussian 16\n")
            self.assertEqual(parse_output(path)["imaginary_modes"], 1)

    def test_second_order_step_minimum_message_completes_irc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irc.log"
            path.write_text("PES minimum detected within a second-order step.\n"
                            "Calculation of REVERSE path complete.\nNormal termination of Gaussian 16\n")
            self.assertTrue(parse_output(path)["irc_path_complete"])

    def test_incomplete_ase_frequency_forces_are_excluded_from_export(self):
        atoms = Atoms("CHNOH", positions=np.arange(15).reshape(5, 3))
        atoms.calc = SinglePointCalculator(atoms, energy=-1, forces=np.zeros((4, 3)))
        with tempfile.TemporaryDirectory() as directory, patch("validate_ts_dft.read", return_value=atoms):
            clean = read_geometry(Path("frequency.log"), atoms.numbers, -2)
            path = Path(directory) / "geometry.extxyz"
            write(path, clean)
            restored = read(path)
            self.assertAlmostEqual(restored.get_potential_energy(), -2 * units.Hartree)
            self.assertNotIn("forces", restored.calc.results)

    def test_allcheck_restart_has_separate_checkpoint_and_no_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irc_reverse.gjf"
            write_input(path, "irc=(rcfc,reverse) geom=allcheck guess=read", "IRC",
                        {"dft_mem": "4GB", "dft_nproc": 4, "dft_method": "wB97X/6-31G(d)"},
                        old_checkpoint="ts.chk")
            content = path.read_text()
            self.assertIn("%oldchk=ts.chk\n%chk=irc_reverse.chk", content)
            self.assertNotIn("0 1", content)
            self.assertTrue(content.endswith("\n\n"))

    def test_known_corrector_failure_uses_smaller_step_from_original_ts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "irc_forward.log").write_text("Maximum number of corrector steps exceded.\n")
            with patch("validate_ts_dft.run_job", return_value={"normal_termination": True}) as run:
                result = run_irc_job(folder, "forward", {}, "nosymm")
            self.assertEqual(result["job"], "irc_forward_smallstep")
            self.assertEqual(result["step_size"], 5)
            self.assertEqual(len(result["previous_attempts"]), 1)
            self.assertEqual(run.call_args.kwargs["old_checkpoint"], "ts.chk")
            self.assertIn("maxcycle=80", run.call_args.kwargs["route"])

    def test_unknown_gaussian_failure_is_preserved_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "irc_forward.log").write_text("SCF did not converge\nError termination\n")
            with patch("validate_ts_dft.run_job") as run, self.assertRaises(RuntimeError):
                run_irc_job(folder, "forward", {}, "nosymm")
            run.assert_not_called()

    def test_corrector_recovery_has_a_finite_number_of_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for suffix in ("", "_smallstep", "_lqa"):
                (folder / f"irc_forward{suffix}.log").write_text("Maximum number of corrector steps exceded.\n")
            with patch("validate_ts_dft.run_job") as run, self.assertRaises(RuntimeError):
                run_irc_job(folder, "forward", {}, "nosymm")
            run.assert_not_called()

    def test_fixed_lqa_policy_preserves_earlier_hpc_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "rxn0109"
            folder.mkdir()
            (folder / "irc_forward.log").write_text("Maximum number of corrector steps exceded.\n")
            (folder / "irc_forward_smallstep.log").write_text("Partial HPC output\n")
            (folder / "irc_forward_smallstep.interrupted.json").write_text("{}")
            with patch("validate_ts_dft.run_job", return_value={"normal_termination": True}) as run:
                result = run_irc_job(folder, "forward", {}, "nosymm")
            self.assertEqual(result["algorithm"], "LQA")
            self.assertEqual(result["job"], "irc_forward_lqa")
            self.assertEqual(len(result["previous_attempts"]), 2)
            self.assertIn(",lqa)", run.call_args.kwargs["route"])

    def test_point_limit_continues_from_the_irc_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "rxn0109"
            folder.mkdir()
            outputs = [{"normal_termination": True, "maximum_steps_reached": True, "irc_minimum_detected": False},
                       {"normal_termination": True, "maximum_steps_reached": False, "irc_minimum_detected": True}]
            with patch("validate_ts_dft.run_job", side_effect=outputs) as run:
                result = run_irc_job(folder, "reverse", {}, "nosymm")
            self.assertEqual(result["resumed_from"], "irc_reverse_lqa")
            self.assertEqual(result["max_points"], 320)
            self.assertEqual(run.call_args.kwargs["old_checkpoint"], "irc_reverse_lqa.chk")
            self.assertIn("irc=(restart,maxpoints=320)", run.call_args.kwargs["route"])

    def test_completed_irc_with_postprocessing_index_error_is_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart.log"
            path.write_text("PES minimum detected on this side of the pathway.\n"
                            "Calculation of REVERSE path complete.\nReaction path calculation complete.\n"
                            "PESDtA: Invalid IPoint.\nError termination via Lnk1e\n")
            result = parse_output(path)
            self.assertFalse(result["normal_termination"])
            self.assertTrue(result["irc_postprocessing_error"])
            self.assertTrue(result["irc_path_complete"])
            path.write_text("PESDtA: Invalid IPoint.\nError termination via Lnk1e\n")
            self.assertFalse(parse_output(path)["irc_path_complete"])
            path.write_text("PESDtA: Invalid IPoint.\nError termination via Lnk1e\n"
                            "PES minimum detected on this side of the pathway.\n"
                            "Calculation of REVERSE path complete.\nReaction path calculation complete.\n")
            self.assertFalse(parse_output(path)["irc_path_complete"])


if __name__ == "__main__":
    unittest.main()
