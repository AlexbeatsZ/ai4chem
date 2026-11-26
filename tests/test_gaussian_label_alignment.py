"""Do not accept partial forces or lose a coordinate rotation during label return."""
import unittest
import numpy as np
from gaussian_label_alignment import parse_label, HARTREE_EV, BOHR_A


class LabelAlignment(unittest.TestCase):
    def fixture(self):
        source = "#p wb97x/6-31g(d) force nosymm 5d 7f\n\nTitle\n\n0 1\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        log = """Standard basis: 6-31G(d) (5D, 7F)
Ex+Corr=wB97X
Input orientation:
 header
 1 8 0 0.000000 0.000000 0.000000
 2 1 0 0.000000 1.000000 0.000000
 3 1 0 -1.000000 0.000000 0.000000
 -----
SCF Done: E(RwB97X) = -76.0 A.U.
Center Atomic Forces (Hartrees/Bohr)
 1 8 0.0 1.0 0.0
 2 1 0.0 0.0 0.0
 3 1 0.0 0.0 0.0
Cartesian Forces:
Normal termination of Gaussian
"""
        return source, log

    def test_full_forces_rotate_into_original_mapped_input(self):
        source, log = self.fixture()
        label = parse_label(source, log, ["O", "H", "H"])
        np.testing.assert_allclose(label["forces_eV_A"][0], [HARTREE_EV/BOHR_A, 0, 0], atol=1e-10)

    def test_partial_force_basis_mismatch_and_failed_job_are_rejected(self):
        source, log = self.fixture()
        for invalid in (log.replace(" 3 1 0.0 0.0 0.0\n", ""), log.replace("5D", "6D"),
                        log+"Error termination", log.replace("-1.000000", "-1.010000")):
            with self.assertRaises(ValueError):
                parse_label(source, invalid, ["O", "H", "H"])
