"""Regression checks for reconstructing the writer's concatenated NEB bands."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from prepare_transition1x import write_group
from transition1x_data import final_band_indices, read_reaction


class FinalBandTests(unittest.TestCase):
    def test_single_band_keeps_all_eight_interior_images(self):
        np.testing.assert_array_equal(final_band_indices(10), np.arange(10))

    def test_multiple_bands_use_last_interior_and_first_endpoints(self):
        np.testing.assert_array_equal(final_band_indices(26), [0, 18, 19, 20, 21, 22, 23, 24, 25, 9])

    def test_invalid_frame_counts_are_rejected(self):
        for count in (0, 8, 9, 11, 17, 25):
            with self.subTest(count=count), self.assertRaises(ValueError):
                final_band_indices(count)

    def test_reader_checks_endpoints_and_transition_state(self):
        temporary_root = Path(tempfile.gettempdir()) / ".agents"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            with h5py.File(Path(directory) / "reaction.h5", "w") as data:
                numbers = np.array([8, 1, 1])
                for count in (10, 26):
                    with self.subTest(count=count):
                        positions = np.arange(count * 3 * 3, dtype=float).reshape(count, 3, 3) / 100
                        energies = np.arange(count, dtype=float)
                        forces = np.zeros_like(positions)
                        group = data.create_group(f"reaction{count}")
                        write_group(group, numbers, positions, energies, forces)
                        for name, index in (("reactant", 0), ("product", 9), ("transition_state", count - 1)):
                            write_group(group.create_group(name), numbers, positions[index:index + 1],
                                        energies[index:index + 1], forces[index:index + 1])
                        path = read_reaction(group, "test", "H2O", "rxn0000")
                        expected = np.arange(10) if count == 10 else [0, 18, 19, 20, 21, 22, 23, 24, 25, 9]
                        np.testing.assert_array_equal(path.energies, expected)
                        self.assertEqual(path.reference_ts_image, 9 if count == 10 else 8)
                        self.assertEqual(len(path.atoms()), 10)
                        group["transition_state"]["positions"][0] = positions[count - 2]
                        with self.assertRaises(AssertionError):
                            read_reaction(group, "test", "H2O", "rxn0000")


if __name__ == "__main__":
    unittest.main()
