"""Physics, gradient and data-role boundaries for the conservative mixture."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scipy.optimize import check_grad

from soft_mixture import (SCHEMA, FEATURE_NAMES, CompositionMixtureCalculator,
    composition_features, fit_gate, gate_weights, quadratic_loss_gradient, validate_gate)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_soft_mixture import CachedPaths, mixture_metrics, role_records, static_guard


MEMBERS = [{"name": str(i), "model_file": f"{i}.model", "size_bytes": 1} for i in range(3)]
SETTINGS = {"max_iterations": 300, "ftol": 1e-12, "gtol": 1e-9, "slope_l2": 1e-5, "coefficient_bound": 8.}


def example_gate():
    return {"schema": SCHEMA, "kind": "composition_softmax", "feature_names": FEATURE_NAMES,
        "members": MEMBERS, "center": [0.]*5, "scale": [1.]*5,
        "coefficients": [[.7, -.5], [.4, .2], [.3, -.1], [0., .5], [-.3, .1], [.1, -.2]]}


class PairPotential(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, stiffness):
        super().__init__()
        self.stiffness = stiffness

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions[1]-atoms.positions[0]
        force = self.stiffness*displacement
        self.results = {"energy": .5*self.stiffness*np.dot(displacement, displacement),
                        "forces": np.array([force, -force])}


class SoftMixtureTests(unittest.TestCase):
    def test_feature_allowlist_is_invariant_to_order_and_rejects_other_elements(self):
        np.testing.assert_array_equal(composition_features([1, 6, 7, 8]), composition_features([8, 1, 7, 6]))
        with self.assertRaises(ValueError):
            composition_features([1, 16])
        gate = example_gate()
        np.testing.assert_allclose(gate_weights(gate, [1, 8]), gate_weights(gate, [8, 1]))
        self.assertAlmostEqual(gate_weights(gate, [1, 8]).sum(), 1.)

    def test_gate_roundtrip_and_invalid_weights(self):
        gate = json.loads(json.dumps(example_gate()))
        np.testing.assert_array_equal(gate_weights(gate, [1, 8]), gate_weights(example_gate(), [1, 8]))
        invalid = {**gate, "kind": "constant", "weights": [.5, .5, .5]}
        with self.assertRaises(ValueError):
            validate_gate(invalid)

    def test_analytic_gate_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        design = np.column_stack((np.ones(6), rng.normal(size=(6, 5))))
        factors = rng.normal(size=(6, 4, 3))
        gram = np.einsum("rki,rkj->rij", factors, factors)
        flat = rng.normal(size=12)
        error = check_grad(lambda x: quadratic_loss_gradient(x, design, gram, 1e-5)[0],
                           lambda x: quadratic_loss_gradient(x, design, gram, 1e-5)[1], flat)
        self.assertLess(error, 2e-6)

    def test_constant_fit_can_choose_a_pure_member_and_scaling_uses_only_supplied_rows(self):
        features = np.array([composition_features([1, 6]), composition_features([1, 8])])
        gram = np.broadcast_to(np.diag([0., 1., 2.]), (2, 3, 3)).copy()
        gate = fit_gate("constant", features, gram, MEMBERS, SETTINGS)
        self.assertGreater(gate["weights"][0], .999)
        conditioned = fit_gate("composition_softmax", features, gram, MEMBERS, SETTINGS)
        np.testing.assert_allclose(conditioned["center"], features.mean(axis=0))

    def test_live_ase_mixture_is_conservative_and_rotates_and_translates_correctly(self):
        atoms = Atoms("HO", positions=[[.2, .1, .3], [1.1, -.2, .7]])
        atoms.calc = CompositionMixtureCalculator(example_gate(), [PairPotential(k) for k in (1., 2., 3.)])
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        original = atoms.positions.copy()
        delta = 1e-5
        atoms.positions[1, 2] += delta
        plus = atoms.get_potential_energy()
        atoms.positions[1, 2] -= 2*delta
        minus = atoms.get_potential_energy()
        self.assertAlmostEqual(-(plus-minus)/(2*delta), forces[1, 2], places=9)
        atoms.positions = original+[3., -2., 1.]
        self.assertAlmostEqual(atoms.get_potential_energy(), energy, places=12)
        np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-12)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        atoms.positions = original @ rotation.T
        self.assertAlmostEqual(atoms.get_potential_energy(), energy, places=12)
        np.testing.assert_allclose(atoms.get_forces(), forces @ rotation.T, atol=1e-12)
        atoms.info["charge"] = 1
        with self.assertRaises(ValueError):
            atoms.get_potential_energy()

    def test_role_validation_refuses_holdout_and_identity_mismatch(self):
        partition = {"meta_development": [4], "validation": [], "holdout_train": [8], "holdout_val": [],
            "endpoint_records": {"meta_development": [{"sid": 4}], "validation": []}}
        self.assertEqual(role_records(partition, "meta_development"), [{"sid": 4}])
        with self.assertRaises(ValueError):
            role_records(partition, "holdout_train")
        partition["holdout_train"] = [4]
        with self.assertRaises(ValueError):
            role_records(partition, "meta_development")

    def test_metrics_mix_curves_before_taking_barriers_and_include_force_covariance(self):
        data = CachedPaths([{}, {}], [np.ones(2), np.ones(4)], np.zeros((2, 5)),
            np.array([[[0., 0.], [2., 0.], [0., 2.]]]*2), np.zeros((2, 3)),
            np.array([[[18., -18.], [-18., 18.]], [[36., -36.], [-36., 36.]]]),
            np.array([[[18., -18.], [-18., 18.]], [[36., -36.], [-36., 36.]]]),
            np.array([18., 36.]), np.array([18., 36.]), np.zeros((2, 2, 2)), [])
        metrics = mixture_metrics(data, [[.5, .5], [.5, .5]])
        self.assertEqual(metrics["barrier_MAE_eV"], 1.)
        self.assertEqual(metrics["force_component_RMSE_eV_per_A"], 0.)
        self.assertEqual(metrics["ts_force_component_RMSE_eV_per_A"], 0.)
        baseline = {"barrier_MAE_eV": 2., "force_component_RMSE_eV_per_A": 1., "ts_force_component_RMSE_eV_per_A": 1.}
        self.assertTrue(static_guard(metrics, baseline))
        self.assertFalse(static_guard({**metrics, "ts_force_component_RMSE_eV_per_A": 2.}, baseline))


if __name__ == "__main__":
    unittest.main()
