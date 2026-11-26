"""Conservative, composition-conditioned mixtures of frozen molecular potentials.

The gate never reads coordinates or labels. For a fixed molecular composition its
weights are constant, so ASE's energy/force linear combination is conservative.
This is an external mixture, not an internally trained sparse MoE backbone.
"""

import json
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import LinearCombinationCalculator
from scipy.optimize import minimize
from scipy.special import softmax


FEATURE_NAMES = ["log_atoms", "fraction_H", "fraction_C", "fraction_N", "fraction_O"]
SCHEMA = "ai4chem-composition-mixture-1"


def composition_features(numbers):
    numbers = np.asarray(numbers)
    if numbers.ndim != 1 or len(numbers) < 2 or not np.isin(numbers, [1, 6, 7, 8]).all():
        raise ValueError("Mixture requires a nonperiodic CHNO molecule with at least two atoms")
    return np.array([np.log(len(numbers))] + [np.mean(numbers == z) for z in (1, 6, 7, 8)])


def validate_gate(gate):
    if gate.get("schema") != SCHEMA or gate.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Unknown gate schema or feature allowlist")
    members = gate.get("members", [])
    if len(members) < 2 or len({m["name"] for m in members}) != len(members):
        raise ValueError("Gate requires unique, explicitly budgeted members")
    if gate["kind"] == "constant":
        weights = np.asarray(gate["weights"], float)
        if weights.shape != (len(members),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Invalid convex mixture weights")
        if not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("Mixture weights must sum to one")
    elif gate["kind"] == "composition_softmax":
        center, scale = np.asarray(gate["center"]), np.asarray(gate["scale"])
        coefficients = np.asarray(gate["coefficients"])
        if center.shape != (5,) or scale.shape != (5,) or coefficients.shape != (6, len(members)-1):
            raise ValueError("Invalid gate feature or coefficient dimensions")
        if not all(np.isfinite(x).all() for x in (center, scale, coefficients)) or np.any(scale <= 0):
            raise ValueError("Nonfinite gate parameters or nonpositive feature scale")
    else:
        raise ValueError("Unsupported gate kind")


def gate_weights(gate, numbers):
    validate_gate(gate)
    features = composition_features(numbers)
    if gate["kind"] == "constant":
        return np.asarray(gate["weights"], float)
    design = np.r_[1., (features - gate["center"]) / gate["scale"]]
    # The last logit is fixed to zero, removing the common-logit redundancy.
    return softmax(np.r_[design @ np.asarray(gate["coefficients"]), 0.])


def quadratic_loss_gradient(flat, design, gram, penalty):
    """Exact energy/force cached-prediction loss and gradient for a softmax gate."""
    member_count = gram.shape[1]
    coefficients = np.asarray(flat).reshape(design.shape[1], member_count-1)
    logits = np.column_stack((design @ coefficients, np.zeros(len(design))))
    weights = softmax(logits, axis=1)
    product = np.einsum("rij,rj->ri", gram, weights)
    loss = np.mean(np.sum(weights * product, axis=1)) + penalty*np.sum(coefficients[1:]**2)
    weighted_gradient = 2*product
    logit_gradient = weights*(weighted_gradient - np.sum(weights*weighted_gradient, axis=1, keepdims=True))
    gradient = design.T @ logit_gradient[:, :-1] / len(design)
    gradient[1:] += 2*penalty*coefficients[1:]
    return float(loss), gradient.ravel()


def fit_gate(kind, features, gram, members, settings):
    """Only the caller's training rows enter scaling or optimization."""
    features, gram = np.asarray(features, float), np.asarray(gram, float)
    count = len(members)
    if features.ndim != 2 or features.shape != (len(gram), 5) or gram.shape[1:] != (count, count):
        raise ValueError("Gate training dimensions differ")
    if not len(features) or not np.isfinite(features).all() or not np.isfinite(gram).all():
        raise ValueError("Gate training data must be finite and nonempty")
    gate = {"schema": SCHEMA, "kind": kind, "feature_names": FEATURE_NAMES, "members": members}
    initial = np.array([.8] + [.2/(count-1)]*(count-1))
    if kind == "constant":
        matrix = gram.mean(axis=0)
        result = minimize(lambda w: (float(w @ matrix @ w), 2*matrix @ w), initial,
            jac=True, method="SLSQP", bounds=[(0., 1.)]*count,
            constraints=[{"type": "eq", "fun": lambda w: w.sum()-1., "jac": lambda w: np.ones(count)}],
            options={"maxiter": settings["max_iterations"], "ftol": settings["ftol"]})
        weights = np.maximum(result.x, 0.)
        gate["weights"] = (weights/weights.sum()).tolist()
    elif kind == "composition_softmax":
        center, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale < 1e-12] = 1.
        design = np.column_stack((np.ones(len(features)), (features-center)/scale))
        coefficients = np.zeros((design.shape[1], count-1))
        coefficients[0] = np.log(initial[:-1]/initial[-1])
        result = minimize(quadratic_loss_gradient, coefficients.ravel(),
            args=(design, gram, settings["slope_l2"]), jac=True, method="L-BFGS-B",
            bounds=[(-settings["coefficient_bound"], settings["coefficient_bound"])]*coefficients.size,
            options={"maxiter": settings["max_iterations"], "ftol": settings["ftol"], "gtol": settings["gtol"]})
        gate.update(center=center.tolist(), scale=scale.tolist(),
                    coefficients=result.x.reshape(coefficients.shape).tolist())
    else:
        raise ValueError("Only the two frozen learned gate controls may be fitted")
    gate["optimization"] = {"success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev), "loss": float(result.fun)}
    validate_gate(gate)
    if not result.success:
        raise RuntimeError(f"Gate optimization failed within its fixed budget: {gate['optimization']}")
    return gate


class CompositionMixtureCalculator(Calculator):
    """Reuse ASE mixing with a coordinate-independent, frozen composition gate."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, gate, calculators=None, root=None, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.gate = json.loads(Path(gate).read_text()) if isinstance(gate, (str, Path)) else json.loads(json.dumps(gate))
        validate_gate(self.gate)
        if calculators is None:
            from mace.calculators import MACECalculator
            root = Path(root) if root else Path(__file__).resolve().parent
            calculators = []
            for member in self.gate["members"]:
                path = root/member["model_file"]
                if path.stat().st_size != member["size_bytes"]:
                    raise ValueError(f"Frozen model size differs: {member['name']}")
                calculators.append(MACECalculator(model_paths=str(path), device=device, default_dtype="float64"))
        if len(calculators) != len(self.gate["members"]):
            raise ValueError("Calculator pool does not match the explicitly budgeted members")
        self.calculators = calculators
        self.last_weights = None

    @staticmethod
    def validate_atoms(atoms):
        if np.any(atoms.pbc) or np.any(atoms.get_initial_charges()) or np.any(atoms.get_initial_magnetic_moments()):
            raise ValueError("Pilot calculator supports only nonperiodic neutral-singlet molecules")
        if atoms.info.get("charge", 0) != 0 or atoms.info.get("multiplicity", atoms.info.get("spin", 1)) != 1:
            raise ValueError("Pilot calculator supports only neutral singlets")
        composition_features(atoms.numbers)
        if not np.isfinite(atoms.positions).all():
            raise ValueError("Nonfinite atomic positions")

    def get_property(self, name, atoms=None, allow_calculation=True):
        # ASE does not include arbitrary info fields in check_state. Revalidate
        # charge/spin even when energy/forces could otherwise be served from cache.
        self.validate_atoms(atoms if atoms is not None else self.atoms)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        atoms = atoms if atoms is not None else self.atoms
        self.validate_atoms(atoms)
        weights = gate_weights(self.gate, atoms.numbers)
        super().calculate(atoms, properties, system_changes)
        mixed = LinearCombinationCalculator(self.calculators, weights.tolist())
        mixed.calculate(atoms, properties, system_changes)
        self.results = {key: mixed.results[key] for key in self.implemented_properties if key in mixed.results}
        self.last_weights = weights
        if any(not np.isfinite(value).all() for value in self.results.values()):
            raise ValueError("Nonfinite mixture output")
