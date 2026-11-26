"""Verify fitted external-mixture artifacts with real reloaded MACE members."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ase.io import read

from soft_mixture import CompositionMixtureCalculator, gate_weights


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    torch.set_num_threads(1)
    output = ROOT/"results/v0.11.0"
    destination = output/"runtime_acceptance.json"
    if destination.exists():
        raise FileExistsError("Existing acceptance evidence must not be overwritten")
    summary = json.loads((output/"summary.json").read_text())
    gate_file = output/"models/composition_softmax.json"
    calc = CompositionMixtureCalculator(gate_file, root=ROOT, device="cpu")
    gate = json.loads(gate_file.read_text())
    # One already-open development reaction is only a numerical acceptance case.
    reaction = "rxn0097"
    frames = [read(ROOT/f"results/v0.8.0/meta_development/{member['name']}/{member['name']}_{reaction}.extxyz", index=0)
              for member in gate["members"]]
    atoms = frames[0].copy()
    atoms.calc = calc
    energy, force = atoms.get_potential_energy(), atoms.get_forces()
    weights = gate_weights(gate, atoms.numbers)
    expected_energy = float(weights @ np.array([frame.get_potential_energy() for frame in frames]))
    expected_force = np.einsum("k,kij->ij", weights, np.array([frame.get_forces() for frame in frames]))
    np.testing.assert_allclose(energy, expected_energy, atol=2e-6, rtol=0)
    np.testing.assert_allclose(force, expected_force, atol=2e-6, rtol=0)
    origin = atoms.positions.copy()
    delta = 1e-4
    atoms.positions[0, 0] += delta
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2*delta
    minus = atoms.get_potential_energy()
    derivative = -(plus-minus)/(2*delta)
    difference = abs(derivative-force[0, 0])
    if difference > 2e-5:
        raise ValueError(f"Mixture energy/force finite-difference mismatch: {difference}")
    atoms.positions = origin+[2.1, -1.7, .3]
    translation_energy = abs(atoms.get_potential_energy()-energy)
    translation_force = float(np.max(np.abs(atoms.get_forces()-force)))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    atoms.positions = origin @ rotation.T
    rotation_energy = abs(atoms.get_potential_energy()-energy)
    rotation_force = float(np.max(np.abs(atoms.get_forces()-force @ rotation.T)))
    permutation = np.arange(len(atoms))[::-1]
    permuted = frames[0][permutation]
    permuted.calc = calc
    permutation_energy = abs(permuted.get_potential_energy()-energy)
    permutation_force = float(np.max(np.abs(permuted.get_forces()-force[permutation])))
    if max(translation_energy, translation_force, rotation_energy, rotation_force,
           permutation_energy, permutation_force) > 2e-6:
        raise ValueError("Mixture symmetry acceptance failed")
    timings = {}
    for name, calculator in (("single_general", calc.calculators[0]), ("composition_softmax", calc)):
        measurements = []
        for repeat in range(4):
            sample = frames[0].copy()
            sample.positions[0, 0] += (repeat+1)*1e-5
            sample.calc = calculator
            calculator.reset()
            started = time.perf_counter()
            sample.get_potential_energy(); sample.get_forces()
            measurements.append(time.perf_counter()-started)
        timings[name] = {"seconds": measurements, "median_seconds": float(np.median(measurements))}
    constant = json.loads((output/"models/constant.json").read_text())
    constant_calc = CompositionMixtureCalculator(constant, calculators=calc.calculators)
    constant_atoms = frames[0].copy(); constant_atoms.calc = constant_calc
    cw = gate_weights(constant, constant_atoms.numbers)
    np.testing.assert_allclose(constant_atoms.get_potential_energy(), cw @ [f.get_potential_energy() for f in frames], atol=2e-6, rtol=0)
    np.testing.assert_allclose(constant_atoms.get_forces(), np.einsum("k,kij->ij", cw, [f.get_forces() for f in frames]), atol=2e-6, rtol=0)
    result = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real CPU float64 three-member reload and numerical acceptance; no new reaction accuracy or TS-search result",
        "reaction": reaction, "atoms": len(atoms), "weights": weights.tolist(),
        "cache_energy_difference_eV": abs(energy-expected_energy),
        "cache_force_max_difference_eV_per_A": float(np.max(np.abs(force-expected_force))),
        "finite_difference_force_error_eV_per_A": difference,
        "translation_energy_difference_eV": translation_energy, "translation_force_max_difference_eV_per_A": translation_force,
        "rotation_energy_difference_eV": rotation_energy, "rotation_force_max_difference_eV_per_A": rotation_force,
        "permutation_energy_difference_eV": permutation_energy, "permutation_force_max_difference_eV_per_A": permutation_force,
        "timing": timings, "mixture_to_single_median_time_ratio": timings["composition_softmax"]["median_seconds"]/timings["single_general"]["median_seconds"],
        "constant_gate_reload_passed": True, "torch_threads": 1, "holdout_labels_evaluated": False}
    save(destination, result)
    summary["runtime_acceptance"] = result
    save(output/"summary.json", summary)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
