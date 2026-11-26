"""Strict fixed-geometry Gaussian energy/force extraction in input atom order."""
import re

import numpy as np
from ase.data import atomic_numbers

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903


def parse_label(input_text, log, symbols, basis_d=5):
    if "Normal termination of Gaussian" not in log or log.rfind("Error termination") > log.rfind("Normal termination of Gaussian"):
        raise ValueError("Gaussian label has no successful final termination")
    route = " ".join(line.strip().lower() for line in input_text.splitlines() if line.startswith("#"))
    if not all(token in route for token in ("wb97x/6-31g(d)", "force", "nosymm")):
        raise ValueError("Unexpected force method or orientation policy")
    if f"({basis_d}D, 7F)" not in log or "Ex+Corr=wB97X" not in log:
        raise ValueError("Gaussian basis convention or functional differs")
    n = len(symbols)
    lines = input_text.splitlines()
    start = next(i+1 for i, line in enumerate(lines) if line.strip() == "0 1")
    rows = [line.split() for line in lines[start:start+n]]
    if len(rows) != n or [row[0] for row in rows] != symbols:
        raise ValueError("Gaussian input atom mapping differs")
    positions = np.asarray([[float(value) for value in row[1:4]] for row in rows])
    numbers = np.asarray([atomic_numbers[symbol] for symbol in symbols])
    marker = log.rfind("Forces (Hartrees/Bohr)")
    if marker < 0:
        raise ValueError("No Cartesian force table")
    force_text = log[marker:].split("Cartesian Forces:")[0]
    force_rows = [line.split() for line in force_text.splitlines()
                  if re.match(r"^\s*\d+\s+\d+\s+[-+0-9.]", line)]
    if len(force_rows) != n or [int(row[0]) for row in force_rows] != list(range(1, n+1)):
        raise ValueError("Force table does not contain every mapped atom")
    if [int(row[1]) for row in force_rows] != numbers.tolist():
        raise ValueError("Force atom mapping differs")
    forces = np.asarray([[float(value.replace("D", "E")) for value in row[2:5]] for row in force_rows])
    orientations = list(re.finditer(r"(?:Input|Standard) orientation:", log[:marker]))
    if not orientations:
        raise ValueError("No force coordinate orientation")
    block = log[orientations[-1].end():marker]
    orientation_rows = []
    entered = False
    for line in block.splitlines():
        fields = line.split()
        if len(fields) == 6 and all(value.isdigit() for value in fields[:3]):
            entered = True
            orientation_rows.append(fields)
        elif entered:
            break
    if len(orientation_rows) != n or [int(row[0]) for row in orientation_rows] != list(range(1, n+1)):
        raise ValueError("Incomplete coordinate orientation")
    if [int(row[1]) for row in orientation_rows] != numbers.tolist():
        raise ValueError("Orientation mapping differs")
    output_positions = np.asarray([[float(value) for value in row[3:]] for row in orientation_rows])
    x = output_positions-output_positions.mean(axis=0)
    y = positions-positions.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T@y)
    rotation = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
    alignment_error = float(np.max(np.abs(x@rotation-y)))
    if alignment_error > 2e-6:
        raise ValueError("Gaussian label geometry differs from fixed input")
    energies = re.findall(r"SCF Done:.*?=\s*([-+\d.DEd]+)", log[:marker])
    if not energies:
        raise ValueError("No SCF energy preceding force table")
    energy = float(energies[-1].replace("D", "E"))*HARTREE_EV
    forces = forces@rotation*(HARTREE_EV/BOHR_A)
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("Nonfinite Gaussian label")
    return dict(energy_eV=energy, forces_eV_A=forces, positions_A=positions,
                alignment_max_A=alignment_error, basis_d=basis_d)
