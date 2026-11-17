"""Endpoint-only CI-NEB, saddle refinement, projected vibrations and MLIP IRC."""

import argparse
import json
import tempfile
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms, units
from ase.build.rotate import minimize_rotation_and_translation
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.mep import NEB
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import FIRE
from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from analyze_benchmark import endpoint_change
from mace_neb import MACEBand, check_batch_parity


def save_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def bond_set(atoms):
    left, right = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
    return {tuple(sorted((int(i), int(j)))) for i, j in zip(left, right) if i != j}


def aligned_rmsd(reference, candidate):
    aligned = candidate.copy()
    minimize_rotation_and_translation(reference, aligned)
    return float(np.sqrt(np.mean(np.sum((reference.positions - aligned.positions) ** 2, axis=1))))


def fragment_rmsds(reference, candidate):
    """Compare mapped connected components independently of fragment arrangement."""
    adjacency = np.zeros((len(reference), len(reference)), dtype=bool)
    for left, right in bond_set(reference):
        adjacency[left, right] = adjacency[right, left] = True
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    return [aligned_rmsd(reference[labels == index], candidate[labels == index]) for index in range(count)]


def assess_irc_branches(branches, minima, endpoints, tolerance):
    for branch, atoms in zip(branches, minima, strict=True):
        branch["matches_endpoint_bonds"] = [bond_set(atoms) == bond_set(endpoint) for endpoint in endpoints]
        branch["endpoint_rmsd_A"] = [aligned_rmsd(endpoint, atoms) for endpoint in endpoints]
        branch["endpoint_fragment_rmsd_A"] = [fragment_rmsds(endpoint, atoms) for endpoint in endpoints]
    bonds = [branch["matches_endpoint_bonds"] for branch in branches]
    matches = [[bonds[side][index] and max(branch["endpoint_fragment_rmsd_A"][index]) <= tolerance
                for index in range(2)] for side, branch in enumerate(branches)]
    bond_connected = (bonds[0][0] and bonds[1][1]) or (bonds[0][1] and bonds[1][0])
    connected = (matches[0][0] and matches[1][1]) or (matches[0][1] and matches[1][0])
    complete = all(branch["converged"] and branch["minimum_converged"] for branch in branches)
    return {"branches": branches, "expected_bond_connectivity": bool(bond_connected),
            "expected_connectivity": bool(connected), "converged_and_connected": bool(complete and connected),
            "geometry_assessment": "proper alignment per mapped connected component"}


def verified_saddle(result, config):
    endpoints_valid = all(item["converged"] and item["input_bonds_preserved"] for item in result.get("endpoints", []))
    saddle = result.get("saddle", {})
    return bool(result.get("endpoints") and endpoints_valid
                and saddle.get("converged", False)
                and saddle.get("fmax_eV_A", 1e10) <= config["ts_fmax_eV_A"]
                and result.get("vibrations", {}).get("imaginary_modes_above_threshold") == 1
                and result.get("irc", {}).get("converged_and_connected", False))


def verified_result(result, config):
    return bool(result.get("neb", {}).get("converged", False) and verified_saddle(result, config))


def recheck_saved_result(folder, config):
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text())
    if "irc" in result:
        endpoints = [read(folder / f"{name}_relaxed.extxyz") for name in ("reactant", "product")]
        minima = [read(folder / f"irc_{direction}_minimum.extxyz") for direction in ("forward", "reverse")]
        result["irc"] = assess_irc_branches(
            result["irc"]["branches"], minima, endpoints, config["irc_endpoint_rmsd_A"])
    result["mlip_verified"] = verified_result(result, config)
    result["saddle_irc_verified"] = verified_saddle(result, config)
    save_json(result_path, result)
    return result


def endpoint_pool(config):
    """Deliberately access endpoint positions and atomic numbers only."""
    pool = []
    with h5py.File(config["input"], "r") as source:
        for formula in sorted(source[config["split"]]):
            for reaction, group in sorted(source[config["split"]][formula].items()):
                numbers = group["atomic_numbers"][()]
                endpoints = [Atoms(numbers, positions=group[name]["positions"][0])
                             for name in ("reactant", "product")]
                kind, formed, broken = endpoint_change(*endpoints)
                pool.append({"formula": formula, "reaction": reaction, "atoms": len(numbers),
                             "kind": kind, "formed": formed, "broken": broken,
                             "endpoints": endpoints})
    return pool


def select_reactions(pool, count):
    if count < 1 or count > len(pool):
        raise ValueError("Reaction count must be between one and the available pool size")
    ordered = sorted(pool, key=lambda item: (item["formed"] + item["broken"], item["reaction"]))
    selected = []
    formulas = sorted({item["formula"] for item in pool})
    for formula in formulas:
        selected.append(next(item for item in ordered if item["formula"] == formula))
        if len(selected) == count:
            return selected
    missing = {item["kind"] for item in pool} - {item["kind"] for item in selected}
    for kind in sorted(missing):
        selected.append(next(item for item in ordered if item["kind"] == kind))
        if len(selected) == count:
            return selected
    # Additional cases increase endpoint-change diversity before reusing a signature.
    remaining = [item for item in ordered if item not in selected]
    signatures = {(item["kind"], item["formed"], item["broken"]) for item in selected}
    remaining.sort(key=lambda item: ((item["kind"], item["formed"], item["broken"]) in signatures,
                                     item["atoms"], item["reaction"]))
    return selected + remaining[:count - len(selected)]


def prepare(config, output):
    selection_path = output / "selection.json"
    if selection_path.exists():
        manifest = json.loads(selection_path.read_text())
        if manifest["config"] != config:
            raise ValueError("Existing search has a different protocol, choose a new output directory")
        selected = []
        for item in manifest["reactions"]:
            folder = output / "inputs" / item["reaction"]
            endpoints = [read(folder / f"{name}.xyz") for name in ("reactant", "product")]
            np.testing.assert_array_equal(endpoints[0].numbers, endpoints[1].numbers)
            selected.append({**item, "endpoints": endpoints})
        return selected
    pool = endpoint_pool(config)
    selected = select_reactions(pool, config["count"])
    entries = []
    for item in selected:
        folder = output / "inputs" / item["reaction"]
        folder.mkdir(parents=True, exist_ok=True)
        endpoints = item["endpoints"]
        for name, atoms in zip(("reactant", "product"), endpoints):
            write(folder / f"{name}.xyz", atoms)
        entries.append({key: value for key, value in item.items() if key != "endpoints"})
    save_json(selection_path, {"config": config, "reactions": entries,
                              "evaluation_role": "previously exposed v0.3.0 test reactions, development only",
                              "initialization": "mapped endpoints only, proper rigid alignment then IDPP"})
    return selected


def frozen_result(atoms):
    saved = atoms.copy()
    saved.calc = SinglePointCalculator(saved, energy=atoms.get_potential_energy(), forces=atoms.get_forces())
    return saved


def max_force(atoms):
    forces = atoms.get_forces()
    if not np.isfinite(forces).all():
        raise ValueError("Non-finite model forces")
    return float(np.linalg.norm(forces, axis=1).max())


def vibrational_subspace(atoms):
    """Remove mass-weighted rigid translations and rotations, including linear cases."""
    masses = atoms.get_masses()
    centered = atoms.positions - atoms.get_center_of_mass()
    sqrt_mass = np.sqrt(masses)[:, None]
    translations = [(np.tile(axis, (len(atoms), 1)) * sqrt_mass).ravel() for axis in np.eye(3)]
    rotations = [(np.cross(np.tile(axis, (len(atoms), 1)), centered) * sqrt_mass).ravel()
                 for axis in np.eye(3)]
    return null_space(np.asarray(translations + rotations), rcond=1e-8)


def projected_frequencies(atoms, hessian):
    weights = np.repeat(atoms.get_masses() ** -0.5, 3)
    mass_weighted = weights[:, None] * hessian * weights[None, :]
    basis = vibrational_subspace(atoms)
    eigenvalues, modes = np.linalg.eigh(basis.T @ mass_weighted @ basis)
    conversion = units._hbar * 1e10 / np.sqrt(units._e * units._amu) / units.invcm
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion
    cartesian_modes = weights[:, None] * (basis @ modes)
    return frequencies, cartesian_modes


def numerical_hessian(atoms, delta):
    positions = atoms.positions.copy()
    hessian = np.empty((3 * len(atoms), 3 * len(atoms)))
    try:
        for coordinate in range(3 * len(atoms)):
            atom, axis = divmod(coordinate, 3)
            atoms.positions[:] = positions
            atoms.positions[atom, axis] += delta
            plus = atoms.get_forces().ravel().copy()
            atoms.positions[atom, axis] -= 2 * delta
            minus = atoms.get_forces().ravel().copy()
            hessian[:, coordinate] = -(plus - minus) / (2 * delta)
    finally:
        atoms.positions[:] = positions
    if not np.isfinite(hessian).all():
        raise ValueError("Non-finite numerical Hessian")
    return (hessian + hessian.T) / 2


def relax_endpoint(atoms, calc, folder, name, config):
    relaxed = atoms.copy()
    relaxed.calc = calc
    optimizer = FIRE(relaxed, logfile=str(folder / f"{name}_relax.log"),
                     trajectory=str(folder / f"{name}_relax.traj"), maxstep=0.1)
    converged = bool(optimizer.run(fmax=config["endpoint_fmax_eV_A"], steps=config["endpoint_steps"]))
    write(folder / f"{name}_relaxed.extxyz", frozen_result(relaxed))
    status = {"converged": converged, "steps": optimizer.nsteps,
              "fmax_eV_A": max_force(relaxed), "input_bonds_preserved": bond_set(atoms) == bond_set(relaxed),
              "input_rmsd_A": aligned_rmsd(atoms, relaxed)}
    return relaxed, status


def run_irc(ts, calc, endpoints, folder, config):
    from sella import IRC
    branches, minima = [], []
    for direction in ("forward", "reverse"):
        atoms = ts.copy()
        atoms.calc = calc
        optimizer = IRC(atoms, dx=config["irc_dx_A_sqrt_amu"], eta=1e-4, gamma=0.4,
                        logfile=str(folder / f"irc_{direction}.log"),
                        trajectory=str(folder / f"irc_{direction}.traj"))
        converged = bool(optimizer.run(fmax=config["irc_fmax_eV_A"], steps=config["irc_steps"],
                                      direction=direction))
        # Finish each descent in the adjacent minimum with the common endpoint tolerance.
        relaxation = FIRE(atoms, logfile=str(folder / f"irc_{direction}_minimum.log"), maxstep=0.1)
        minimum_converged = bool(relaxation.run(fmax=config["endpoint_fmax_eV_A"],
                                               steps=config["endpoint_steps"]))
        write(folder / f"irc_{direction}_minimum.extxyz", frozen_result(atoms))
        minima.append(atoms.copy())
        branches.append({"direction": direction, "converged": converged, "steps": optimizer.nsteps,
                         "minimum_converged": minimum_converged, "fmax_eV_A": max_force(atoms)})
    return assess_irc_branches(branches, minima, endpoints, config["irc_endpoint_rmsd_A"])


def search_one(item, calc, folder, config):
    from sella import Sella
    folder.mkdir(parents=True, exist_ok=True)
    result = {"reaction": item["reaction"], "formula": item["formula"], "status": "running",
              "mlip_verified": False, "stage": "endpoints", "neb_backend": "batched MACE"}
    start = time.monotonic()
    try:
        endpoints, endpoint_status = [], []
        for name, atoms in zip(("reactant", "product"), item["endpoints"]):
            relaxed, status = relax_endpoint(atoms, calc, folder, name, config)
            endpoints.append(relaxed)
            endpoint_status.append(status)
        minimize_rotation_and_translation(endpoints[0], endpoints[1])
        result["endpoints"] = endpoint_status
        images = [endpoints[0]] + [endpoints[0].copy() for _ in range(config["images"] - 2)] + [endpoints[1]]
        for image in images:
            image.calc = calc
        neb = NEB(images, k=config["spring_eV_A2"], method="improvedtangent",
                  allow_shared_calculator=True, remove_rotation_and_translation=True)
        result["stage"] = "idpp"
        neb.interpolate(method="idpp")
        write(folder / "initial_band.extxyz", images, write_results=False)
        result["batch_parity"] = check_batch_parity(images, calc)
        MACEBand(images, calc).attach()
        result["stage"] = "neb"
        warmup = FIRE(neb, logfile=str(folder / "neb_warmup.log"), maxstep=0.1)
        warmup.run(fmax=config["neb_warmup_fmax_eV_A"], steps=config["neb_warmup_steps"])
        neb.climb = True
        climbing = FIRE(neb, logfile=str(folder / "neb_climb.log"),
                        trajectory=str(folder / "neb.traj"), maxstep=0.1)
        converged = bool(climbing.run(fmax=config["neb_fmax_eV_A"], steps=config["neb_climb_steps"]))
        neb_force = float(np.linalg.norm(neb.get_forces().reshape(-1, 3), axis=1).max())
        energies = np.array([image.get_potential_energy() for image in images])
        write(folder / "final_band.extxyz", [frozen_result(image) for image in images])
        peak = int(np.argmax(energies[1:-1])) + 1
        result["neb"] = {"converged": converged, "warmup_steps": warmup.nsteps,
                         "climb_steps": climbing.nsteps, "fmax_eV_A": neb_force,
                         "peak_image": peak, "relative_energies_eV": (energies - energies[0]).tolist(),
                         "sampled_barrier_eV": float(max(energies) - energies[0])}
        result["stage"] = "saddle"
        ts = images[peak].copy()
        ts.calc = calc
        saddle = Sella(ts, order=1, internal=True, logfile=str(folder / "sella.log"),
                       trajectory=str(folder / "sella.traj"))
        saddle_converged = bool(saddle.run(fmax=config["ts_fmax_eV_A"], steps=config["ts_steps"]))
        write(folder / "ts_candidate.extxyz", frozen_result(ts))
        result["saddle"] = {"converged": saddle_converged, "steps": saddle.nsteps,
                            "fmax_eV_A": max_force(ts),
                            "barrier_eV": float(ts.get_potential_energy() - endpoints[0].get_potential_energy())}
        result["stage"] = "hessian"
        hessian = numerical_hessian(ts, config["hessian_delta_A"])
        frequencies, modes = projected_frequencies(ts, hessian)
        np.savez(folder / "ts_hessian.npz", hessian_eV_A2=hessian, frequencies_cm1=frequencies,
                 cartesian_modes=modes)
        strong_negative = int(np.count_nonzero(frequencies < -config["imaginary_threshold_cm1"]))
        result["vibrations"] = {"frequencies_cm1": frequencies.tolist(),
                                "imaginary_modes_above_threshold": strong_negative,
                                "all_negative_modes": int(np.count_nonzero(frequencies < 0)),
                                "rigid_modes_removed": 3 * len(ts) - len(frequencies)}
        stationary = max_force(ts) <= config["ts_fmax_eV_A"]
        if saddle_converged and stationary and strong_negative == 1:
            result["stage"] = "irc"
            result["irc"] = run_irc(ts, calc, endpoints, folder, config)
        result["mlip_verified"] = verified_result(result, config)
        result["saddle_irc_verified"] = verified_saddle(result, config)
        result["status"] = "completed"
        result["stage"] = "finished"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result["elapsed_seconds"] = time.monotonic() - start
    save_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v0.4.0.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v0.4.0"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--reactions", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    selected = prepare(config, args.output)
    if args.models and set(args.models) - set(config["models"]):
        parser.error("Unknown model requested")
    if args.reactions and set(args.reactions) - {item["reaction"] for item in selected}:
        parser.error("Requested reaction is outside the fixed selection")
    if args.prepare_only:
        print(json.dumps([{key: value for key, value in item.items() if key != "endpoints"}
                          for item in selected], indent=2), flush=True)
        return
    if args.recheck:
        for name in args.models or config["models"]:
            for item in selected:
                if args.reactions and item["reaction"] not in args.reactions:
                    continue
                result = recheck_saved_result(args.output / name / item["reaction"], config)
                print(f"Rechecked {name} {item['reaction']} verified={result['mlip_verified']}", flush=True)
        return
    from mace.calculators import MACECalculator
    for name in args.models or config["models"]:
        spec = config["models"][name]
        calc = MACECalculator(model_paths=spec["path"], head=spec["head"],
                              device=config["device"], default_dtype=config["dtype"])
        for item in selected:
            if args.reactions and item["reaction"] not in args.reactions:
                continue
            folder = args.output / name / item["reaction"]
            result_path = folder / "result.json"
            previous = None
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous["status"] != "failed" or not args.retry_failed:
                    print(f"Keeping recorded outcome {name} {item['reaction']}", flush=True)
                    continue
            print(f"Starting {name} {item['reaction']}", flush=True)
            result = search_one(item, calc, folder, config)
            if previous is not None:
                result["previous_attempt"] = previous
                save_json(result_path, result)
            print(f"Finished {name} {item['reaction']} {result['status']} "
                  f"verified={result['mlip_verified']} time={result['elapsed_seconds']:.1f}s", flush=True)
        del calc


if __name__ == "__main__":
    main()
