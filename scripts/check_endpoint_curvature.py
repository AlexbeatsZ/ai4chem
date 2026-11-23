"""Check a saved failed endpoint's negative mode with displaced analytic gradients.

This is a finite-difference diagnostic at fixed geometries, never a change to
the frozen full-chain verdict. No optimization or replacement Hessian is run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

from check_mac_qc import save_json, utc_now
from check_mac_ts_irc import setup
from run_mac_dft_chain import method


def evaluate(point, folder, config):
    import numpy as np
    output = Path(folder)/point["name"]
    output.mkdir()
    setup(output, 4)
    started = time.perf_counter()
    mf = method(point, output, config)
    gradient = np.asarray(mf.nuc_grad_method().kernel())
    if gradient.shape != (len(point["symbols"]), 3) or not np.isfinite(gradient).all():
        raise ValueError("Invalid displaced analytic gradient")
    result = {"name": point["name"], "displacement_A": point["displacement_A"],
              "gradient_hartree_per_bohr": gradient.tolist(), "energy_hartree": float(mf.e_tot),
              "scf_converged": bool(mf.converged), "elapsed_seconds": time.perf_counter()-started}
    save_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    diagnostic = json.loads(args.diagnostic.read_text())
    config = json.loads(args.protocol.read_text())["mac"]
    config["scf_conv_tol"] = diagnostic["scf_conv_tol"]
    setup(output, diagnostic["threads_per_worker"])
    import numpy as np
    from pyscf import gto, lib
    from pyscf.hessian import thermo
    source = json.loads((args.source/"result.json").read_text())
    xyz = np.asarray(source["positions_A"])
    mol = gto.M(atom=list(zip(source["symbols"], xyz)), unit="Angstrom", basis=config["basis"],
                cart=config["cart"], charge=config["charge"], spin=config["spin"], verbose=0)
    hessian = np.load(args.source/"hessian.npy")
    modes = thermo.harmonic_analysis(mol, hessian, imaginary_freq=False)
    index = int(np.argmin(modes["freq_wavenumber"]))
    displacement = modes["norm_mode"][index].real
    displacement /= np.linalg.norm(displacement)
    matrix = hessian.transpose(0,2,1,3).reshape(3*mol.natm, 3*mol.natm)
    analytic_curvature = float(displacement.ravel()@matrix@displacement.ravel())
    if float(modes["freq_wavenumber"][index]) >= 0 or analytic_curvature >= 0:
        raise ValueError("Source does not have a negative vibrational curvature")
    points = [{"name": f"{amplitude:g}-{sign:+d}", "symbols": source["symbols"],
               "positions_A": (xyz+sign*amplitude*displacement).tolist(),
               "displacement_A": sign*amplitude}
              for amplitude in diagnostic["displacements_A"] for sign in (-1,1)]
    state = {"started_at_utc": utc_now(), "status": "running", "diagnostic": diagnostic,
             "source": str(args.source), "original_full_chain_verdict_changed": False,
             "mode_frequency_cm1": float(modes["freq_wavenumber"][index]),
             "cartesian_direction": displacement.tolist(),
             "analytic_directional_curvature_hartree_per_bohr2": analytic_curvature,
             "results": [], "finite_differences": []}
    save_json(output/"summary.json", state)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=diagnostic["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, point, str(output), config) for point in points]
        for future in as_completed(futures):
            state["results"].append(future.result())
            save_json(output/"summary.json", state)
    for amplitude in diagnostic["displacements_A"]:
        pair = {int(np.sign(point["displacement_A"])): point for point in state["results"]
                if abs(point["displacement_A"]) == amplitude}
        delta_gradient = np.asarray(pair[1]["gradient_hartree_per_bohr"])-np.asarray(pair[-1]["gradient_hartree_per_bohr"])
        numerical = float(np.sum(displacement*delta_gradient)/(2*amplitude/lib.param.BOHR))
        state["finite_differences"].append({"amplitude_A": amplitude,
            "directional_curvature_hartree_per_bohr2": numerical,
            "relative_difference_from_analytic": (numerical-analytic_curvature)/abs(analytic_curvature),
            "negative_curvature": numerical < 0})
    state.update(status="completed", finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-started,
                 negative_curvature_confirmed=all(x["negative_curvature"] for x in state["finite_differences"]))
    save_json(output/"summary.json", state)
    print(json.dumps({k:state[k] for k in ("status", "mode_frequency_cm1", "finite_differences", "negative_curvature_confirmed")}))


if __name__ == "__main__":
    main()
