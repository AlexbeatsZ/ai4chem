"""Run the separately frozen native Mac DFT TS/frequency/IRC pilot.

PySCF supplies analytic derivatives; geomeTRIC supplies optimizers and IRC.
Raw-chain completion is distinct from the analysis environment's mapped-endpoint
and Gaussian-reference checks. Every invocation requires a fresh output folder.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import time
import traceback

from check_mac_qc import save_json, utc_now
from check_mac_ts_irc import setup


def coordinates(mol):
    return {"symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
            "positions_A": mol.atom_coords(unit="Angstrom").tolist()}


def method(atoms, folder, config):
    import numpy as np
    from pyscf import dft, gto
    xyz = np.asarray(atoms["positions_A"], dtype=float)
    if xyz.shape != (len(atoms["symbols"]), 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite mapped Nx3 Cartesian coordinates")
    mol = gto.M(atom=list(zip(atoms["symbols"], xyz)), unit="Angstrom",
                basis=config["basis"], cart=config["cart"], charge=config["charge"],
                spin=config["spin"], max_memory=config["memory_mb"], verbose=4,
                output=str(folder / "pyscf.log"))
    mf = dft.RKS(mol)
    mf.xc = config["xc"]
    mf.conv_tol = config["scf_conv_tol"]
    mf.max_cycle = config["scf_max_cycles"]
    mf.grids.atom_grid = (config["grid_radial"], config["grid_angular"])
    mf.grids.prune = config["grid_pruning"]
    energy = mf.kernel()
    if not mf.converged or not np.isfinite(energy):
        raise RuntimeError("Initial SCF did not converge to a finite energy")
    return mf


def analytic_hessian(mf, folder):
    import numpy as np
    tensor = np.asarray(mf.Hessian().kernel())
    n = mf.mol.natm
    if tensor.shape != (n, n, 3, 3) or not np.isfinite(tensor).all():
        raise RuntimeError("Incomplete/nonfinite analytic Hessian")
    matrix = tensor.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
    if not np.allclose(matrix, matrix.T, atol=1e-7, rtol=1e-7):
        raise RuntimeError("Analytic Hessian is not symmetric")
    np.save(folder / "hessian.npy", tensor)
    path = folder / "initial_hessian.txt"
    np.savetxt(path, matrix)
    return tensor, path


def optimize(atoms, folder, config, kind="minimum", direction=None, ts_hessian=None):
    import numpy as np
    import geometric.optimize
    from pyscf.geomopt import geometric_solver
    folder.mkdir()
    start = time.perf_counter()
    evaluations = 0
    with (folder / "optimizer.log").open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        mf = method(atoms, folder, config)

        def callback(env):
            nonlocal evaluations
            gradient = np.asarray(env["gradients"])
            if not np.isfinite(gradient).all() or not np.isfinite(env["energy"]):
                raise RuntimeError("Nonfinite geometry-optimization energy/gradient")
            evaluations += 1
            record = {"evaluation": evaluations, "energy_hartree": float(env["energy"]),
                      "max_gradient_hartree_per_bohr": float(np.linalg.norm(gradient, axis=1).max()),
                      **coordinates(env["mol"])}
            with (folder / "evaluations.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

        engine = geometric_solver.PySCFEngine(mf.nuc_grad_method().as_scanner())
        engine.mol = engine.mol.copy()
        engine.M.build_topology()
        engine.callback = callback
        engine.assert_convergence = True
        options = {"maxiter": config[f"max_{kind}_steps"] if kind != "ts" else config["max_ts_steps"]}
        if kind == "ts":
            _, hessian_path = analytic_hessian(mf, folder)
            options.update(transition=True, hessian=f"file:{hessian_path}")
        elif kind == "irc":
            # Both directions use the same optimized TS geometry and Hessian.
            matrix = np.load(ts_hessian).transpose(0, 2, 1, 3).reshape(3*mf.mol.natm, 3*mf.mol.natm)
            hessian_path = folder / "initial_hessian.txt"
            np.savetxt(hessian_path, matrix)
            options.update(irc=True, irc_direction=direction, trust=config["irc_trust_A"],
                           hessian=f"file:{hessian_path}")
        geometric.optimize.run_optimizer(customengine=engine, input=str(folder / "geometry"), **options)
        final = coordinates(engine.mol)
    result = {"kind": kind, "direction": direction, "converged": True,
              "evaluations": evaluations, "elapsed_seconds": time.perf_counter()-start, **final}
    save_json(folder / "result.json", result)
    return final


def frequencies(atoms, folder, config, expected_modes):
    import numpy as np
    from pyscf.hessian import thermo
    folder.mkdir()
    start = time.perf_counter()
    with (folder / "analysis.log").open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        mf = method(atoms, folder, config)
        hessian, _ = analytic_hessian(mf, folder)
        modes = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
        gradient = np.asarray(mf.nuc_grad_method().kernel())
    values = np.asarray(modes["freq_wavenumber"], dtype=float)
    if len(values) != expected_modes or not np.isfinite(values).all():
        raise RuntimeError(f"Expected exactly {expected_modes} finite vibrational modes")
    if gradient.shape != (len(atoms["symbols"]), 3) or not np.isfinite(gradient).all():
        raise RuntimeError("Expected finite final Nx3 gradient")
    result = {"energy_hartree": float(mf.e_tot), "basis_functions": mf.mol.nao_nr(),
              "frequencies_cm1": values.tolist(), "imaginary_modes": int((values < 0).sum()),
              "significant_imaginary_modes": int((values < -config["imaginary_threshold_cm1"]).sum()),
              "max_gradient_hartree_per_bohr": float(np.linalg.norm(gradient, axis=1).max()),
              "elapsed_seconds": time.perf_counter()-start, **atoms}
    save_json(folder / "result.json", result)
    return result


def branch(ts, direction, output, config, expected_modes):
    root = Path(output) / direction
    root.mkdir()
    setup(root, config["threads"])
    start = time.perf_counter()
    result = {"direction": direction, "pid": os.getpid(), "status": "running", "stage": "irc",
              "started_at_utc": utc_now()}
    save_json(root / "result.json", result)
    try:
        end = optimize(ts, root / "irc", config, kind="irc", direction=direction,
                       ts_hessian=Path(output)/"ts_frequencies/hessian.npy")
        result["stage"] = "minimum"
        save_json(root / "result.json", result)
        minimum = optimize(end, root / "minimum", config)
        result["stage"] = "minimum_frequencies"
        save_json(root / "result.json", result)
        analysis = frequencies(minimum, root / "frequencies", config, expected_modes)
        if analysis["imaginary_modes"]:
            raise RuntimeError("IRC endpoint has negative vibrational frequencies")
        result.update(status="passed", stage="finished", analysis=analysis)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result.update(finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-start,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3)
    save_json(root / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reaction", default="rxn0666")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    job = next(j for j in protocol["jobs"] if j["reaction"] == args.reaction and j["engine"] == "PySCF-geomeTRIC")
    config = protocol["mac"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup(output, config["threads"])
    import pyscf
    import geometric
    start = time.perf_counter()
    result = {"job_id": job["job_id"], "reaction": job["reaction"], "host": platform.node(),
              "pid": os.getpid(), "status": "running", "stage": "ts", "raw_chain_completed": False,
              "dft_verified": False, "purpose": job["purpose"], "started_at_utc": utc_now(),
              "protocol_frozen_at_utc": protocol["frozen_at_utc"], "configuration": config,
              "pyscf_version": pyscf.__version__, "geometric_version": geometric.__version__, "branches": []}
    save_json(output / "protocol.json", protocol)
    save_json(output / "summary.json", result)
    try:
        ts = optimize(job, output / "ts", config, kind="ts")
        result["stage"] = "ts_frequencies"
        save_json(output / "summary.json", result)
        result["ts"] = frequencies(ts, output / "ts_frequencies", config, job["expected_vibrational_modes"])
        if result["ts"]["imaginary_modes"] != 1 or result["ts"]["significant_imaginary_modes"] != 1:
            raise RuntimeError("Optimized TS must have exactly one significant negative vibrational mode")
        result["stage"] = "irc_and_minima"
        save_json(output / "summary.json", result)
        print("TS/DFT frequency passed; launching both IRC directions", flush=True)
        with ProcessPoolExecutor(max_workers=config["irc_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(branch, ts, direction, str(output), config, job["expected_vibrational_modes"])
                       for direction in ("forward", "backward")]
            for future in as_completed(pending):
                record = future.result()
                result["branches"].append(record)
                save_json(output / "summary.json", result)
                print(f"{record['direction']}: {record['status']}", flush=True)
        if len(result["branches"]) != 2 or any(b["status"] != "passed" for b in result["branches"]):
            raise RuntimeError("Both IRC/minimum/frequency branches must pass")
        result.update(status="raw_chain_completed", stage="awaiting_geometry_validation", raw_chain_completed=True)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result.update(finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-start,
                  parent_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3)
    save_json(output / "summary.json", result)
    print(json.dumps({k: result[k] for k in ("reaction", "status", "stage", "elapsed_seconds")}), flush=True)
    return 0 if result["raw_chain_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
