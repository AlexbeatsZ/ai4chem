"""Exercise PySCF/geomeTRIC TS, frequencies, both IRC directions and minima.

Uses the upstream HCN/HNC HF/3-21G example as a capability test, independent of
the frozen v0.5.0 Gaussian reaction matrix.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from check_mac_qc import save_json, utc_now


def setup(output, threads):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(threads)
    from pyscf import lib
    lib.num_threads(threads)
    scratch = output / "scratch"
    scratch.mkdir(exist_ok=True)
    lib.param.TMPDIR = str(scratch)


def method(atoms, folder):
    from pyscf import gto, scf
    mol = gto.M(atom=atoms, basis="3-21g", unit="Angstrom", charge=0, spin=0,
                max_memory=2000, verbose=4, output=str(folder / "pyscf.log"))
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Initial SCF did not converge")
    return mf


def coordinates(mol):
    return [(mol.atom_symbol(i), xyz.tolist())
            for i, xyz in enumerate(mol.atom_coords(unit="Angstrom"))]


def optimize(atoms, folder, kind="minimum", direction=None):
    import numpy as np
    import geometric.optimize
    from pyscf.geomopt import geometric_solver
    folder.mkdir()
    with (folder / "optimizer.log").open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        mf = method(atoms, folder)
        trajectory = folder / "evaluations.jsonl"

        def callback(env):
            entry = {"cycle": env["self"].cycle, "energy_hartree": float(env["energy"]),
                     "positions_A": env["mol"].atom_coords(unit="Angstrom").tolist()}
            with trajectory.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")

        options = {"maxsteps": 300 if kind == "irc" else 80, "callback": callback}
        if kind == "ts":
            options.update(transition=True, hessian=True)
        if kind == "irc":
            # PySCF 2.15.1's generic wrapper does not build Molecule topology,
            # while geomeTRIC 1.1.1's IRC setup reads M.molecules immediately.
            # Use the public custom-engine interface and prepare it explicitly.
            engine = geometric_solver.PySCFEngine(mf.nuc_grad_method().as_scanner())
            engine.mol = engine.mol.copy()
            engine.M.build_topology()
            engine.callback = callback
            engine.assert_convergence = True
            hessian = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9)
            hessian_file = folder / "initial_hessian.txt"
            np.savetxt(hessian_file, hessian)
            geometric.optimize.run_optimizer(customengine=engine, input=str(folder / "path"),
                                             irc=True, irc_direction=direction, trust=0.05,
                                             hessian=f"file:{hessian_file}", maxiter=300)
            converged, mol = True, engine.mol
        else:
            converged, mol = geometric_solver.kernel(mf, **options)
    result = {"kind": kind, "direction": direction, "converged": bool(converged),
              "atoms": coordinates(mol)}
    save_json(folder / "result.json", result)
    if not converged:
        raise RuntimeError(f"geomeTRIC {kind} {direction} did not converge")
    return result["atoms"]


def frequencies(atoms, folder):
    import numpy as np
    from pyscf.hessian import thermo
    folder.mkdir()
    with (folder / "analysis.log").open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        mf = method(atoms, folder)
        hessian = mf.Hessian().kernel()
        modes = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
    values = np.asarray(modes["freq_wavenumber"], dtype=float)
    if not np.isfinite(values).all() or len(values) < 3:
        raise RuntimeError("Incomplete/nonfinite triatomic vibrational frequencies")
    np.save(folder / "hessian.npy", hessian)
    result = {"energy_hartree": float(mf.e_tot), "frequencies_cm1": values.tolist(),
              "imaginary_modes": int((values < 0).sum()), "atoms": atoms}
    save_json(folder / "result.json", result)
    return result


def branch(ts, direction, output, threads):
    import numpy as np
    root = Path(output) / direction
    root.mkdir()
    setup(root, threads)
    result = {"direction": direction, "started_at_utc": utc_now(), "status": "running"}
    save_json(root / "result.json", result)
    try:
        end = optimize(ts, root / "irc", kind="irc", direction=direction)
        minimum = optimize(end, root / "minimum")
        analysis = frequencies(minimum, root / "frequencies")
        xyz = np.array([item[1] for item in minimum])
        hc = float(np.linalg.norm(xyz[2] - xyz[0]))
        hn = float(np.linalg.norm(xyz[2] - xyz[1]))
        cn = float(np.linalg.norm(xyz[0] - xyz[1]))
        if analysis["imaginary_modes"] or cn >= 1.5 or min(hc, hn) >= 1.3:
            raise RuntimeError("Endpoint is not a stable expected HCN/HNC minimum")
        result.update(status="passed", endpoint="HCN" if hc < hn else "HNC",
                      distances_A={"HC": hc, "HN": hn, "CN": cn}, analysis=analysis)
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result["finished_at_utc"] = utc_now()
    save_json(root / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup(output, args.threads)
    import geometric
    import pyscf
    source = Path(__file__).resolve().parents[1] / "examples/mac-qc/hcn_hnc_ts.xyz"
    lines = source.read_text().splitlines()[2:]
    atoms = [(line.split()[0], list(map(float, line.split()[1:]))) for line in lines if line.strip()]
    started = time.perf_counter()
    summary = {"started_at_utc": utc_now(), "status": "running", "stage": "ts",
               "method": "RHF/3-21G", "purpose": "capability_test_not_v05_gaussian_validation",
               "pyscf_version": pyscf.__version__, "geometric_version": geometric.__version__,
               "threads_per_process": args.threads, "irc_workers": 2, "branches": []}
    save_json(output / "summary.json", summary)
    try:
        ts = optimize(atoms, output / "ts", kind="ts")
        summary["ts"] = frequencies(ts, output / "ts_frequencies")
        values = summary["ts"]["frequencies_cm1"]
        if summary["ts"]["imaginary_modes"] != 1 or min(values) >= -50:
            raise RuntimeError("Optimized TS must have exactly one significant imaginary mode")
        summary["stage"] = "irc_and_minima"
        save_json(output / "summary.json", summary)
        print("TS optimization/frequencies passed; running both IRC branches", flush=True)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(branch, ts, direction, str(output), args.threads)
                       for direction in ("forward", "backward")]
            for future in as_completed(pending):
                result = future.result()
                summary["branches"].append(result)
                save_json(output / "summary.json", summary)
                print(f"{result['direction']}: {result['status']} {result.get('endpoint', '')}", flush=True)
        if any(r["status"] != "passed" for r in summary["branches"]) or {
            r.get("endpoint") for r in summary["branches"]
        } != {"HCN", "HNC"}:
            raise RuntimeError("Both converged IRC branches must connect the two distinct stable endpoints")
        summary.update(status="passed", stage="finished")
    except Exception:
        summary.update(status="failed", error=traceback.format_exc())
    summary.update(finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
