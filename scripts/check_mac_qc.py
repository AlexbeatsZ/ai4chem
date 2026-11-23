"""Run independent PySCF single-point/gradient checks in bounded worker processes.

This is a node-readiness and cross-engine check, never a replacement for the
frozen v0.5.0 Gaussian TS/frequency/IRC validation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import time
import traceback


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_one(job, output, threads, memory_mb):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    folder = Path(output) / job["reaction"]
    folder.mkdir()
    started = time.perf_counter()
    result = {"reaction": job["reaction"], "pid": os.getpid(),
              "started_at_utc": utc_now(), "status": "running",
              "purpose": "readiness_and_cross_engine_check_only",
              "source_candidate": job["candidate_key_prefix"],
              "threads": threads, "memory_mb": memory_mb,
              "method": "wb97x/6-31g(d)", "charge": 0, "spin_2S": 0,
              "cartesian_basis": True,
              "grid": {"radial": 99, "angular": 590, "pruning": None},
              "scf_convergence_hartree": 1e-9,
              "python_version": platform.python_version(), "platform": platform.platform()}
    save_json(folder / "result.json", result)
    try:
        import numpy as np
        import pyscf
        from pyscf import dft, gto, lib

        lib.num_threads(threads)
        result.update(threads=lib.num_threads(), pyscf_version=pyscf.__version__,
                      numpy_version=np.__version__)
        coordinates = np.asarray(job["positions_A"], dtype=float)
        if coordinates.shape != (len(job["symbols"]), 3) or not np.isfinite(coordinates).all():
            raise ValueError("Expected finite mapped Cartesian coordinates in Angstrom")
        mol = gto.M(atom=list(zip(job["symbols"], coordinates)), unit="Angstrom",
                    basis="6-31g(d)", cart=True, charge=0, spin=0, max_memory=memory_mb,
                    verbose=4, output=str(folder / "pyscf.log"))
        # Gaussian's saved 6-31G(d) logs use Cartesian 6D functions. PySCF's
        # spherical default otherwise changes the basis despite the same name.
        result["basis_functions"] = mol.nao_nr()
        if result["basis_functions"] != job["gaussian_basis_functions"]:
            raise ValueError("PySCF and Gaussian basis function counts differ")
        mf = dft.RKS(mol)
        mf.xc = "wb97x"
        mf.conv_tol = 1e-9
        mf.max_cycle = 128
        mf.grids.atom_grid = (99, 590)
        mf.grids.prune = None
        energy = float(mf.kernel())
        if not mf.converged or not np.isfinite(energy):
            raise ValueError("SCF did not converge to a finite energy")
        gradient = np.asarray(mf.nuc_grad_method().kernel())
        if gradient.shape != coordinates.shape or not np.isfinite(gradient).all():
            raise ValueError("Expected a finite Nx3 analytic gradient")
        reference = float(job["gaussian_energy_hartree"])
        result.update(status="passed", scf_converged=True, energy_hartree=energy,
                      gaussian_energy_hartree=reference,
                      energy_difference_hartree=energy - reference,
                      max_gradient_hartree_per_bohr=float(np.linalg.norm(gradient, axis=1).max()),
                      forces_hartree_per_bohr=(-gradient).tolist())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started,
                      process_peak_rss_gib=rss / (1024 ** (3 if platform.system() == "Darwin" else 2)))
        save_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=4000)
    args = parser.parse_args()
    if min(args.jobs, args.threads, args.memory_mb) < 1:
        parser.error("jobs, threads and memory must be positive")
    manifest = json.loads(args.manifest.read_text())
    names = [job["reaction"] for job in manifest["jobs"]]
    if not names or len(set(names)) != len(names):
        parser.error("manifest needs unique reaction job IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "inputs.json", manifest)
    started = time.perf_counter()
    summary = {"started_at_utc": utc_now(), "status": "running", "workers": args.jobs,
               "threads_per_worker": args.threads, "memory_mb_per_worker": args.memory_mb,
               "purpose": "readiness_and_cross_engine_check_only", "results": []}
    save_json(args.output / "summary.json", summary)
    with ProcessPoolExecutor(max_workers=args.jobs,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [pool.submit(check_one, job, str(args.output), args.threads, args.memory_mb)
                   for job in manifest["jobs"]]
        for future in as_completed(pending):
            result = future.result()
            summary["results"].append(result)
            save_json(args.output / "summary.json", summary)
            print(f"{result['reaction']}: {result['status']} ({result['elapsed_seconds']:.1f} s)", flush=True)
    events = []
    for result in summary["results"]:
        events.extend(((result["started_at_utc"], 1), (result["finished_at_utc"], -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(active, peak)
    summary.update(status="passed" if all(r["status"] == "passed" for r in summary["results"]) else "failed",
                   finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started,
                   observed_peak_concurrent_jobs=peak)
    save_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
