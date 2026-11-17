"""Validate MACE-OFF23-small CUDA inference and ASE result serialization."""

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from ase.build import molecule
from ase.io import read, write
from mace.calculators import mace_off


def main():
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this OMEN smoke test")

    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}")
    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    matrix = torch.randn(2048, 2048, device="cuda")
    product = matrix @ matrix.T
    if not torch.isfinite(product).all().item():
        raise RuntimeError("CUDA matrix multiplication produced non-finite values")
    del matrix, product
    print("CUDA matrix multiplication PASS", flush=True)

    started = time.perf_counter()
    calculator = mace_off(model="small", device="cuda", default_dtype="float32")
    model_device = next(calculator.models[0].parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"Model loaded on unexpected device {model_device}")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    atoms = molecule("H2O")
    atoms.calc = calculator
    started = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    if not np.isfinite(energy):
        raise RuntimeError("Energy is non-finite")
    if forces.shape != (3, 3) or not np.isfinite(forces).all():
        raise RuntimeError("Expected finite force vectors for three atoms")

    structure_path = output_dir / "smoke_h2o.extxyz"
    write(structure_path, atoms, format="extxyz")
    restored = read(structure_path, format="extxyz")
    if restored.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise RuntimeError("Saved atom symbols differ from the input")
    np.testing.assert_allclose(restored.positions, atoms.positions, atol=1e-8, rtol=0)
    np.testing.assert_allclose(restored.get_potential_energy(), energy, atol=1e-8, rtol=0)
    np.testing.assert_allclose(restored.get_forces(), forces, atol=1e-8, rtol=0)

    summary = {
        "status": "PASS",
        "model": "MACE-OFF23-small",
        "device": str(model_device),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "mace_torch": version("mace-torch"),
        "ase": version("ase"),
        "symbols": atoms.get_chemical_symbols(),
        "positions_angstrom": atoms.positions.tolist(),
        "energy_eV": energy,
        "forces_eV_per_angstrom": forces.tolist(),
        "model_load_seconds": load_seconds,
        "first_inference_seconds": inference_seconds,
        "peak_allocated_gpu_MiB": torch.cuda.max_memory_allocated() / 1024**2,
        "output_file": structure_path.name,
        "saved_results_verified": True,
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("PASS - CUDA inference and saved energy/forces verified", flush=True)


if __name__ == "__main__":
    main()
