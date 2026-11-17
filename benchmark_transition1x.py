"""Compare pretrained MACE models on fixed final Transition1x NEB bands."""

import argparse
import csv
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from mace.calculators import MACECalculator, mace_off

from transition1x_data import load_paths


MODELS = {
    "off23-small": {"label": "OFF23-small", "head": None},
    "mh1-omol": {"label": "MH-1 / omol", "file": "mace-mh-1.model", "head": "omol"},
    "mh0-rgd1": {"label": "MH-0 / RGD1", "file": "mace-mh-0.model", "head": "rgd1_b3lyp"},
}


def calculator_for(name, device, dtype):
    spec = MODELS[name]
    if name == "off23-small":
        return mace_off(model="small", device=device, default_dtype=dtype)
    path = Path(__file__).resolve().parent / "models" / spec["file"]
    return MACECalculator(model_paths=str(path), device=device, default_dtype=dtype, head=spec["head"])


def error_metrics(predictions, reference):
    error = np.asarray(predictions) - np.asarray(reference)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/transition1x_test_10.h5"))
    parser.add_argument("--output", type=Path, default=Path("results/v0.2.0"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    paths = load_paths(args.input, args.split, args.count, args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reference_structures = [atoms for path in paths for atoms in path.atoms()]
    write(output / "reference_paths.extxyz", reference_structures)
    restored = read(output / "reference_paths.extxyz", index=":")
    for original, saved in zip(reference_structures, restored, strict=True):
        np.testing.assert_array_equal(original.numbers, saved.numbers)
        np.testing.assert_allclose(original.positions, saved.positions, atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.get_potential_energy(), saved.get_potential_energy(), atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.get_forces(), saved.get_forces(), atol=1e-8, rtol=0)
    with h5py.File(args.input, "r") as source:
        provenance = json.loads(source.attrs.get("provenance", "{}"))
    summary = {
        "input": str(args.input), "split": args.split, "seed": args.seed,
        "reactions": len(paths), "path_images": len(reference_structures), "dtype": args.dtype,
        "device": args.device, "python": platform.python_version(), "torch": torch.__version__,
        "mace_torch": version("mace-torch"), "ase": version("ase"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "reference_method": "omegaB97X/6-31G(d)", "source_provenance": provenance,
        "energy_reference": "each model's own reactant energy on the same fixed path",
        "barrier_definition": "maximum sampled band energy minus reactant energy",
        "force_error_definition": "Cartesian component error in eV/angstrom",
        "ts_region_definition": "writer-defined DFT TS image and adjacent band images, clipped at endpoints",
        "scope": "ten-reaction cross-DFT zero-shot feasibility benchmark",
        "models": {},
    }
    image_rows, reaction_rows = [], []
    predicted_curves = {}
    for name in args.models:
        print(f"Evaluating {name} on {len(paths)} reactions", flush=True)
        calc = calculator_for(name, args.device, args.dtype)
        model = calc.models[0]
        actual_head = MODELS[name]["head"]
        if actual_head is not None and actual_head not in model.heads:
            raise ValueError(f"Requested {actual_head} is absent from checkpoint heads")
        if next(model.parameters()).device.type != torch.device(args.device).type:
            raise RuntimeError("Model is on an unexpected device")
        model_info = {"head": actual_head, "available_heads": list(getattr(model, "heads", [])),
                      "curves": [], "reaction_metrics": []}
        relative_errors, force_errors, ts_force_errors = [], [], []
        started = time.perf_counter()
        for path in paths:
            energies, forces, predicted_structures = [], [], []
            for atoms in path.atoms():
                reference_energy, reference_forces = atoms.get_potential_energy(), atoms.get_forces()
                atoms.calc = calc
                energy, force = float(atoms.get_potential_energy()), atoms.get_forces()
                if not np.isfinite(energy) or force.shape != (len(atoms), 3) or not np.isfinite(force).all():
                    raise ValueError(f"Invalid predictions for {name}/{path.reaction}")
                atoms.info["reference_energy_eV"] = reference_energy
                atoms.arrays["reference_forces"] = reference_forces.copy()
                # Freeze calculated results before assigning this calculator to the next image.
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=force.copy())
                predicted_structures.append(atoms)
                energies.append(energy)
                forces.append(force)
            energies, forces = np.asarray(energies), np.asarray(forces)
            predicted_relative = energies - energies[0]
            reference_relative = path.energies - path.energies[0]
            energy_error = predicted_relative - reference_relative
            force_error = forces - path.forces
            dft_ts = path.reference_ts_image
            ts_region = slice(max(0, dft_ts - 1), min(10, dft_ts + 2))
            predicted_barrier, reference_barrier = float(predicted_relative.max()), float(reference_relative.max())
            metrics = {
                "model": name, "split": path.split, "formula": path.formula, "reaction": path.reaction,
                "atoms": len(path.numbers), "source_frames": path.source_frame_count,
                "dft_ts_image": dft_ts, "predicted_max_image": int(predicted_relative.argmax()),
                "relative_energy_mae_eV": float(np.mean(np.abs(energy_error))),
                "relative_energy_rmse_eV": float(np.sqrt(np.mean(energy_error**2))),
                "reference_barrier_eV": reference_barrier, "predicted_barrier_eV": predicted_barrier,
                "barrier_error_eV": predicted_barrier - reference_barrier,
                "energy_error_at_dft_ts_eV": float(energy_error[dft_ts]),
                "reaction_energy_error_eV": float(energy_error[-1]),
                "force_component_mae_eV_per_A": float(np.mean(np.abs(force_error))),
                "force_component_rmse_eV_per_A": float(np.sqrt(np.mean(force_error**2))),
                "ts_region_force_component_rmse_eV_per_A": float(np.sqrt(np.mean(force_error[ts_region]**2))),
            }
            reaction_rows.append(metrics)
            model_info["reaction_metrics"].append(metrics)
            model_info["curves"].append({"reaction": path.reaction, "formula": path.formula,
                                         "reference_relative_eV": reference_relative.tolist(),
                                         "predicted_relative_eV": predicted_relative.tolist()})
            relative_errors.extend(energy_error.tolist())
            force_errors.extend(force_error.ravel().tolist())
            ts_force_errors.extend(force_error[ts_region].ravel().tolist())
            for image in range(10):
                image_rows.append({"model": name, "split": path.split, "formula": path.formula,
                                   "reaction": path.reaction, "image": image,
                                   "source_frame": int(path.source_indices[image]),
                                   "reference_relative_eV": float(reference_relative[image]),
                                   "predicted_relative_eV": float(predicted_relative[image]),
                                   "relative_error_eV": float(energy_error[image]),
                                   "force_component_rmse_eV_per_A": float(np.sqrt(np.mean(force_error[image]**2)))})
            write(output / f"{name}_{path.reaction}.extxyz", predicted_structures)
            print(f"{name} {path.reaction} barrier error {metrics['barrier_error_eV']:.4f} eV, "
                  f"force RMSE {metrics['force_component_rmse_eV_per_A']:.4f} eV/A", flush=True)
        model_info["inference_seconds"] = time.perf_counter() - started
        energy_stats = error_metrics(relative_errors, np.zeros(len(relative_errors)))
        force_stats = error_metrics(force_errors, np.zeros(len(force_errors)))
        barrier_errors = [entry["barrier_error_eV"] for entry in model_info["reaction_metrics"]]
        model_info["aggregate"] = {
            "relative_energy_mae_eV": energy_stats["mae"],
            "relative_energy_rmse_eV": energy_stats["rmse"],
            "barrier_mae_eV": float(np.mean(np.abs(barrier_errors))),
            "force_component_mae_eV_per_A": force_stats["mae"],
            "force_component_rmse_eV_per_A": force_stats["rmse"],
            "ts_region_force_component_rmse_eV_per_A": float(np.sqrt(np.mean(np.square(ts_force_errors)))),
        }
        summary["models"][name] = model_info
        predicted_curves[name] = model_info["curves"]
        del model, calc
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_csv(output / "reaction_metrics.csv", reaction_rows)
    write_csv(output / "image_predictions.csv", image_rows)
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    columns = 2
    rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3.1), squeeze=False, layout="constrained")
    colors = {"off23-small": "#d47720", "mh1-omol": "#19746a", "mh0-rgd1": "#72539b"}
    for index, path in enumerate(paths):
        axis = axes.flat[index]
        reference = path.energies - path.energies[0]
        axis.plot(range(10), reference, "o-", color="#202733", linewidth=2, markersize=3, label="DFT reference")
        for name in args.models:
            axis.plot(range(10), predicted_curves[name][index]["predicted_relative_eV"],
                      ".--", color=colors[name], linewidth=1.4, label=MODELS[name]["label"])
        axis.set_title(f"{path.reaction}  |  {path.formula}", fontsize=11)
        axis.set_xlabel("Final NEB image index")
        axis.set_ylabel("Relative energy (eV)")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(fontsize=8)
    for index in range(len(paths), rows * columns):
        axes.flat[index].set_visible(False)
    fig.suptitle("Transition1x test subset | fixed geometries | reactant-referenced energies\n"
                 f"Cross-DFT zero-shot comparison, {len(paths)} reactions", fontsize=14)
    fig.savefig(output / "reaction_energy_profiles.png", dpi=180)
    fig.savefig(output / "reaction_energy_profiles.pdf")
    plt.close(fig)
    print(json.dumps({name: data["aggregate"] for name, data in summary["models"].items()}, indent=2))
    print("PASS - reference readback and pretrained path benchmark completed", flush=True)


if __name__ == "__main__":
    main()
