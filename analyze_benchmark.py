"""Diagnose fixed-path errors and bootstrap reaction-level statistics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list


def endpoint_change(reactant, product):
    def bonds(atoms):
        left, right = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
        return {tuple(sorted((int(i), int(j)))) for i, j in zip(left, right) if i != j}
    initial, final = bonds(reactant), bonds(product)
    formed, broken = len(final - initial), len(initial - final)
    if formed and broken:
        kind = "rearrangement"
    elif formed:
        kind = "formation"
    elif broken:
        kind = "cleavage"
    else:
        kind = "no_detected_change"
    return kind, formed, broken


def bootstrap_mean(values, groups, replicates=2000, seed=42, observation_weights=None, root=False):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    membership = np.asarray([groups == group for group in unique], dtype=float)
    draws = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), replicates)
    weights = draws @ membership
    population_weights = np.ones(len(values)) if observation_weights is None else np.asarray(observation_weights)
    weights *= population_weights
    estimates = weights @ values / weights.sum(axis=1)
    point = float(np.average(values, weights=population_weights))
    if root:
        estimates = np.sqrt(estimates)
        point = float(np.sqrt(point))
    return {"mean": point, "lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)), "clusters": len(unique)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    reactions = pd.read_csv(args.input / "reaction_metrics.csv")
    images = pd.read_csv(args.input / "image_predictions.csv")
    metadata = {}
    structures = read(args.input / "reference_paths.extxyz", index=":")
    for start in range(0, len(structures), 10):
        band = structures[start:start + 10]
        kind, formed, broken = endpoint_change(band[0], band[-1])
        metadata[band[0].info["reaction"]] = (kind, formed, broken)
    additions = []
    for _, row in reactions.iterrows():
        band = images[(images.model == row.model) & (images.reaction == row.reaction)].sort_values("image")
        if len(band) != 10:
            raise ValueError("Each reaction must have exactly ten path images")
        ts = band[band.image == row.dft_ts_image].iloc[0]
        peak_gain = float(band.predicted_relative_eV.max() - ts.predicted_relative_eV)
        reference_peak_gain = float(band.reference_relative_eV.max() - ts.reference_relative_eV)
        np.testing.assert_allclose(row.barrier_error_eV,
                                   row.energy_error_at_dft_ts_eV + peak_gain - reference_peak_gain, atol=1e-9)
        nonreactant = band[band.image != 0].relative_error_eV.to_numpy()
        kind, formed, broken = metadata[row.reaction]
        additions.append({"nonreactant_relative_energy_mae_eV": float(np.abs(nonreactant).mean()),
                          "nonreactant_relative_energy_rmse_eV": float(np.sqrt(np.mean(nonreactant**2))),
                          "ts_relative_energy_abs_error_eV": abs(row.energy_error_at_dft_ts_eV),
                          "barrier_abs_error_eV": abs(row.barrier_error_eV),
                          "peak_shift_images": int(row.predicted_max_image - row.dft_ts_image),
                          "peak_selection_gain_eV": peak_gain,
                          "reference_peak_gain_eV": reference_peak_gain,
                          "endpoint_change": kind, "formed_bonds": formed, "broken_bonds": broken})
    reactions = pd.concat([reactions.reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    reactions.to_csv(output / "diagnosed_reactions.csv", index=False)
    summary = {"bootstrap_replicates": 2000, "models": {},
               "energy_identifiability": "TS relative-energy error equals TS absolute discrepancy minus reactant absolute discrepancy; their individual accuracy is unidentifiable across reference methods",
               "reaction_type_definition": "endpoint connectivity changes using ASE covalent radii times 1.2",
               "force_statistics": "global component RMSE uses atom-count weights; bootstrap intervals below describe means of reaction-level metrics"}
    metrics = ["barrier_abs_error_eV", "nonreactant_relative_energy_mae_eV",
               "ts_relative_energy_abs_error_eV", "force_component_rmse_eV_per_A"]
    slices = []
    for model, data in reactions.groupby("model", sort=True):
        data = data.sort_values("reaction")
        aggregate = {metric: float(data[metric].mean()) for metric in metrics}
        aggregate.update(reactions=len(data), barrier_signed_mean_eV=float(data.barrier_error_eV.mean()),
                         underestimated_barriers=int((data.barrier_error_eV < 0).sum()),
                         shifted_peaks=int((data.peak_shift_images != 0).sum()),
                         force_global_component_rmse_eV_per_A=float(np.sqrt(np.average(data.force_component_rmse_eV_per_A**2, weights=data.atoms))))
        region_lengths = np.minimum(9, data.dft_ts_image + 1) - np.maximum(0, data.dft_ts_image - 1) + 1
        aggregate["ts_region_global_force_rmse_eV_per_A"] = float(np.sqrt(np.average(
            data.ts_region_force_component_rmse_eV_per_A**2, weights=data.atoms * region_lengths)))
        aggregate["mean_reaction_force_rmse_eV_per_A"] = aggregate.pop("force_component_rmse_eV_per_A")
        intervals = {}
        for unit in ("reaction", "formula"):
            intervals[unit] = {metric: bootstrap_mean(data[metric].to_numpy(), data[unit].to_numpy()) for metric in metrics}
            intervals[unit]["global_force_component_rmse_eV_per_A"] = bootstrap_mean(
                data.force_component_rmse_eV_per_A.to_numpy()**2, data[unit].to_numpy(),
                observation_weights=data.atoms.to_numpy(), root=True)
        summary["models"][model] = {"aggregate": aggregate, "bootstrap": intervals}
        for column in ("formula", "atoms", "endpoint_change"):
            for key, group in data.groupby(column):
                slices.append({"model": model, "group_by": column, "group": str(key), "reactions": len(group),
                               **{metric: float(group[metric].mean()) for metric in metrics}})
    pd.DataFrame(slices).to_csv(output / "stratified_metrics.csv", index=False)
    models = sorted(summary["models"])
    paired = {}
    for model in models:
        if model == "mh1-omol":
            continue
        reference = reactions[reactions.model == "mh1-omol"].set_index("reaction")
        comparison = reactions[reactions.model == model].set_index("reaction")
        if set(reference.index) != set(comparison.index):
            continue
        reference = reference.sort_index()
        comparison = comparison.loc[reference.index]
        paired[model + " minus mh1-omol"] = {
            unit: bootstrap_mean(comparison.barrier_abs_error_eV.to_numpy() - reference.barrier_abs_error_eV.to_numpy(),
                                 reference[unit].to_numpy() if unit == "formula" else reference.index.to_numpy())
            for unit in ("reaction", "formula")}
    summary["paired_barrier_mae_differences"] = paired
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    values = [reactions[reactions.model == model].barrier_error_eV.to_numpy() for model in models]
    axes[0].boxplot(values, tick_labels=models, showmeans=True)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Signed sampled-barrier error (eV)")
    for model in models:
        data = reactions[reactions.model == model]
        axes[1].scatter(data.energy_error_at_dft_ts_eV, data.barrier_error_eV, label=model, alpha=0.65, s=18)
    limits = axes[1].get_xlim()
    axes[1].plot(limits, limits, "k--", linewidth=1)
    axes[1].set_xlabel("Relative-energy error at DFT TS (eV)")
    axes[1].set_ylabel("Sampled-barrier error (eV)")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(alpha=0.2)
    fig.savefig(output / "barrier_diagnostics.png", dpi=180)
    fig.savefig(output / "barrier_diagnostics.pdf")
    plt.close(fig)
    print(json.dumps({name: info["aggregate"] for name, info in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
