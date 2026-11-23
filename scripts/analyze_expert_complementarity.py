"""Unattainable headroom with explicit force/multi-metric nondegradation guards."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from expert_routing import EXPERTS
from model_study_data import save_json


def main():
    directory = ROOT / "results/v0.7.0/router"
    source = ROOT / "results/v0.7.0/meta_development_mac"
    table = pd.read_csv(source / "reaction_metrics.csv")
    names = sorted(table.reaction.unique())
    matrices = {}
    for key in ("barrier_error_eV", "force_component_rmse_eV_per_A",
                "ts_region_force_component_rmse_eV_per_A", "reaction_energy_error_eV"):
        matrices[key] = table.pivot(index="reaction", columns="model", values=key).loc[names, EXPERTS].abs().to_numpy()
    errors, forces = matrices["barrier_error_eV"], matrices["force_component_rmse_eV_per_A"]
    atoms = table[table.model == EXPERTS[0]].set_index("reaction").loc[names, "atoms"].to_numpy()
    baseline = EXPERTS.index("legacy_mh1_256")
    force_mask = forces <= forces[:, baseline:baseline+1] + 1e-12
    all_mask = force_mask.copy()
    for key in ("ts_region_force_component_rmse_eV_per_A", "reaction_energy_error_eV"):
        values = matrices[key]
        all_mask &= values <= values[:, baseline:baseline+1] + 1e-12
    rows = np.arange(len(names))
    choices = {"fixed_finetuned": np.full(len(names), baseline),
        "barrier_oracle": errors.argmin(axis=1),
        "force_guarded_oracle": np.where(force_mask, errors, np.inf).argmin(axis=1),
        "multi_metric_guarded_oracle": np.where(all_mask, errors, np.inf).argmin(axis=1)}
    summary = {"scope": "post-hoc diagnostic on meta-development only; every oracle uses true evaluation labels and cannot be deployed",
        "force_best_expert_counts": dict(zip(EXPERTS, np.bincount(forces.argmin(axis=1), minlength=len(EXPERTS)).tolist())),
        "alternate_force_improvement_cases": int(np.sum(np.min(np.delete(forces, baseline, axis=1), axis=1) < forces[:, baseline])),
        "guards": {"force": "per-reaction force RMSE no worse than fixed fine-tuned expert",
            "multi_metric": "also TS-region force RMSE and reaction-energy absolute error no worse per reaction"},
        "reactions": len(names), "results": {}}
    for label, selected in choices.items():
        summary["results"][label] = {"barrier_MAE_eV": float(np.mean(errors[rows, selected])),
            "force_global_component_RMSE_eV_per_A": float(np.sqrt(np.average(forces[rows, selected]**2, weights=atoms))),
            "mean_reaction_TS_force_RMSE_eV_per_A": float(np.mean(matrices["ts_region_force_component_rmse_eV_per_A"][rows, selected])),
            "reaction_energy_MAE_eV": float(np.mean(matrices["reaction_energy_error_eV"][rows, selected])),
            "changed_expert_reactions": int(np.sum(selected != baseline))}
    save_json(directory / "complementarity.json", summary)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    shown = [("fixed_finetuned", "fixed fine-tuned / both guarded oracles", "o", "#236d8c"),
             ("barrier_oracle", "barrier-only oracle (unattainable)", "X", "#bc563d")]
    for key, label, marker, color in shown:
        row = summary["results"][key]
        ax.scatter(row["barrier_MAE_eV"], row["force_global_component_RMSE_eV_per_A"], s=95,
            marker=marker, color=color, label=label)
    ax.set(xlabel="Barrier MAE (eV), lower is better", ylabel="Force component RMSE (eV/A), lower is better",
        title="Expert complementarity: barrier gains can sacrifice forces\n128 meta-development reactions; oracle points are not deployable")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(directory / "complementarity.png", dpi=180)
    fig.savefig(directory / "complementarity.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
