"""Compare the original MACE pool with added architectures on the same meta set."""

from datetime import datetime, timezone
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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from expert_routing import EXPERTS
from model_study_data import save_json

NEW = ["aimnet2-wb97m", "aimnet2-rxn", "ani2x", "orbmol-conservative"]


def analyze(table, experts, baseline="legacy_mh1_256"):
    reactions = sorted(table.reaction.unique())
    fixed = table[table.model == baseline].set_index("reaction").loc[reactions]
    atoms = fixed.atoms.to_numpy(float)
    values = {}
    for key in ("barrier_error_eV", "force_component_rmse_eV_per_A",
                "ts_region_force_component_rmse_eV_per_A", "reaction_energy_error_eV"):
        values[key] = table.pivot(index="reaction", columns="model", values=key).loc[reactions, experts].abs().to_numpy()
    errors, forces = values["barrier_error_eV"], values["force_component_rmse_eV_per_A"]
    index, rows = experts.index(baseline), np.arange(len(reactions))

    def metrics(choices):
        return {"barrier_MAE_eV": float(errors[rows, choices].mean()),
            "barrier_absolute_error_p90_eV": float(np.quantile(errors[rows, choices], .9)),
            "force_global_component_RMSE_eV_per_A": float(np.sqrt(np.average(forces[rows, choices]**2, weights=atoms))),
            "mean_reaction_TS_force_RMSE_eV_per_A": float(values["ts_region_force_component_rmse_eV_per_A"][rows, choices].mean()),
            "reaction_energy_MAE_eV": float(values["reaction_energy_error_eV"][rows, choices].mean()),
            "choices": dict(zip(experts, np.bincount(choices, minlength=len(experts)).tolist()))}

    force_guard = forces <= forces[:, index:index+1] + 1e-12
    joint_guard = force_guard.copy()
    for key in ("ts_region_force_component_rmse_eV_per_A", "reaction_energy_error_eV"):
        joint_guard &= values[key] <= values[key][:, index:index+1] + 1e-12
    alternatives = np.arange(len(experts)) != index
    report = {"experts": experts, "fixed": {name: metrics(np.full(len(rows), i)) for i,name in enumerate(experts)},
        "barrier_oracle_unattainable": metrics(errors.argmin(axis=1)),
        "force_guarded_oracle_unattainable": metrics(np.where(force_guard, errors, np.inf).argmin(axis=1)),
        "multi_metric_guarded_oracle_unattainable": metrics(np.where(joint_guard, errors, np.inf).argmin(axis=1)),
        "force_best_counts": dict(zip(experts, np.bincount(forces.argmin(axis=1), minlength=len(experts)).tolist())),
        "alternative_force_improvement_reactions": int(np.sum(forces[:, alternatives].min(axis=1) < forces[:, index])),
        "joint_barrier_force_improvement_cases": {name: [reactions[j] for j in np.flatnonzero(
            (errors[:, i] < errors[:, index]) & (forces[:, i] < forces[:, index]))]
            for i,name in enumerate(experts) if i != index}}
    # An aggregate guard is weaker than a per-reaction guard. Solve its explicit
    # hard-choice upper bound too, rather than mistaking the strict guard for a
    # proof that no aggregate tradeoff can improve both metrics.
    n, k = errors.shape
    constraints = lil_matrix((n+3, n*k))
    for j in range(n):
        constraints[j, j*k:(j+1)*k] = 1
    cost_arrays = [forces**2 * atoms[:, None],
        values["ts_region_force_component_rmse_eV_per_A"],
        values["reaction_energy_error_eV"]]
    upper = [float(cost[:, index].sum()) for cost in cost_arrays]
    for j,cost in enumerate(cost_arrays):
        scale = max(upper[j], 1e-12)
        constraints[n+j] = cost.ravel()/scale
    solved = milp(errors.ravel(), integrality=np.ones(n*k), bounds=Bounds(0, 1),
        constraints=LinearConstraint(constraints.tocsr(), np.r_[np.ones(n), [-np.inf]*3], np.ones(n+3)),
        options={"time_limit": 45, "mip_rel_gap": 1e-6})
    aggregate = {"solver_status": int(solved.status), "message": solved.message,
        "optimality_proved": bool(solved.success), "mip_gap": float(solved.mip_gap) if solved.x is not None else None,
        "guards": "aggregate force MSE, mean per-reaction TS force RMSE and reaction-energy MAE no worse than baseline",
        "warning": "uses evaluation labels and is unattainable; not a deployable router"}
    if solved.x is not None:
        choice = solved.x.reshape(n, k).argmax(axis=1)
        for cost,budget in zip(cost_arrays, upper):
            if cost[rows, choice].sum() > budget + 1e-6*max(budget, 1):
                raise ValueError("MILP incumbent violates an aggregate metric guard")
        aggregate["metrics"] = metrics(choice)
    report["aggregate_guarded_oracle_unattainable"] = aggregate
    return report


def main():
    runtime = ROOT / "results/v0.7.0/expert_diversity"
    base = pd.read_csv(ROOT / "results/v0.7.0/meta_development_mac/reaction_metrics.csv")
    tables = [base]
    for name in NEW:
        path = runtime / name
        if json.loads((path / "state.json").read_text())["stage"] != "completed":
            raise RuntimeError(f"External expert is not complete: {name}")
        tables.append(pd.read_csv(path / "meta_development/reaction_metrics.csv"))
    table = pd.concat(tables, ignore_index=True)
    partition = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    expected = {f"rxn{sid:04d}" for sid in partition["meta_development"]}
    for name in EXPERTS+NEW:
        subset = table[table.model == name]
        if len(subset) != 128 or set(subset.reaction) != expected or subset.reaction.duplicated().any():
            raise ValueError(f"Wrong reaction identity for {name}")
        reference = subset.set_index("reaction").sort_index()
        original = base[base.model == EXPERTS[0]].set_index("reaction").sort_index()
        np.testing.assert_allclose(reference.reference_barrier_eV, original.reference_barrier_eV, atol=1e-10, rtol=0)
        if not reference.formula.equals(original.formula) or not reference.atoms.equals(original.atoms):
            raise ValueError("Formula/atom mapping differs across experts")
    report = {"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory same-meta expansion after original results; all oracles are unattainable",
        "holdout_labels_evaluated": False, "reactions": 128, "architecture_families": ["MACE", "AIMNet2", "ANI", "Orb"],
        "original_four": analyze(base, EXPERTS), "expanded_eight": analyze(table, EXPERTS+NEW),
        "raw_cross_DFT_comparison": True, "foundation_reaction_overlap_verified": False}
    save_json(runtime / "summary.json", report)
    table.to_csv(runtime / "reaction_metrics.csv", index=False)
    destination = ROOT / "docs/reports/v0.7.0-diversity"
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "summary.json", report)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for name, result in report["expanded_eight"]["fixed"].items():
        x, y = result["barrier_MAE_eV"], result["force_global_component_RMSE_eV_per_A"]
        ax.scatter(x, y, s=55, color="#266e8b" if name in EXPERTS else "#c37736")
        ax.annotate(name, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    for pool, marker, color in (("original_four", "x", "#747779"), ("expanded_eight", "*", "#2f8753")):
        result = report[pool]["aggregate_guarded_oracle_unattainable"].get("metrics")
        if result:
            ax.scatter(result["barrier_MAE_eV"], result["force_global_component_RMSE_eV_per_A"],
                s=130, marker=marker, color=color, label=pool+" aggregate-guarded oracle (unattainable)")
    ax.set(xlabel="Barrier MAE (eV), lower is better", ylabel="Force component RMSE (eV/A), lower is better",
        title="Does architecture diversity create complementary experts?\nSame 128 meta-development reactions, fixed DFT geometries")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(destination / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({pool: {key: value for key,value in report[pool].items() if key != "fixed"}
                      for pool in ("original_four", "expanded_eight")}, indent=2))


if __name__ == "__main__":
    main()
