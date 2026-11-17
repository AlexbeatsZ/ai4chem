"""Create the v0.3.0 validation/final tables and experiment figures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/v0.3.0"


def main():
    selection = json.loads((OUTPUT / "selection.json").read_text())
    final = json.loads((OUTPUT / "final/analysis/analysis_summary.json").read_text())
    validation = json.loads((OUTPUT / "validation/analysis/analysis_summary.json").read_text())
    rows = []
    for role in ("validation", "development", "final"):
        summary = json.loads((OUTPUT / role / "analysis/analysis_summary.json").read_text())
        for name, information in summary["models"].items():
            metrics = information["aggregate"]
            interval = information["bootstrap"]["reaction"]["barrier_abs_error_eV"]
            formula_interval = information["bootstrap"]["formula"]["barrier_abs_error_eV"]
            rows.append({"role": role, "model": name, "reactions": metrics["reactions"],
                         "barrier_mae_eV": metrics["barrier_abs_error_eV"],
                         "barrier_reaction_ci_lower": interval["lower_95"],
                         "barrier_reaction_ci_upper": interval["upper_95"],
                         "barrier_formula_ci_lower": formula_interval["lower_95"],
                         "barrier_formula_ci_upper": formula_interval["upper_95"],
                         "barrier_signed_mean_eV": metrics["barrier_signed_mean_eV"],
                         "nonreactant_energy_mae_eV": metrics["nonreactant_relative_energy_mae_eV"],
                         "ts_energy_mae_eV": metrics["ts_relative_energy_abs_error_eV"],
                         "force_component_rmse_eV_per_A": metrics["force_global_component_rmse_eV_per_A"],
                         "ts_region_force_rmse_eV_per_A": metrics["ts_region_global_force_rmse_eV_per_A"],
                         "underestimated_barriers": metrics["underestimated_barriers"],
                         "shifted_peaks": metrics["shifted_peaks"]})
    pd.DataFrame(rows).to_csv(OUTPUT / "experiment_comparison.csv", index=False)
    model_names = list(final["models"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for directory in sorted((OUTPUT / "training").iterdir()):
        if not directory.is_dir():
            continue
        metrics_files = list((directory / "metrics").glob("*_train.txt"))
        if not metrics_files:
            continue
        events = [json.loads(line) for line in metrics_files[0].read_text().splitlines()]
        evaluations = [event for event in events if event.get("mode") == "eval"]
        epochs = [-1 if event.get("epoch") is None else event["epoch"] for event in evaluations]
        forces = [event["rmse_f"] for event in evaluations]
        axes[0, 0].plot(epochs, forces, ".-", label=directory.name)
    axes[0, 0].set_xlabel("Epoch (-1 = initial validation)")
    axes[0, 0].set_ylabel("Validation force component RMSE (eV/A)")
    axes[0, 0].legend(fontsize=7)
    candidates = list(validation["models"])
    axes[0, 1].bar(candidates, [validation["models"][name]["aggregate"]["barrier_abs_error_eV"] for name in candidates], color="#356b78")
    axes[0, 1].set_ylabel("Validation sampled-barrier MAE (eV)")
    axes[0, 1].tick_params(axis="x", labelrotation=55, labelsize=8)
    for axis, key, label in (
        (axes[1, 0], "barrier_abs_error_eV", "Final sampled-barrier MAE (eV)"),
        (axes[1, 1], "global_force_component_rmse_eV_per_A", "Final force component RMSE (eV/A)"),
    ):
        intervals = [final["models"][name]["bootstrap"]["reaction"][key] for name in model_names]
        point = np.asarray([interval["mean"] for interval in intervals])
        errors = np.asarray([[interval["mean"] - interval["lower_95"] for interval in intervals],
                             [interval["upper_95"] - interval["mean"] for interval in intervals]])
        axis.bar(model_names, point, yerr=np.maximum(errors, 0), capsize=4, color="#c07732")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelrotation=25, labelsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Transition1x small-data fine-tuning | fixed validation selection\n"
                 "Final error bars: 95% reaction-cluster bootstrap intervals", fontsize=14)
    fig.savefig(OUTPUT / "finetuning_generalization.png", dpi=180)
    fig.savefig(OUTPUT / "finetuning_generalization.pdf")
    plt.close(fig)
    report = ["# v0.3.0 微调与泛化验证", "", f"验证集选定模型为 `{selection['selected']}`。", "",
              "开发集 77 条、validation 20 条、最终集 210 条。原十条包含在开发集中。", "",
              "| 最终模型 | 能垒 MAE / eV | 非反应物能量 MAE / eV | 力分量 RMSE / (eV/Å) |", "| --- | --- | --- | --- |"]
    for row in rows:
        if row["role"] == "final":
            report.append(f"| {row['model']} | {row['barrier_mae_eV']:.6f} | {row['nonreactant_energy_mae_eV']:.6f} | {row['force_component_rmse_eV_per_A']:.6f} |")
    report += ["", "完整区间、分层指标和诊断位于各 role 的 analysis 子目录，所有角色对比位于 experiment_comparison.csv。", "",
               "源数据采用固定 revision 的公开 JMP 导出副本，尚待与官方完整 HDF5 逐数值对照。", "",
               "本轮采用一个 train shard、单随机种子和八个 epochs。预训练重叠、进一步收敛、多种子结果和模型独立优化 NEB 待后续验证。", ""]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("PASS - research comparison table, figures and report saved", flush=True)


if __name__ == "__main__":
    main()
