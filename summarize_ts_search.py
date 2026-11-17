"""Report endpoint-only search outcomes and post-search DFT-path comparisons."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from transition1x_data import read_reaction
from ts_search import aligned_rmsd, save_json


def arc_length(images):
    distances = [aligned_rmsd(left, right) for left, right in zip(images[:-1], images[1:])]
    coordinate = np.r_[0.0, np.cumsum(distances)]
    return coordinate / coordinate[-1] if coordinate[-1] else np.linspace(0, 1, len(images))


def overview_plot(root, rows, dft_rows, summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    stages = ["NEB", "TS force", "One imag.\n>50 cm$^{-1}$", "IRC + geometry"]
    keys = ["neb_converged", "saddle_converged", "single_significant_imaginary", "irc_connected"]
    x = np.arange(len(stages))
    for index, (model, counts) in enumerate(summary["models"].items()):
        values = [counts[key] for key in keys]
        bars = axes[0].bar(x + (index - 0.5) * 0.35, values, 0.35, label=model)
        axes[0].bar_label(bars, fontsize=8)
    axes[0].set(xticks=x, xticklabels=stages, ylim=(0, 11), ylabel="Reactions / 10", title="Endpoint-only searches")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(fontsize=8, loc="lower left")
    validated = {item["reaction"]: item["electronic_barrier_eV"] for item in dft_rows
                 if item["dft_verified"] and item["electronic_barrier_eV"] is not None}
    values = list(validated.values())
    for model in summary["models"]:
        subset = [row for row in rows if row["model"] == model and row["mlip_verified"]
                  and row["reaction"] in validated]
        reference = [validated[row["reaction"]] for row in subset]
        prediction = [row["mlip_barrier_eV"] for row in subset]
        values.extend(prediction)
        axes[1].scatter(reference, prediction, label=model, s=35)
        if subset:
            summary.setdefault("selected_dft_barrier_comparison", {})[model] = {
                "reactions": len(subset), "mae_eV": float(np.mean(np.abs(np.array(prediction) - reference))),
                "role": "conditional on selected DFT-validated candidates and valid model searches"}
    if values:
        low, high = min(values) - 0.15, max(values) + 0.15
        axes[1].plot([low, high], [low, high], "--", color="0.6", linewidth=1)
        axes[1].set(xlim=(low, high), ylim=(low, high))
    axes[1].set(xlabel="Gaussian electronic barrier / eV", ylabel="Model-surface electronic barrier / eV",
                title=f"Selected DFT-validated reactions (n={len(validated)})")
    axes[1].legend(fontsize=8)
    geometries = [item for item in dft_rows if item["mlip_to_dft_ts_rmsd_A"] is not None]
    axes[2].barh([item["reaction"] for item in geometries],
                 [item["mlip_to_dft_ts_rmsd_A"] for item in geometries],
                 color=["#ff7f0e" if item["dft_verified"] else "#888888" for item in geometries])
    axes[2].set(xlabel="Mapped TS geometry RMSD / angstrom", title="Fine-tuned MLIP to Gaussian TS")
    fig.savefig(root / "ts_search_overview.png", dpi=180)
    fig.savefig(root / "ts_search_overview.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.4.0"))
    args = parser.parse_args()
    root = args.output
    selection = json.loads((root / "selection.json").read_text())
    config = selection["config"]
    rows, outcomes, dft_rows, reference_frames = [], {}, [], []
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    with h5py.File(config["input"], "r") as source:
        for axis, item in zip(axes.flat, selection["reactions"]):
            reaction = item["reaction"]
            reference = read_reaction(source[config["split"]][item["formula"]][reaction],
                                      config["split"], item["formula"], reaction)
            reference_atoms = reference.atoms()
            reference_frames.extend(reference_atoms)
            reference_barrier = float(np.max(reference.energies) - reference.energies[0])
            axis.plot(arc_length(reference_atoms), reference.energies - reference.energies[0],
                      ".--", color="#555555", label="DFT reference band")
            for model in config["models"]:
                folder = root / model / reaction
                path = folder / "result.json"
                if not path.exists():
                    raise RuntimeError(f"Missing planned outcome {path}")
                result = json.loads(path.read_text())
                outcomes.setdefault(model, []).append(result)
                saddle = result.get("saddle", {})
                vibrations = result.get("vibrations", {})
                neb = result.get("neb", {})
                candidate_path = folder / "ts_candidate.extxyz"
                rmsd = None
                if candidate_path.exists():
                    rmsd = aligned_rmsd(reference_atoms[reference.reference_ts_image], read(candidate_path))
                barrier = saddle.get("barrier_eV")
                rows.append({"model": model, "reaction": reaction, "formula": item["formula"],
                             "kind": item["kind"], "atoms": item["atoms"], "status": result["status"],
                             "neb_backend": result.get("neb_backend", "serial MACECalculator"),
                             "endpoints_converged": all(entry["converged"] for entry in result.get("endpoints", [])),
                             "endpoint_bonds_preserved": all(entry["input_bonds_preserved"] for entry in result.get("endpoints", [])),
                             "neb_converged": neb.get("converged", False), "neb_fmax_eV_A": neb.get("fmax_eV_A"),
                             "saddle_converged": saddle.get("converged", False),
                             "saddle_fmax_eV_A": saddle.get("fmax_eV_A"),
                             "imaginary_modes_above_50_cm1": vibrations.get("imaginary_modes_above_threshold"),
                             "all_negative_modes": vibrations.get("all_negative_modes"),
                             "irc_connected": result.get("irc", {}).get("converged_and_connected", False),
                             "irc_bond_connected": result.get("irc", {}).get("expected_bond_connectivity", False),
                             "saddle_irc_verified": result.get("saddle_irc_verified", False),
                             "mlip_verified": result["mlip_verified"],
                             "strict_single_negative_and_verified": result["mlip_verified"] and vibrations.get("all_negative_modes") == 1,
                             "mlip_barrier_eV": barrier, "reference_sampled_barrier_eV": reference_barrier,
                             "barrier_difference_eV": None if barrier is None else barrier - reference_barrier,
                             "reference_ts_geometry_rmsd_A": rmsd, "elapsed_seconds": result["elapsed_seconds"],
                             "error": result.get("error", "")})
                band_path = folder / "final_band.extxyz"
                if band_path.exists():
                    band = read(band_path, index=":")
                    energy = np.array([atoms.get_potential_energy() for atoms in band])
                    label = model + (" [pass]" if result["mlip_verified"] else " [unverified]")
                    axis.plot(arc_length(band), energy - energy[0], ".-", label=label)
            axis.set_title(f"{reaction} · {item['formula']} · {item['kind']}")
            axis.set_xlabel("Normalized arc length of each path")
            axis.set_ylabel("Relative electronic energy / eV")
            axis.legend(fontsize=7)
            dft_path = root / "dft" / reaction / "result.json"
            if dft_path.exists():
                dft = json.loads(dft_path.read_text())
                dft_rows.append({"reaction": reaction, "status": dft["status"], "stage": dft["stage"],
                                 "imaginary_modes": dft.get("ts", {}).get("imaginary_modes"),
                                 "dft_verified": dft["dft_verified"],
                                 "expected_connectivity": dft.get("expected_connectivity", False),
                                 "stable_minima": dft.get("stable_minima", False),
                                 "irc_minima_detected": dft.get("irc_minima_detected", False),
                                 "irc_point_budget_reached": any(branch["irc"].get("maximum_steps_reached", False)
                                                                  for branch in dft.get("branches", [])),
                                 "recovered_irc_postprocessing_error": dft.get("recovered_irc_postprocessing_error", False),
                                 "mlip_to_dft_ts_rmsd_A": dft.get("mlip_to_dft_ts_rmsd_A"),
                                 "electronic_barrier_eV": dft.get("electronic_barrier_eV"),
                                 "gaussian_reported_elapsed_seconds": dft.get("gaussian_reported_elapsed_seconds"),
                                 "irc_algorithms": ",".join(sorted({branch["irc"]["algorithm"] for branch in dft.get("branches", [])})),
                                 "error": dft.get("error", "")})
    write(root / "reference_paths.extxyz", reference_frames)
    fig.savefig(root / "optimized_paths.png", dpi=180)
    fig.savefig(root / "optimized_paths.pdf")
    plt.close(fig)
    for name, entries in (("search_metrics.csv", rows), ("dft_metrics.csv", dft_rows)):
        if entries:
            with (root / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                writer.writeheader()
                writer.writerows(entries)
    summary = {"role": "development pilot on previously evaluated reactions", "planned_reactions": config["count"],
               "models": {}, "dft_planned": config["dft_count"], "dft_recorded": len(dft_rows),
               "dft_completed": sum(item["status"] == "completed" for item in dft_rows),
               "dft_verified": sum(item["dft_verified"] for item in dft_rows),
               "scope": "endpoint-only CI-NEB and model-surface verification, selected Gaussian refinement"}
    lines = ["# v0.4.0 端点驱动过渡态搜索实测", "",
             "十条反应来自已评价的 v0.3.0 开发池，覆盖八个化学式和三种几何成断键类别。",
             "搜索以两个端点为输入，端点在各自模型势能面优化，再通过 IDPP 初始化 CI-NEB。",
             "DFT 路径仅在搜索后用于评价。所有预定搜索均计入分母。", "",
             "| 模型 | NEB 收敛 | 鞍点优化收敛 | 单显著虚频 | IRC 收敛并连接 | 鞍点验证通过 | 全阶段通过 | 严格单负频全阶段通过 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model, results in outcomes.items():
        selected_rows = [row for row in rows if row["model"] == model]
        counts = {"planned": len(results),
                  "neb_converged": sum(result.get("neb", {}).get("converged", False) for result in results),
                  "saddle_converged": sum(result.get("saddle", {}).get("converged", False) for result in results),
                  "single_significant_imaginary": sum(result.get("vibrations", {}).get("imaginary_modes_above_threshold") == 1 for result in results),
                  "irc_connected": sum(result.get("irc", {}).get("converged_and_connected", False) for result in results),
                  "irc_bond_connected": sum(result.get("irc", {}).get("expected_bond_connectivity", False) for result in results),
                  "saddle_irc_verified": sum(result.get("saddle_irc_verified", False) for result in results),
                  "mlip_verified": sum(result["mlip_verified"] for result in results),
                  "strict_single_negative_verified": sum(row["strict_single_negative_and_verified"] for row in selected_rows)}
        summary["models"][model] = counts
        lines.append(f"| {model} | {counts['neb_converged']}/10 | {counts['saddle_converged']}/10 | "
                     f"{counts['single_significant_imaginary']}/10 | {counts['irc_connected']}/10 | "
                     f"{counts['saddle_irc_verified']}/10 | {counts['mlip_verified']}/10 | {counts['strict_single_negative_verified']}/10 |")
    lines += ["", "全阶段通过要求端点收敛且保持输入连接、NEB 收敛、精修最大原子受力不超过",
              "0.01 eV/Å、恰有一个超过 50 cm⁻¹ 的虚频，且双向 IRC 和末端优化收敛并匹配预期端点。",
              "端点几何逐连通片段配准，完整体系 RMSD 另存。严格单负频列进一步要求投影振动空间中",
              "所有负频率的数量恰为一。鞍点验证列允许 Sella 从达到 NEB 步数预算的候选继续精修后通过，",
              "其余驻点、频率、端点及 IRC 条件相同。", "",
              "## 逐反应全阶段结果", "",
              "| 反应 | 原始 MH-1 | 微调 MH-1 |", "| --- | --- | --- |"]
    for item in selection["reactions"]:
        statuses = []
        for model in config["models"]:
            row = next(row for row in rows if row["model"] == model and row["reaction"] == item["reaction"])
            if row["mlip_verified"]:
                reason = "通过"
            elif row["status"] == "failed":
                reason = row["error"]
            elif not row["endpoint_bonds_preserved"]:
                reason = "端点优化改变键连接"
            elif not row["neb_converged"]:
                reason = "NEB 达到步数预算"
            elif not row["saddle_converged"]:
                reason = "鞍点精修达到步数预算"
            elif not row["irc_bond_connected"]:
                reason = "IRC 指向其他端点"
            elif not row["irc_connected"]:
                reason = "IRC 收敛或端点几何检查失败"
            else:
                reason = row["error"] or "驻点或频率检查失败"
            statuses.append(reason)
        lines.append(f"| {item['reaction']} | {statuses[0]} | {statuses[1]} |")
    lines += ["",
              "## Gaussian DFT 验证", "",
              f"已完成 {summary['dft_completed']}/{config['dft_count']} 条 DFT 验证，"
              f"综合通过 {summary['dft_verified']}/{config['dft_count']}。",
              "使用 ωB97X/6-31G(d)，中性单重态，TS 优化、频率、双向 IRC 及末端最小值优化和频率。", "",
              "rxn4439 使用完成的 HPC IRC，其余四条在 HPC 校正器反复迭代后改用 LQA、步长 5。",
              "保留中止的 HPC 日志、检查点及调整记录，当前属于自适应开发方案。", "",
              "| 反应 | 状态 | 阶段 | TS 负频率数 | DFT 综合通过 | MLIP→DFT TS RMSD / Å | 电子能垒 / eV |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in dft_rows:
        rmsd = row["mlip_to_dft_ts_rmsd_A"]
        barrier = row["electronic_barrier_eV"]
        lines.append(f"| {row['reaction']} | {row['status']} | {row['stage']} | {row['imaginary_modes']} | "
                     f"{row['dft_verified']} | {'' if rmsd is None else f'{rmsd:.4f}'} | "
                     f"{'' if barrier is None else f'{barrier:.4f}'} |")
    for row in dft_rows:
        if row["recovered_irc_postprocessing_error"]:
            lines += ["", f"{row['reaction']} 的反向 IRC 在第 160 点预算后从 checkpoint 续算，"
                      "第 161 点检测到 PES 最小值并完成路径追踪，随后 Gaussian 汇总阶段出现 "
                      "`PESDtA: Invalid IPoint`。该作业保留 normal_termination=false。"
                      "保存的末端另行执行了正常结束的 DFT 最小值优化与频率计算，"
                      f"稳定最小值检查为 {row['stable_minima']}，端点匹配为 {row['expected_connectivity']}，"
                      f"综合验证为 {row['dft_verified']}。原始错误日志与检查点保留在结果包。"]
        if row["status"] == "completed" and not row["dft_verified"]:
            lines += ["", f"{row['reaction']} 的端点匹配为 {row['expected_connectivity']}，两端稳定最小值检查为 "
                      f"{row['stable_minima']}，IRC 双向检测到最小值为 {row['irc_minima_detected']}，"
                      f"IRC 达到路径点预算为 {row['irc_point_budget_reached']}。"]
    overview_plot(root, rows, dft_rows, summary)
    if summary.get("selected_dft_barrier_comparison"):
        lines += ["", "## 所选 DFT 反应的能垒对照", "",
                  "该比较限定在已选择并完成 DFT 验证、相应模型搜索也通过的反应。", "",
                  "| 模型 | 反应数 | 电子能垒 MAE / eV |", "| --- | --- | --- |"]
        for model, comparison in summary["selected_dft_barrier_comparison"].items():
            lines.append(f"| {model} | {comparison['reactions']} | {comparison['mae_eV']:.6f} |")
    lines += ["", "## 解释范围", "",
              "优化路径图各曲线使用自身归一化弧长和各自反应物能量，横坐标相同位置对应各自路径的进度。",
              "参考 DFT 离散峰值保留为数据集路径诊断，原始峰值尚未在本轮逐条验证为 DFT 一阶鞍点。",
              "Gaussian 电子能垒以 IRC 所连的优化反应物为基准，热力学及零点能校正另行处理。",
              "本轮十反应、单次初始化属于开发试验。预训练数据重叠、外部分布和多种子搜索仍待检验。", ""]
    save_json(root / "summary.json", summary)
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
