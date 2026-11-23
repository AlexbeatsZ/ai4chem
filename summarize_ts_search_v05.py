"""Report all planned strategy units, paired outcomes, actual costs and DFT checks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from benchmark_transition1x import write_csv
from diagnose_ts_search import force_history
from prepare_v05_reactions import ROOT
from ts_search import save_json


def collect(output, manifest, models):
    rows, records, unique = [], [], {}
    for item in manifest["reactions"]:
        for model in models:
            for strategy in manifest["protocol"]["strategies"]:
                path = output / item["role"] / model / item["reaction"] / strategy / "result.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                row = {"role": item["role"], "reaction": item["reaction"], "formula": item["formula"],
                       "kind": item["kind"], "atoms": item["atoms"], "model": model, "strategy": strategy,
                       "status": result.get("status", "pending"),
                       "saddle_irc_verified": result.get("saddle_irc_verified"),
                       "mlip_verified": result.get("mlip_verified"), "neb_converged": result.get("neb_converged"),
                       "attempts": len(result["attempts"]) if result else None,
                       "theoretical_seconds": result.get("theoretical_seconds"),
                       "endpoint_seconds": result.get("endpoint_seconds"),
                       "image_step_cost": result.get("image_step_cost"),
                       "primary_failure": result.get("primary_failure", "pending"),
                       "selected_attempt": result.get("selected_attempt")}
                rows.append(row)
                if result:
                    records.append(result)
                    for attempt in result["attempts"]:
                        unique[attempt["cache_key"]] = {**attempt, "role": item["role"],
                                                        "model": model, "reaction": item["reaction"]}
    return rows, records, unique


def group_summary(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    times = [row["theoretical_seconds"] for row in completed]
    verified = sum(row["saddle_irc_verified"] is True for row in completed)
    return {"planned": len(rows), "completed": len(completed), "pending": len(rows) - len(completed),
            "verified_saddles": verified,
            "strict_pipeline": sum(row["mlip_verified"] is True for row in completed),
            "neb_converged": sum(row["neb_converged"] is True for row in completed),
            "average_attempts": float(np.mean([row["attempts"] for row in completed])) if completed else None,
            "total_theoretical_seconds": float(sum(times)),
            "time_median_q1_q3_seconds": np.percentile(times, [50, 25, 75]).tolist() if times else None,
            "seconds_per_verified_saddle": float(sum(times) / verified) if verified else None,
            "total_image_step_cost": sum(row["image_step_cost"] for row in completed),
            "failures": dict(Counter(row["primary_failure"] for row in completed))}


def paired_comparisons(rows, models):
    output = []
    for role in ("development", "holdout"):
        for model in models:
            selected = [row for row in rows if row["role"] == role and row["model"] == model]
            for left, right in (("S1", "S0"), ("S2", "S0"), ("S2", "S1")):
                lookup = {(row["reaction"], row["strategy"]): row for row in selected}
                pairs = [(lookup[(reaction, left)], lookup[(reaction, right)])
                         for reaction in sorted({row["reaction"] for row in selected})]
                pairs = [(a, b) for a, b in pairs if a["status"] == b["status"] == "completed"]
                for metric in ("saddle_irc_verified", "mlip_verified"):
                    gained = sum(a[metric] is True and b[metric] is False for a, b in pairs)
                    lost = sum(a[metric] is False and b[metric] is True for a, b in pairs)
                    discordant = gained + lost
                    differences = [a["theoretical_seconds"] - b["theoretical_seconds"] for a, b in pairs]
                    output.append({"role": role, "model": model, "comparison": f"{left}-{right}",
                                   "metric": metric, "paired_reactions": len(pairs), "gained": gained, "lost": lost,
                                   "exact_two_sided_p": float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0,
                                   "median_cost_difference_seconds": float(np.median(differences)) if differences else None})
    return output


def dft_metrics(output, manifest):
    path = output / "dft/selection.json"
    selection = json.loads(path.read_text()) if path.exists() else {"strategy_units": []}
    lookup = {(item["reaction"], item["strategy"]): item for item in selection["strategy_units"]}
    rows = []
    for reaction in manifest["dft_reactions"]:
        for strategy in manifest["protocol"]["strategies"]:
            item = lookup.get((reaction, strategy), {})
            folder = output / "dft" / reaction / item.get("dft_job", "")[:16]
            result_path = folder / "result.json"
            result = json.loads(result_path.read_text()) if item.get("dft_job") and result_path.exists() else {}
            strategy_path = output / "holdout/mh1-finetuned" / reaction / strategy / "result.json"
            strategy_result = json.loads(strategy_path.read_text()) if strategy_path.exists() else {}
            attempt_path = Path(strategy_result.get("selected_folder", str(output / "missing"))) / "result.json"
            attempt = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
            predicted = attempt.get("saddle", {}).get("barrier_eV")
            reference = result.get("electronic_barrier_eV")
            error = predicted - reference if predicted is not None and reference is not None else None
            rows.append({"reaction": reaction, "strategy": strategy,
                         "status": result.get("status", item.get("status", "pending")),
                         "dft_job": item.get("dft_job"), "dft_verified": result.get("dft_verified", False),
                         "ts_imaginary_modes": result.get("ts", {}).get("imaginary_modes"),
                         "stable_minima": result.get("stable_minima"),
                         "irc_connected": result.get("expected_connectivity"),
                         "irc_minima_detected": result.get("irc_minima_detected"),
                         "recovered_irc_postprocessing_error": result.get("recovered_irc_postprocessing_error", False),
                         "mlip_to_dft_ts_rmsd_A": result.get("mlip_to_dft_ts_rmsd_A"),
                         "mlip_barrier_eV": predicted, "dft_electronic_barrier_eV": reference,
                         "barrier_error_eV": error, "error": result.get("error")})
    return rows


def make_plots(output, rows, summary, models):
    colors = {"S0": "#587185", "S1": "#dc9b45", "S2": "#318c83"}
    strategies = list(colors)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for axis, model in zip(axes[0], models):
        for index, strategy in enumerate(strategies):
            counts = summary["groups"][f"holdout/{model}/{strategy}"]
            values = [counts[key] for key in ("neb_converged", "verified_saddles", "strict_pipeline")]
            bars = axis.bar(np.arange(3) + (index - 1) * 0.24, values, 0.24,
                            color=colors[strategy], label=strategy)
            axis.bar_label(bars, fontsize=8)
        axis.set(xticks=np.arange(3), xticklabels=["NEB", "Saddle + IRC", "Full pipeline"],
                 ylim=(0, 13), ylabel="Verified reactions / 12", title=model)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    x = np.arange(6)
    keys = [(model, strategy) for model in models for strategy in strategies]
    medians, lower, upper = [], [], []
    for model, strategy in keys:
        stats = summary["groups"][f"holdout/{model}/{strategy}"]["time_median_q1_q3_seconds"]
        median, q1, q3 = stats or (0, 0, 0)
        medians.append(median)
        lower.append(median - q1)
        upper.append(q3 - median)
    axes[1, 0].bar(x, medians, yerr=[lower, upper], color=[colors[s] for _, s in keys], capsize=3)
    axes[1, 0].set(xticks=x, xticklabels=[f"{m.replace('mh1-', '')}\n{s}" for m, s in keys],
                   ylabel="Seconds per reaction", title="Counterfactual strategy cost, median and IQR")
    costs = [summary["groups"][f"holdout/{m}/{s}"]["total_image_step_cost"] for m, s in keys]
    axes[1, 1].bar(x, costs, color=[colors[s] for _, s in keys])
    axes[1, 1].set(xticks=x, xticklabels=[f"{m.replace('mh1-', '')}\n{s}" for m, s in keys],
                   ylabel="Image × optimizer steps", title="Cumulative NEB work")
    fig.suptitle("Failure-aware TS search | 12 project-held-out reactions | two frozen models")
    fig.savefig(output / "strategy_comparison.png", dpi=180)
    fig.savefig(output / "strategy_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), layout="constrained")
    reactions = [item["reaction"] for item in rows if item["role"] == "holdout"
                 and item["model"] == models[0] and item["strategy"] == "S0"]
    lookup = {(row["model"], row["reaction"], row["strategy"]): row for row in rows}
    palette = matplotlib.colors.ListedColormap(["#c9ced3", "#bf6b61", "#dfb466", "#389784"])
    for axis, model in zip(axes, models):
        cells = np.zeros((3, len(reactions)))
        for i, strategy in enumerate(strategies):
            for j, reaction in enumerate(reactions):
                row = lookup[(model, reaction, strategy)]
                if row["status"] == "completed":
                    cells[i, j] = 3 if row["mlip_verified"] else 2 if row["saddle_irc_verified"] else 1
                    axis.text(j, i, str(row["attempts"]), ha="center", va="center", color="white", fontsize=9)
        axis.imshow(cells, cmap=palette, vmin=0, vmax=3, aspect="auto")
        axis.set(yticks=np.arange(3), yticklabels=strategies, xticks=np.arange(len(reactions)),
                 xticklabels=reactions, title=model)
        axis.tick_params(axis="x", labelrotation=35)
    fig.suptitle("Per-reaction outcomes | green full pipeline, amber saddle + IRC, red failed\nCell values give attempts; gray indicates pending")
    fig.savefig(output / "reaction_outcomes.png", dpi=180)
    fig.savefig(output / "reaction_outcomes.pdf")
    plt.close(fig)


def case_plot(output, records):
    candidates = sorted((record for record in records if record["strategy"] == "S2" and len(record["attempts"]) > 1),
                        key=lambda record: (-len(record["attempts"]), record["reaction"], record["model"]))[:3]
    if not candidates:
        return
    fig, axes = plt.subplots(len(candidates), 1, figsize=(10, 3.1 * len(candidates)), layout="constrained", squeeze=False)
    for axis, record in zip(axes[:, 0], candidates):
        for number, attempt in enumerate(record["attempts"]):
            history = force_history(Path(attempt["folder"]) / "neb_climb.log")
            if history:
                axis.semilogy(history, label=f"{number + 1} {attempt['plan']['type']}")
        axis.axhline(0.05, color="0.5", linestyle="--", linewidth=1)
        axis.set(xlabel="CI-NEB step within attempt", ylabel="Effective fmax / (eV/A)",
                 title=f"{record['model']} {record['reaction']} | {record['primary_failure']}")
        axis.legend(fontsize=8)
    fig.suptitle("Post-hoc diagnostic examples, ordered by attempt count then reaction ID")
    fig.savefig(output / "failure_cases.png", dpi=180)
    fig.savefig(output / "failure_cases.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
    baseline = json.loads((ROOT / "configs/v0.4.0.json").read_text())
    models = list(baseline["models"])
    rows, records, unique = collect(args.output, manifest, models)
    if not args.allow_incomplete and any(row["status"] == "pending" for row in rows):
        raise RuntimeError("Complete all 96 planned strategy units before the final report")
    summary = {"planned_strategy_units": len(rows), "completed_strategy_units": len(records), "groups": {}}
    for role in ("development", "holdout"):
        for model in models:
            for strategy in manifest["protocol"]["strategies"]:
                group = [row for row in rows if (row["role"], row["model"], row["strategy"]) == (role, model, strategy)]
                summary["groups"][f"{role}/{model}/{strategy}"] = group_summary(group)
    endpoint_seconds = sum(json.loads(path.read_text())["elapsed_seconds"]
                           for path in (args.output / "endpoints").glob("mh1-*/rxn*/endpoints.json"))
    actual = endpoint_seconds + sum(attempt["elapsed_seconds"] for attempt in unique.values())
    theoretical = sum(record["theoretical_seconds"] for record in records)
    summary["cost_accounting"] = {"referenced_unique_attempts": len(unique),
                                   "unique_endpoint_seconds": endpoint_seconds,
                                   "actual_referenced_compute_seconds": actual,
                                   "all_strategies_theoretical_seconds": theoretical,
                                   "reuse_saved_compute_seconds": theoretical - actual,
                                   "definition": "sum of measured jobs, excluding driver/model-load overhead"}
    summary["paired_comparisons"] = paired_comparisons(rows, models)
    dft_rows = dft_metrics(args.output, manifest)
    summary["dft"] = {}
    for strategy in manifest["protocol"]["strategies"]:
        group = [row for row in dft_rows if row["strategy"] == strategy]
        errors = [abs(row["barrier_error_eV"]) for row in group if row["dft_verified"] and row["barrier_error_eV"] is not None]
        summary["dft"][strategy] = {"preselected_reactions": len(group),
                                   "candidates": sum(bool(row["dft_job"]) for row in group),
                                   "verified": sum(row["dft_verified"] for row in group),
                                   "conditional_barrier_mae_eV": float(np.mean(errors)) if errors else None,
                                   "statuses": dict(Counter(row["status"] for row in group))}
    attempt_rows = []
    for attempt in unique.values():
        result = json.loads((Path(attempt["folder"]) / "result.json").read_text())
        attempt_rows.append({"cache_key": attempt["cache_key"], "role": attempt["role"], "model": attempt["model"],
                             "reaction": attempt["reaction"], "initialization": attempt["plan"]["type"],
                             "images": attempt["plan"]["images"], "elapsed_seconds": attempt["elapsed_seconds"],
                             "image_step_cost": attempt["image_step_cost"],
                             "saddle_irc_verified": attempt["saddle_irc_verified"], "mlip_verified": attempt["mlip_verified"],
                             "primary_failure": attempt["diagnosis"]["primary_failure"],
                             "peak_gpu_allocated_MiB": result.get("peak_gpu_allocated_MiB"),
                             **{f"{stage}_seconds": result.get("stage_seconds", {}).get(stage, 0)
                                for stage in ("initialization", "neb", "saddle", "hessian", "irc")}})
    write_csv(args.output / "strategy_metrics.csv", rows)
    if attempt_rows:
        write_csv(args.output / "attempt_metrics.csv", attempt_rows)
        write_csv(args.output / "diagnostics.csv", [
            {key: attempt[key] for key in ("cache_key", "role", "model", "reaction", "initialization", "primary_failure")}
            for attempt in attempt_rows])
    write_csv(args.output / "dft_metrics.csv", dft_rows)
    save_json(args.output / "policy_trace.json", records)
    save_json(args.output / "summary.json", summary)
    make_plots(args.output, rows, summary, models)
    case_plot(args.output, records)
    report = ["# v0.5.0 失败感知自适应过渡态搜索", "",
              "四条开发反应与十二条项目保留反应，比较两个固定模型和三种搜索策略。",
              "源数据为固定 revision 的 train 构型分片 0 和 1，排除全部既有项目反应。",
              "端点和保留反应清单在搜索前固定，控制器在开发阶段后冻结。", "",
              f"计划 {len(rows)} 个策略评价单元，已完成 {len(records)} 个。", "",
              "## 保留反应结果", "",
              "| 模型 | 策略 | 完成 / 计划 | NEB | 鞍点与 IRC | 全阶段 | 平均尝试 | 成本中位数 / s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model in models:
        for strategy in manifest["protocol"]["strategies"]:
            group = summary["groups"][f"holdout/{model}/{strategy}"]
            attempts = f"{group['average_attempts']:.2f}" if group["average_attempts"] is not None else "pending"
            median = f"{group['time_median_q1_q3_seconds'][0]:.1f}" if group["time_median_q1_q3_seconds"] else "pending"
            report.append(f"| {model} | {strategy} | {group['completed']}/{group['planned']} | "
                          f"{group['neb_converged']}/12 | {group['verified_saddles']}/12 | {group['strict_pipeline']}/12 | {attempts} | {median} |")
    report += ["", "鞍点与 IRC 通过允许 NEB 达到步数预算，要求端点、驻点受力、单显著虚频及双向连接通过。",
               "全阶段通过额外要求 NEB 收敛。全部预定反应保留在分母中。", "",
               "## 成本与配对比较", "",
               f"策略引用 {len(unique)} 次唯一搜索，含共享端点的实际计算累计 {actual:.1f} s，"
               f"各策略独立执行的理论累计 {theoretical:.1f} s，复用节省 {theoretical - actual:.1f} s。", "",
               "成本采用同一 OMEN 的实测作业时间，包含初始化与 CPU 数值步骤，排除驱动和模型加载开销。",
               "图像步是工作量代理。各阶段耗时、成本四分位距和每个验证鞍点的平均成本见 CSV 与 summary.json。", "",
               "| 模型 | 比较 | 指标 | 配对反应 | 新增通过 | 丢失通过 | 双侧精确 p |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in summary["paired_comparisons"]:
        if item["role"] == "holdout":
            report.append(f"| {item['model']} | {item['comparison']} | {item['metric']} | {item['paired_reactions']} | "
                          f"{item['gained']} | {item['lost']} | {item['exact_two_sided_p']:.4f} |")
    report += ["", "精确检验采用配对不一致对的二项分布，属于多指标小样本探索性分析。", "",
               "## Gaussian 验证", "",
               "DFT 反应在搜索前按三种端点变化类别固定为 " + ", ".join(manifest["dft_reactions"]) + "。",
               "各策略的相同坐标候选复用同一 Gaussian 计算，保留全部九个预选评价单元。", "",
               "| 策略 | 候选 / 预选 | DFT 通过 / 预选 | 条件电子能垒 MAE / eV | 状态 |",
               "| --- | --- | --- | --- | --- |"]
    for strategy, values in summary["dft"].items():
        error = values["conditional_barrier_mae_eV"]
        error_text = f"{error:.6f}" if error is not None else "pending"
        report.append(f"| {strategy} | {values['candidates']}/3 | {values['verified']}/3 | "
                      f"{error_text} | {values['statuses']} |")
    report += ["", "DFT 使用 ωB97X/6-31G(d)，电荷零、单重态遵循原始 Transition1x ORCA 生成器及其 ASE 默认值。",
               "LQA 步长 5、160 点和最多 320 点 checkpoint 续算在保留阶段之前固定，保留全部异常及恢复记录。",
               "电子能垒为精修 TS 减去 IRC 所连优化反应物的电子能量，采用 DFT 验证通过的候选计算条件误差。", "",
               "## 解释范围", "",
               "十二条反应用于项目保留评价，两个模型的预训练重叠尚未排除。端点变化类别采用共价半径启发式。",
               "仅两片中具有完整端点对的反应参与候选池，构型分片覆盖会影响抽样分布。",
               "搜索使用单个固定扰动种子和有限预算。报告完整失败及无候选记录，结论限于当前反应与参数。",
               "Gaussian 验证限定在预选的三条反应及数据集参考电子态。另行比较自旋面与参考波函数稳定性仍需后续计算。", ""]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {len(records)}/{len(rows)} strategy units, {len(unique)} referenced unique attempts", flush=True)


if __name__ == "__main__":
    main()
