"""Write the reproducible developmental routing record; preserve pending status."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_study_data import save_json


def main():
    source = ROOT / "results/v0.7.0/router"
    output = ROOT / "docs/reports/v0.7.0-routing"
    output.mkdir(parents=True, exist_ok=True)
    headroom = json.loads((source / "headroom.json").read_text())
    complementary = json.loads((source / "complementarity.json").read_text())
    smoke = json.loads((ROOT / "results/v0.7.0/decision-smoke/summary.json").read_text())
    results = {}
    for name in ("ridge", "extratrees", "tabpfn", "tabicl", "jevmlx"):
        path = source / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())["summary"]
    status = "complete" if len(results) == 5 else "partial"
    record = {"status": status, "scope": headroom["scope"], "prospective_holdout_opened": False,
        "reactions": 128, "formulas": 42, "endpoint_features": 38,
        "folds": "5, grouped by chemical formula; one row per reaction",
        "data_limits": {"rearrangement": 118, "cleavage": 10, "formation": 0},
        "fixed_expert_metrics": headroom["fixed_expert_all_metrics"],
        "headroom": complementary, "routing_metrics": results,
        "pending_methods": sorted(set(("ridge", "extratrees", "tabpfn", "tabicl", "jevmlx")) - set(results)),
        "local_runtime_acceptance": smoke,
        "conclusion": "No demonstrated joint barrier/force improvement with this fixed cross-DFT expert pool. This developmental static benchmark does not establish endpoint-search success or rule out later expert routing."}
    save_json(output / "summary.json", record)
    lines = ["# v0.7 本地决策与专家路由先导记录", "",
        "这是计算研究记录，使用 128 条独立于 MACE train 的 meta-development 反应、42 个化学式，",
        "不是后续 holdout 或端点搜索/DFT 通过率。128 条中 118 重排、10 断键，没有成键类。",
        "旧 287 条 test 未用于拟合路由器；新冻结的 64+64 后续 holdout 仍未打开。", "",
        "## 本地运行", "",
        "TabPFN v2（软件 10.2.0）、TabICLv2（2.2.0）实际回归和分类均在 Mac MPS 通过。",
        "jevmlx 固定源码版本，使用本地 Qwen2.5-7B-Instruct-4bit，经 MLX 输出受约束选项；",
        "它不是 Jev 原权重或 RLCD。两次短已知答案运行约 2.70/0.248 秒，只证明运行能力。",
        "真实长表格上下文另计耗时，概率不等同 DFT 精度。权重 revision 在 configs/decision-models.json。", "",
        "## 专家互补性", "",
        "| 方案 | 能垒 MAE (eV) | 全局力分量 RMSE (eV/Å) | 反应能 MAE (eV) |",
        "| --- | ---: | ---: | ---: |"]
    labels = {"fixed_finetuned": "固定微调 MH-1", "barrier_oracle": "只优化能垒的理想选择（不可部署）",
        "force_guarded_oracle": "要求每条反应力不变差的理想选择", "multi_metric_guarded_oracle": "再要求 TS 力、反应能不变差"}
    for key, values in complementary["results"].items():
        lines.append(f"| {labels[key]} | {values['barrier_MAE_eV']:.6f} | {values['force_global_component_RMSE_eV_per_A']:.6f} | {values['reaction_energy_MAE_eV']:.6f} |")
    lines += ["", "只看能垒的 oracle 约有 32% 改善空间，却把力 RMSE 增大到约 2.8 倍。",
        "固定微调 MH-1 在这 128 条反应上逐条具有最低的力 RMSE，因此加入力不变差约束后没有额外收益。",
        "这表明当前不同 DFT 来源的专家不是已经验证的互补力场。参考 oracle 读取评价标签，只能作上限。", "",
        "## 实际路由", "",
        "仅用 38 个端点几何/元素特征，五折按化学式分组。每个专家分别回归预计绝对能垒误差，再选最小者。",
        "预处理只在该折训练部分拟合。固定专家基线也只用每折训练标签选择。",
        "Jev 风格比较器使用相同训练折的 8 个特征最近邻实例和训练均值；它的上下文规模与表格模型不同。", "",
        "| 方法 | 能垒 MAE (eV) | 力分量 RMSE (eV/Å) | 五折拟合及批量推理总秒数 |",
        "| --- | ---: | ---: | ---: |"]
    for name, values in results.items():
        lines.append(f"| {name} | {values['barrier_MAE_eV']:.6f} | {values['force_global_component_RMSE_eV_per_A']:.6f} | {values['oof_fit_and_batch_predict_seconds']:.3f} |")
    if "jevmlx" in results:
        values = results["jevmlx"]
        lines += ["", f"Jev 风格比较器五折共用时 {values['oof_fit_and_batch_predict_seconds']/60:.2f} 分钟，",
            f"选中参考能垒最优专家的比例为 {values['expert_selection_accuracy_vs_oracle']:.2%}。"]
        if "restricted_score_diagnostic" in values:
            diagnostic = values["restricted_score_diagnostic"]
            lines += [f"所选答案平均受约束概率为 {diagnostic['mean_chosen_option_probability']:.2%}，",
                "高于该最优专家选择正确率；它不是已校准的化学正确率。"]
    lines += ["", f"状态：{status}；待完成方法：{', '.join(record['pending_methods']) or '无'}。",
        "时间包含折内拟合与批量推理，不是每个新反应的部署延迟，也不是跨后端速度基准。",
        "TabPFN/ExtraTrees 在本轮选择所有反应均使用原微调专家，因此没有提高精度。",
        "完整逐反应选择、配对化学式 bootstrap 和原始运行在 results/v0.7.0/router。", "",
        "## 下一步", "",
        "优先提高底层模型及共同 DFT 标准下专家的真实性能，再比较能垒、受力、反应能和尾部风险。",
        "当前静态路径上的负结果不否定搜索成功率上的互补；后者必须用独立端点搜索和完整 DFT 链验证。",
        "样本都来自本项目的中性单重态 CHNO 数据，未测试金属、周期材料、离子或其他自旋。",
        "MACE 的 256/2048/4096 帧对照正在 OMEN 按冻结预算继续。后续需增加独立反应、优化/力权重对照、",
        "不同基础模型在同一 DFT 标准下的微调，以及有效方案的多种子重复。最终方案冻结后才打开新 holdout。", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Saved {status} routing record with {len(results)} completed methods")


if __name__ == "__main__":
    main()
