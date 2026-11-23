"""Close the bounded pilot using terminal raw evidence, without rerunning searches.

Keep v0.5 statistics immutable. This writes a separate diagnostic report,
verification, stationary-point figure and cost interpretation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import units

from analyze_search_costs import analyze
from assess_dft_chain import assess, atoms_from_record, bonds, fragment_errors, parse_gaussian

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def log_elapsed(folder):
    total = 0.0
    for path in folder.glob("*.log"):
        if path.name.endswith(".launcher.log"):
            continue
        for d, h, m, s in re.findall(r"Elapsed time:\s+(\d+) days\s+(\d+) hours\s+(\d+) minutes\s+([\d.]+) seconds", path.read_text(errors="replace")):
            total += int(d)*86400+int(h)*3600+int(m)*60+float(s)
    return total


def summarize(root):
    base = root/"results/v0.6.0-pilot"
    original = load(root/"configs/v0.6.0-pilot/protocol.json")
    replay = load(root/"configs/v0.6.0-pilot/protocol-r2.json")
    gaussian_folder = base/"gaussian/rxn9456-return-r2"
    mac_folder = base/"mac/rxn9456-r2"
    gaussian = assess(replay, "rxn9456", gaussian_folder, "gaussian")
    calibration = assess(original, "rxn0666", base/"mac/rxn0666-r1", "mac")
    saved_gaussian = load(gaussian_folder/"assessment.json")
    job = next(j for j in replay["jobs"] if j["engine"] == "Gaussian16W-A.03")
    task = load(gaussian_folder/"task.json")
    mac = load(mac_folder/"summary.json")
    curvature = load(base/"analysis/endpoint-curvature/summary.json")
    verification = {
        "gaussian_fresh_raw_log_assessment_passes": gaussian["dft_verified"],
        "gaussian_matches_previous_assessment": all(gaussian[k] == saved_gaussian[k] for k in
            ("dft_verified", "expected_connectivity", "electronic_barrier_eV", "mlip_to_dft_ts_rmsd_A")),
        "gaussian_same_frozen_candidate": task["numbers"] == job["numbers"] and task["positions_A"] == job["positions_A"],
        "gaussian_same_frozen_settings": all(task["configuration"][k] == replay[k] for k in
            ("dft_method", "dft_nproc", "dft_mem", "charge", "multiplicity", "dft_policy")),
        "mac_same_frozen_protocol": load(mac_folder/"protocol.json") == replay,
        "positive_control_full_chain_passes": calibration["dft_verified"],
        "positive_control_all_preset_tolerances_pass": calibration["gaussian_comparison"]["within_predefined_tolerances"],
        "failed_mac_candidate_remains_failed": mac["status"] == "failed" and not mac["raw_chain_completed"] and not mac["dft_verified"],
        "negative_curvature_independently_confirmed": curvature["negative_curvature_confirmed"],
        "curvature_does_not_change_frozen_verdict": not curvature["original_full_chain_verdict_changed"],
    }
    if not all(verification.values()):
        raise ValueError({k:v for k,v in verification.items() if not v})
    state = load(gaussian_folder/"worker-state.json")
    irc = [{"direction":b["direction"], "points": next(s["irc_points"] for s in state["stages"] if s["job"] == b["irc_job"]),
            "normal_termination": next(s["normal_termination"] for s in state["stages"] if s["job"] == b["irc_job"])}
           for b in state["branches"]]
    candidate = atoms_from_record(job, job["numbers"])
    ts = parse_gaussian(gaussian_folder/"ts.log", job["numbers"])[1]
    endpoints = [atoms_from_record(job["endpoints"][n], job["numbers"]) for n in ("reactant", "product")]
    changes = [{"pair_1based": [i+1,j+1], "elements": candidate[i].symbol+candidate[j].symbol,
                "reactant_A": endpoints[0].get_distance(i,j), "mlip_ts_A": candidate.get_distance(i,j),
                "dft_ts_A": ts.get_distance(i,j), "product_input_A": endpoints[1].get_distance(i,j)}
               for i,j in sorted(bonds(endpoints[0]) ^ bonds(endpoints[1]))]
    unstable = load(mac_folder/"forward/frequencies/result.json")
    unstable_atoms = atoms_from_record(unstable, job["numbers"])
    last = json.loads((mac_folder/"backward/irc/evaluations.jsonl").read_text().splitlines()[-1])
    first_failure = load(base/"gaussian/rxn9456-return-r1/worker-state.json")
    early = list((base/"gaussian/rxn9456-return-r1").glob("previous-worker-state-*.json"))
    r1_owner = first_failure["elapsed_seconds"] + sum(load(p)["elapsed_seconds"] for p in early)
    selected = load(root/"results/v0.5.0/attempts/mh1-finetuned/rxn9456/498df471bf2dd325/result.json")
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": replay["study_type"], "changes_v05_results": False,
        "source_versions": ["v0.5.0", "v0.6.0-pilot-r1", "v0.6.0-pilot-r2"],
        "verification": verification, "gaussian_candidate": gaussian, "irc_points": irc,
        "positive_control": calibration, "changing_mapped_bonds": changes,
        "mlip_barrier_eV": selected["saddle"]["barrier_eV"],
        "mlip_minus_dft_barrier_eV": selected["saddle"]["barrier_eV"]-gaussian["electronic_barrier_eV"],
        "mac_failed_candidate": {"status": mac["status"], "elapsed_seconds": mac["elapsed_seconds"],
            "ts_negative_cm1": min(mac["ts"]["frequencies_cm1"]),
            "forward": {"irc_converged": True, "minimum_optimization_converged": True,
                "negative_cm1": min(unstable["frequencies_cm1"]),
                "matches_input_bonds": [bonds(unstable_atoms)==bonds(e) for e in endpoints],
                "fragment_rmsd_A": [fragment_errors(e,unstable_atoms) for e in endpoints]},
            "backward": {"reason": "IRC maximum iteration budget reached without convergence",
                "maxiter": replay["mac"]["max_irc_steps"],
                "energy_gradient_evaluations": len((mac_folder/"backward/irc/evaluations.jsonl").read_text().splitlines()),
                "last_gradient_hartree_per_bohr":last["max_gradient_hartree_per_bohr"]}},
        "endpoint_curvature": {k:curvature[k] for k in ("mode_frequency_cm1", "analytic_directional_curvature_hartree_per_bohr2",
            "finite_differences", "negative_curvature_confirmed", "elapsed_seconds")},
        "compute": {"gaussian_r1_all_owner_seconds": r1_owner,
            "gaussian_r1_raw_reported_elapsed_seconds": log_elapsed(base/"gaussian/rxn9456-return-r1"),
            "gaussian_r2_owner_seconds": state["elapsed_seconds"],
            "gaussian_r2_raw_reported_elapsed_seconds": log_elapsed(gaussian_folder),
            "gaussian_all_revision_owner_seconds": r1_owner+state["elapsed_seconds"],
            "mac_calibration_wall_seconds": calibration["source_elapsed_seconds"],
            "mac_candidate_failed_wall_seconds": mac["elapsed_seconds"],
            "curvature_wall_seconds": curvature["elapsed_seconds"],
            "accounting_note": "Owner/session intervals and Gaussian-reported times are separate; overlapping hosts are not summed as study elapsed time. Replays and failures remain in cost."},
        "search_cost_analysis": analyze(root/"results/v0.5.0"),
        "scope": "One post-hoc extra candidate, one known-positive calibration; not a new unseen holdout or a general success rate."}
    return result


def plot(result, output):
    fig, axes = plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    branch = result["gaussian_candidate"]["branches"]
    reactant = next(b for b in branch if b["matches_input_geometry"][0])
    product = next(b for b in branch if b["matches_input_geometry"][1])
    barrier = result["gaussian_candidate"]["electronic_barrier_eV"]
    relative_product = (product["energy_hartree"]-reactant["energy_hartree"])*units.Hartree
    axes[0].plot([0,1,2],[0,barrier,relative_product],"o-",color="#365f86",lw=2,label="Gaussian DFT")
    axes[0].scatter([1],[result["mlip_barrier_eV"]],marker="x",s=75,color="#bf623b",label="MACE TS barrier")
    axes[0].annotate(f"{barrier:.4f} eV",(1,barrier),xytext=(0,12),textcoords="offset points",ha="center")
    axes[0].set(xticks=[0,1,2],xticklabels=["Reactant","TS","Product\ncomplex"],
                ylabel="Electronic energy / eV",ylim=(-.3,5.5),title="DFT-verified extra recovery")
    axes[0].legend(frameon=False,fontsize=8,loc="upper left")
    axes[0].text(.5,-.24,"Stationary energies; lines guide the eye",transform=axes[0].transAxes,ha="center",fontsize=8)
    for change,color in zip(result["changing_mapped_bonds"],["#365f86","#658d5a","#bf623b"]):
        distances=[change[k] for k in ("reactant_A","mlip_ts_A","dft_ts_A","product_input_A")]
        label="-".join(map(str,change["pair_1based"]))+" "+change["elements"]
        axes[1].plot(range(4),distances,"o-",color=color,label=label,lw=1.8)
    axes[1].set(xticks=range(4),xticklabels=["Input R","MACE TS","DFT TS","Input P"],ylabel="Mapped pair distance / Angstrom",title="Three breaking bonds")
    axes[1].legend(frameon=False,fontsize=8)
    curvature=result["endpoint_curvature"]
    numbers=[curvature["analytic_directional_curvature_hartree_per_bohr2"]]+[r["directional_curvature_hartree_per_bohr2"] for r in curvature["finite_differences"]]
    axes[2].bar(range(3),np.array(numbers)*1000,color=["#bf623b","#365f86","#658d5a"],width=.6)
    axes[2].axhline(0,color="black",lw=.7)
    axes[2].set(xticks=range(3),xticklabels=["Analytic","FD 0.005 A","FD 0.010 A"],
                ylabel="Curvature / (10^-3 Hartree/Bohr^2)",title="Mac endpoint: real negative curvature",ylim=(-1.5,.1))
    axes[2].text(.5,.12,"-119.69 cm^-1: failed stability check",transform=axes[2].transAxes,ha="center",fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    for extension in ("png","pdf"):
        fig.savefig(output/f"pilot_diagnosis.{extension}",dpi=200)
    plt.close(fig)


def report(result):
    g=result["gaussian_candidate"];cal=result["positive_control"];cost=result["search_cost_analysis"];compute=result["compute"]
    lines=["# v0.6.0-pilot：额外恢复候选的 DFT 真值与完整链诊断", "",
        "rxn9456 的第二版 Gaussian TS/频率、双向 IRC、稳定最小值及映射端点全部通过。",
        "Mac 已知正对照 rxn0666 全链通过；Mac 独立 rxn9456 全链失败，保持失败判定。",
        "本报告为事后定向诊断与引擎校准，v0.5.0 冻结结果和分母不变。", "",
        "## 1. 唯一额外恢复得到 DFT 支持", "",
        f"- 唯一 TS 虚频 {min(g['ts']['frequencies_cm1']):.4f} cm⁻¹，完整 24 模，优化与频率均正常结束。",
        f"- 双向 LQA 分别 {result['irc_points'][0]['points']} / {result['irc_points'][1]['points']} 点检测到最小值，日志正常结束。",
        "- 两端各完整 24 模且无负频率，映射键与逐片段 RMSD 均通过 ≤0.5 Å 判据。",
        f"- 电子势垒 {g['electronic_barrier_eV']:.9f} eV；MACE {result['mlip_barrier_eV']:.9f} eV，差 {result['mlip_minus_dft_barrier_eV']:+.9f} eV。",
        f"- MLIP→DFT TS 对齐 RMSD {g['mlip_to_dft_ts_rmsd_A']:.8f} Å。",
        "- 输入的产物图对应 CO + 2 CH2O；末端是有限距离的三片段复合体，不能据此声称无限分离产物热力学或自由能。", "",
        "第一版 160 点耗尽，续算第 162 点在尚未检测到最小值时触发 PESDtA。",
        "第二版事前固定初始 320 点后重算，实际到第 265 点才检测到正向最小值。",
        "第一版属于预算/实现数值失败，不能据它判定候选化学通道错误；第二版单独报告。", "",
        "## 2. Mac 的能力与失败边界", "",
        f"正对照 rxn0666 完成全链，电子势垒 {cal['electronic_barrier_eV']:.9f} eV，",
        f"与 Gaussian 差 {cal['gaussian_comparison']['electronic_barrier_difference_eV']:+.9g} eV，四项事前容差全部满足。",
        "这证实原生 Mac 能完成十原子 DFT 全链，但不能保证任意反应自动收敛，也不能据不同主机/网格的一次时长宣称引擎加速倍数。", "",
        "rxn9456 的 Mac TS/频率通过，之后：", "",
        "- forward IRC 和末端优化停止，但末端有 -119.6857 cm⁻¹ 虚频，虽然反应物映射键一致，仍不是通过验收的稳定最小值。",
        "- backward 在 300 步预算内未收敛，367 次能量/梯度评价不等于 367 个已接受 IRC 步。",
        "- 对失败端点沿该负振动模式，在 ±0.005 / ±0.010 Å 做四次固定几何能量/梯度计算，",
        "  有限差分曲率均为负，与解析曲率相差不到 1%。这支持负曲率真实存在，不能将其当作纯频率解析噪声忽略。",
        "- 不放宽频率阈值、不覆盖失败、不继续调预算直到成功。负模逃逸等优化策略留待独立协议。", "",
        "## 3. 成本与搜索控制的解释", "",
        "| 模型 | 策略 | 鞍点+IRC /12 | 全阶段 /12 | 独立策略成本估计 /min | 每个有效鞍点 /min |", "| --- | --- | --- | --- | --- | --- |"]
    for row in cost["costs"]:
        lines.append(f"| {row['model']} | {row['strategy']} | {row['verified_saddles']} | {row['strict_pipeline']} | {row['estimated_seconds']/60:.3f} | {row['seconds_per_saddle']/60:.3f} |")
    d=cost["decomposition"][0]
    lines += ["", "S1/S2 的微调鞍点数 5→6 来自同一个 rxn9456，严格全阶段仍为 5/12。",
        "S2 相对 S1 在原始/微调模型分别节省 12.821% / 1.198%，不能概括为普遍大幅提速。",
        f"原始模型的 {d['S1_minus_S2_seconds']:.3f} 秒节省中，{d['endpoint_gate_seconds']:.3f} 秒（{d['endpoint_gate_seconds']/d['S1_minus_S2_seconds']*100:.2f}%）",
        "来自两个共享端点已经失败的反应上提前停止。对 S1 事后采用相同端点门槛、保留其已测首次成本的记账反事实，",
        f"原始模型剩余节省约 {d['remaining_saving_fraction_counterfactual']*100:.3f}%；它不是新策略运行，不能取代真实 S1 对照。",
        "微调对原始模型的逐反应配对增益 S0 为 2 增/0 损，S1/S2 各 3 增/0 损；",
        "对应双侧精确检验 p=0.5 / 0.25 / 0.25。小样本不能支持统计显著的普遍优势。", "",
        "## 4. 保留失败和重算成本", "",
        f"- Gaussian r1 所有 owner 区间合计 {compute['gaussian_r1_all_owner_seconds']/60:.3f} min；r2 {compute['gaussian_r2_owner_seconds']/60:.3f} min；",
        f"  合计 {compute['gaussian_all_revision_owner_seconds']/60:.3f} min，包含失败与 TS 重算。",
        f"- Mac 正对照墙钟 {compute['mac_calibration_wall_seconds']/60:.3f} min；rxn9456 失败链 {compute['mac_candidate_failed_wall_seconds']/60:.3f} min；",
        f"  负曲率诊断另用 {compute['curvature_wall_seconds']:.3f} s。",
        "- 主机间并发不能相加为实验流逝时间，程序 owner 时长与 Gaussian 自报时长也分别保留。", "",
        "## 5. 复现与后续交付", "",
        "使用既有分析 .venv：", "", "```bash",
        ".venv/bin/python scripts/analyze_search_costs.py",
        ".venv/bin/python scripts/summarize_v06_pilot.py", "```", "",
        "原始证据位于 results/v0.6.0-pilot/mac 和 gaussian；协议在 configs/v0.6.0-pilot。",
        "verification.json 检查候选/参数绑定、原始日志再解析、正对照和失败保留；此通过表示报告正确保留通过与失败，",
        "不表示 Mac rxn9456 全链通过。接下来将已完成的受控实验整理为课程报告、图表和汇报叙事。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=ROOT/"results/v0.6.0-pilot/final")
    args=parser.parse_args()
    result=summarize(args.root)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in (("summary.json",result),("verification.json",{"passed":all(result["verification"].values()),"checks":result["verification"]})):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    (args.output/"report.md").write_text(report(result))
    plot(result,args.output)
    print(json.dumps({"verification_passed":all(result["verification"].values()),"gaussian_rxn9456":result["gaussian_candidate"]["dft_verified"],"mac_calibration":result["positive_control"]["dft_verified"],"mac_rxn9456":result["mac_failed_candidate"]["status"]}))


if __name__=="__main__":
    main()
