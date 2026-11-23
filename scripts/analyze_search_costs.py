"""Derive the holdout cost decomposition from the completed v0.5 results.

This reads saved experiments only. The equal-endpoint-gate calculation is an
explicit post-hoc accounting counterfactual, not an additional strategy run.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mh1-original", "mh1-finetuned")
STRATEGIES = ("S0", "S1", "S2")


def analyze(source):
    summary = json.loads((source/"summary.json").read_text())
    with (source/"strategy_metrics.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["role"] == "holdout"]
    lookup = {(r["model"], r["reaction"], r["strategy"]): r for r in rows}
    if len(rows) != 72 or len(lookup) != 72 or any(r["status"] != "completed" for r in rows):
        raise ValueError("Require all 72 unique completed holdout units")
    reaction_sets = [{r["reaction"] for r in rows if r["model"] == m and r["strategy"] == s}
                     for m in MODELS for s in STRATEGIES]
    if any(len(ids) != 12 or ids != reaction_sets[0] for ids in reaction_sets):
        raise ValueError("All six groups must use the same twelve mapped reactions")
    costs, differences, paired = [], [], []
    for model in MODELS:
        groups = {s: summary["groups"][f"holdout/{model}/{s}"] for s in STRATEGIES}
        for strategy, group in groups.items():
            measured = sum(float(r["theoretical_seconds"]) for r in rows
                           if r["model"] == model and r["strategy"] == strategy)
            if not np.isclose(measured, group["total_theoretical_seconds"], atol=1e-7, rtol=0):
                raise ValueError("Reaction costs disagree with the published group total")
            costs.append({"model": model, "strategy": strategy,
                          "reactions": group["planned"], "verified_saddles": group["verified_saddles"],
                          "strict_pipeline": group["strict_pipeline"],
                          "estimated_seconds": group["total_theoretical_seconds"],
                          "seconds_per_saddle": group["seconds_per_verified_saddle"],
                          "relative_to_S0": group["total_theoretical_seconds"]/groups["S0"]["total_theoretical_seconds"]})
        reactions = sorted(r["reaction"] for r in rows if r["model"] == model and r["strategy"] == "S0")
        for reaction in reactions:
            first, adaptive = (lookup[(model, reaction, s)] for s in ("S1", "S2"))
            difference = float(first["theoretical_seconds"])-float(adaptive["theoretical_seconds"])
            if difference:
                endpoint_gate = first["primary_failure"] == adaptive["primary_failure"] == "endpoint_failure"
                baseline = lookup[(model, reaction, "S0")]
                if endpoint_gate and (int(adaptive["attempts"]) != 1 or
                    not np.isclose(float(adaptive["theoretical_seconds"]),
                                   float(baseline["theoretical_seconds"]), atol=1e-7, rtol=0)):
                    raise ValueError("Endpoint-gate decomposition requires measured S2 = S0 first-attempt cost")
                differences.append({"model": model, "reaction": reaction, "S1_minus_S2_seconds": difference,
                                    "S1_failure": first["primary_failure"], "S2_failure": adaptive["primary_failure"],
                                    "endpoint_gate": endpoint_gate})
    for strategy in STRATEGIES:
        original = {r["reaction"]: r for r in rows if r["model"] == MODELS[0] and r["strategy"] == strategy}
        fine = {r["reaction"]: r for r in rows if r["model"] == MODELS[1] and r["strategy"] == strategy}
        gained = [r for r in original if fine[r]["saddle_irc_verified"] == "True" and original[r]["saddle_irc_verified"] == "False"]
        lost = [r for r in original if fine[r]["saddle_irc_verified"] == "False" and original[r]["saddle_irc_verified"] == "True"]
        count = len(gained)+len(lost)
        paired.append({"strategy": strategy, "gained": gained, "lost": lost,
                       "exact_two_sided_p": float(binomtest(len(gained), count, .5).pvalue) if count else 1.0})
    decomposition = []
    for model in MODELS:
        by = {c["strategy"]: c for c in costs if c["model"] == model}
        gate = sum(d["S1_minus_S2_seconds"] for d in differences if d["model"] == model and d["endpoint_gate"])
        total = by["S1"]["estimated_seconds"]-by["S2"]["estimated_seconds"]
        adjusted = by["S1"]["estimated_seconds"]-gate
        decomposition.append({"model": model, "S1_minus_S2_seconds": total,
                              "saving_fraction_S1": total/by["S1"]["estimated_seconds"],
                              "endpoint_gate_seconds": gate, "other_policy_seconds": total-gate,
                              "S1_equal_endpoint_gate_counterfactual_seconds": adjusted,
                              "remaining_saving_fraction_counterfactual": (adjusted-by["S2"]["estimated_seconds"])/adjusted})
    return {"source_version": "v0.5.0", "analysis_type": "post_hoc_saved_result_decomposition",
            "cost_definition": summary["cost_accounting"]["definition"], "costs": costs,
            "reaction_cost_differences": differences, "decomposition": decomposition,
            "model_paired_comparisons": paired,
            "counterfactual_assumption": "Skip S1 extra attempts when its shared fixed endpoints already fail, retaining its measured first-attempt cost. No new calculation or success reassessment."}


def plot(result, output):
    colors = ("#365f86", "#bf623b")
    labels = ("Original MH-1", "Fine-tuned MH-1")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    x = np.arange(3)
    for index, model in enumerate(MODELS):
        rows = [c for c in result["costs"] if c["model"] == model]
        positions = x+(index-.5)*.32
        total = np.array([r["verified_saddles"] for r in rows])
        strict = np.array([r["strict_pipeline"] for r in rows])
        axes[0].bar(positions, strict, .29, color=colors[index], label=labels[index])
        axes[0].bar(positions, total-strict, .29, bottom=strict, color=colors[index], hatch="///", alpha=.35)
        for position, count in zip(positions, total):
            axes[0].text(position, count+.12, f"{count}/12", ha="center", fontsize=10)
        minutes = [r["estimated_seconds"]/60 for r in rows]
        axes[1].plot(x, minutes, "o-", color=colors[index], label=labels[index], linewidth=2)
        for position, value in zip(x, minutes):
            axes[1].annotate(f"{value:.1f}", (position, value), xytext=(0, 8 if index == 0 else -15),
                             textcoords="offset points", ha="center", color=colors[index])
    axes[0].set(ylabel="Verified saddle + IRC count", ylim=(0, 8), title="Same 12 project holdout reactions")
    axes[0].text(.02, .97, "Hatched cap: NEB unconverged", transform=axes[0].transAxes, va="top", fontsize=10)
    axes[1].set(ylabel="Estimated separate-strategy compute / min", ylim=(0, 140), title="Sum of measured jobs, excluding model loading")
    for axis in axes:
        axis.set_xticks(x, STRATEGIES)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.legend(frameon=False, loc="upper left" if axis == axes[1] else "upper right")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"success_cost.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/v0.5.0")
    parser.add_argument("--output", type=Path, default=ROOT/"results/v0.6.0-pilot/analysis")
    args = parser.parse_args()
    result = analyze(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"cost_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (args.output/"cost_efficiency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["costs"][0]))
        writer.writeheader()
        writer.writerows(result["costs"])
    plot(result, args.output)
    print(json.dumps({"reactions": 12, "completed_units": 72, "output": str(args.output)}))


if __name__ == "__main__":
    main()
