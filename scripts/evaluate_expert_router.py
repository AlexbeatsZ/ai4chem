"""Measure expert headroom and endpoint-only, formula-grouped local routing."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from expert_routing import EXPERTS, endpoint_features, grouped_delta_interval
from model_study_data import save_json


def evaluate(args):
    partition = json.loads((ROOT / "configs/v0.7.0_partitions.json").read_text())
    config = json.loads((ROOT / "configs/decision-models.json").read_text())
    records = sorted(partition["endpoint_records"]["meta_development"], key=lambda x: x["sid"])
    reactions = [x["reaction"] for x in records]
    frame = pd.DataFrame([endpoint_features(x) for x in records])
    # The geometry allowlist above never reads CSV/reference metrics as features.
    metrics = pd.read_csv(args.benchmark / "reaction_metrics.csv")
    if set(metrics.reaction) != set(reactions) or set(metrics.model) != set(EXPERTS):
        raise ValueError("Benchmark must contain exactly the 128 frozen meta-development reactions and four experts")
    if len(metrics) != len(reactions) * len(EXPERTS) or metrics.duplicated(["reaction", "model"]).any():
        raise ValueError("Expected one independent metric per reaction/expert")
    losses = metrics.pivot(index="reaction", columns="model", values="barrier_error_eV").loc[reactions, EXPERTS].abs().to_numpy()
    if not np.isfinite(losses).all():
        raise ValueError("Missing or non-finite expert errors")
    groups = np.array([x["formula"] for x in records])
    x = frame.to_numpy(float)
    best_fixed = int(np.argmin(np.mean(losses, axis=0)))
    winner = np.argmin(losses, axis=1)
    oracle = np.min(losses, axis=1)
    fixed_errors = losses[:, best_fixed]
    force_rmse = metrics.pivot(index="reaction", columns="model", values="force_component_rmse_eV_per_A").loc[reactions, EXPERTS].to_numpy()
    ts_force_rmse = metrics.pivot(index="reaction", columns="model", values="ts_region_force_component_rmse_eV_per_A").loc[reactions, EXPERTS].to_numpy()
    reaction_energy_errors = metrics.pivot(index="reaction", columns="model", values="reaction_energy_error_eV").loc[reactions, EXPERTS].abs().to_numpy()
    atom_counts = np.asarray([item["atoms"] for item in records], float)

    def selected_metrics(choice):
        indices = np.arange(len(records))
        errors = losses[indices, choice]
        return {"barrier_MAE_eV": float(np.mean(errors)), "barrier_absolute_error_p90_eV": float(np.quantile(errors, .9)),
            "force_global_component_RMSE_eV_per_A": float(np.sqrt(np.average(force_rmse[indices, choice]**2, weights=atom_counts))),
            "mean_reaction_TS_region_force_RMSE_eV_per_A": float(np.mean(ts_force_rmse[indices, choice])),
            "reaction_energy_MAE_eV": float(np.mean(reaction_energy_errors[indices, choice])),
            "error_over_0.1eV_fraction": float(np.mean(errors>.1))}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "scope": "meta-development formula-grouped OOF on fixed DFT path geometries; router inputs are endpoint-only",
        "prospective_holdout_opened": False, "legacy_test_used_for_fitting": False,
        "reactions": len(records), "formulas": len(set(groups)), "feature_count": x.shape[1],
        "feature_names": list(frame.columns), "experts": EXPERTS,
        "best_fixed_descriptive": EXPERTS[best_fixed],
        "fixed_expert_barrier_MAE_eV": dict(zip(EXPERTS, np.mean(losses, axis=0).tolist())),
        "oracle_unattainable": {"barrier_MAE_eV": float(np.mean(oracle)),
            "headroom_eV": float(np.mean(fixed_errors-oracle)),
            "winner_counts": dict(zip(EXPERTS, np.bincount(winner, minlength=len(EXPERTS)).tolist())),
            "warning": "selects using each evaluation reference label; cannot be deployed"},
        "fixed_expert_all_metrics": {name: selected_metrics(np.full(len(records), i)) for i,name in enumerate(EXPERTS)},
        "barrier_oracle_all_metrics": selected_metrics(winner),
        "routing_objective": "expected absolute barrier error; other reported metrics are outcomes, not guaranteed improvements",
        "selection": "all learners fixed before meta outcomes; no tuning on these out-of-fold predictions",
        "models": {}, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / "headroom.json", summary)
    print(json.dumps({"best_fixed": summary["best_fixed_descriptive"], "fixed": float(np.mean(fixed_errors)), "oracle": float(np.mean(oracle))}), flush=True)
    if args.headroom_only:
        return
    splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(x, groups=groups))
    fold_ids = np.full(len(records), -1)
    baseline_choices = np.empty(len(records), dtype=int)
    for fold, (train, test) in enumerate(splits):
        assert not set(groups[train]) & set(groups[test])
        fold_ids[test] = fold
        baseline_choices[test] = int(np.argmin(np.mean(losses[train], axis=0)))
    selected_errors = {"fold_selected_fixed": losses[np.arange(len(records)), baseline_choices], "oracle_unattainable": oracle}
    choices = {"fold_selected_fixed": baseline_choices}
    for expert in range(len(EXPERTS)):
        selected_errors[f"always_{EXPERTS[expert]}"] = losses[:, expert]
    baseline = selected_errors["fold_selected_fixed"]
    torch.set_num_threads(4)
    for method in args.methods:
        path = output / f"{method}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["reactions"] != reactions or cached["folds"] != fold_ids.tolist():
                raise ValueError(f"Cached identity mismatch for {method}")
            choice = np.asarray(cached["choices"], dtype=int)
            summary["models"][method] = cached["summary"]
        else:
            choice = np.empty(len(records), dtype=int)
            confidence = np.full(len(records), np.nan)
            times = []
            for fold, (train, test) in enumerate(splits):
                start = time.perf_counter()
                if method in ("ridge", "extratrees"):
                    estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                        StandardScaler(), Ridge(alpha=10.0)) if method == "ridge" else make_pipeline(
                        SimpleImputer(strategy="median", keep_empty_features=True),
                        ExtraTreesRegressor(n_estimators=256, min_samples_leaf=4, max_features=.8, random_state=config["seed"]+fold, n_jobs=4))
                    estimator.fit(x[train], losses[train])
                    predicted = np.maximum(estimator.predict(x[test]), 0.)
                    choice[test] = np.argmin(predicted, axis=1)
                elif method in ("tabpfn", "tabicl"):
                    predicted = np.empty((len(test), len(EXPERTS)))
                    for expert in range(len(EXPERTS)):
                        item = config["models"][f"{method}_regressor"]
                        settings = {"device": "mps", "n_estimators": 4, "random_state": config["seed"]+fold,
                            "model_path": str(ROOT / item["local"] / item["file"])}
                        if method == "tabpfn":
                            from tabpfn import TabPFNRegressor
                            from tabpfn.constants import ModelVersion
                            estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2, **settings)
                        else:
                            from tabicl import TabICLRegressor
                            estimator = TabICLRegressor(allow_auto_download=False, use_amp=False, n_jobs=1, **settings)
                        estimator.fit(x[train], losses[train, expert])
                        predicted[:, expert] = np.maximum(estimator.predict(x[test]), 0.)
                        del estimator
                        gc.collect()
                        torch.mps.empty_cache()
                    if not np.isfinite(predicted).all():
                        raise RuntimeError(f"Non-finite {method} risk prediction")
                    choice[test] = np.argmin(predicted, axis=1)
                elif method == "jevmlx":
                    from jevmlx import choose
                    imputer = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
                    z_train, z_test = imputer.fit_transform(x[train]), imputer.transform(x[test])
                    codes = ["OFF23", "MH1", "MH0", "FINETUNED"]
                    options = dict(zip(codes, EXPERTS))
                    means = dict(zip(codes, np.mean(losses[train], axis=0).round(5).tolist()))
                    for index, j in enumerate(test):
                        nearest = np.argsort(np.mean((z_train-z_test[index])**2, axis=1))[:8]
                        examples = [{"features": {k: (round(float(v), 5) if np.isfinite(v) else None) for k,v in frame.iloc[train[n]].items()},
                            "observed_absolute_barrier_errors_eV": dict(zip(codes, losses[train[n]].round(5).tolist()))} for n in nearest]
                        query = {k: (round(float(v), 5) if np.isfinite(v) else None) for k,v in frame.iloc[j].items()}
                        context = json.dumps({"task": "Choose the expert expected to minimize absolute wB97X/6-31G(d) reaction barrier error for a neutral singlet CHNO reaction. The true error for this query is unknown. Use endpoint geometry features and training examples; option probabilities are not chemistry precision guarantees.",
                            "training_mean_error_eV": means, "historical_examples": examples, "query_endpoint_features": query})
                        answer = choose(context, options, model=str(ROOT/config["models"]["jevmlx"]["local"]))
                        choice[j] = codes.index(answer.value)
                        confidence[j] = answer.probability
                else:
                    raise ValueError(method)
                times.append(time.perf_counter()-start)
                print(f"{method}: formula fold {fold+1}/5 finished", flush=True)
            errors = losses[np.arange(len(records)), choice]
            result = {**selected_metrics(choice),
                "expert_selection_accuracy_vs_oracle": float(np.mean(choice==winner)),
                "paired_vs_fold_selected_fixed": grouped_delta_interval(errors-baseline, groups),
                "oof_fit_and_batch_predict_seconds": sum(times), "fold_seconds": times,
                "timing_scope": "includes fold fitting and batch prediction; not a deployment single-query latency"}
            if method == "jevmlx":
                correct = (choice == winner).astype(float)
                result["restricted_score_diagnostic"] = {
                    "mean_chosen_option_probability": float(np.mean(confidence)),
                    "chosen_expert_matches_reference_best_fraction": float(np.mean(correct)),
                    "mean_score_minus_accuracy": float(np.mean(confidence-correct)),
                    "top_label_binary_brier": float(np.mean((confidence-correct)**2)),
                    "scope": "OOF developmental diagnostic for matching the minimum-error expert; not calibrated DFT precision probability"}
            cached = {"reactions": reactions, "folds": fold_ids.tolist(), "choices": choice.tolist(),
                "confidence": [float(v) if np.isfinite(v) else None for v in confidence], "summary": result}
            save_json(path, cached)
            summary["models"][method] = result
        choices[method] = choice
        selected_errors[method] = losses[np.arange(len(records)), choice]
        save_json(output/"summary.json", summary)
    rows = []
    for index, reaction in enumerate(reactions):
        row = {"reaction": reaction, "formula": groups[index], "fold": int(fold_ids[index])}
        row.update({name: float(values[index]) for name, values in selected_errors.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/"selected_errors.csv", index=False)
    summary["baseline"] = {"barrier_MAE_eV": float(np.mean(baseline)),
        **selected_metrics(baseline_choices),
        "rule": "select best fixed expert using only the current fold's training labels"}
    summary.update(status="meta_development_oof_complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(output/"summary.json", summary)
    print(json.dumps(summary["models"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT/"results/v0.7.0/meta_development_mac")
    parser.add_argument("--output", type=Path, default=ROOT/"results/v0.7.0/router")
    parser.add_argument("--headroom-only", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=["ridge","extratrees","tabpfn","tabicl","jevmlx"], default=["ridge","extratrees","tabpfn","tabicl"])
    evaluate(parser.parse_args())
