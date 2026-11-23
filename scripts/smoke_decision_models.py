"""Download pinned public weights and verify real local MPS/MLX inference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_study_data import save_json


def prepare_weights(config):
    from huggingface_hub import hf_hub_download, snapshot_download

    def fetch(item):
        folder = ROOT / item["local"]
        if "file" in item:
            for filename in [item["file"], "README.md"] + ([item["license_file"]] if "license_file" in item else []):
                hf_hub_download(item["repo"], filename, revision=item["revision"],
                                local_dir=folder, token=False)
        else:
            snapshot_download(item["repo"], revision=item["revision"], local_dir=folder,
                token=False, max_workers=3, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "README.md", "LICENSE*"])
        print(f"Pinned weights ready: {item['repo']}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(fetch, config["models"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--model", choices=["all", "tabpfn", "tabicl", "jevmlx"], default="all")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/decision-models.json").read_text())
    output = ROOT / "results/v0.7.0/decision-smoke"
    output.mkdir(parents=True, exist_ok=True)
    selected = {key: value for key, value in config["models"].items()
                if args.model == "all" or key == args.model or key.startswith(args.model + "_")}
    prepare_weights({"models": selected})
    if args.download_only:
        return
    import numpy as np
    import torch
    from sklearn.datasets import make_regression, make_classification

    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Expected native Apple Silicon MPS")
    report = {"host": "Mac", "platform": platform.platform(), "machine": platform.machine(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "api_inference_used": False,
        "scope": "synthetic/specified-answer runtime acceptance only, not chemistry accuracy evidence",
        "versions": {p: version(p) for p in ["torch", "tabpfn", "tabicl", "mlx", "mlx-lm", "jevmlx", "scikit-learn"]},
        "mps_available": True, "weight_manifest": "configs/decision-models.json", "models": {}}
    prior = output / "summary.json"
    if prior.exists():
        cached = json.loads(prior.read_text())
        if cached["versions"] == report["versions"]:
            report["models"].update(cached["models"])
    x, y = make_regression(n_samples=96, n_features=6, noise=.1, random_state=config["seed"])
    xc, yc = make_classification(n_samples=96, n_features=6, n_informative=4, n_redundant=0,
                                 random_state=config["seed"])
    for name in ("tabpfn", "tabicl"):
        if args.model not in ("all", name):
            continue
        started = time.perf_counter()
        if name == "tabpfn":
            from tabpfn import TabPFNRegressor, TabPFNClassifier
            from tabpfn.constants import ModelVersion
            estimators = (TabPFNRegressor, TabPFNClassifier)
        else:
            from tabicl import TabICLRegressor, TabICLClassifier
            estimators = (TabICLRegressor, TabICLClassifier)
        result = {}
        for kind, cls, features, labels in zip(("regressor", "classifier"), estimators, (x, xc), (y, yc)):
            item = config["models"][f"{name}_{kind}"]
            path = ROOT / item["local"] / item["file"]
            settings = {"model_path": str(path), "device": "mps", "n_estimators": 4, "random_state": config["seed"]}
            if name == "tabpfn":
                estimator = cls.create_default_for_version(ModelVersion.V2, **settings)
            else:
                estimator = cls(allow_auto_download=False, use_amp=False, n_jobs=1, **settings)
            t0 = time.perf_counter()
            estimator.fit(features[:64], labels[:64])
            predicted = np.asarray(estimator.predict(features[64:]))
            if predicted.shape != (32,) or not np.isfinite(predicted).all():
                raise RuntimeError(f"Invalid {name}/{kind} output")
            result[kind] = {"requested_device": "mps", "fit_predict_seconds": time.perf_counter() - t0,
                "predictions": predicted.tolist(), "finite": True,
                "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                "score_on_synthetic_example": float(np.mean((predicted-labels[64:])**2)) if kind == "regressor" else float(np.mean(predicted==labels[64:]))}
            if kind == "classifier":
                probabilities = np.asarray(estimator.predict_proba(features[64:]))
                if probabilities.shape != (32, 2) or not np.isfinite(probabilities).all():
                    raise RuntimeError("Invalid local classification probabilities")
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                result[kind]["probabilities_normalized"] = True
            print(f"PASS - {name}/{kind} actual MPS predictions", flush=True)
            del estimator
            torch.mps.empty_cache()
        report["models"][name] = {"status": "passed", "elapsed_seconds": time.perf_counter()-started, **result}
        save_json(output / f"{name}.json", report)
    if args.model in ("all", "jevmlx"):
        from typing import Literal
        from pydantic import BaseModel, Field
        from jevmlx import decide

        class ExpertProbe(BaseModel):
            expert: Literal["MOLECULAR", "MATERIALS"] = Field(description="Select the expert explicitly supporting the stated elements and neutral molecular reaction task.")

        context = ("Task: neutral singlet gas-phase CHNO molecular reaction barrier. "
            "MOLECULAR supports H,C,N,O and gas-phase molecular reactions. "
            "MATERIALS supports only Fe and Ni bulk crystals; it does not support C,H,N,O. "
            "Choose a supported expert. This is a runtime probe, not a DFT accuracy assessment.")
        item = config["models"]["jevmlx"]
        model_path = str(ROOT / item["local"])
        results = []
        for repetition in range(2):
            t0 = time.perf_counter()
            decision = decide(ExpertProbe, context, model=model_path)
            field = decision.fields["expert"]
            results.append({"repetition": repetition, "wall_seconds": time.perf_counter()-t0,
                "expert": decision.value.expert, "probability": field.probability,
                "probability_margin": field.probability_margin,
                "expected_probe_answer": "MOLECULAR", "correct": decision.value.expert == "MOLECULAR"})
        report["models"]["jevmlx"] = {"status": "passed", "device": "Apple Metal via MLX",
            "results": results, "probability_scope": "restricted option score; not calibrated chemistry correctness"}
        save_json(output / "jevmlx.json", report)
        print("PASS - jevmlx actual local MLX decisions", flush=True)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "summary.json", report)
    print(json.dumps({k: v["status"] for k, v in report["models"].items()}), flush=True)


if __name__ == "__main__":
    main()
