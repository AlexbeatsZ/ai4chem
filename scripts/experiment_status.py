"""Read saved experiment progress without loading models or changing results."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def experiment_status(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "configs/v0.5.0_reactions.json").read_text())
    frozen = json.loads((root / "configs/v0.5.0_frozen.json").read_text())
    groups = {}
    active_attempts = []
    for role in ("development", "holdout"):
        reactions = [x["reaction"] for x in manifest["reactions"] if x["role"] == role]
        for model in frozen["models"]:
            for strategy in manifest["protocol"]["strategies"]:
                records = []
                pending = []
                for reaction in reactions:
                    path = root / "results/v0.5.0" / role / model / reaction / strategy / "result.json"
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record.get("status") == "completed":
                            records.append(record)
                            continue
                    pending.append(reaction)
                groups[f"{role}/{model}/{strategy}"] = {
                    "planned": len(reactions), "completed": len(records),
                    "verified_saddles": sum(x.get("saddle_irc_verified") is True for x in records),
                    "strict_pipeline": sum(x.get("mlip_verified") is True for x in records),
                    "failures": dict(Counter(x.get("primary_failure", "unknown") for x in records)),
                    "pending_reactions": pending,
                }
    for path in sorted((root / "results/v0.5.0/attempts").glob("*/*/*/identity.json")):
        if not path.with_name("result.json").exists():
            identity = json.loads(path.read_text())
            active_attempts.append({"folder": str(path.parent.relative_to(root)),
                                    "model": identity["model"]["name"],
                                    "reaction": identity["reaction"],
                                    "initialization": identity["plan"]["type"]})
    dft_root = root / "results/v0.5.0/dft"
    selection_path = dft_root / "selection.json"
    dft = {"planned_strategy_units": 9, "selection_prepared": selection_path.exists(), "jobs": []}
    provisional_path = root / "results/coordination/dft-ready-selection.json"
    readable_selection = selection_path if selection_path.exists() else provisional_path
    dft["provisional_selection"] = not selection_path.exists() and provisional_path.exists()
    if readable_selection.exists():
        selection = json.loads(readable_selection.read_text())
        if selection_path.exists():
            dft["planned_strategy_units"] = len(selection["strategy_units"])
        for key, item in selection["unique_jobs"].items():
            folder = dft_root / item["reaction"] / key[:16]
            path = folder / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            dft["jobs"].append({"reaction": item["reaction"], "job": key,
                                "status": result.get("status", "pending"),
                                "stage": result.get("stage"),
                                "dft_verified": result.get("dft_verified", False)})
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "frozen_at_utc": frozen["freeze_time_utc"],
            "planned_strategy_units": sum(x["planned"] for x in groups.values()),
            "completed_strategy_units": sum(x["completed"] for x in groups.values()),
            "groups": groups, "attempts_without_terminal_result": active_attempts, "dft": dft}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(experiment_status(args.root), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
