"""Export only terminal strategy units and their referenced artifacts for offline analysis."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile


def export_snapshot(root, destination):
    root = root.resolve()
    results = root / "results/v0.5.0"
    files = set()
    records = []

    def include(path):
        if path.is_file():
            if not path.name.endswith((".lock", ".tmp")) and not path.name.startswith("Gau-"):
                path.relative_to(root)
                files.add(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    include(child)

    for role in ("development", "holdout"):
        for path in sorted((results / role).glob("*/*/*/result.json")):
            record = json.loads(path.read_text())
            if record.get("status") != "completed":
                continue
            records.append(record)
            include(path.parent)
            include(results / "endpoints" / record["model"] / record["reaction"])
            include(results / "inputs" / record["reaction"])
            for attempt in record["attempts"]:
                folder = root / attempt["folder"]
                saved = json.loads((folder / "result.json").read_text())
                if saved.get("status") not in {"completed", "failed"}:
                    raise ValueError(f"Referenced attempt has no terminal state: {folder}")
                include(folder)
    for path in (root / "configs").glob("*.json"):
        include(path)
    for name in ("environment.json", "pilot_verification.json", "legacy_diagnosis.json"):
        include(results / name)
    include(results / "baseline_regression")
    selection_path = results / "dft/selection.json"
    provisional_path = root / "results/coordination/dft-ready-selection.json"
    readable_selection = selection_path if selection_path.exists() else provisional_path
    if readable_selection.exists():
        include(readable_selection)
        selection = json.loads(readable_selection.read_text())
        for key, item in selection["unique_jobs"].items():
            folder = results / "dft" / item["reaction"] / key[:16]
            result_path = folder / "result.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("status") in {"completed", "failed"}:
                include(folder)
    manifest = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "terminal strategy units and referenced files only; not a final release",
                "completed_strategy_units": len(records), "source": str(root),
                "files": [str(path.relative_to(root)) for path in sorted(files)]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("results/coordination/snapshot-manifest.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    return {"archive": str(destination), "bytes": destination.stat().st_size,
            "files": len(files), "completed_strategy_units": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_snapshot(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
