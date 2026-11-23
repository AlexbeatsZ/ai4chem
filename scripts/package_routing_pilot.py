"""Package the completed decision pilot without unpublished training or weights."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    record = json.loads((ROOT / "docs/reports/v0.7.0-routing/summary.json").read_text())
    if record["status"] != "complete" or record["pending_methods"]:
        raise ValueError("All five routing methods must finish before pilot packaging")
    if record["prospective_holdout_opened"]:
        raise ValueError("This pilot must not open prospective holdouts")
    files = [ROOT / "configs/decision-models.json", ROOT / "configs/v0.7.0.json",
        ROOT / "configs/v0.7.0_partitions.json", ROOT / "docs/design/decision-routing.md",
        ROOT / "docs/design/model-improvement.md"]
    files += list((ROOT / "docs/reports/v0.7.0-routing").glob("*"))
    for folder in ("environments/decision-models", "environments/mac-mace"):
        files += [ROOT / folder / "pixi.toml", ROOT / folder / "pixi.lock"]
    for folder, names in {
        "results/v0.7.0/meta_development_mac": ["reaction_metrics.csv", "image_predictions.csv", "benchmark_summary.json"],
        "results/v0.7.0/router": ["headroom.json", "complementarity.json", "complementarity.png", "complementarity.pdf", "summary.json", "selected_errors.csv", "ridge.json", "extratrees.json", "tabpfn.json", "tabicl.json", "jevmlx.json"],
        "results/v0.7.0/decision-smoke": ["summary.json"],
        "results/v0.7.0/validation-cpu-probe": ["cross_device_verification.json"],
    }.items():
        files += [ROOT / folder / name for name in names]
    directory = ROOT / "results/releases/v0.7.0-routing-pilot"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "ai4chem-v0.7.0-routing-pilot.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("SCOPE.txt", "Completed local decision-model pilot on 128 meta-development reactions.\nNot the completed v0.7 MACE training study or prospective test.\nWeights are not redistributed; use the pinned original public repositories and their licenses.\nSource code is in the corresponding Git release archive.\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Pilot ZIP CRC failed")
        entries = len(archive.namelist())
    print(json.dumps({"artifact": str(output), "bytes": output.stat().st_size, "entries": entries,
                      "scope": "routing pilot only", "all_methods_complete": True}))


if __name__ == "__main__":
    main()
