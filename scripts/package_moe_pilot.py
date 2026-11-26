"""Package completed v0.11 pilot models and evidence without environments/data."""

import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = ROOT/"results/v0.11.0"
    soft = json.loads((result/"summary.json").read_text())
    residual = json.loads((result/"residual_pilot/summary.json").read_text())
    if soft["status"] != "completed" or residual["status"] != "completed" or soft["runtime_acceptance"]["status"] != "passed":
        raise RuntimeError("Require completed training and real calculator acceptance")
    if soft["holdout_labels_evaluated"] or residual["holdout_labels_evaluated"]:
        raise ValueError("This pilot package must not contain prospective evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/"ai4chem-v0.11.0-moe-pilot-models.zip"
    if destination.exists():
        raise FileExistsError("Do not overwrite an existing release package")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for member in soft["protocol"]["members"]:
            path = ROOT/member["model_file"]
            if path.stat().st_size != member["size_bytes"]:
                raise ValueError("Frozen weight changed before packaging")
            archive.write(path, member["model_file"])
        for path in (result/"models").glob("*.json"):
            archive.write(path, f"models/v0.11.0/{path.name}")
        for path in (result/"residual_pilot").glob("*.pt"):
            archive.write(path, f"models/v0.11.0/{path.name}")
        for path in (ROOT/"soft_mixture.py", ROOT/"residual_moe.py"):
            archive.write(path, path.name)
        for path in (ROOT/"configs/v0.11.0.json", ROOT/"configs/residual-moe-pilot-20261125.json"):
            archive.write(path, str(path.relative_to(ROOT)))
        for path in (result/"summary.json", result/"runtime_acceptance.json", result/"frozen_protocol.json",
                     result/"residual_interface/acceptance.json", result/"residual_pilot/summary.json",
                     result/"residual_pilot/frozen_protocol.json"):
            archive.write(path, str(path.relative_to(ROOT)))
        for path in sorted((result/"residual_pilot").glob("*-training.json")):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(ROOT/"docs/reports/v0.11.0-moe/README.md", "README.md")
        archive.write(ROOT/"docs/design/soft-mixture.md", "DESIGN.md")
    with zipfile.ZipFile(destination) as archive:
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError(f"Corrupt release entry: {invalid}")
        count = len(archive.infolist())
    metadata = {"name": destination.name, "size_bytes": destination.stat().st_size, "entries": count,
        "scope": "Frozen external gates, two actual residual-adapter pilot weights and three required MACE members; development evidence only",
        "holdout_labels_evaluated": False}
    (args.output/"package-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
