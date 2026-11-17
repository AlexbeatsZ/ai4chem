"""Fetch official MH checkpoints and inspect their actual serialized heads."""

import json
import subprocess
import sys
from pathlib import Path

import torch
import mace.calculators  # Initialize the checkpoint loader compatibility settings.

from download_file import download_file


CHECKPOINTS = {
    "mace-mh-1": {"size": 59208139, "requested_head": "omol"},
    "mace-mh-0": {"size": 39921174, "requested_head": "rgd1_b3lyp"},
}
BASE_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1"


def main():
    root = Path(__file__).resolve().parent
    reports = []
    for name, spec in CHECKPOINTS.items():
        url = f"{BASE_URL}/{name}.model"
        path = download_file(url, root / "models" / f"{name}.model", spec["size"])
        model = torch.load(path, map_location="cpu", weights_only=False)
        heads = list(model.heads)
        head = spec["requested_head"]
        if head not in heads:
            raise ValueError(f"{name} has no requested head {head}, available {heads}")
        report = {"model": name, "url": url, "size_bytes": path.stat().st_size,
                  "heads": heads, "selected_head": head}
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
        del model
        subprocess.run(
            [str(Path(sys.executable).with_name("mace_select_head")), "--list_heads", str(path)],
            check=True,
        )
    output = root / "results" / "v0.2.0"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_heads.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
