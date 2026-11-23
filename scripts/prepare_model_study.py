"""Prepare only frozen train and meta-development labels on CPU."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_study_data import freeze_roles, prepare_meta, prepare_training

if __name__ == "__main__":
    protocol = json.loads((ROOT / "configs/v0.7.0.json").read_text())
    partition = freeze_roles(protocol)
    prepare_training(partition)
    prepare_meta(partition)
    print("PASS - complete-path train/meta datasets prepared; holdouts unopened", flush=True)
