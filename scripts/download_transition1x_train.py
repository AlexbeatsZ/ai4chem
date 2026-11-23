"""Fetch all frame-sharded training files at the already-pinned source revision."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_file import download_file
from prepare_transition1x import REVISION


def download_all(workers=3):
    api = f"https://huggingface.co/api/datasets/nimashoghi/transition1x/tree/{REVISION}/data"
    with urllib.request.urlopen(api, timeout=30) as response:
        listing = json.load(response)
    files = []
    for index in range(13):
        remote = f"data/train-{index:05d}-of-00013.parquet"
        entry = next(item for item in listing if item["path"] == remote)
        files.append({"index": index, "source": remote, "bytes": entry["size"],
                      "local": f"data/raw/transition1x_train_{index:02d}.parquet"})
    state_path = ROOT / "results/v0.7.0/download.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"revision": REVISION, "source": api, "total_bytes": sum(x["bytes"] for x in files),
             "status": "running", "files": files, "completed": []}

    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    save()
    base = f"https://huggingface.co/datasets/nimashoghi/transition1x/resolve/{REVISION}/"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_file, base + x["source"], ROOT / x["local"], x["bytes"]): x
                   for x in files}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception as error:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                save()
                raise
            state["completed"].append(item["index"])
            state["completed"].sort()
            save()
            print(f"Training shards ready: {len(state['completed'])}/13", flush=True)
    state["status"] = "completed"
    save()
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    download_all(args.workers)
