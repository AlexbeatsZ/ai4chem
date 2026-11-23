"""Attach to the OMEN search, then run the unchanged DFT and acceptance commands."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiment_status import experiment_status


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_pids(root, script, required_argument=None):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().split("\0")
            if (argv and Path(argv[0]).name.startswith("python")
                    and script in argv and (required_argument is None or required_argument in argv)
                    and (entry / "cwd").resolve() == root):
                matches.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dft-jobs", type=int, default=2)
    args = parser.parse_args()
    if not os.environ.get("WSL_DISTRO_NAME") or not sys.platform.startswith("linux"):
        parser.error("Execution belongs to the existing OMEN WSL environment; use experiment_status.py elsewhere")
    if args.dft_jobs < 1:
        parser.error("--dft-jobs must be positive")
    root = args.root.resolve()
    os.chdir(root)
    runtime = root / "results/coordination"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="4", JAX_PLATFORMS="cpu", TMPDIR="/tmp/.agents")
    os.environ["PYTHONPATH"] = str(root)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    state = {"host": "OMEN", "coordinator_pid": os.getpid(), "started_at_utc": timestamp(),
             "status": "running", "stage": "attach_search", "root": str(root)}

    def update(**changes):
        state.update(changes, updated_at_utc=timestamp(), progress=experiment_status(root))
        atomic_json(runtime / "omen-state.json", state)

    def event(message):
        record = {"time_utc": timestamp(), "host": "OMEN", "message": message}
        with (runtime / "omen-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{record['time_utc']} {message}", flush=True)

    def run(stage, command):
        log_path = runtime / f"{stage}.log"
        event(f"Starting {stage}: {' '.join(command)}")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{timestamp()} {' '.join(command)}\n")
            stream.flush()
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=process.pid, command=command, log=str(log_path))
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update()
            update(child_pid=None, last_exit_code=code)
        if code:
            raise RuntimeError(f"{stage} failed with exit {code}; inspect {log_path}")
        event(f"Completed {stage}")

    with (runtime / "omen-coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An OMEN coordinator is already running")
        try:
            event("Attached to existing frozen experiment; scientific failures are retained")
            while (pids := task_pids(root, "adaptive_ts_search.py", "holdout")):
                update(stage="attach_search", attached_search_pids=pids)
                time.sleep(30)
            update(attached_search_pids=[])
            if state["progress"]["completed_strategy_units"] != 96:
                run("holdout_resume", [sys.executable, "-u", "adaptive_ts_search.py", "--role", "holdout"])
            if experiment_status(root)["completed_strategy_units"] != 96:
                raise RuntimeError("The complete 96-unit matrix is required before DFT")
            run("regression", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            while (pids := task_pids(root, "validate_ts_dft_v05.py")):
                update(stage="wait_ready_dft", attached_dft_pids=pids)
                time.sleep(30)
            update(attached_dft_pids=[])
            run("dft", [sys.executable, "-u", "validate_ts_dft_v05.py", "--jobs", str(args.dft_jobs)])
            run("report", [sys.executable, "summarize_ts_search_v05.py"])
            run("dependencies", [sys.executable, "-m", "pip", "check"])
            run("cuda_smoke", [sys.executable, "smoke_test.py"])
            run("verification", [sys.executable, "scripts/verify_v05_results.py"])
            update(status="ready_for_review", stage="review_and_release", command=None)
            event("Scientific pipeline and acceptance checks finished; review plots and publish release")
        except BaseException as error:
            update(status="stopped", error=f"{type(error).__name__}: {error}")
            event(state["error"])
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
