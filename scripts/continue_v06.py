"""One-shot handoff coordinator for the two frozen v0.6 pilot chains.

Observe existing jobs, export only terminal Gaussian evidence, independently
assess geometry, and write a review report. No retries of calculations, policy
changes, installations, commits, publication, cron or scheduled tasks.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

from assess_dft_chain import assess
from check_mac_qc import save_json

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def write_report(state, folder, deliverables):
    lines = ["# v0.6.0-pilot 后续诊断", "", f"更新于 {utc_now()}", "",
             "本轮分析事后选择候选，并单独记录引擎校准。结果沿用相应版本的协议和任务身份。", ""]
    for name, entry in state["jobs"].items():
        lines += [f"## {name}", "", f"当前状态为 {entry['status']}。", ""]
        if "assessment" in entry:
            result = entry["assessment"]
            lines += [f"完整 DFT 验证为 {result['dft_verified']}，预期端点连接为 {result['expected_connectivity']}。", "",
                      f"TS 负频率为 {[f for f in result['ts']['frequencies_cm1'] if f < 0]} cm⁻¹。", ""]
            if "electronic_barrier_eV" in result:
                lines += [f"电子势垒为 {result['electronic_barrier_eV']:.8f} eV。", ""]
            if "gaussian_comparison" in result:
                lines += ["Gaussian 对照结果如下。", "", "```json", json.dumps(result["gaussian_comparison"], indent=2), "```", ""]
        if entry.get("error"):
            lines += ["原始异常已保留。", "", "```text", entry["error"], "```", ""]
    lines += ["## 交接", "", f"原始日志与判定保留在 results/v0.6.0-pilot，运行状态位于 {folder}。",
              "读取根 AGENTS.md 和 docs/design/v0.6.0-pilot.md 后继续。后续策略改变需要独立评价。", ""]
    content = "\n".join(lines)
    (folder/"report.md").write_text(content)
    if deliverables:
        deliverables.mkdir(parents=True, exist_ok=True)
        (deliverables/"ai4chem-next-investigation.md").write_text(content)
        save_json(deliverables/"ai4chem-next-investigation.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-ssh", required=True)
    parser.add_argument("--worker-root", required=True)
    parser.add_argument("--worker-job", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/v0.6.0-pilot/protocol.json")
    parser.add_argument("--mac-reaction", default="rxn0666")
    parser.add_argument("--gaussian-reaction", default="rxn9456")
    parser.add_argument("--received-folder", type=Path, default=ROOT/"results/v0.6.0-pilot/gaussian/rxn9456-return-r1")
    parser.add_argument("--mac-folder", type=Path, default=ROOT/"results/v0.6.0-pilot/mac/rxn0666-r1")
    parser.add_argument("--output", type=Path, default=ROOT/"results/coordination/v06")
    parser.add_argument("--deliverables", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-hours", type=float, default=8)
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.max_hours <= 0:
        parser.error("Use at least 30 seconds between polls and a positive finite deadline")
    # Paths deliberately use the worker's space-free project root. Structured
    # data goes through JSON/files, never nested PowerShell command quoting.
    for value in (args.worker_ssh, args.worker_root, args.worker_job):
        if any(c.isspace() or c in "\"';&|`$" for c in value):
            parser.error("Worker transport/path must be a simple space-free value")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    mac_key = args.mac_reaction+"_mac"
    gaussian_key = args.gaussian_reaction+"_gaussian"
    state = {"status": "running", "pid": os.getpid(), "started_at_utc": utc_now(),
             "protocol_frozen_at_utc": protocol["frozen_at_utc"],
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "jobs": {mac_key: {"status": "waiting"}, gaussian_key: {"status": "waiting"}}}
    started = time.monotonic()

    def remote(script, *extra):
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                 args.worker_ssh, "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                 "-File", args.worker_root+"/scripts/"+script, "-JobDirectory", args.worker_job, *extra],
                                capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout.lstrip("\ufeff")

    while any(j["status"] not in ("assessed", "failed") for j in state["jobs"].values()):
        if time.monotonic()-started > args.max_hours*3600:
            state["status"] = "stopped_deadline"
            break
        mac = state["jobs"][mac_key]
        if mac["status"] not in ("assessed", "failed"):
            try:
                source = json.loads((args.mac_folder/"summary.json").read_text())
                mac.update(status="running", stage=source["stage"], source_pid=source["pid"])
                if source["raw_chain_completed"]:
                    result = assess(protocol, args.mac_reaction, args.mac_folder, "mac")
                    save_json(args.mac_folder/"assessment.json", result)
                    mac.update(status="assessed", assessment=result)
                elif source["status"] == "failed" or not alive(source["pid"]):
                    mac.update(status="failed", error=source.get("error", "Recorded Mac process no longer exists; inspect raw evidence"))
            except Exception:
                mac.update(status="failed", error=traceback.format_exc())
        gaussian = state["jobs"][gaussian_key]
        if gaussian["status"] not in ("assessed", "failed"):
            try:
                observed = json.loads(remote("status_gaussian_chain.ps1"))
                gaussian.update(status="running", stage=observed["state"]["stage"], last_observation=observed)
                gaussian.pop("connection_error", None)
                source = observed["state"]
                if source["status"] != "running" and not observed["owner_alive"]:
                    exported = json.loads(remote("export_gaussian_chain.ps1"))
                    archive = args.output/"worker-return.zip"
                    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                    args.worker_ssh+":"+exported["archive"].replace("\\", "/"), str(archive)],
                                   capture_output=True, text=True, timeout=180, check=True)
                    received = args.received_folder
                    received.mkdir(parents=True, exist_ok=False)
                    with zipfile.ZipFile(archive) as bundle:
                        for name in bundle.namelist():
                            if Path(name).is_absolute() or ".." in Path(name).parts:
                                raise ValueError("Unsafe result archive member")
                        bundle.extractall(received)
                    archive.unlink()
                    remote("export_gaussian_chain.ps1", "-RemoveArchive")
                    gaussian["received_folder"] = str(received)
                    if source["raw_chain_completed"]:
                        result = assess(protocol, args.gaussian_reaction, received, "gaussian")
                        save_json(received/"assessment.json", result)
                        gaussian.update(status="assessed", assessment=result)
                    else:
                        gaussian.update(status="failed", error=source.get("error", "Worker failed; raw logs collected"))
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError, RuntimeError):
                gaussian["connection_error"] = traceback.format_exc()
            except Exception:
                gaussian.update(status="failed", error=traceback.format_exc())
        state.update(updated_at_utc=utc_now(), elapsed_seconds=time.monotonic()-started)
        save_json(args.output/"state.json", state)
        write_report(state, args.output, args.deliverables)
        if any(j["status"] not in ("assessed", "failed") for j in state["jobs"].values()):
            time.sleep(args.poll_seconds)
    if state["status"] == "running":
        state["status"] = "ready_for_review"
    state["finished_at_utc"] = utc_now()
    save_json(args.output/"state.json", state)
    write_report(state, args.output, args.deliverables)
    print(json.dumps({"status":state["status"], "jobs":{k:v["status"] for k,v in state["jobs"].items()}}), flush=True)


if __name__ == "__main__":
    main()
