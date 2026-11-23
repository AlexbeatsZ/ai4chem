"""Run frozen single-start, fixed multi-start and bounded failure-aware policies."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from ase.io import read, write
import numpy as np

from diagnose_ts_search import diagnose, force_history
from prepare_v05_reactions import ROOT, load_inputs, prepare
from ts_search import relax_endpoint, save_json, search_one


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def job_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".job.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def environment():
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "mace-torch", "ase", "sella", "jax", "numpy", "scipy")}
    return {"python": platform.python_version(), "packages": versions,
            "gpu": subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                text=True).strip(),
            "host": "OMEN Windows / WSL Ubuntu-24.04", "platform": platform.platform()}


def implementation_identity():
    return {name: file_digest(ROOT / name) for name in (
        "ts_search.py", "mace_neb.py", "adaptive_ts_search.py", "diagnose_ts_search.py",
        "prepare_v05_reactions.py")}


def model_identity(name, config):
    spec = config["models"][name]
    return {"name": name, **spec, "sha256": file_digest(ROOT / spec["path"])}


def plan_for(action, item, policy, parent=None):
    plan = {"type": action, "images": 10}
    if action == "perturbed_idpp":
        plan.update(seed=policy["seed"] + item["sid"], rms_A=policy["perturbation_rms_A"])
    elif action == "high_resolution_idpp":
        plan["images"] = policy["high_resolution_images"]
    elif action == "continuation":
        if parent is None:
            raise ValueError("Continuation needs a saved parent attempt")
        band = Path(parent["folder"]) / "final_band.extxyz"
        plan.update(images=parent["plan"]["images"], parent=parent["cache_key"],
                    band=str(band), band_sha256=file_digest(band), steps=policy["continuation_steps"])
    elif action != "idpp":
        raise ValueError(f"Unknown search initialization {action}")
    return plan


def next_action(strategy, records, policy):
    if not records:
        return "idpp"
    last = records[-1]
    diagnosis = last["diagnosis"]
    if (last["saddle_irc_verified"] or len(records) >= policy["maximum_attempts"]
            or diagnosis["primary_failure"] == "infrastructure_failure"):
        return None
    if strategy == "S0":
        return None
    if strategy == "S1":
        return ["idpp", "perturbed_idpp", "high_resolution_idpp"][len(records)]
    if not diagnosis["retry_allowed"]:
        return None
    used = {record["plan"]["type"] for record in records}
    action = diagnosis["recommended_action"]
    if action not in used:
        return action
    # Exhaust each distinct recovery at most once. Continuation has a 250-step total cap.
    for alternative in ("perturbed_idpp", "high_resolution_idpp"):
        if alternative not in used:
            return alternative
    return None


def cached_endpoints(item, calc, folder, config, identity, software, code):
    expected = {"model": identity, "software": software, "implementation": code,
                "endpoint_parameters": {
        key: config[key] for key in ("endpoint_fmax_eV_A", "endpoint_steps")},
        "numbers": item["atomic_numbers"], "reactant": item["reactant_positions"],
        "product": item["product_positions"]}
    with job_lock(folder):
        path = folder / "endpoints.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != expected:
                raise ValueError("Cached endpoint identity differs")
        else:
            start = time.monotonic()
            statuses = []
            for name, atoms in zip(("reactant", "product"), item["endpoints"]):
                _, status = relax_endpoint(atoms, calc, folder, name, config)
                statuses.append(status)
            record = {"identity": expected, "statuses": statuses,
                      "elapsed_seconds": time.monotonic() - start}
            save_json(path, record)
        endpoints = [read(folder / f"{name}_relaxed.extxyz") for name in ("reactant", "product")]
        for atoms in endpoints:
            np.testing.assert_array_equal(atoms.numbers, item["atomic_numbers"])
            if not np.isfinite(atoms.positions).all():
                raise ValueError("Non-finite cached endpoint positions")
    return endpoints, record


def execute_attempt(item, calc, output, config, identity, software, code, endpoints, endpoint_record, plan, policy):
    settings = {**config, "images": plan["images"]}
    if plan["type"] == "continuation":
        settings.update(neb_warmup_steps=0, neb_climb_steps=policy["continuation_steps"])
    expected = {"model": identity, "reaction": item["reaction"], "plan": plan,
                "settings": settings, "software": software, "implementation": code,
                "endpoints": endpoint_record["identity"]}
    key = digest(expected)
    folder = output / "attempts" / identity["name"] / item["reaction"] / key[:16]
    with job_lock(folder):
        identity_path = folder / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != expected:
            raise ValueError("Attempt cache collision or configuration mismatch")
        save_json(identity_path, expected)
        path = folder / "result.json"
        cache_hit = path.exists()
        if cache_hit:
            result = json.loads(path.read_text())
        else:
            print(f"Running {identity['name']} {item['reaction']} {plan['type']}", flush=True)
            import torch
            torch.cuda.reset_peak_memory_stats()
            result = search_one(item, calc, folder, settings,
                                prepared_endpoints=(endpoints, endpoint_record["statuses"]), initialization=plan)
            result.update(cache_key=key, initialization=plan,
                          peak_gpu_allocated_MiB=torch.cuda.max_memory_allocated() / 2 ** 20)
            save_json(path, result)
        diagnosis = diagnose(result, folder, policy)
        save_json(folder / "diagnosis.json", diagnosis)
    neb = result.get("neb", {})
    warmup_steps = neb.get("warmup_steps", max(0, len(force_history(folder / "neb_warmup.log")) - 1))
    climb_steps = neb.get("climb_steps", max(0, len(force_history(folder / "neb_climb.log")) - 1))
    cost = plan["images"] * (warmup_steps + climb_steps)
    record = {"cache_key": key, "folder": str(folder), "plan": plan, "cache_hit": cache_hit,
              "elapsed_seconds": result["elapsed_seconds"], "stage_seconds": result.get("stage_seconds", {}),
              "image_step_cost": cost, "neb_converged": neb.get("converged", False),
              "saddle_irc_verified": result.get("saddle_irc_verified", False),
              "mlip_verified": result.get("mlip_verified", False), "diagnosis": diagnosis}
    print(f"Outcome {identity['name']} {item['reaction']} {plan['type']} "
          f"saddle={record['saddle_irc_verified']} {diagnosis['primary_failure']} "
          f"seconds={record['elapsed_seconds']:.1f} cached={cache_hit}", flush=True)
    return record


def run_strategy(strategy, item, calc, output, config, identity, software, code, endpoints, endpoint_record, policy):
    folder = output / item["role"] / identity["name"] / item["reaction"] / strategy
    with job_lock(folder):
        records = []
        action = next_action(strategy, records, policy)
        while action is not None:
            plan = plan_for(action, item, policy, records[-1] if records else None)
            record = execute_attempt(item, calc, output, config, identity, software, code,
                                     endpoints, endpoint_record, plan, policy)
            records.append(record)
            action = next_action(strategy, records, policy)
            save_json(folder / "policy_trace.json", {"strategy": strategy, "attempts": records,
                                                      "next_action": action})
        successful = next((record for record in records if record["saddle_irc_verified"]), None)
        selected = successful or records[-1]
        result = {"model": identity["name"], "reaction": item["reaction"], "role": item["role"],
                  "strategy": strategy, "attempts": records, "selected_attempt": selected["cache_key"],
                  "selected_folder": selected["folder"], "status": "completed",
                  "saddle_irc_verified": selected["saddle_irc_verified"],
                  "mlip_verified": selected["mlip_verified"],
                  "neb_converged": any(record["neb_converged"] for record in records),
                  "endpoint_seconds": endpoint_record["elapsed_seconds"],
                  "theoretical_seconds": endpoint_record["elapsed_seconds"] + sum(record["elapsed_seconds"] for record in records),
                  "image_step_cost": sum(record["image_step_cost"] for record in records),
                  "primary_failure": selected["diagnosis"]["primary_failure"]}
        save_json(folder / "result.json", result)
        return result


def freeze(output, policy, config, manifest, software, code):
    for item in manifest["reactions"]:
        if item["role"] != "development":
            continue
        for name in config["models"]:
            for strategy in policy["strategies"]:
                path = output / "development" / name / item["reaction"] / strategy / "result.json"
                if not path.exists() or json.loads(path.read_text())["status"] != "completed":
                    raise RuntimeError("Finish every development unit before freezing holdout policy")
    frozen = {"policy": policy, "baseline": config, "manifest_sha256": digest(manifest),
              "models": {name: model_identity(name, config) for name in config["models"]},
              "software": software, "implementation": code,
              "freeze_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = ROOT / "configs/v0.5.0_frozen.json"
    if path.exists():
        previous = json.loads(path.read_text())
        frozen["freeze_time_utc"] = previous["freeze_time_utc"]
        if frozen != previous:
            raise ValueError("The holdout protocol is already frozen with different content")
    else:
        save_json(path, frozen)
    print("Frozen development-reviewed policy before holdout execution", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("development", "holdout"), default="development")
    parser.add_argument("--output", type=Path, default=Path("results/v0.5.0"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--reactions", nargs="+")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    policy = json.loads((ROOT / "configs/v0.5.0.json").read_text())
    config = json.loads((ROOT / policy["baseline_config"]).read_text())
    manifest = prepare(policy, args.output)
    code = implementation_identity()
    software_path = args.output / "environment.json"
    software = environment()
    if software_path.exists() and json.loads(software_path.read_text()) != software:
        raise ValueError("Runtime environment changed, use a separately versioned experiment")
    save_json(software_path, software)
    if args.freeze:
        freeze(args.output, policy, config, manifest, software, code)
        return
    if args.role == "holdout":
        frozen = json.loads((ROOT / "configs/v0.5.0_frozen.json").read_text())
        current = {"policy": policy, "baseline": config, "manifest_sha256": digest(manifest),
                   "models": {name: model_identity(name, config) for name in config["models"]},
                   "software": software, "implementation": code}
        if current != {key: value for key, value in frozen.items() if key != "freeze_time_utc"}:
            raise ValueError("Holdout execution differs from the frozen protocol")
    if args.models and set(args.models) - set(config["models"]):
        parser.error("Unknown model")
    items = load_inputs(manifest, args.output, args.role)
    if args.reactions and set(args.reactions) - {item["reaction"] for item in items}:
        parser.error("Reaction is outside the chosen role")
    from mace.calculators import MACECalculator
    for name in args.models or config["models"]:
        identity = model_identity(name, config)
        calc = MACECalculator(model_paths=identity["path"], head=identity["head"],
                              device=config["device"], default_dtype=config["dtype"])
        for item in items:
            if args.reactions and item["reaction"] not in args.reactions:
                continue
            endpoint_folder = args.output / "endpoints" / name / item["reaction"]
            endpoints, record = cached_endpoints(item, calc, endpoint_folder, config, identity, software, code)
            for strategy in policy["strategies"]:
                run_strategy(strategy, item, calc, args.output, config, identity, software, code,
                             endpoints, record, policy)
        del calc


if __name__ == "__main__":
    main()
