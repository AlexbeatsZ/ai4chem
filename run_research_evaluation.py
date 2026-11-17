"""Evaluate the fixed protocol, select on validation, then open reserved final data."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/v0.3.0"


def run_benchmark(role, split, count, names, checkpoints):
    directory = OUTPUT / role
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "benchmark_transition1x.py", "--input", f"data/v0.3.0/{role}.h5",
               "--output", str(directory), "--split", split, "--count", str(count), "--models", *names]
    for name, path in checkpoints.items():
        command.extend(["--checkpoint", f"{name}={path}"])
    print(f"Starting {role} evaluation, {count} reactions, models {names}", flush=True)
    with (directory / "console.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (directory / "analysis.log").open("w") as log:
        subprocess.run([sys.executable, "analyze_benchmark.py", str(directory)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((directory / "analysis/analysis_summary.json").read_text())


def main():
    protocol = json.loads((ROOT / "configs/v0.3.0.json").read_text())
    partitions = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
    training = json.loads((OUTPUT / "training/manifest.json").read_text())
    if {entry["name"] for entry in training} != {entry["name"] for entry in protocol["experiments"]}:
        raise ValueError("All planned training experiments must complete before model selection")
    checkpoints = {entry["name"]: str(ROOT / entry["model"]) for entry in training}
    baseline = ["off23-small", "mh1-omol", "mh0-rgd1"]
    models = baseline + list(checkpoints)
    validation = run_benchmark("validation", "val", len(partitions["validation"]), models, checkpoints)
    ranked = sorted(validation["models"], key=lambda name: (
        validation["models"][name]["aggregate"]["barrier_abs_error_eV"],
        validation["models"][name]["aggregate"]["nonreactant_relative_energy_mae_eV"], name))
    selected = ranked[0]
    selection = {"criterion": protocol["selection"], "selected": selected,
                 "validation_ranking": [{"model": name, **validation["models"][name]["aggregate"]} for name in ranked],
                 "final_used_for_selection": False, "protocol": "configs/v0.3.0.json"}
    (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Validation-selected candidate {selected}, before final evaluation", flush=True)
    run_benchmark("development", "test", len(partitions["development"]), models, checkpoints)
    final_models = baseline + ([selected] if selected not in baseline else [])
    final_checkpoints = {selected: checkpoints[selected]} if selected in checkpoints else {}
    run_benchmark("final", "test", len(partitions["final"]), final_models, final_checkpoints)
    subprocess.run([sys.executable, "summarize_research.py"], cwd=ROOT, check=True)
    (OUTPUT / "evaluation_complete.json").write_text(json.dumps({
        "status": "PASS", "selected": selected, "development_reactions": len(partitions["development"]),
        "final_reactions": len(partitions["final"]), "validation_reactions": len(partitions["validation"]),
        "final_models": final_models, "selection_source": "validation only",
    }, indent=2) + "\n")
    print("PASS - validation selection, expanded development and independent final evaluation complete", flush=True)


if __name__ == "__main__":
    main()
