"""Independently inspect saved v0.5 artifacts without loading a potential.

Only terminal attempts referenced by completed strategy records are inspected.
An integrity pass neither establishes matrix completeness nor scientific success.
The only write is the explicitly requested JSON output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import ase
from ase import units
from ase.io import read
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.base = self.root / "results" / "v0.5.0"
        self.issues = []
        self.checks = Counter()
        self.inputs = 0
        self.attempts = {}
        self.strategies = []
        self.skipped = []
        self.policy = {}
        self.frozen = {}
        self.items = {}

    def check(self, condition, code, path, detail=""):
        self.checks[code] += 1
        if not bool(condition):
            self.issues.append({"code": code, "path": str(path), "detail": str(detail)})
        return bool(condition)

    def load(self, path):
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))

    def finite(self, value, path, label):
        a = np.asarray(value, dtype=float)
        self.check(np.isfinite(a).all(), "finite_" + label, path)
        return a

    def equal(self, a, b, path, label, atol=1e-7, rtol=0):
        a, b = np.asarray(a), np.asarray(b)
        ok = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)
        self.check(ok, label, path, f"shapes {a.shape} / {b.shape}")

    def relocated(self, saved):
        # Stored paths are OMEN WSL paths. Anchor on the experiment subdirectory.
        text = str(saved).replace("\\", "/")
        anchor = "results/v0.5.0/"
        if anchor not in text:
            raise ValueError(f"Unrecognized artifact path: {saved}")
        candidate = (self.base / text.split(anchor, 1)[1]).resolve()
        if not candidate.is_relative_to(self.base.resolve()):
            raise ValueError("Artifact reference escapes experiment root")
        return candidate

    def atoms(self, path, numbers, results=False, many=False):
        images = read(path, index=":" if many else -1)
        if not many:
            images = [images]
        for atom in images:
            self.equal(atom.numbers, numbers, path, "atom_mapping", atol=0)
            self.finite(atom.positions, path, "positions")
            if results:
                self.finite(atom.get_potential_energy(), path, "energy")
                forces = self.finite(atom.get_forces(), path, "forces")
                self.check(forces.shape == (len(numbers), 3), "forces_shape", path)
        return images if many else images[0]

    def guard(self, path, function):
        try:
            return function()
        except Exception as exc:
            self.issues.append({"code": "artifact_read_or_schema", "path": str(path),
                                "detail": f"{type(exc).__name__}: {exc}"})
            return None

    def input_check(self, item):
        nums = item["atomic_numbers"]
        self.check(len(nums) == item["atoms"], "input_atom_count", item["reaction"])
        for name in ("reactant", "product"):
            pos = self.finite(item[name + "_positions"], item["reaction"], "manifest_positions")
            self.check(pos.shape == (len(nums), 3), "input_positions_shape", item["reaction"])
            path = self.base / "inputs" / item["reaction"] / (name + ".xyz")
            if path.exists():
                atom = self.atoms(path, nums)
                self.equal(atom.positions, pos, path, "input_matches_manifest")
            else:
                self.skipped.append({"path": str(path), "reason": "input file absent from snapshot"})
        self.inputs += 1

    def endpoint_check(self, model, reaction):
        folder = self.base / "endpoints" / model / reaction
        path = folder / "endpoints.json"
        endpoint = self.load(path)
        item = self.items[reaction]
        ident = endpoint["identity"]
        self.check(ident["model"] == self.frozen["models"][model], "endpoint_model_identity", path)
        self.check(ident["software"] == self.frozen["software"], "endpoint_software_identity", path)
        self.check(ident["implementation"] == self.frozen["implementation"], "endpoint_code_identity", path)
        self.equal(ident["numbers"], item["atomic_numbers"], path, "endpoint_input_mapping", atol=0)
        self.equal(ident["reactant"], item["reactant_positions"], path, "endpoint_reactant_input", atol=0)
        self.equal(ident["product"], item["product_positions"], path, "endpoint_product_input", atol=0)
        self.check(len(endpoint["statuses"]) == 2, "endpoint_status_count", path)
        for name in ("reactant", "product"):
            self.atoms(folder / (name + "_relaxed.extxyz"), item["atomic_numbers"], results=True)
        self.check(endpoint["elapsed_seconds"] >= 0, "endpoint_nonnegative_time", path)
        return endpoint

    def hessian_check(self, folder, result, numbers):
        path = folder / "ts_hessian.npz"
        with np.load(path, allow_pickle=False) as saved:
            hessian = self.finite(saved["hessian_eV_A2"], path, "hessian")
            freq = self.finite(saved["frequencies_cm1"], path, "frequencies")
            modes = self.finite(saved["cartesian_modes"], path, "modes")
        self.check(hessian.shape == (3 * len(numbers), 3 * len(numbers)), "hessian_shape", path)
        self.equal(hessian, hessian.T, path, "hessian_symmetric", atol=1e-10)
        self.equal(freq, result["vibrations"]["frequencies_cm1"], path, "frequency_json_npz", atol=1e-10)
        self.check(modes.shape == (3 * len(numbers), len(freq)), "mode_shape", path)
        threshold = self.frozen["baseline"]["imaginary_threshold_cm1"]
        self.check(int(np.count_nonzero(freq < -threshold)) == result["vibrations"]["imaginary_modes_above_threshold"],
                   "saved_imaginary_count", path)
        # Independent linear algebra projection; no project calculation module is imported.
        ts = self.atoms(folder / "ts_candidate.extxyz", numbers, results=True)
        centered = ts.positions - ts.get_center_of_mass()
        sqrt_mass = np.sqrt(ts.get_masses())[:, None]
        rigid = np.asarray([(np.tile(axis, (len(ts), 1)) * sqrt_mass).ravel() for axis in np.eye(3)] +
                           [(np.cross(np.tile(axis, (len(ts), 1)), centered) * sqrt_mass).ravel() for axis in np.eye(3)])
        _, singular, vh = np.linalg.svd(rigid, full_matrices=True)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-8))
        basis = vh[rank:].T
        weights = np.repeat(ts.get_masses() ** -0.5, 3)
        eigen = np.linalg.eigvalsh(basis.T @ (weights[:, None] * hessian * weights[None, :]) @ basis)
        conversion = units._hbar * 1e10 / np.sqrt(units._e * units._amu) / units.invcm
        reconstructed = np.sign(eigen) * np.sqrt(np.abs(eigen)) * conversion
        self.equal(freq, reconstructed, path, "frequency_from_saved_hessian", atol=1e-4, rtol=1e-7)

    def log_steps(self, path):
        if not path.exists():
            return 0
        steps = [int(line.split()[1]) for line in path.read_text().splitlines()
                 if line.startswith("FIRE:")]
        return max(steps, default=0)

    def attempt_check(self, folder, expected_key=None):
        if folder in self.attempts:
            if expected_key:
                self.check(self.attempts[folder]["cache_key"] == expected_key,
                           "strategy_attempt_reference", folder)
            return self.attempts[folder]
        ident = self.load(folder / "identity.json")
        result = self.load(folder / "result.json")
        plan, settings = ident["plan"], ident["settings"]
        key = digest(ident)
        self.check(folder.name == key[:16], "attempt_directory_identity", folder)
        self.check(result["cache_key"] == key, "attempt_result_identity", folder)
        if expected_key:
            self.check(key == expected_key, "strategy_attempt_reference", folder)
        model, reaction = ident["model"]["name"], ident["reaction"]
        item = self.items[reaction]
        self.check(ident["model"] == self.frozen["models"][model], "attempt_model_identity", folder)
        self.check(ident["software"] == self.frozen["software"], "attempt_software_identity", folder)
        self.check(ident["implementation"] == self.frozen["implementation"], "attempt_code_identity", folder)
        self.check(result["reaction"] == reaction, "attempt_reaction", folder)
        self.check(result["initialization"] == plan, "attempt_saved_plan", folder)
        self.check(result["status"] in ("completed", "failed"), "attempt_terminal", folder)
        expected_settings = {**self.frozen["baseline"], "images": plan["images"]}
        if plan["type"] == "continuation":
            expected_settings.update(neb_warmup_steps=0, neb_climb_steps=self.policy["continuation_steps"])
        self.check(settings == expected_settings, "attempt_settings_frozen", folder)
        endpoint = self.endpoint_check(model, reaction)
        self.check(ident["endpoints"] == endpoint["identity"], "attempt_endpoint_identity", folder)
        nums = item["atomic_numbers"]
        initial_path, final_path = folder / "initial_band.extxyz", folder / "final_band.extxyz"
        initial = self.atoms(initial_path, nums, many=True) if initial_path.exists() else None
        final = self.atoms(final_path, nums, results=True, many=True) if final_path.exists() else None
        if result["status"] == "completed":
            self.check(initial is not None and final is not None, "completed_band_files", folder)
            self.hessian_check(folder, result, nums)
        if initial:
            self.check(len(initial) == plan["images"], "initial_image_count", folder)
        if final:
            self.check(len(final) == plan["images"], "final_image_count", folder)
            energies = np.array([image.get_potential_energy() for image in final])
            self.equal(energies - energies[0], result["neb"]["relative_energies_eV"], folder,
                       "saved_band_relative_energies", atol=1e-7)
            self.equal(max(energies) - energies[0], result["neb"]["sampled_barrier_eV"], folder,
                       "saved_band_barrier", atol=1e-7)
        if initial and final:
            for a, b in ((initial[0], final[0]), (initial[-1], final[-1])):
                self.equal(a.get_all_distances(), b.get_all_distances(), folder,
                           "band_endpoint_geometry_unchanged", atol=1e-6)
        if initial and plan["type"] == "continuation":
            parent_folder = self.relocated(plan["band"]).parent
            parent_ident = self.load(parent_folder / "identity.json")
            self.check(digest(parent_ident) == plan["parent"], "continuation_parent_identity", folder)
            parent_path = parent_folder / "final_band.extxyz"
            self.check(hashlib.sha256(parent_path.read_bytes()).hexdigest() == plan["band_sha256"],
                       "continuation_saved_band_identity", folder)
            parent_band = self.atoms(parent_path, nums, many=True)
            self.check(len(initial) == len(parent_band), "continuation_image_count", folder)
            for a, b in zip(initial, parent_band):
                self.equal(a.positions, b.positions, folder, "continuation_initial_matches_parent", atol=1e-7)
        if initial and plan["type"] == "perturbed_idpp":
            self.check(plan["seed"] == self.policy["seed"] + item["sid"], "perturbation_seed", folder)
            self.check(plan["rms_A"] == self.policy["perturbation_rms_A"], "perturbation_declared_rms", folder)
            # Standard IDPP and perturbed IDPP share the same aligned endpoints.
            candidates = []
            for identity_path in folder.parent.glob("*/identity.json"):
                other = self.load(identity_path)
                if other["plan"]["type"] == "idpp" and other["plan"]["images"] == plan["images"]:
                    candidates.append(identity_path.parent / "initial_band.extxyz")
            if not candidates:
                self.skipped.append({"path": str(folder), "reason": "standard initial band unavailable for perturbation comparison"})
            else:
                standard = self.atoms(candidates[0], nums, many=True)
                self.equal(initial[0].positions, standard[0].positions, folder, "perturbation_reactant_unchanged", atol=1e-8)
                self.equal(initial[-1].positions, standard[-1].positions, folder, "perturbation_product_unchanged", atol=1e-8)
                displacement = np.asarray([a.positions - b.positions for a, b in zip(initial[1:-1], standard[1:-1])])
                rms = np.sqrt(np.mean(np.sum(displacement ** 2, axis=2), axis=1))
                self.equal(rms, np.full(len(rms), plan["rms_A"]), folder, "perturbation_actual_rms", atol=3e-8)
                self.equal(displacement.mean(axis=1), np.zeros((len(rms), 3)), folder,
                           "perturbation_zero_translation", atol=2e-8)
        neb = result.get("neb", {})
        warmup = neb.get("warmup_steps", self.log_steps(folder / "neb_warmup.log"))
        climb = neb.get("climb_steps", self.log_steps(folder / "neb_climb.log"))
        self.check(0 <= warmup <= settings["neb_warmup_steps"], "warmup_budget", folder)
        self.check(0 <= climb <= settings["neb_climb_steps"], "climb_budget", folder)
        elapsed = float(result["elapsed_seconds"])
        stages = result.get("stage_seconds", {})
        self.check(elapsed >= 0 and all(v >= 0 and np.isfinite(v) for v in stages.values()), "attempt_nonnegative_time", folder)
        self.check(sum(stages.values()) <= elapsed + 1e-5, "attempt_stage_time_sum", folder)
        info = {"folder": str(folder), "cache_key": key, "model": model, "reaction": reaction,
                "type": plan["type"], "status": result["status"], "image_step_cost": plan["images"] * (warmup + climb),
                "elapsed_seconds": elapsed, "saddle_irc_verified": result.get("saddle_irc_verified", False),
                "mlip_verified": result.get("mlip_verified", False)}
        self.attempts[folder] = info
        return info

    def strategy_check(self, path):
        unit = self.load(path)
        if unit.get("status") != "completed":
            self.skipped.append({"path": str(path), "reason": "strategy has no completed terminal record"})
            return
        model, reaction, strategy, role = unit["model"], unit["reaction"], unit["strategy"], unit["role"]
        self.check(path.relative_to(self.base).parts[:-1] == (role, model, reaction, strategy), "strategy_directory_fields", path)
        self.check(role == self.items[reaction]["role"], "strategy_role", path)
        records = unit["attempts"]
        self.check(1 <= len(records) <= (1 if strategy == "S0" else self.policy["maximum_attempts"]), "strategy_attempt_budget", path)
        self.check(len({r["plan"]["type"] for r in records}) == len(records), "strategy_distinct_initializations", path)
        self.check(records[0]["plan"]["type"] == "idpp", "strategy_starts_idpp", path)
        if strategy == "S1":
            self.check([r["plan"]["type"] for r in records] == ["idpp", "perturbed_idpp", "high_resolution_idpp"][:len(records)],
                       "fixed_strategy_order", path)
        cost, elapsed, infos = 0, 0.0, []
        for record in records:
            folder = self.relocated(record["folder"])
            info = self.attempt_check(folder, record["cache_key"])
            self.check(record["plan"]["type"] == info["type"], "strategy_plan_reference", path)
            self.check(record["image_step_cost"] == info["image_step_cost"], "strategy_record_image_cost", path)
            self.equal(record["elapsed_seconds"], info["elapsed_seconds"], path, "strategy_record_time", atol=1e-7)
            for flag in ("saddle_irc_verified", "mlip_verified"):
                self.check(record[flag] == info[flag], "strategy_record_" + flag, path)
            cost += info["image_step_cost"]
            elapsed += info["elapsed_seconds"]
            infos.append(info)
        self.check(unit["image_step_cost"] == cost, "strategy_cost_sum", path)
        endpoint = self.load(self.base / "endpoints" / model / reaction / "endpoints.json")
        self.equal(unit["endpoint_seconds"], endpoint["elapsed_seconds"], path, "strategy_shared_endpoint_time", atol=1e-7)
        self.equal(unit["theoretical_seconds"], endpoint["elapsed_seconds"] + elapsed, path, "strategy_time_sum", atol=1e-6)
        selected = next((x for x in infos if x["saddle_irc_verified"]), infos[-1])
        self.check(unit["selected_attempt"] == selected["cache_key"], "strategy_selection", path)
        self.check(self.relocated(unit["selected_folder"]) == Path(selected["folder"]), "strategy_selected_folder", path)
        for flag in ("saddle_irc_verified", "mlip_verified"):
            self.check(unit[flag] == selected[flag], "strategy_selected_" + flag, path)
        self.strategies.append({"role": role, "model": model, "reaction": reaction, "strategy": strategy,
                                "attempts": len(records), "image_step_cost": cost,
                                "saddle_irc_verified": unit["saddle_irc_verified"]})

    def run(self):
        self.frozen = self.load(self.root / "configs/v0.5.0_frozen.json")
        self.policy = self.frozen["policy"]
        manifest_path = self.root / "configs/v0.5.0_reactions.json"
        manifest = self.load(manifest_path)
        self.check(digest(manifest) == self.frozen["manifest_sha256"], "frozen_manifest_identity", manifest_path)
        self.check(self.load(self.root / "configs/v0.5.0.json") == self.policy, "frozen_policy_unchanged", self.root)
        self.check(self.load(self.root / self.policy["baseline_config"]) == self.frozen["baseline"], "frozen_baseline_unchanged", self.root)
        self.items = {x["reaction"]: x for x in manifest["reactions"]}
        self.check(len(self.items) == len(manifest["reactions"]), "unique_input_reactions", manifest_path)
        for item in self.items.values():
            self.guard(item["reaction"], lambda item=item: self.input_check(item))
        for role in ("development", "holdout"):
            for path in sorted((self.base / role).glob("*/rxn*/S*/result.json")):
                self.guard(path, lambda path=path: self.strategy_check(path))
        expected = len(self.items) * len(self.frozen["models"]) * len(self.policy["strategies"])
        covered = Counter(x["role"] for x in self.strategies)
        return {"audit_schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__, "ase": ase.__version__,
                "root": str(self.root), "scope": "terminal attempts referenced by completed saved strategy units only",
                "interpretation": "Artifact consistency only; does not establish complete evaluation or scientific success. No potential or DFT calculations run.",
                "model_files_loaded": False, "source_artifacts_modified": False,
                "integrity_passed": not self.issues, "matrix_complete": len(self.strategies) == expected,
                "expected_strategy_units": expected, "audited_strategy_units": len(self.strategies),
                "audited_units_by_role": dict(covered), "audited_unique_attempts": len(self.attempts),
                "manifest_inputs_checked": self.inputs, "check_counts": dict(self.checks),
                "issues": self.issues, "skipped_checks": self.skipped,
                "strategy_units": self.strategies, "attempts": list(self.attempts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="project or relocated snapshot root containing configs/ and results/")
    parser.add_argument("--output", type=Path, required=True, help="only file the auditor will write")
    args = parser.parse_args()
    output = args.output.resolve()
    audit = Audit(args.root)
    if output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
        if not isinstance(previous, dict) or previous.get("audit_schema_version") != 1:
            parser.error("output exists and is not a previous audit JSON; source artifacts are read-only")
    if output.is_relative_to(audit.root / "configs"):
        parser.error("output cannot be written into source configuration directory")
    try:
        report = audit.run()
    except Exception as exc:
        report = {"audit_schema_version": 1, "integrity_passed": False, "matrix_complete": False,
                  "issues": [{"code": "audit_setup", "detail": f"{type(exc).__name__}: {exc}"}],
                  "scope": "audit setup failed; no scientific conclusion"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("integrity_passed", "matrix_complete", "audited_strategy_units", "audited_unique_attempts")}))
    print(f"Audit JSON: {output}")
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
