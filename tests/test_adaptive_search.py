"""Guard endpoint-only selection, deterministic starts, budgets and cache identity."""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase import Atoms

from adaptive_ts_search import digest, execute_attempt, next_action, plan_for
from diagnose_ts_search import diagnose
from prepare_v05_reactions import diverse_selection, endpoint_candidates, excluded_ids
from ts_search import perturb_band
from ase.io import write
from validate_ts_dft_v05 import candidates, main as dft_main, ready_reactions


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "configs/v0.5.0.json").read_text())


def valid_result():
    return {"status": "completed", "stage": "finished", "elapsed_seconds": 10,
            "endpoints": [{"converged": True, "input_bonds_preserved": True}] * 2,
            "neb": {"converged": True, "fmax_eV_A": 0.04, "warmup_steps": 3, "climb_steps": 5},
            "saddle": {"converged": True, "fmax_eV_A": 0.001},
            "vibrations": {"imaginary_modes_above_threshold": 1},
            "irc": {"converged_and_connected": True},
            "mlip_verified": True, "saddle_irc_verified": True}


class AdaptiveSearchTests(unittest.TestCase):
    def test_manifest_excludes_entire_prior_reactions_and_roles_are_disjoint(self):
        partitions = json.loads((ROOT / "configs/v0.3.0_partitions.json").read_text())
        manifest = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
        prior = excluded_ids(partitions)
        ids = {item["sid"] for item in manifest["reactions"]}
        self.assertEqual(len(ids), 16)
        self.assertFalse(ids & prior)
        for role, count in (("development", 4), ("holdout", 12)):
            items = [item for item in manifest["reactions"] if item["role"] == role]
            self.assertEqual(len(items), count)
            self.assertEqual({item["kind"] for item in items}, {"formation", "cleavage", "rearrangement"})
        previous = json.loads((ROOT / "configs/v0.4.0_dft_selection.json").read_text())
        self.assertFalse({item["reaction"] for item in manifest["reactions"]}
                         & set(previous["reactions"]))

    def test_reader_accepts_endpoint_only_columns_and_handles_frame_sharding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoints.parquet"
            rows = [{"sid": sid, "fid": fid, "atomic_numbers": [1, 1],
                     "pos": [[0., 0., 0.], [0.7 if fid == 0 else 2., 0., 0.]]}
                    for sid, fid in ((1, 0), (1, 9), (2, 0), (3, 0), (3, 9))]
            pq.write_table(pa.Table.from_pylist(rows), path)
            pool, incomplete = endpoint_candidates(path, {3}, {"H2"})
            self.assertEqual([item["sid"] for item in pool], [1])
            self.assertEqual(incomplete, [2])
            self.assertEqual(pool[0]["kind"], "cleavage")

    def test_selection_is_seeded_and_covers_classes(self):
        pool = [{"sid": index, "kind": ("formation", "cleavage", "rearrangement")[index % 3],
                 "formula": str(index), "atoms": 5 + index, "formed": index % 3, "broken": 1}
                for index in range(20)]
        first = diverse_selection(pool, 4, 42)
        self.assertEqual(first, diverse_selection(pool, 4, 42))
        self.assertNotEqual(first, diverse_selection(pool, 4, 43))

    def test_perturbation_has_fixed_rms_and_preserves_endpoints_and_mapping(self):
        atoms = Atoms("CHNO", positions=np.arange(12).reshape(4, 3))
        images = [atoms.copy() for _ in range(10)]
        other = [atoms.copy() for _ in range(10)]
        different = [atoms.copy() for _ in range(10)]
        perturb_band(images, 42, 0.03)
        perturb_band(other, 42, 0.03)
        perturb_band(different, 43, 0.03)
        for left, right in zip(images, other):
            np.testing.assert_array_equal(left.positions, right.positions)
            np.testing.assert_array_equal(left.numbers, atoms.numbers)
        for image in (images[0], images[-1]):
            np.testing.assert_array_equal(image.positions, atoms.positions)
        for image in images[1:-1]:
            delta = image.positions - atoms.positions
            np.testing.assert_allclose(delta.mean(axis=0), 0, atol=1e-15)
            self.assertAlmostEqual(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))), 0.03)
        self.assertFalse(np.array_equal(images[1].positions, different[1].positions))

    def test_stage_diagnosis_and_recovery_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            original = valid_result()
            self.assertEqual(diagnose(original, directory, POLICY)["primary_failure"], "none")
            for failure, patch_value, action in (
                ("endpoint_failure", {"endpoints": []}, "stop"),
                ("saddle_failure", {"saddle": {"converged": False}}, "high_resolution_idpp"),
                ("frequency_failure", {"vibrations": {"imaginary_modes_above_threshold": 2}}, "high_resolution_idpp"),
                ("irc_wrong_endpoint", {"irc": {"branches": [{"converged": True, "minimum_converged": True}] * 2}}, "perturbed_idpp"),
                ("irc_unconverged", {"irc": {}}, "perturbed_idpp"),
                ("infrastructure_failure", {"error": "RuntimeError: CUDA out of memory"}, "stop")):
                result = {**original, "saddle_irc_verified": False, **patch_value}
                diagnosis = diagnose(result, directory, POLICY)
                self.assertEqual(diagnosis["primary_failure"], failure)
                self.assertEqual(diagnosis["recommended_action"], action)

    def test_improving_neb_uses_two_nonoverlapping_tail_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neb_climb.log"
            values = [0.2] * 60 + [0.1] * 20 + [0.08] * 20
            path.write_text("\n".join(f"FIRE: {index} 00:00:00 -1 {force}" for index, force in enumerate(values)))
            result = {**valid_result(), "saddle_irc_verified": False,
                      "neb": {"converged": False, "fmax_eV_A": 0.08}}
            diagnosis = diagnose(result, directory, POLICY)
            self.assertEqual(diagnosis["primary_failure"], "neb_improving")
            self.assertAlmostEqual(diagnosis["evidence"]["force_trend_ratio"], 0.8)
            result["neb"]["fmax_eV_A"] = 0.11
            self.assertEqual(diagnose(result, directory, POLICY)["primary_failure"], "neb_stagnation")

    def test_verified_saddle_stops_even_when_neb_budget_is_exhausted(self):
        record = {"saddle_irc_verified": True, "diagnosis": {"primary_failure": "none"}}
        for strategy in ("S0", "S1", "S2"):
            self.assertIsNone(next_action(strategy, [record], POLICY))

    def test_retry_budget_and_no_duplicate_continuations(self):
        records = []
        for _ in range(3):
            action = next_action("S2", records, POLICY)
            records.append({"plan": {"type": action}, "saddle_irc_verified": False,
                            "diagnosis": {"primary_failure": "neb_improving", "recommended_action": "continuation",
                                          "retry_allowed": True}})
        self.assertEqual([record["plan"]["type"] for record in records], ["idpp", "continuation", "perturbed_idpp"])
        self.assertIsNone(next_action("S2", records, POLICY))

    def test_fixed_sequence_and_infrastructure_stop(self):
        records = []
        for action in ("idpp", "perturbed_idpp", "high_resolution_idpp"):
            self.assertEqual(next_action("S1", records, POLICY), action)
            records.append({"saddle_irc_verified": False, "diagnosis": {"primary_failure": "irc_wrong_endpoint"}})
        self.assertIsNone(next_action("S1", records, POLICY))
        records[0]["diagnosis"]["primary_failure"] = "infrastructure_failure"
        for strategy in ("S0", "S1", "S2"):
            self.assertIsNone(next_action(strategy, records[:1], POLICY))

    def test_cache_reuses_identical_attempts_and_keeps_their_theoretical_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = {"sid": 1, "reaction": "rxn0001"}
            identity = {"name": "model", "sha256": "abc"}
            config = json.loads((ROOT / "configs/v0.4.0.json").read_text())
            plan = plan_for("idpp", item, POLICY)
            args = (item, None, root, config, identity, {}, {}, [], {"identity": {}, "statuses": []})
            with patch("adaptive_ts_search.search_one", return_value=valid_result()) as search, \
                    patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0):
                first = execute_attempt(*args, plan, POLICY)
                second = execute_attempt(*args, plan, POLICY)
                changed = execute_attempt(*args, plan_for("perturbed_idpp", item, POLICY), POLICY)
            self.assertEqual(search.call_count, 2)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["elapsed_seconds"], second["elapsed_seconds"])
            self.assertEqual(first["image_step_cost"], 80)
            self.assertNotEqual(first["cache_key"], changed["cache_key"])
            self.assertNotEqual(digest({"model": "a"}), digest({"model": "b"}))

    def test_continuation_identity_includes_parent_geometry_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_band.extxyz"
            path.write_text("parent geometry")
            parent = {"folder": directory, "plan": {"images": 14}, "cache_key": "parent"}
            plan = plan_for("continuation", {"sid": 1}, POLICY, parent)
            self.assertEqual(plan["images"], 14)
            self.assertEqual(plan["steps"], 250)
            path.write_text("changed geometry")
            self.assertNotEqual(plan["band_sha256"], plan_for("continuation", {"sid": 1}, POLICY, parent)["band_sha256"])

    def test_dft_preserves_nine_units_and_reuses_identical_candidates(self):
        manifest = {"dft_reactions": ["rxn1", "rxn2", "rxn3"],
                    "protocol": {"strategies": ["S0", "S1", "S2"]},
                    "reactions": [{"reaction": f"rxn{i}", "formula": "H2", "atomic_numbers": [1, 1]}
                                  for i in range(1, 4)]}
        config = {"electronic_state_verified": True, "charge": 0, "multiplicity": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for reaction in manifest["dft_reactions"]:
                folder = root / "attempts" / reaction
                folder.mkdir(parents=True)
                write(folder / "ts_candidate.extxyz", Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]]))
                for strategy in manifest["protocol"]["strategies"]:
                    strategy_folder = root / "holdout/mh1-finetuned" / reaction / strategy
                    strategy_folder.mkdir(parents=True)
                    (strategy_folder / "result.json").write_text(json.dumps({
                        "selected_folder": str(folder), "selected_attempt": reaction,
                        "mlip_verified": True, "saddle_irc_verified": True}))
            jobs = candidates(root, manifest, config)
            self.assertEqual(len(jobs), 3)
            selection = json.loads((root / "dft/selection.json").read_text())
            self.assertEqual(len(selection["strategy_units"]), 9)
            self.assertEqual(candidates(root, manifest, config), jobs)
            with self.assertRaises(ValueError):
                candidates(root, manifest, {**config, "electronic_state_verified": False})

    def test_partial_dft_candidates_leave_canonical_selection_and_final_identity_unchanged(self):
        manifest = {"dft_reactions": ["rxn1", "rxn2", "rxn3"],
                    "protocol": {"strategies": ["S0", "S1", "S2"]},
                    "reactions": [{"reaction": f"rxn{i}", "formula": "H2", "atomic_numbers": [1, 1]}
                                  for i in range(1, 4)]}
        config = {"electronic_state_verified": True, "charge": 0, "multiplicity": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for reaction in manifest["dft_reactions"]:
                folder = root / "attempts" / reaction
                folder.mkdir(parents=True)
                write(folder / "ts_candidate.extxyz", Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]]))
                for strategy in manifest["protocol"]["strategies"]:
                    strategy_folder = root / "holdout/mh1-finetuned" / reaction / strategy
                    strategy_folder.mkdir(parents=True)
                    (strategy_folder / "result.json").write_text(json.dumps({
                        "selected_folder": str(folder), "selected_attempt": reaction, "status": "completed",
                        "mlip_verified": True, "saddle_irc_verified": True}))
            partial_manifest = {**manifest, "dft_reactions": ["rxn1"]}
            partial = candidates(root, partial_manifest, config, save_selection=False)
            canonical = root / "dft/selection.json"
            self.assertFalse(canonical.exists())
            complete = candidates(root, manifest, config)
            self.assertEqual(len(complete), 3)
            self.assertTrue(all(complete[key] == job for key, job in partial.items()))
            saved = canonical.read_bytes()
            self.assertEqual(candidates(root, partial_manifest, config, save_selection=False), partial)
            self.assertEqual(canonical.read_bytes(), saved)
            self.assertEqual(len(json.loads(saved)["strategy_units"]), 9)

    def test_ready_only_preparation_waits_for_all_strategy_results_and_preserves_no_candidate(self):
        manifest = {"dft_reactions": ["rxn1", "rxn2", "rxn3"],
                    "protocol": {"strategies": ["S0", "S1", "S2"]},
                    "reactions": [{"reaction": f"rxn{i}", "formula": "H2", "atomic_numbers": [1, 1]}
                                  for i in range(1, 4)]}
        config = {"electronic_state_verified": True, "charge": 0, "multiplicity": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/v0.5.0"
            configs = root / "configs"
            configs.mkdir()
            (configs / "v0.5.0_reactions.json").write_text(json.dumps(manifest))
            (configs / "v0.5.0_dft.json").write_text(json.dumps(config))
            for reaction in ("rxn1", "rxn2"):
                for strategy in manifest["protocol"]["strategies"]:
                    folder = output / "holdout/mh1-finetuned" / reaction / strategy
                    folder.mkdir(parents=True)
                    (folder / "result.json").write_text(json.dumps({
                        "status": "running" if reaction == "rxn2" and strategy == "S2" else "completed",
                        "selected_folder": str(output / "attempts" / reaction), "selected_attempt": reaction,
                        "mlip_verified": False, "saddle_irc_verified": False}))
            self.assertEqual(ready_reactions(output, manifest), ["rxn1"])
            with patch("validate_ts_dft_v05.ROOT", root), \
                    patch("sys.argv", ["validate_ts_dft_v05.py", "--output", str(output),
                                       "--ready-only", "--prepare-only"]), redirect_stdout(io.StringIO()):
                dft_main()
            self.assertFalse((output / "dft/selection.json").exists())
            scheduling = json.loads((root / "results/coordination/dft-ready-selection.json").read_text())
            self.assertEqual(scheduling["ready_reactions"], ["rxn1"])
            self.assertEqual(scheduling["pending_reactions"], ["rxn2", "rxn3"])
            self.assertFalse(scheduling["canonical_selection_written"])
            self.assertTrue(scheduling["provisional"])
            self.assertIn("recorded_at_utc", scheduling)
            self.assertEqual(scheduling["unique_jobs"], {})
            self.assertEqual(scheduling["ready_search_strategy_units"], 3)
            for reaction in manifest["dft_reactions"]:
                for strategy in manifest["protocol"]["strategies"]:
                    folder = output / "holdout/mh1-finetuned" / reaction / strategy
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "result.json").write_text(json.dumps({
                        "status": "completed", "selected_folder": str(output / "attempts" / reaction),
                        "selected_attempt": reaction, "mlip_verified": False, "saddle_irc_verified": False}))
            with patch("validate_ts_dft_v05.ROOT", root), \
                    patch("sys.argv", ["validate_ts_dft_v05.py", "--output", str(output),
                                       "--ready-only", "--prepare-only"]), redirect_stdout(io.StringIO()):
                dft_main()
            selection = json.loads((output / "dft/selection.json").read_text())
            self.assertEqual(len(selection["strategy_units"]), 9)
            self.assertTrue(all(unit["status"] == "no_candidate" for unit in selection["strategy_units"]))
            scheduling = json.loads((root / "results/coordination/dft-ready-selection.json").read_text())
            self.assertTrue(scheduling["canonical_selection_written"])
            self.assertEqual(scheduling["pending_reactions"], [])

    def test_explicit_dft_retries_preserve_each_failure_and_keep_completed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/v0.5.0"
            configs = root / "configs"
            configs.mkdir()
            (configs / "v0.5.0_reactions.json").write_text(json.dumps({"dft_reactions": ["rxn1", "rxn2"]}))
            (configs / "v0.5.0_dft.json").write_text("{}")
            failed = output / "dft/rxn1/failed-job"
            completed = output / "dft/rxn2/completed-job"
            failed.mkdir(parents=True)
            completed.mkdir(parents=True)
            old_result = b'{"status":"failed","error":"first failure","dft_verified":false}\n'
            old_error = b"first traceback\nwith detail\n"
            completed_result = b'{"status":"completed","dft_verified":true}\n'
            (failed / "result.json").write_bytes(old_result)
            (failed / "error.txt").write_bytes(old_error)
            (completed / "result.json").write_bytes(completed_result)
            jobs = {"failed-key": {"reaction": "rxn1", "output_folder": str(failed)},
                    "completed-key": {"reaction": "rxn2", "output_folder": str(completed)}}

            def fail_again(output, item, config):
                self.assertEqual(item["reaction"], "rxn1")
                result = {"status": "failed", "error": "retry failure", "dft_verified": False}
                (failed / "result.json").write_text(json.dumps(result))
                (failed / "error.txt").write_text("retry traceback\n")
                return result

            with patch("validate_ts_dft_v05.ROOT", root), \
                    patch("validate_ts_dft_v05.candidates", return_value=jobs), \
                    patch("validate_ts_dft_v05.validate_one", side_effect=fail_again) as validate, \
                    redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["validate_ts_dft_v05.py", "--output", str(output)]):
                    dft_main()
                self.assertEqual(validate.call_count, 0)
                self.assertFalse((failed / "previous_attempts").exists())
                for _ in range(2):
                    with patch("sys.argv", ["validate_ts_dft_v05.py", "--output", str(output), "--retry-failed"]):
                        dft_main()
                self.assertEqual(validate.call_count, 2)
            archives = sorted((failed / "previous_attempts").iterdir())
            self.assertEqual(len(archives), 2)
            self.assertEqual((archives[0] / "result.json").read_bytes(), old_result)
            self.assertEqual((archives[0] / "error.txt").read_bytes(), old_error)
            self.assertEqual(json.loads((archives[1] / "result.json").read_text())["error"], "retry failure")
            self.assertEqual((archives[1] / "error.txt").read_text(), "retry traceback\n")
            self.assertEqual((completed / "result.json").read_bytes(), completed_result)
            self.assertFalse((completed / "previous_attempts").exists())


if __name__ == "__main__":
    unittest.main()
