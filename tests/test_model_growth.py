"""Scientific role isolation and immutable nesting for coverage expansion."""

import copy
import fcntl
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_growth_data import excluded_roles, validate_partition
from prepare_transition1x import REVISION
from run_model_growth import valid_metrics, wait_for_cuda


class GrowthBoundaries(unittest.TestCase):
    def fixture(self):
        protocol = dict(source_revision=REVISION, reaction_counts=[4, 6])
        base = dict(training_ids=[1, 2], frozen_at_utc="baseline")
        partition = dict(protocol=protocol, additional_training_ids=[3, 4, 5, 6],
            training_ids=[1, 2, 3, 4, 5, 6], baseline_partition_frozen_at_utc="baseline",
            endpoint_records=[dict(sid=i) for i in range(1, 7)],
            datasets={"4": [1, 2, 3, 4], "6": [1, 2, 3, 4, 5, 6]})
        roles = {"validation": {7}, "meta_development": {8}, "holdout_train": {9},
                 "holdout_val": {10}, "historic_evaluation": {11}}
        return protocol, base, partition, roles

    def test_every_evaluation_role_is_excluded_even_from_the_smaller_dataset(self):
        protocol, base, original, roles = self.fixture()
        validate_partition(original, protocol, base, roles)
        for role, ids in roles.items():
            partition = copy.deepcopy(original)
            partition["training_ids"][2] = next(iter(ids))
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "evaluation role"):
                validate_partition(partition, protocol, base, roles)

    def test_replacing_old_training_reactions_and_nonnested_membership_are_rejected(self):
        protocol, base, original, roles = self.fixture()
        for mutation in (
            lambda p: p["datasets"]["4"].__setitem__(0, 5),
            lambda p: p["datasets"]["4"].__setitem__(3, 5),
            lambda p: p["additional_training_ids"].__setitem__(0, 1),
            lambda p: p.__setitem__("baseline_partition_frozen_at_utc", "changed"),
        ):
            partition = copy.deepcopy(original)
            mutation(partition)
            with self.assertRaises(ValueError):
                validate_partition(partition, protocol, base, roles)

    def test_incomplete_endpoint_records_and_changed_source_are_rejected(self):
        protocol, base, partition, roles = self.fixture()
        partition["endpoint_records"].pop()
        with self.assertRaisesRegex(ValueError, "Endpoint"):
            validate_partition(partition, protocol, base, roles)
        protocol, base, partition, roles = self.fixture()
        protocol["source_revision"] = "different"
        with self.assertRaisesRegex(ValueError, "source revision"):
            validate_partition(partition, protocol, base, roles)

    def test_historic_reactions_and_all_new_roles_remain_excluded(self):
        previous = dict(validation=[1], meta_development=[2], holdout_train=[3], holdout_val=[4])
        old = dict(development=[5], final=[6], validation=[7])
        roles = excluded_roles(previous, old, dict(reactions=[dict(sid=8)]))
        self.assertEqual(set().union(*roles.values()), set(range(1, 9)))

    def test_invalid_metrics_cannot_pass_selection_through_nan_comparisons(self):
        good = dict(barrier_mae_eV=.1, force_component_rmse_eV_per_A=.08,
                    ts_region_force_component_rmse_eV_per_A=.12)
        valid_metrics(good)
        for key in good:
            for value in (math.nan, math.inf, -.01):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    valid_metrics({**good, key: value})

    def test_real_cuda_locks_remain_exclusive_after_the_old_queue_drains(self):
        with tempfile.TemporaryDirectory(dir="/tmp/.agents") as directory:
            root = Path(directory)
            queue = root / "results/overnight-20261124/omen"
            queue.mkdir(parents=True)
            for version in ("v0.9.0", "v0.7.0"):
                (root / "results" / version).mkdir()
            path = queue / "owner.lock"
            old_owner = path.open("a")
            fcntl.flock(old_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed = []

            def drain(_):
                (queue / "state.json").write_text(json.dumps(dict(status="completed")))
                old_owner.close()

            with patch("run_model_growth.ROOT", root), patch("run_model_growth.time.sleep", drain):
                locks = wait_for_cuda(dict(predecessor_wait_hours=1), lambda stage, **kw: observed.append(stage))
            try:
                self.assertEqual(observed, ["waiting_for_existing_cuda_owner"])
                self.assertEqual(len(locks), 3)
                for owned in locks:
                    with Path(owned.name).open("a") as contender:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                for owned in locks:
                    owned.close()

    def test_unlocked_but_nonterminal_queue_cannot_admit_new_training(self):
        with tempfile.TemporaryDirectory(dir="/tmp/.agents") as directory:
            root = Path(directory)
            queue = root / "results/overnight-20261124/omen"
            queue.mkdir(parents=True)
            (queue / "state.json").write_text(json.dumps(dict(status="running")))
            for version in ("v0.9.0", "v0.7.0"):
                (root / "results" / version).mkdir()
            with patch("run_model_growth.ROOT", root), self.assertRaisesRegex(RuntimeError, "terminal"):
                wait_for_cuda(dict(predecessor_wait_hours=1), lambda *a, **kw: None)


if __name__ == "__main__":
    unittest.main()
