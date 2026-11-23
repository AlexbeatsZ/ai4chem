"""Keep missing and failed searches in the planned denominator and compare paired reactions."""

import json
from pathlib import Path
import tempfile
import unittest

from ase import Atoms
from ase.io import write

from summarize_ts_search_v05 import collect, endpoint_changes, group_summary, paired_comparisons


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveSummaryTests(unittest.TestCase):
    def test_all_planned_units_exist_before_any_computation(self):
        manifest = json.loads((ROOT / "configs/v0.5.0_reactions.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            rows, records, unique = collect(Path(directory), manifest, ["mh1-original", "mh1-finetuned"])
        self.assertEqual(len(rows), 96)
        self.assertEqual(len(records), 0)
        self.assertEqual(len(unique), 0)
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        group = [row for row in rows if (row["role"], row["model"], row["strategy"]) == ("holdout", "mh1-original", "S0")]
        summary = group_summary(group)
        self.assertEqual(summary["planned"], 12)
        self.assertEqual(summary["pending"], 12)
        self.assertEqual(summary["verified_saddles"], 0)

    def test_paired_exact_test_counts_discordant_reactions(self):
        rows = []
        for reaction in range(4):
            for strategy in ("S0", "S1", "S2"):
                passed = strategy != "S0"
                rows.append({"role": "holdout", "model": "model", "strategy": strategy,
                             "reaction": str(reaction), "status": "completed", "saddle_irc_verified": passed,
                             "mlip_verified": passed, "theoretical_seconds": 10 if passed else 5})
        comparisons = paired_comparisons(rows, ["model"])
        item = next(item for item in comparisons if item["role"] == "holdout"
                    and item["comparison"] == "S1-S0" and item["metric"] == "saddle_irc_verified")
        self.assertEqual((item["gained"], item["lost"], item["paired_reactions"]), (4, 0, 4))
        self.assertEqual(item["exact_two_sided_p"], 0.125)
        self.assertEqual(item["median_cost_difference_seconds"], 5)

    def test_hh_cutoff_diagnostic_waits_for_completed_shared_endpoints(self):
        positions = [[0, 0, 0], [0.74469, 0, 0]]
        manifest = {"reactions": [{"reaction": "rxn1", "role": "development", "atomic_numbers": [1, 1],
                                    "reactant_positions": positions, "product_positions": positions}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "endpoints/model/rxn1"
            folder.mkdir(parents=True)
            for side in ("reactant", "product"):
                write(folder / f"{side}_relaxed.extxyz", Atoms("H2", positions=[[0, 0, 0], [0.74270, 0, 0]]))
            self.assertEqual(endpoint_changes(root, manifest, ["model"]), [])
            (folder / "endpoints.json").write_text(json.dumps({"statuses": [{"converged": True}] * 2}))
            changes = endpoint_changes(root, manifest, ["model"])
            self.assertEqual(len(changes), 2)
            self.assertTrue(all(change["symbols"] == ["H", "H"] and change["both_endpoints_converged"] for change in changes))
            self.assertAlmostEqual(changes[0]["bond_cutoff_A"], 0.744)


if __name__ == "__main__":
    unittest.main()
