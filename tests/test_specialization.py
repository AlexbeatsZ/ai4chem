"""Scientific gates: identity-free routing and equal total training budgets."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
import warnings
import numpy as np
from specialization_gate import fit_gate, predict_gate
from run_specialization import validate_budgets, training_capacity


class SpecializationTests(unittest.TestCase):
    def records(self):
        return [{"atomic_numbers":[1,1], "reactant_positions":[[0,0,0],[r,0,0]],
                 "product_positions":[[0,0,0],[r+.1,0,0]], "sid":i}
                for i,r in enumerate([.60,.63,.66,.69,1.60,1.63,1.66,1.69])]

    def test_gate_is_balanced_and_predicts_without_identity_or_reference_labels(self):
        records = self.records()
        gate, trained = fit_gate(records, 12)
        self.assertEqual(np.bincount(trained).tolist(), [4,4])
        poisoned = [{**row, "sid":999, "reference_barrier":1000., "REF_energy":-1e20} for row in records]
        predicted, _ = predict_gate(poisoned,gate)
        np.testing.assert_array_equal(predicted, trained)

    def test_geometry_outlier_uses_fallback_heuristic(self):
        gate,_ = fit_gate(self.records(),12)
        outside = self.records()[0]
        outside["reactant_positions"][1][0] = 500.
        outside["product_positions"][1][0] = 600.
        _,accepted = predict_gate([outside],gate)
        self.assertFalse(accepted[0])

    def test_identical_geometries_cannot_be_partitioned_by_reaction_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError,"identical boundary"):
                fit_gate([deepcopy(self.records()[0]) for _ in range(8)],12)

    def test_shared_budget_rejects_an_extra_epoch_in_only_one_system(self):
        protocol = json.loads((Path(__file__).resolve().parents[1]/"configs/v0.8.0.json").read_text())
        self.assertEqual(set(validate_budgets(protocol).values()),{40960})
        protocol["experiments"][0]["training_overrides"] = {"epochs":41}
        with self.assertRaisesRegex(ValueError,"equal-update"):
            validate_budgets(protocol)

    def test_search_reservations_bound_training_and_failures_block_dispatch(self):
        execution = {"cpu_thread_budget":12,"search_threads":3,
                     "training_threads":4,"max_training_workers":3}
        for completed, expected in ((0,0),(1,1),(2,2),(3,3)):
            stages = {str(i):"completed" if i < completed else "searching" for i in range(3)}
            self.assertEqual(training_capacity(stages,execution),expected)
        self.assertEqual(training_capacity({"a":"missing","b":"completed","c":"completed"},execution),2)
        self.assertEqual(training_capacity({"a":"failed","b":"completed","c":"completed"},execution),0)


if __name__ == "__main__":
    unittest.main()
