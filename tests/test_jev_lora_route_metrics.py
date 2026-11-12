from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "jev_lora" / "evaluate_route_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_route_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(metrics)


class RouteMetricTests(unittest.TestCase):
    def test_auc_and_average_precision_use_slm_as_positive_class(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.35, 0.8]

        self.assertAlmostEqual(metrics.roc_auc(labels, probabilities), 0.75)
        self.assertAlmostEqual(metrics.average_precision(labels, probabilities), 5 / 6)

    def test_threshold_metrics_report_precision_recall_and_predicted_offload_rate(self):
        result = metrics.threshold_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)

        self.assertEqual(result, {
            "threshold": 0.5,
            "precision": 1.0,
            "recall": 0.5,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 1,
            "true_negative": 2,
            "predicted_slm": 1,
            "predicted_llm": 3,
            "gold_slm": 2,
            "gold_llm": 2,
            "sample_count": 4,
        })

    def test_threshold_table_avoids_unrequested_derived_metrics(self):
        result = metrics.threshold_metrics([0, 1], [0.2, 0.8], 0.5)

        self.assertNotIn("f1", result)
        self.assertNotIn("accuracy", result)
        self.assertNotIn("specificity", result)
        self.assertNotIn("offload_rate", result)

    def test_tied_scores_receive_average_ranks(self):
        self.assertAlmostEqual(metrics.roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(metrics.average_precision([0, 1], [0.5, 0.5]), 0.5)

    def test_auc_and_ap_are_null_when_one_class_is_absent(self):
        self.assertIsNone(metrics.roc_auc([0, 0], [0.2, 0.9]))
        self.assertIsNone(metrics.average_precision([0, 0], [0.2, 0.9]))

    def test_threshold_requires_a_finite_probability_cutoff(self):
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                metrics.threshold_metrics([0, 1], [0.1, 0.9], threshold)


if __name__ == "__main__":
    unittest.main()
