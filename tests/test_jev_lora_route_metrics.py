from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
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

    def test_test_evaluation_reads_test_split_and_preserves_point_identity(self):
        torch = ModuleType('torch')
        torch.bfloat16, torch.float32 = 'bf16', 'fp32'
        data = ModuleType('wev.data')
        inference = ModuleType('wev.inference')
        used_splits = []

        def split_file(directory, split):
            used_splits.append(split)
            return directory / f'{split}.jsonl'

        data.split_file = split_file
        data.load_items = lambda *args, **kwargs: ([{
            'meta': [{'type': 'choice', 'keys': ['LLM', 'SLM']}],
            'enc': {'labels': [1]}, '_meta': {'task_id': 'test-task', 'k': 3},
        }], 0)
        model = SimpleNamespace(tokenizer=object(), max_state=4096, max_branch=2048,
                                model=SimpleNamespace(probs=lambda _: [[[0.2, 0.8]]]))
        inference.load = lambda *args, **kwargs: model
        modules = {'torch': torch, 'wev': ModuleType('wev'), 'wev.data': data, 'wev.inference': inference}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            out = Path(tmp) / 'evaluation'
            result = metrics.evaluate_split(run='frozen-checkpoint', data_dir=Path(tmp),
                                            out_dir=out, thresholds=[0.65], split='test')
            prediction = json.loads((out / 'predictions.jsonl').read_text())
        self.assertEqual(used_splits, ['test'])
        self.assertEqual(result['split'], 'test')
        self.assertEqual(prediction['task_id'], 'test-task')
        self.assertEqual(prediction['k'], 3)
        self.assertEqual(prediction['gold_route'], 'SLM')
        self.assertEqual(prediction['p_slm'], 0.8)

    def test_cli_forwards_test_split_without_loading_gpu(self):
        summary = {'split': 'test'}
        with patch.object(metrics, 'evaluate_split', return_value=summary) as evaluate:
            result = metrics.main(['--run', 'checkpoint', '--data', 'test-dir',
                                   '--out', 'new-eval', '--split', 'test', '--thresholds', '0.65'])
        self.assertEqual(result, 0)
        self.assertEqual(evaluate.call_args.kwargs['split'], 'test')


if __name__ == "__main__":
    unittest.main()
