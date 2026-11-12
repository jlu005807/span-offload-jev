from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


if importlib.util.find_spec("yaml") is None:
    yaml_compat = types.ModuleType("yaml")
    yaml_compat.safe_dump = lambda value, **kwargs: json.dumps(value)
    yaml_compat.safe_load = json.loads
    sys.modules["yaml"] = yaml_compat

compare = importlib.import_module("experiments.compare_routers")


class LoraThresholdProfileTests(unittest.TestCase):
    def setUp(self):
        self.old_profile = os.environ.get("JEV_COMPARE_PROFILE")

    def tearDown(self):
        if self.old_profile is None:
            os.environ.pop("JEV_COMPARE_PROFILE", None)
        else:
            os.environ["JEV_COMPARE_PROFILE"] = self.old_profile

    def test_lora_profiles_use_v1_and_exact_threshold(self):
        expected = {
            "lora-v1-t50": ("jev_lora_v1_t50", 0.50),
            "lora-v1-t55": ("jev_lora_v1_t55", 0.55),
            "lora-v1-t60": ("jev_lora_v1_t60", 0.60),
            "lora-v1-t65": ("jev_lora_v1_t65", 0.65),
            "lora-v1-t70": ("jev_lora_v1_t70", 0.70),
            "lora-v1-t75": ("jev_lora_v1_t75", 0.75),
            "lora-v1-t80": ("jev_lora_v1_t80", 0.80),
        }
        for profile, (group, threshold) in expected.items():
            with self.subTest(profile=profile):
                os.environ["JEV_COMPARE_PROFILE"] = profile
                self.assertEqual(compare.active_groups(), (group,))
                config = compare.build_configs(_config())[group]
                self.assertEqual(config["runner"]["jev"]["prompt_version"], "v1")
                self.assertEqual(config["runner"]["jev"]["threshold"], threshold)

    def test_phase_one_run_requires_all_56_tasks_to_be_official_dev(self):
        os.environ["JEV_COMPARE_PROFILE"] = "lora-v1-t50"
        calls = []
        tasks = [f"dev-task-{index}" for index in range(56)]
        splits = {task: "dev" for task in tasks}
        splits[tasks[0]] = "train"

        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "56 official dev"):
            compare.run_comparison(
                _config(), tasks, Path(temp) / "bad-split",
                execute=lambda *args: calls.append(args), health=lambda _: None,
                task_splits=splits,
            )
        self.assertEqual(calls, [])

    def test_phase_one_run_records_56_dev_rows_for_one_threshold(self):
        os.environ["JEV_COMPARE_PROFILE"] = "lora-v1-t50"
        tasks = [f"dev-task-{index}" for index in range(56)]
        calls = []

        def execute(group, config_path, task, output, experiment, split):
            calls.append((group, task, split))
            return _valid_result(task, 0.50)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            report = compare.run_comparison(
                _config(), tasks, output, execute=execute, health=lambda _: None,
                task_splits={task: "dev" for task in tasks}, smoke_task=tasks[0],
            )
            rows = json.loads((output / "results.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())

        self.assertEqual(len(calls), 57)
        self.assertEqual(len(rows), 56)
        self.assertTrue(report["complete"])
        self.assertTrue(report["development_experiment"])
        self.assertEqual(manifest["prompt_version"], "v1")
        self.assertEqual(manifest["task_ids"], tasks)
        self.assertEqual(manifest["groups"], ["jev_lora_v1_t50"])
        self.assertEqual(rows[0]["experiment_group_id"], "LoRA-V1-T50")


def _config():
    return {
        "models": {"large": {"model_id": "large"}, "small": {"model_id": "small"}},
        "agent": {"step_limit": 40},
        "runner": {
            "m4_memory": {"enabled": True, "embedding_mode": "m1_online"},
            "l2_add4": {"enabled": True},
            "jev": {"enabled": True, "input_mode": "combined", "threshold": 0.8},
        },
    }


def _valid_result(task_id, threshold):
    group = "jev_lora_v1_t50"
    return {
        "instance_id": task_id,
        "task_success": True,
        "framework_model_calls_by_target": {"small_model": 1, "large_model": 1},
        "framework_tokens_by_target_io": {
            "small_model": {"in": 20, "out": 5}, "large_model": {"in": 30, "out": 8},
        },
        "e2e_wall_clock_ms": 1000,
        "jev": {"http_successes": 1, "routing_audit_complete": True,
                "input_mode": "context", "threshold": threshold},
        "jev_audit": {"input_tokens_known": 10, "known_usage_responses": 1,
                       "prompt_sha256": compare._context_prompt_sha256("v1")},
    }


if __name__ == "__main__":
    unittest.main()
