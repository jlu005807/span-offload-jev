from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "jev_lora" / "convert_appworld_labels.py"
SPEC = importlib.util.spec_from_file_location("convert_appworld_labels", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(converter)


def _trace():
    # AppWorld initial_messages contains 22 template messages to skip, then
    # the real task instruction that must remain in online history.
    initial = [
        {"role": "system", "content": f"template-{index}"}
        for index in range(22)
    ] + [{"role": "user", "content": "Find the order"}]
    messages = initial + [
        {"role": "assistant", "content": "I will search", "reasoning_content": None},
        {"role": "tool", "content": "Order found", "tool_call_id": "call-1"},
        {"role": "assistant", "content": None, "tool_calls": [{"tool_call_id": "call-2"}]},
    ]
    return {
        "trace_metadata": {
            "task_id": "task-1",
            "trace_kind": "original",
            "task_instruction": "Find the order",
            "initial_messages": initial,
            "agent_configuration": {
                "models": {
                    "large": {"model_id": "Qwen3.8-27B"},
                    "small": {"model_id": "Qwen3.5-9B"},
                }
            },
        },
        "steps": [
            {"step_id": 1, "input_context": {"messages": messages}},
        ],
    }


class ConversionTests(unittest.TestCase):
    def test_labels_map_to_online_choice_keys(self):
        for label, expected in [(0, "LLM"), (1, "SLM"), (2, "SLM")]:
            with self.subTest(label=label):
                row = converter.convert_label_row(
                    {"task_id": "task-1", "k": 1, "label": label, "mask": 1}, _trace()
                )
                self.assertEqual(row["labels"], {"route": expected})

    def test_masked_unknown_is_excluded(self):
        self.assertIsNone(converter.convert_label_row(
            {"task_id": "task-1", "k": 1, "label": "unknown", "mask": 0}, _trace()
        ))

    def test_request_rebuilds_v1_context_and_keeps_meta_outside_model_input(self):
        row = converter.convert_label_row(
            {
                "task_id": "task-1", "k": 1, "label": 2, "mask": 1,
                "label_version": "counterfactual-l123-v1", "reasons": [],
            },
            _trace(),
        )

        request = row["request"]
        self.assertEqual(request["model"], "wev-1.7b")
        self.assertEqual(request["state"], {
            "model_info": {
                "large_model": "Qwen3.8-27B",
                "small_model": "Qwen3.5-9B",
                "execution_policy": converter.EXECUTION_POLICY,
            },
            "context": {
                "goal_text": "Find the order",
                "goal_available": True,
                "history": [
                    {"messages": [{"role": "user", "content": "Find the order"}]},
                    {"messages": [
                        {"role": "assistant", "content": "I will search"},
                        {"role": "tool", "content": "Order found", "tool_call_id": "call-1"},
                    ]},
                    {"messages": [{"role": "assistant",
                                   "tool_calls": [{"tool_call_id": "call-2"}]}]},
                ],
            },
        })
        route = request["questions"]["route"]
        self.assertEqual(route["type"], "choice")
        self.assertEqual(route["instructions"], converter.V1_INSTRUCTIONS)
        self.assertEqual(route["criteria"], converter.ROUTE_CRITERIA)
        self.assertEqual(row["labels"], {"route": "SLM"})
        self.assertEqual(row["_meta"], {
            "task_id": "task-1", "k": 1, "original_label": 2, "mask": 1,
            "label_version": "counterfactual-l123-v1", "reasons": [],
        })

    def test_prompt_and_criteria_hash_matches_v1_experiment(self):
        self.assertEqual(
            converter.v1_prompt_sha256(),
            "a520f9c99e990a6ce282dbcf29d5347ff085685953c86e394dfc1f3f21362b2a",
        )

    def test_inconsistent_trace_fails_closed(self):
        mutations = [
            (lambda trace: trace["steps"].__setitem__(0, {"step_id": 2}), "step_id"),
            (lambda trace: trace["steps"][0]["input_context"].__setitem__("messages", []), "prefix"),
        ]
        for mutation, message in mutations:
            with self.subTest(message=message):
                trace = _trace()
                mutation(trace)
                with self.assertRaisesRegex(ValueError, message):
                    converter.convert_label_row(
                        {"task_id": "task-1", "k": 1, "label": 1, "mask": 1}, trace
                    )

    def test_initial_messages_must_contain_template_prefix_and_task_goal(self):
        trace = _trace()
        trace["trace_metadata"]["initial_messages"] = trace["trace_metadata"]["initial_messages"][:22]
        trace["steps"][0]["input_context"]["messages"] = trace["steps"][0]["input_context"]["messages"][:22]
        with self.assertRaisesRegex(ValueError, "at least 23"):
            converter.convert_label_row(
                {"task_id": "task-1", "k": 1, "label": 1, "mask": 1}, trace
            )

    def test_missing_goal_is_explicitly_marked_unavailable(self):
        trace = _trace()
        trace["trace_metadata"]["task_instruction"] = "  "
        row = converter.convert_label_row(
            {"task_id": "task-1", "k": 1, "label": 1, "mask": 1}, trace
        )
        self.assertIsNone(row["request"]["state"]["context"]["goal_text"])
        self.assertFalse(row["request"]["state"]["context"]["goal_available"])

    def test_conversion_refuses_to_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({"task_id": "task-1", "k": 1, "label": 1, "mask": 1}) + "\n")
            data = root / "data"
            (data / "task-1" / "original").mkdir(parents=True)
            (data / "task-1" / "original" / "trace.json").write_text(json.dumps(_trace()))
            dev_data = root / "dev-data"
            (dev_data / "task-2" / "original").mkdir(parents=True)
            dev_trace = _trace()
            dev_trace["trace_metadata"]["task_id"] = "task-2"
            (dev_data / "task-2" / "original" / "trace.json").write_text(json.dumps(dev_trace))
            dev_labels = root / "dev-labels.jsonl"
            dev_labels.write_text(json.dumps({"task_id": "task-2", "k": 1, "label": 1, "mask": 1}) + "\n")
            out = root / "out"
            out.mkdir()

            with self.assertRaises(FileExistsError):
                converter.convert_splits(
                    train_data=data,
                    train_labels=labels,
                    dev_data=dev_data,
                    dev_labels=dev_labels,
                    out=out,
                )

    def test_oversized_online_state_is_excluded_and_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            trace_dir = data / "task-1" / "original"
            trace_dir.mkdir(parents=True)
            trace = _trace()
            (trace_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({
                "task_id": "task-1", "k": 1, "label": 1, "mask": 1,
            }) + "\n", encoding="utf-8")

            rows, summary = converter.convert_split(
                data, labels, max_history_spans=10, max_state_chars=1,
                request_model_id="wev-1.7b",
            )

            self.assertEqual(rows, [])
            self.assertEqual(summary["excluded_online_unavailable"], 1)
            self.assertEqual(summary["excluded_by_reason"], {"state_too_large": 1})


if __name__ == "__main__":
    unittest.main()
