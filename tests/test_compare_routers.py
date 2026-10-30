from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def module():
    return importlib.import_module("experiments.compare_routers")


def config():
    return {"models": {"large": {"model_id": "large"}, "small": {"model_id": "small"},
                       "decoding": {"seed": 42}}, "agent": {"step_limit": 40},
            "runner": {"m4_memory": {"enabled": True, "embedding_mode": "m1_online"},
                       "l2_add4": {"enabled": True, "model_path": "weights.json"},
                       "jev": {"enabled": True, "input_mode": "combined", "threshold": .8}}}


def result(group, task, **changes):
    row = {"instance_id": task, "task_success": True, "total_tokens": 25,
           "framework_model_calls_by_target": {"large_model": 2},
           "framework_tokens_by_target_io": {"large_model": {"in": 20, "out": 5}},
           "e2e_wall_clock_ms": 1000}
    if group == "legacy":
        row["l2_add4"] = {"hits": 2, "miss_no_vector": 0}
    if group.startswith("jev"):
        row["jev"] = {"http_successes": 2, "routing_audit_complete": True,
                      "input_mode": "context", "threshold": .8 if group == "jev08" else .7}
        row["jev_audit"] = {"input_tokens_known": 13, "known_usage_responses": 2}
    row.update(changes)
    return row


def test_configs_change_only_route_keep_models_and_disable_memory():
    raw = config()
    before = deepcopy(raw)
    configs = module().build_configs(raw)
    assert raw == before
    assert set(configs) == {"native", "legacy", "jev08", "jev07"}
    assert all(c["models"] == raw["models"] and not c["runner"]["m4_memory"]["enabled"] for c in configs.values())
    assert configs["legacy"]["runner"]["l2_add4"]["enabled"] is True
    assert configs["legacy"]["runner"]["jev"]["enabled"] is False
    assert configs["jev08"]["runner"]["jev"]["threshold"] == .8
    assert configs["jev07"]["runner"]["jev"]["threshold"] == .7
    assert configs["jev07"]["runner"]["jev"]["input_mode"] == "context"


def test_gate_does_not_require_quality_success_but_rejects_missing_vectors():
    mod = module()
    assert mod.acceptance_errors("legacy", result("legacy", "a", task_success=False)) == []
    errors = mod.acceptance_errors("legacy", result("legacy", "a", l2_add4={"hits": 1, "miss_no_vector": 1}))
    assert errors
    assert mod.acceptance_errors("native", {"instance_id": "a", "task_success": True})
    assert mod.acceptance_errors("jev07", result("jev07", "a", jev={"http_successes": 0}))


def test_failed_task_is_kept_and_task_orders_rotate(tmp_path):
    mod = module()
    calls = []
    def execute(group, cfg, task, output, experiment, split):
        calls.append((task, group, str(output), experiment))
        return result(group, task, task_success=not (task == "one" and group == "jev07"),
                      task_failure_reason="evaluation_failed" if task == "one" and group == "jev07" else None)
    report = mod.run_comparison(config(), ["one", "two"], tmp_path / "out", execute=execute,
                                health=lambda _: None, task_splits={"one": "dev", "two": "train", "099174d_1": "test_challenge"})
    assert len(calls) == 12  # Four smoke cells, then two tasks times four groups.
    assert [g for t, g, *_ in calls if t == "one"] != [g for t, g, *_ in calls if t == "two"]
    assert len({c[2] for c in calls}) == len({c[3] for c in calls}) == 12
    assert report["groups"]["jev07"]["tasks"] == 2
    assert report["groups"]["jev07"]["successes"] == 1
    assert report["groups"]["native"]["successes"] == 2
    assert report["groups"]["jev07"]["jev_input_tokens_known"] == 26
    assert len(json.loads((tmp_path / "out/results.json").read_text())) == 8


def test_bad_original_router_smoke_stops_before_full_batch(tmp_path):
    calls = []
    def execute(group, cfg, task, output, experiment, split):
        calls.append(group)
        row = result(group, task)
        if group == "legacy":
            row["l2_add4"]["miss_no_vector"] = 1
        return row
    with pytest.raises(RuntimeError, match="acceptance"):
        module().run_comparison(config(), ["one"], tmp_path / "out", execute=execute,
                                health=lambda _: None, task_splits={"one": "dev", "099174d_1": "test_challenge"})
    assert len(calls) == 4
    assert not json.loads((tmp_path / "out/acceptance.json").read_text())["ready_for_batch"]


def test_duplicate_tasks_or_existing_output_never_execute(tmp_path):
    mod = module()
    def never(*args):
        pytest.fail("Unexpected inference")
    with pytest.raises(ValueError):
        mod.run_comparison(config(), ["one", "one"], tmp_path / "out", execute=never)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        mod.run_comparison(config(), ["one"], existing, execute=never)


def test_normalizing_native_calls_does_not_invent_usage_for_crashes():
    mod = module()
    row = mod.normalize_native({"model_calls": 2, "prompt_tokens": 10, "completion_tokens": 3})
    assert row["framework_model_calls_by_target"] == {"large_model": 2}
    assert row["framework_tokens_by_target_io"] == {"large_model": {"in": 10, "out": 3}}
    assert "framework_model_calls_by_target" not in mod.normalize_native({"task_success": False})


def test_embedding_usage_counts_batch_once_and_preserves_vectors():
    from types import SimpleNamespace
    vectors = (SimpleNamespace(values=(1.0, 2.0), prompt_eval_count=12),
               SimpleNamespace(values=(3.0, 4.0), prompt_eval_count=12))
    embedder = SimpleNamespace(embed_many=lambda texts: vectors)
    stats = module().measure_embedding(embedder)
    assert embedder.embed_many(["goal", "actions"]) is vectors
    assert stats["requests"] == 1
    assert stats["input_tokens_known"] == 12
    assert stats["texts"] == 2


def test_timed_out_cell_is_preserved_without_inventing_counts_and_next_cell_runs(tmp_path):
    def execute(group, cfg, task, output, experiment, split):
        if group == "jev07" and task == "one":
            return {"instance_id": task, "task_success": False, "task_failure_reason": "task_timeout"}
        return result(group, task)
    report = module().run_comparison(config(), ["one", "two"], tmp_path / "out", execute=execute,
        health=lambda _: None, task_splits={"one": "train", "two": "dev", "099174d_1": "test_challenge"})
    assert report["complete"] is True
    assert report["groups"]["jev07"]["tasks_with_call_counts"] == 1
    assert report["groups"]["jev07"]["invalid_measurement_tasks"] == 1
    assert report["groups"]["jev07"]["failures"] == {"task_timeout": 1}


def test_native_runner_imports_in_real_module_worker_environment():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-c", "from experiments.run_native import run_native_task"],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_preflight_attempts_are_separate_from_server_usage_and_ignore_unsent_stale_metadata(tmp_path):
    rows = [
        {"status": "ok", "http_attempted": True, "input_metadata": {"input_tokens": 100},
         "response": {"answers": {"route": {"probabilities": {"SLM": .6}}}}},
        {"status": "unavailable", "http_attempted": True, "input_metadata": {"input_tokens": 200}},
        {"status": "input_unavailable", "http_attempted": False, "input_metadata": {"input_tokens": 999}},
    ]
    (tmp_path / "routing-decisions-test.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    stats = module().audit_usage(tmp_path)
    assert stats["successful_responses_without_usage"] == 1
    assert stats["attempted_preflight_input_tokens"] == 300
    assert stats["failed_attempt_preflight_input_tokens"] == 200
    assert stats["known_usage_responses"] == 0


def test_timeout_recovers_checkpoints_and_marks_group_totals_incomplete(tmp_path):
    mod = module()
    (tmp_path / "partial-execution.json").write_text(json.dumps({
        "framework_model_calls_by_target": {"large_model": 3},
        "framework_tokens_by_target_io": {"large_model": {"in": 100, "out": 10}}}))
    (tmp_path / "partial-embedding.json").write_text(json.dumps({"requests": 2, "input_tokens_known": 80}))
    row = mod.recover_interrupted(tmp_path, "one", "task_timeout", 500)
    assert row["framework_model_calls_by_target"] == {"large_model": 3}
    assert row["embedding_usage"]["input_tokens_known"] == 80
    row.update(group="legacy", valid_for_comparison=False)
    report = mod.summarize([row])["groups"]["legacy"]
    assert report["consumption_complete"] is False
    assert report["actual_slm_rate"] is None
    assert report["execution_tokens_by_target_io"]["large_model"]["in"] == 100


def test_embedding_observation_persists_completed_usage(tmp_path):
    from types import SimpleNamespace
    vectors = (SimpleNamespace(values=(1.0,), prompt_eval_count=15),)
    embedder = SimpleNamespace(embed_many=lambda texts: vectors)
    stats = module().measure_embedding(embedder, checkpoint=tmp_path/"partial-embedding.json")
    assert embedder.embed_many(["goal"]) is vectors
    assert json.loads((tmp_path/"partial-embedding.json").read_text()) == stats


def test_gateway_checkpoint_preserves_response_and_records_completed_usage(tmp_path):
    from types import SimpleNamespace
    from experiments.run_appworld import _GatewayMeter
    response = SimpleNamespace(input_tokens=12, output_tokens=3, latency_ms=5, cost=None)
    inner = SimpleNamespace(complete=lambda *a, **kw: response)
    meter = module().checkpoint_meter_type(_GatewayMeter, tmp_path/"partial.json")(inner)
    target = SimpleNamespace(kind=SimpleNamespace(value="small_model"))
    assert meter.complete(target, None, model_attempt_id="a", deadline_at=None) is response
    saved = json.loads((tmp_path/"partial.json").read_text())
    assert saved["framework_model_calls_by_target"] == {"small_model": 1}
    assert saved["framework_tokens_by_target_io"] == {"small_model": {"in": 12, "out": 3}}


def test_native_checkpoint_uses_original_measurement_and_preserves_message(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from time import perf_counter
    from experiments import run_native
    monkeypatch.setattr(run_native, "_real_time", perf_counter)
    usage = SimpleNamespace(tokens=SimpleNamespace(input_cache_miss=10, input_cache_hit=2,
        input_cache_write=0, output=3), cost=SimpleNamespace(total=0))
    response = {"standardized_usage": usage}
    model = SimpleNamespace(generate=lambda **kw: response)
    totals = run_native._empty_usage()
    module().native_measurement_with_checkpoints(run_native._install_measurement, tmp_path/"partial.json")(model, totals)
    assert model.generate(messages=[]) is response
    saved = json.loads((tmp_path/"partial.json").read_text())
    assert saved["framework_model_calls_by_target"] == {"large_model": 1}
    assert saved["framework_tokens_by_target_io"]["large_model"] == {"in": 12, "out": 3}


@pytest.mark.parametrize("existing", [None, "http://existing-service:8000"])
def test_native_direct_entry_initializes_model_server_environment(monkeypatch, tmp_path, existing):
    from experiments import run_native
    if existing is None:
        monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    else:
        monkeypatch.setenv("MODEL_SERVER_URL", existing)
    class StopBeforeAgentImport(Exception):
        pass
    def check_environment():
        assert os.environ["MODEL_SERVER_URL"] == (existing or "")
        raise StopBeforeAgentImport
    monkeypatch.setattr(run_native, "_ensure_appworld_agents_importable", check_environment)
    with pytest.raises(StopBeforeAgentImport):
        run_native.run_native_task(raw_config={}, model_kind="large", task_id="test",
            output_dir=tmp_path, step_limit=40, agent_type="react_code", experiment_name="test")


def test_threshold_sweep_configs_and_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_COMPARE_PROFILE", "thresholds")
    mod = module()
    expected = {"jev055": .55, "jev060": .6, "jev065": .65, "jev075": .75}
    configs = mod.build_configs(config())
    assert set(configs) == set(expected)
    def execute(group, cfg, task, output, experiment, split):
        c = __import__("yaml").safe_load(Path(cfg).read_text())
        assert c["runner"]["jev"]["threshold"] == expected[group]
        assert c["models"] == config()["models"]
        assert not c["runner"]["l2_add4"]["enabled"]
        row = result(group, task)
        row["jev"]["threshold"] = expected[group]
        assert mod.acceptance_errors(group, row) == []
        return row
    report = mod.run_comparison(config(), ["one"], tmp_path/"sweep", execute=execute,
        health=lambda _:None, task_splits={"one":"train", "099174d_1":"train"})
    assert set(report["groups"]) == set(expected)
    assert report["complete"] is True
    assert len(json.loads((tmp_path/"sweep/results.json").read_text())) == 4
