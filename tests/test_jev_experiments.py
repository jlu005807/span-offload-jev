from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments import repair, report, run_round


def _jev(**overrides):
    value = {
        "mode": "active",
        "input_mode": "combined",
        "decisions": 2,
        "http_attempts": 2,
        "http_successes": 1,
        "http_errors": 1,
        "input_unavailable": 0,
        "budget_unavailable": 0,
        "slm_intent": 1,
        "llm_intent": 0,
        "client_total_ms": 125.0,
        "state_chars_total": 90,
        "audit_errors": 0,
        "routing_audit_complete": True,
    }
    value.update(overrides)
    return value


def test_report_uses_real_target_counts_and_aggregates_jev_metrics():
    summary = report.summarize([
        {
            "instance_id": "a",
            "task_success": True,
            "framework_model_calls": 100,
            "framework_model_calls_by_target": {"small_model": 2, "large_model": 3},
            "jev": _jev(),
        },
        {
            "instance_id": "b",
            "task_success": False,
            "framework_model_calls": 1,
            "framework_model_calls_by_target": {"small_model": 1, "large_model": 0},
            "jev": _jev(http_attempts=0, http_successes=0, http_errors=0,
                         decisions=1, client_total_ms=20.0, state_chars_total=10,
                         slm_intent=0, llm_intent=1),
        },
    ])

    assert summary["offload"] == 3 / 6
    assert summary["offload_detail"] == "3/6"
    assert summary["calls_per_run"] == 101 / 2
    assert summary["jev"]["decisions"] == 3
    assert summary["jev"]["http_attempts"] == 2
    assert summary["jev"]["client_total_ms"] == 145.0
    assert summary["jev_routing_latency_ms"] == 145.0
    assert summary["jev_http_failure_rate"] == 0.5
    assert summary["cost"] is None


def test_report_does_not_infer_offload_from_predictions_or_jev_intent():
    summary = report.summarize([{
        "instance_id": "shadow",
        "task_success": True,
        "framework_model_calls": 3,
        "l2_add4": {"p_gate_values": [1.0], "p_net_values": [1.0],
                     "tau_gate": 0.1, "tau_net": 0.1},
        "jev": _jev(slm_intent=1),
    }])

    assert summary["offload"] is None
    assert summary["offload_detail"] is None
    assert summary["offload_missing_rows"] == 1


def _write_config(root: Path, *, enabled: bool, mode: str = "active") -> Path:
    path = root / "configs" / "arms" / "jev-test.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"runner": {"jev": {"enabled": enabled, "mode": mode}}}),
                    encoding="utf-8")
    return path


def test_jev_enabled_is_read_from_loaded_config(tmp_path):
    path = _write_config(tmp_path, enabled=True)
    assert run_round.jev_enabled_from_config(path)
    assert repair.jev_enabled_from_config(path)
    path = _write_config(tmp_path, enabled=False)
    assert not run_round.jev_enabled_from_config(path)


def test_run_round_rejects_nonempty_jev_round_before_subprocess(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    config = _write_config(root, enabled=True)
    (root / "experiments").mkdir()
    instances = root / "instances.txt"
    instances.write_text("task-1\n", encoding="utf-8")
    out = root / "runs" / "jev-test" / "r1"
    out.mkdir(parents=True)
    (out / "existing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_round, "REPO", root)
    monkeypatch.setattr(run_round, "PY", "python")

    called = []
    monkeypatch.setattr(run_round.subprocess, "call", lambda *args, **kwargs: called.append(args) or 0)
    try:
        run_round.main(["--arm", "jev-test", "--rounds", "1", "--instances", str(instances),
                        "--out-root", str(root / "runs"), "--workers", "4"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("nonempty Jev round was accepted")
    assert called == []


def test_run_round_jev_lists_failures_without_automatic_rerun(tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    _write_config(root, enabled=True)
    (root / "experiments").mkdir()
    instances = root / "instances.txt"
    instances.write_text("task-1\ntask-2\n", encoding="utf-8")
    monkeypatch.setattr(run_round, "REPO", root)
    monkeypatch.setattr(run_round, "PY", "python")
    calls = []
    monkeypatch.setattr(run_round, "run_one", lambda cmd, env, log, timeout=1800:
                        (calls.append(cmd) or (cmd[-1], 1)))
    assert run_round.main(["--arm", "jev-test", "--rounds", "1", "--instances", str(instances),
                           "--out-root", str(root / "runs"), "--workers", "4"]) == 0
    assert len(calls) == 2
    output = capsys.readouterr().out
    assert "no automatic rerun" in output
    assert "task-1" in output and "task-2" in output


def test_repair_jev_only_reports_and_preserves_outputs(tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    _write_config(root, enabled=True)
    instances = root / "instances.txt"
    instances.write_text("task-1\ntask-2\n", encoding="utf-8")
    out = root / "runs" / "jev-test" / "r1"
    out.mkdir(parents=True)
    original = [{"instance_id": "task-1", "l2_add4": {"miss_no_vector": 1}}]
    summary = out / "appworld-classifier_route-summary.json"
    summary.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(repair, "REPO", root)
    monkeypatch.setattr(repair, "PY", "python")
    monkeypatch.setattr(repair.subprocess, "call", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Jev repair launched subprocess")))

    assert repair.main(["--arm", "jev-test", "--round", "1", "--instances", str(instances),
                        "--out-root", str(root / "runs")]) == 0
    assert json.loads(summary.read_text(encoding="utf-8")) == original
    assert "fresh output dir" in capsys.readouterr().out
