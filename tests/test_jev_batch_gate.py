import hashlib
import json

import pytest

from deploy.jev import run_batch


def test_rejected_acceptance_never_starts_services_or_tasks(tmp_path, monkeypatch):
    gate = tmp_path / "acceptance.json"
    gate.write_text(json.dumps({"ready_for_batch": False}))
    monkeypatch.setenv("JEV_ACCEPTANCE_FILE", str(gate))
    monkeypatch.setattr(run_batch, "check_services", lambda *_: pytest.fail("Gate was bypassed"))
    with pytest.raises(SystemExit, match="has not passed"):
        run_batch.main()


def test_changed_configuration_cannot_reuse_old_acceptance(tmp_path, monkeypatch):
    gate = tmp_path / "acceptance.json"
    gate.write_text(json.dumps({"ready_for_batch": True,
                                "config_sha256": hashlib.sha256(b"old config").hexdigest()}))
    config = tmp_path / "config.yaml"
    config.write_text("runner: {}\n")
    monkeypatch.setenv("JEV_ACCEPTANCE_FILE", str(gate))
    monkeypatch.setenv("JEV_RUN_CONFIG", str(config))
    monkeypatch.setattr(run_batch, "check_services", lambda *_: pytest.fail("Gate was bypassed"))
    with pytest.raises(SystemExit, match="differs"):
        run_batch.main()


def test_duplicate_manifest_cannot_overwrite_task_trajectory(tmp_path):
    path = tmp_path / "instances.txt"
    path.write_text("099174d_1\n099174d_1\n")
    with pytest.raises(ValueError, match="duplicates"):
        run_batch.load_manifest(path)


def test_batch_retains_context_failure_and_executes_next_task(tmp_path, monkeypatch):
    import yaml
    config_bytes = yaml.safe_dump({"runner": {"jev": {"enabled": True}}}).encode()
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(config_bytes)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"ready_for_batch": True,
                              "config_sha256": hashlib.sha256(config_bytes).hexdigest()}))
    manifest = tmp_path / "ids.txt"
    manifest.write_text("failed-task\nnext-task\n")
    output = tmp_path / "out"
    for name, value in {"JEV_ACCEPTANCE_FILE": gate, "JEV_RUN_CONFIG": cfg,
                        "JEV_TASK_MANIFEST": manifest, "JEV_RUN_OUTPUT": output,
                        "APPWORLD_ROOT": tmp_path, "JEV_MODEL_DIR": tmp_path}.items():
        monkeypatch.setenv(name, str(value))
    calls = []
    class Runner:
        def __init__(self, **kwargs):
            self._appworld_agent_config = {"logger_config": {}}
        def run_task(self, task_id):
            calls.append(task_id)
            return {"instance_id": task_id, "task_success": task_id == "next-task",
                    "task_failure_reason": "context_length_exceeded" if task_id == "failed-task" else None,
                    "total_tokens": 12, "framework_model_calls": 1,
                    "framework_model_calls_by_target": {"large_model": 1}}
        def close(self):
            pass
    monkeypatch.setattr(run_batch, "AppWorldPilotRunner", Runner)
    monkeypatch.setattr(run_batch, "check_services", lambda *_: None)
    monkeypatch.setattr(run_batch.OpenJevPreflight, "from_directory", lambda *_: None)
    run_batch.main()
    rows = json.loads((output / "appworld-classifier_route-summary.json").read_text())
    assert calls == ["failed-task", "next-task"]
    assert rows[0]["task_failure_reason"] == "context_length_exceeded"
    assert rows[1]["task_success"] is True
    report = json.loads((output / "report.json").read_text())
    assert report["n"] == 2 and report["success"] == .5
    assert report["task_failures_by_reason"] == {"context_length_exceeded": 1}
