"""Deployment-only single task entrypoint; paths/config supplied via environment."""
import json
import hashlib
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from deploy.jev.openjev_preflight import OpenJevPreflight
from experiments.run_appworld import AppWorldPilotRunner


def new_experiment_name(task_id):
    return f"jev-{task_id}-{uuid4().hex}"


def acceptance_errors(result, *, jev_enabled):
    errors = []
    if jev_enabled is not True:
        errors.append("Jev backend was not enabled")
    rows = result.get("workflows", [])
    if len(rows) != 1:
        return errors + ["Expected exactly one task result"]
    row = rows[0]
    if not isinstance(row.get("task_success"), bool):
        errors.append("AppWorld task outcome is missing")
    if row.get("failure") is not None and row.get("task_failure_reason") != "context_length_exceeded":
        errors.append("Task has a runtime failure")
    jev = row.get("jev") or {}
    if (jev.get("http_successes") or 0) <= 0:
        errors.append("No successful native Jev HTTP decisions")
    if jev.get("routing_audit_complete") is not True:
        errors.append("Jev decision audit is incomplete")
    calls = row.get("framework_model_calls_by_target") or {}
    if sum(calls.values()) <= 0:
        errors.append("Actual model dispatch counters missing")
    return errors


def main():
    config = Path(os.environ["JEV_RUN_CONFIG"])
    config_bytes = config.read_bytes()
    output = Path(os.environ["JEV_RUN_OUTPUT"])
    if output.exists():
        raise SystemExit("Use a new output directory for each acceptance run")
    output.mkdir(parents=True)
    raw = yaml.safe_load(config_bytes)
    (output / "config.yaml").write_bytes(config_bytes)
    preflight = OpenJevPreflight.from_directory(os.environ["JEV_MODEL_DIR"])
    task_id = os.environ.get("JEV_TASK_ID", "099174d_1")
    experiment = new_experiment_name(task_id)
    appworld_output = Path(os.environ["APPWORLD_ROOT"]) / "experiments" / "outputs" / experiment
    if appworld_output.exists():
        raise SystemExit("AppWorld output directory already exists")
    (output / "manifest.json").write_text(json.dumps({
        "task_id": task_id, "experiment_name": experiment,
        "appworld_output": str(appworld_output), "config": str(config.resolve()),
        "model_dir": os.environ["JEV_MODEL_DIR"],
    }, indent=2), encoding="utf-8")
    runner = AppWorldPilotRunner(raw_config=raw, mode="classifier_route", agent_type="react_code",
                                task_ids=[task_id], output_dir=output,
                                experiment_name=experiment, jev_input_preflight=preflight)
    runner._appworld_agent_config["logger_config"]["verbose"] = True
    try:
        result = runner.run_tasks()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        runner.close()
    errors = acceptance_errors(result, jev_enabled=raw.get("runner", {}).get("jev", {}).get("enabled"))
    (output / "acceptance.json").write_text(json.dumps({
        "acceptance_kind": "integration",
        "ready_for_batch": not errors, "errors": errors,
        "task_success": result["workflows"][0].get("task_success") if len(result.get("workflows", [])) == 1 else None,
        "task_failure_reason": result["workflows"][0].get("task_failure_reason") if len(result.get("workflows", [])) == 1 else None,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
