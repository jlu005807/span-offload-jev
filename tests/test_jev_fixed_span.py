from hashlib import sha256
import importlib
import json
import sqlite3
from types import SimpleNamespace as NS

import pytest
import yaml

from traceoffload.contracts import ModelRequest
from traceoffload.m2.jev.input_builder import JevInputBuilder, _serialize
from traceoffload.m2.jev.types import JevConfig, JevUnavailable


def source(tmp_path, *, tasks=2, steps=5):
    root = tmp_path / "source"
    root.mkdir()
    cfg = {"runner": {"jev": {"enabled": True, "mode": "active", "input_mode": "combined",
                             "model_id": "jev", "endpoint": "http://127.0.0.1:8900"}}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    (root / "appworld-classifier_route-summary.json").write_text(json.dumps([
        {"workflow_run_id": f"w{t}", "instance_id": f"task{t}", "task_success": True}
        for t in range(tasks)]))
    db = root / "history.sqlite3"
    con = sqlite3.connect(db)
    con.execute("create table span_records(sequence integer primary key, payload_json text)")
    audits = []
    for t in range(tasks):
        for k in range(steps):
            messages = [{"role": "system", "content": "template"}]
            for j in range(k):
                messages += [{"role": "assistant", "content": f"```python\nprint({j})\n```"},
                             {"role": "user", "content": "observed result"}]
            req = ModelRequest.model_validate_json(json.dumps({"request_version": "1", "messages": messages,
                "native_payload": {"messages": messages}, "model_parameters": {},
                "request_scope": {"data_policy_ref": {"kind": "policy", "id": "p", "version": "1"}}}))
            profile = NS(workflow_run_id=f"w{t}", episode_id=f"episode-{k+1}",
                         logical_invocation_id=f"invocation-{k+1}",
                         online_features={"prefix_n_A": float(k), "log_est_prompt_tokens": 5.0})
            ctx = NS(**vars(profile), prior_execution_facts=None,
                     workflow_context=NS(workflow_goal=NS(goal_text="goal\nwith blank\n\nlines")))
            builder = JevInputBuilder(JevConfig(model_id="jev", endpoint="http://127.0.0.1:8900"),
                                      large_model="large", small_model="small")
            builder.set_skip_leading_messages(1)
            prepared = builder.build(ctx, req, profile)
            audits.append({**{key: getattr(profile, key) for key in
                ["workflow_run_id", "episode_id", "logical_invocation_id"]},
                "input_mode": "combined", "request": prepared.request,
                "input_metadata": prepared.metadata})
            payload = {"model_request": req.model_dump(mode="json"), "runtime_profile": vars(profile),
                       "execution_summary": {"must_not_leak": "AFTER_CALL_SECRET"}}
            con.execute("insert into span_records(payload_json) values(?)", (json.dumps(payload),))
    con.commit()
    con.close()
    (root / "routing-decisions-test.jsonl").write_text("\n".join(json.dumps(r) for r in audits) + "\n")
    return root


def data_module():
    return importlib.import_module("experiments.jev_fixed_span_data")


def test_export_rebuilds_deduplicated_history_and_samples_deterministically(tmp_path):
    root = source(tmp_path)
    before = sha256((root / "history.sqlite3").read_bytes()).hexdigest()
    mod = data_module()
    mod.export_dataset(root, tmp_path / "a", count=6, task_count=2, seed=42)
    mod.export_dataset(root, tmp_path / "b", count=6, task_count=2, seed=42)
    samples = mod.load_dataset(tmp_path / "a")
    assert len(samples) == 6
    assert {s["task_id"] for s in samples} == {"task0", "task1"}
    assert (tmp_path / "a/samples.jsonl").read_bytes() == (tmp_path / "b/samples.jsonl").read_bytes()
    s = next(s for s in samples if s["episode_id"] != "episode-1")
    assert s["inputs"]["context"]["request"]["state"]["context"]["goal_text"] == "goal\nwith blank\n\nlines"
    assert "print(" in _serialize(s["inputs"]["context"]["request"])
    assert "AFTER_CALL_SECRET" not in _serialize(samples)
    assert set(s["inputs"]["four_features"]["request"]["state"]) == {"model_info", "features"}
    assert sha256((root / "history.sqlite3").read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        mod.export_dataset(root, tmp_path / "a", count=6, task_count=2)


def test_changed_original_combined_input_cannot_silently_be_replaced(tmp_path):
    root = source(tmp_path, tasks=1, steps=1)
    f = root / "routing-decisions-test.jsonl"
    r = json.loads(f.read_text())
    r["request"]["questions"]["route"]["instructions"] = "changed instructions"
    r["input_metadata"]["input_sha256"] = sha256(_serialize(r["request"]).encode()).hexdigest()
    f.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="reproduction"):
        data_module().export_dataset(root, tmp_path / "out", count=1, task_count=1)


def test_duplicate_audit_identity_is_rejected(tmp_path):
    root = source(tmp_path, tasks=1, steps=1)
    f = root / "routing-decisions-test.jsonl"
    f.write_text(f.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        data_module().export_dataset(root, tmp_path / "out", count=1, task_count=1)


def test_dataset_tampering_is_rejected(tmp_path):
    root = source(tmp_path, tasks=1, steps=1)
    data_module().export_dataset(root, tmp_path / "out", count=1, task_count=1)
    f = tmp_path / "out/samples.jsonl"
    f.write_text(f.read_text(encoding="utf-8").replace('"task0"', '"other"'), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        data_module().load_dataset(tmp_path / "out")


def test_runner_keeps_all_modes_after_failure_and_uses_paired_denominator(tmp_path):
    root = source(tmp_path, tasks=1, steps=2)
    data_module().export_dataset(root, tmp_path / "data", count=2, task_count=1)
    calls = []
    class Client:
        def predict(self, body, *, timeout_ms):
            state = body["state"]
            mode = "combined" if "features" in state and "context" in state else (
                "context" if "context" in state else "four_features")
            calls.append(mode)
            if mode == "context" and calls.count(mode) == 1:
                raise JevUnavailable("timeout", attempted=True)
            return {"model": "jev", "answers": {"route": {"type": "choice",
                "probabilities": {"LLM": .3, "SLM": .7}}}, "usage": {"input_tokens": 20}}
    mod = importlib.import_module("experiments.jev_fixed_span")
    report = mod.run_experiment(tmp_path / "data", tmp_path / "result", Client(), seed=42)
    assert len(calls) == 6 and all(calls.count(m) == 2 for m in mod.MODES)
    assert report["paired_samples"] == 1
    assert report["modes"]["context"]["successes"] == 1
    assert report["modes"]["context"]["thresholds"]["0.7"]["all_sample_rate"] == .5
    assert report["modes"]["combined"]["thresholds"]["0.7"]["all_sample_rate"] == 1
    assert report["paired"]["context"]["mean_slm_probability"] == .7
    assert report["local_correctness"] is None and report["net_saving"] is None
    assert len((tmp_path / "result/decisions.jsonl").read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        mod.run_experiment(tmp_path / "data", tmp_path / "result", Client())


def test_report_rejects_duplicate_or_missing_mode_results():
    mod = importlib.import_module("experiments.jev_fixed_span")
    with pytest.raises(ValueError):
        mod.summarize([{ "sample_id": "a", "mode": "context", "status": "unavailable"}] * 3)


def test_insufficient_selected_task_capacity_is_not_silently_replaced(tmp_path):
    root = source(tmp_path, tasks=2, steps=2)
    with pytest.raises(ValueError, match="Not enough"):
        data_module().export_dataset(root, tmp_path / "out", count=3, task_count=1)


def test_original_history_mismatch_rejects_reconstruction(tmp_path):
    root = source(tmp_path, tasks=1, steps=2)
    con = sqlite3.connect(root / "history.sqlite3")
    row = json.loads(con.execute("select payload_json from span_records where sequence=2").fetchone()[0])
    row["model_request"]["messages"][-1]["content"] = "different observation"
    con.execute("update span_records set payload_json=? where sequence=2", (json.dumps(row),))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="reproduction"):
        data_module().export_dataset(root, tmp_path / "out", count=2, task_count=1)


def test_native_http_preflight_rejects_one_mode_without_transmission(tmp_path, monkeypatch):
    import httpx
    from time import perf_counter
    from traceoffload.m2.jev import client as client_module
    root = source(tmp_path, tasks=1, steps=1)
    data_module().export_dataset(root, tmp_path / "data", count=1, task_count=1)
    received = []
    def respond(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "jev", "answers": {"route": {
            "type": "choice", "probabilities": {"LLM": .4, "SLM": .6}}},
            "usage": {"input_tokens": 50}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(client_module, "real_time", perf_counter)
    def preflight(body):
        if set(body["state"]) == {"model_info", "features"}:
            raise JevUnavailable("input_too_long")
        return 50
    client = client_module.HttpJevClient(JevConfig(model_id="jev", endpoint="http://127.0.0.1:8900"),
                                         input_preflight=preflight)
    mod = importlib.import_module("experiments.jev_fixed_span")
    report = mod.run_experiment(tmp_path / "data", tmp_path / "result", client)
    assert len(received) == 2
    assert all(list(r["questions"]["route"]["criteria"]) == ["LLM", "SLM"] for r in received)
    assert received[0]["questions"] == received[1]["questions"]
    assert report["paired_samples"] == 0
    assert report["modes"]["four_features"]["errors"] == {"input_too_long": 1}
    assert report["modes"]["context"]["input_tokens_known"] == 50


def test_wrong_response_model_is_retained_as_unavailable(tmp_path):
    root = source(tmp_path, tasks=1, steps=1)
    data_module().export_dataset(root, tmp_path / "data", count=1, task_count=1)
    class WrongModel:
        def predict(self, body, *, timeout_ms):
            return {"model": "another-model"}
    mod = importlib.import_module("experiments.jev_fixed_span")
    report = mod.run_experiment(tmp_path / "data", tmp_path / "result", WrongModel())
    assert report["paired_samples"] == 0
    assert all(r["errors"] == {"model_mismatch": 1} for r in report["modes"].values())


def test_real_run_requires_acceptance_bound_to_dataset(tmp_path):
    root = source(tmp_path, tasks=1, steps=1)
    mod = data_module()
    mod.export_dataset(root, tmp_path / "data", count=1, task_count=1)
    gate = tmp_path / "acceptance.json"
    gate.write_text(json.dumps({"ready_for_experiment": False}))
    with pytest.raises(ValueError, match="acceptance"):
        mod.check_acceptance(tmp_path / "data", gate)
    gate.write_text(json.dumps({"ready_for_experiment": True, "samples_sha256": "wrong",
                               "dataset_manifest_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash"):
        mod.check_acceptance(tmp_path / "data", gate)
