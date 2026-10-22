"""Real local HTTP through the production Runner/Coordinator/M3 wiring."""
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import yaml

from experiments.run_appworld import AppWorldPilotRunner
from traceoffload.clock import real_utcnow
from traceoffload.contracts import ExecutionMode, GuardOutcome, ModelTarget, ModelTargetKind
from traceoffload.fakes import FakeModelGateway, FakeVerifier
from test_jev_client import _server
from test_jev_runner import REPO
from components.test_coordinator import _request, _workflow


@pytest.mark.parametrize("input_mode", ["context", "four_features", "combined"])
@pytest.mark.parametrize("reject_slm", [False, True])
def test_native_http_reaches_guarded_executor_and_fallback(tmp_path, input_mode, reject_slm):
    payload = {"model": "jev-test", "answers": {"route": {
        "type": "choice", "probabilities": {"LLM": 0.1, "SLM": 0.9}}}}
    with _server(payload=payload) as (endpoint, received):
        raw = yaml.safe_load((REPO / "configs/arms/offload-active.yaml").read_text(encoding="utf-8"))
        raw["runner"]["history_store"]["backend"] = "memory"
        raw["runner"]["jev"] = dict(enabled=True, mode="active", input_mode=input_mode,
                                     endpoint=endpoint, model_id="jev-test", request_timeout_ms=5000)
        checked = []
        def test_preflight(body):
            # A test fixture, never a production tokenizer or length estimate.
            checked.append(body)
            return 42
        runner = AppWorldPilotRunner(raw_config=raw, mode="classifier_route", agent_type="react_code",
                                     task_ids=["demo"], output_dir=tmp_path,
                                     jev_input_preflight=test_preflight)
        try:
            runner._set_skip_leading_messages(0)
            runner._verifier = FakeVerifier(default=GuardOutcome.REJECT if reject_slm else GuardOutcome.PASS)
            gateway = FakeModelGateway()
            runtime = SimpleNamespace(gateway=gateway, tool_proxy=object(), analyzer=None)
            coordinator = runner._build_coordinator(runtime)
            workflow = "workflow-http-test"
            outcome = coordinator.orchestrate_episode(
                _request(), _workflow(), workflow_run_id=workflow,
                caller_deadline_at=real_utcnow() + timedelta(minutes=1))
            assert outcome.response is not None
            assert len(checked) == len(received) == 1
            assert received[0]["path"] == "/v1/systemone"
            body = json.loads(received[0]["body"])
            assert body == checked[0]
            assert list(body["questions"]) == ["route"]
            assert body["state"]["model_info"]["small_model"] == raw["models"]["small"]["model_id"]
            spans = runner._events.span_records()
            guarded = [span for span in spans if span.route_plan.execution_mode == ExecutionMode.GUARDED_SLM]
            assert len(guarded) == 1 and runner._verifier.calls >= 1
            assert gateway.calls == (2 if reject_slm else 1)
            assert outcome.response.output_text == f"fake-{runner._large.id if reject_slm else runner._small.id}"
            if reject_slm:
                assert any(span.route_plan.decision_meta.reason_code == "slm_fallback" for span in spans)
            # Forced Coordinator paths bypass Jev even when the observer sees a prefix.
            coordinator.orchestrate_episode(
                _request(), _workflow(), workflow_run_id=workflow,
                caller_deadline_at=real_utcnow() + timedelta(minutes=1),
                _forced_target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=runner._large))
            assert len(received) == 1
            summary = {}
            runner._finalize_jev_task(workflow, summary)
            assert summary["jev"]["http_attempts"] == 1
            runner._clear_jev_workflow(workflow)
            assert not runner._jev_contexts._inputs
        finally:
            runner.close()
