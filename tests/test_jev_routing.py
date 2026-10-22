from datetime import timedelta
from types import SimpleNamespace
import json

import pytest

from traceoffload.clock import real_utcnow
from traceoffload.contracts import ExecutionMode, ModelTargetKind
from traceoffload.m2.jev.audit import JevAudit
from traceoffload.m2.jev.input_builder import PreparedInput, RoutingContextStore, key_for
from traceoffload.m2.jev.router import JevRouter
from traceoffload.m2.jev.types import JevUnavailable
from test_l2_add4_routing import _profile, _policy, LLM, SLM, _router, _decision


def config(**kwargs):
    values = dict(mode="active", input_mode="combined", model_id="jev-test",
                  expected_revision=None, threshold=0.8, request_timeout_ms=1000,
                  execution_reserve_ms=100, max_state_chars=12000)
    return SimpleNamespace(**(values | kwargs))


class Client:
    def __init__(self, probability=0.9, error=None):
        self.p, self.error, self.calls = probability, error, []

    def predict(self, request, *, timeout_ms):
        self.calls.append((request, timeout_ms))
        if self.error:
            raise self.error
        return {"model": "jev-test", "answers": {"route": {
            "type": "choice", "probabilities": {"LLM": 1 - self.p, "SLM": self.p}}}}


def setup(tmp_path, *, client=None, cfg=None, legacy=None, remaining=20):
    profile = _profile().model_copy(update={"produced_at": real_utcnow()})
    store = RoutingContextStore()
    request = {"model": "jev-test", "state": {}, "questions": {"route": {}}}
    store.put(key_for(profile), PreparedInput(
        request=request, caller_deadline_at=real_utcnow() + timedelta(seconds=remaining),
        error=None, metadata={"state_chars": 2}))
    audit = JevAudit(tmp_path / "routing.jsonl", config=cfg or config())
    client = client or Client()
    router = JevRouter(LLM, SLM, config=cfg or config(), context_store=store,
                       client=client, audit=audit, legacy=legacy)
    return profile, router, client, audit, store


@pytest.mark.parametrize("probability,small", [(0.8, True), (0.79, False)])
def test_active_threshold_preserves_guarded_execution(tmp_path, probability, small):
    p, r, c, a, store = setup(tmp_path, client=Client(probability))
    plan = r.decide(p, (), _policy())
    assert (plan.target.kind == ModelTargetKind.SMALL_MODEL) is small
    assert plan.execution_mode == (ExecutionMode.GUARDED_SLM if small else ExecutionMode.DIRECT)
    assert bool(plan.fallback_plan) is small
    assert bool(plan.verifier_plan.steps) is small
    assert plan.decision_meta.classifier_predictor == "jev_choice_v1"
    assert plan.decision_meta.classifier_p is None
    assert len(c.calls) == 1
    assert store.take(key_for(p)) is None
    assert a.summary(p.workflow_run_id)["http_attempts"] == 1


def test_shadow_calls_legacy_once_and_returns_identical_plan(tmp_path):
    legacy = _router(_decision(offload=False))
    original = legacy.decide
    plans = []
    def decide(*args, **kwargs):
        result = original(*args, **kwargs)
        plans.append(result)
        return result
    legacy.decide = decide
    p, r, c, a, _ = setup(tmp_path, cfg=config(mode="shadow"), legacy=legacy)
    result = r.decide(p, (), _policy())
    assert len(plans) == 1 and result is plans[0]
    assert result.decision_meta.reason_code == "l2_add4_llm"
    row = json.loads((tmp_path / "routing.jsonl").read_text())
    assert row["legacy_intent"] == "LLM" and row["intent"] == "SLM"


def test_client_failure_routes_large_and_retains_attempt(tmp_path):
    error = JevUnavailable("timeout", attempted=True)
    p, r, c, a, _ = setup(tmp_path, client=Client(error=error))
    assert r.decide(p, (), _policy()).decision_meta.reason_code == "jev_unavailable_large"
    stats = a.summary(p.workflow_run_id)
    assert stats["http_attempts"] == 1 and stats["http_errors"] == 1


@pytest.mark.parametrize("error", [None, JevUnavailable("http_error", attempted=True)])
def test_audit_records_complete_preflight_token_count(tmp_path, error):
    client = Client(error=error)
    client.last_input_tokens = 37
    p, r, _c, _a, _store = setup(tmp_path, client=client)
    r.decide(p, (), _policy())
    row = json.loads((tmp_path / "routing.jsonl").read_text())
    assert row["input_metadata"]["input_tokens"] == 37


def test_expired_budget_does_not_call_client(tmp_path):
    p, r, c, a, _ = setup(tmp_path, remaining=-1)
    assert r.decide(p, (), _policy()).target.kind == ModelTargetKind.LARGE_MODEL
    assert c.calls == []
    assert a.summary(p.workflow_run_id)["budget_unavailable"] == 1


def test_route_deadline_is_tighter_than_caller(tmp_path):
    p, r, c, a, _ = setup(tmp_path)
    p = p.model_copy(update={"produced_at": real_utcnow() - timedelta(seconds=10)})
    r.decide(p, (), _policy())
    assert not c.calls


def test_missing_context_is_not_synthesized(tmp_path):
    p, r, c, a, store = setup(tmp_path)
    store.clear()
    r.decide(p, (), _policy())
    assert not c.calls
    assert a.summary(p.workflow_run_id)["input_unavailable"] == 1


def test_audit_failure_is_visible_and_does_not_change_route(tmp_path):
    p, r, c, a, _ = setup(tmp_path)
    a.path = tmp_path  # directory cannot be opened as a file
    assert r.decide(p, (), _policy()).target.kind == ModelTargetKind.SMALL_MODEL
    assert a.summary(p.workflow_run_id)["routing_audit_complete"] is False
