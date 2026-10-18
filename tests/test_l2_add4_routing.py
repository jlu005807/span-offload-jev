"""`_ClassifierRouter` 在 **add4 两阶段**下的出计划行为。

与 `test_classifier_router_execution_mode.py`（单标量 v4）分开：那个钉的是
「一个 p 跟一个 τ 比」，这个钉的是「**两个** p 的与」，以及**两个 τ 都要进
`decision_meta`** —— 只看一个 p 分不出「全没卸载」是门控挡的还是经济挡的。

用的 `Add4Decision` 是**真的那个 dataclass**（不是替身）：判定语义本身由
`tests/components/test_l2_add4_predictor.py` 负责，这里只管「拿到判定之后
怎么变成路由」。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from traceoffload.m2.predictor import Add4Decision
from traceoffload.contracts import (
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    Component,
    DownstreamDependency,
    EffectRisk,
    ExecutionMode,
    ModelRef,
    ModelTargetKind,
    ProducerRef,
    RecoveryLevel,
    RoutePolicy,
    RuntimeProfile,
    SpanDescriptor,
)
from traceoffload.runner import _ClassifierRouter

NOW = datetime.now(timezone.utc)
LLM = ModelRef(kind="model", id="large", version="v1")
SLM = ModelRef(kind="model", id="small", version="v1")

_LINEAGE = {
    "workflow_run_id": "workflow-1",
    "workflow_plan_ref": {"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
    "workflow_plan_revision": 0,
    "episode_id": "episode-1",
    "logical_invocation_id": "invocation-1",
}


def _profile() -> RuntimeProfile:
    return RuntimeProfile(
        schema_version="1.0",
        message_id="message-profile-1",
        producer_ref=ProducerRef(component=Component.M1, version="0.1.0"),
        produced_at=NOW,
        **_LINEAGE,
        request_ref=CapabilityRef(
            kind="request",
            id="request-1",
            version="v1",
            scope=CapabilityScope.PROFILE_READ,
            audience=Component.M1,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(**_LINEAGE),
        ),
        descriptor_key="descriptor/v1|output:none|tools:none",
        descriptor=SpanDescriptor(
            subtask_type="tool_call_generation",
            estimated_input_tokens=100,
            output_contract="none",
            allowed_targets="unknown",
            downstream_dependency=DownstreamDependency.FULL,
            effect_risk=EffectRisk.NONE,
            available_verifiers=(),
            recovery_level=RecoveryLevel.NOT_NEEDED,
        ),
        online_features={"prefix_n_A": 1.0, "log_est_prompt_tokens": 5.0},
    )


def _policy() -> RoutePolicy:
    return RoutePolicy(
        policy_version="policy-1",
        correctness_lower_bound=0.9,
        risk_upper_bound=0.05,
        budget={"max_cost": 10.0, "max_latency_ms": 5000.0},
        utility_weights={"cost": 1.0, "latency": 1.0, "risk": 1.0},
        exploration_constraints={},
        conservative_prior={
            "p_unsafe": 0.02,
            "p_episode_success": 0.5,
            "expected_cost": 0.5,
            "expected_latency_ms": 500.0,
        },
    )


def _counting_artifact() -> Any:
    """旧 artifact 的替身：`predict_proba` 一被调用就记下来。

    两阶段那套**不许**碰它 —— 拿不到向量时应当直接走大模型，而不是让旧
    预测器来出分（那等于中途换了模型）。
    """
    calls: list[Any] = []

    def _predict(features: Any) -> float:
        calls.append(features)
        return 0.99

    artifact = SimpleNamespace(
        threshold=0.5,
        artifact_type="test-artifact",
        feature_names=("a", "b"),
        predict_proba=_predict,
    )
    artifact.calls = calls
    return artifact


def _router(decision: Add4Decision | None, artifact: Any = None) -> _ClassifierRouter:
    return _ClassifierRouter(
        LLM,
        SLM,
        lookup=lambda profile: decision,
    )


def _decision(*, offload: bool, p_gate: float = 0.9, p_net: float = 0.9) -> Add4Decision:
    return Add4Decision(
        p_gate=p_gate, p_net=p_net, tau_gate=0.5381, tau_net=0.5523, offload=offload
    )


# --------------------------------------------------------------------------
# 卸载 ⇒ 小模型 + 守卫；不卸载 ⇒ 大模型 DIRECT
# --------------------------------------------------------------------------

def test_offload_gets_a_guarded_small_model_plan() -> None:
    plan = _router(_decision(offload=True)).decide(_profile(), (), _policy())

    assert plan.target.kind == ModelTargetKind.SMALL_MODEL
    assert plan.execution_mode == ExecutionMode.GUARDED_SLM
    assert plan.verifier_plan.steps, "guard 路径必须有 pre-check"
    assert plan.fallback_plan is not None
    assert plan.fallback_plan.target.kind == ModelTargetKind.LARGE_MODEL


def test_no_offload_stays_direct_on_the_large_model() -> None:
    plan = _router(_decision(offload=False)).decide(_profile(), (), _policy())

    assert plan.target.kind == ModelTargetKind.LARGE_MODEL
    assert plan.execution_mode == ExecutionMode.DIRECT
    assert plan.verifier_plan.steps == ()
    assert plan.fallback_plan is None
    assert plan.decision_meta.reason_code == "l2_add4_llm"


# --------------------------------------------------------------------------
# 两个 p 都必须进 decision_meta
# --------------------------------------------------------------------------

def test_both_probabilities_and_thresholds_are_recorded() -> None:
    """**只看一个 p 分不出是门控挡的还是经济挡的** —— 这两种情况调参方向相反。"""
    plan = _router(
        _decision(offload=False, p_gate=0.10, p_net=0.95)
    ).decide(_profile(), (), _policy())
    meta = plan.decision_meta

    assert meta.classifier_predictor == "l2_add4_two_stage"
    assert meta.classifier_p_gate == 0.10, "门控那一段"
    assert meta.classifier_p == 0.95, "经济那一段（沿用单值字段）"
    assert meta.classifier_tau_gate == 0.5381
    assert meta.classifier_threshold == 0.5523


def test_predictor_name_distinguishes_the_two_stages() -> None:
    """`l2_add4_two_stage` 与 `l2_v4` / `l2_new_v1` 必须能分开 —— 汇总靠它归因。"""
    plan = _router(_decision(offload=True)).decide(_profile(), (), _policy())
    assert plan.decision_meta.classifier_predictor == "l2_add4_two_stage"


# --------------------------------------------------------------------------
# 拿不到向量 ⇒ 大模型，且**不许**碰旧 artifact
# --------------------------------------------------------------------------

def test_missing_vector_falls_back_to_the_large_model() -> None:
    plan = _router(None).decide(_profile(), (), _policy())

    assert plan.target.kind == ModelTargetKind.LARGE_MODEL
    assert plan.execution_mode == ExecutionMode.DIRECT
    assert plan.decision_meta.reason_code == "l2_add4_vector_unavailable"
    assert plan.decision_meta.classifier_predictor == "l2_add4_unavailable"
    assert plan.decision_meta.classifier_p is None, "这一刀没有任何预测器出过分"


def test_missing_vector_never_consults_the_old_artifact() -> None:
    """与 v4 同一条纪律：`predict_proba` 返回 0.99（会卸载），

    所以「计划是大模型」+「一次都没调用」两条一起，才证明真的没退回。
    """
    artifact = _counting_artifact()
    plan = _router(None, artifact).decide(_profile(), (), _policy())

    assert artifact.calls == [], "退回旧预测器了 —— 那等于中途换了模型"
    assert plan.target.kind == ModelTargetKind.LARGE_MODEL


# --------------------------------------------------------------------------
# 回归：没配 add4 时，v4 那条路逐字不变
# --------------------------------------------------------------------------
