"""`shadow_route`：预测器照常算，但**决定不生效**。

为什么做在 router 层而不是预测器层
--------------------------------
「不生效」是对**路由输出**（`RoutePlan`）的操作，不是对预测器的操作。放在
router 层，换预测器（v4 ↔ add4）、换预测器的**数量**、换 mode，这一层
**一行都不用改** —— 这是「可拔插」要求的具体落点。

⚠️ 装饰器**必须保留** `decision_meta` 里的预测器字段（`classifier_p` /
`classifier_p_gate` / `classifier_predictor`）：那是「M2 真的算了」的唯一证据。
只覆盖 `reason_code` 并补记原本的走向，别的字段原样透传。

⚠️ 它还必须把**整个 plan 形状**换掉（`execution_mode` / `verifier_plan` /
`fallback_plan`），不只是换 target。否则内层给出的 `GUARDED_SLM` 会把强制
到「大模型」的那次调用拖进守卫路径 —— 那正好是这组实验要避开的东西。
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from traceoffload.m2.predictor import Add4Decision
from traceoffload.config import load_pilot_config
from traceoffload.fakes import (
    FakeAgentEnv,
    FakeEvaluator,
    FakeMiniSweModel,
    FakeModelGateway,
    FakeVerifier,
)
from traceoffload.contracts import (
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    Component,
    DecisionMeta,
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
from traceoffload.runner import _ClassifierRouter, _ShadowRouter, PilotRunner

NOW = datetime.now(timezone.utc)
LLM = ModelRef(kind="model", id="large", version="v1")
SLM = ModelRef(kind="model", id="small", version="v1")

_LINEAGE: dict[str, Any] = {
    "workflow_run_id": "workflow-1",
    "workflow_plan_ref": {"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
    "workflow_plan_revision": 0,
    "episode_id": "episode-1",
    "logical_invocation_id": "invocation-1",
}


def _meta(**over: Any) -> DecisionMeta:
    base: dict[str, Any] = {
        "policy_version": "policy-1",
        "feedback_version": "feedback-1",
        "reason_code": "l2_add4_slm",
        "is_exploration": False,
        "selection_probability": 1.0,
    }
    return DecisionMeta(**{**base, **over})


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


def _artifact() -> Any:
    return type(
        "Artifact",
        (),
        {
            "threshold": 0.5,
            "artifact_type": "test-artifact",
            "feature_names": ("a", "b"),
            "predict_proba": lambda self, features: 0.99,
        },
    )()


def _inner(offload: bool) -> _ClassifierRouter:
    """**真的** `_ClassifierRouter`，不是替身 —— 它产出的 plan 是生产形状。

    替身会让这组测试失去意义：装饰器要处理的是真实的 `RoutePlan`（带守卫
    计划、带 fallback），自己手搓一个「差不多」的 plan 测不出形状换没换全。
    """
    decision = Add4Decision(
        p_gate=0.9, p_net=0.9, tau_gate=0.5381, tau_net=0.5523, offload=offload
    )
    return _ClassifierRouter(LLM, SLM, lookup=lambda profile: decision)


def _offloading_inner() -> _ClassifierRouter:
    return _inner(offload=True)


def _not_offloading_inner() -> _ClassifierRouter:
    return _inner(offload=False)


def _shadow_of(inner: Any) -> _ShadowRouter:
    return _ShadowRouter(inner, LLM, SLM, force="large")


# --------------------------------------------------------------------------
# 契约：两个新字段
# --------------------------------------------------------------------------

def test_decision_meta_declares_the_shadow_fields() -> None:
    """`extra="forbid"` —— 不声明就写不进去（会抛 ValidationError）。"""
    meta = _meta(shadow_forced_target="large", shadow_would_route="l2_add4_slm")

    assert meta.shadow_forced_target == "large"
    assert meta.shadow_would_route == "l2_add4_slm"


def test_shadow_fields_default_to_none() -> None:
    """没开 shadow 时必须是 None。

    有值的话汇总脚本分不清「这一批开了 shadow」和「没开」—— 而这两批的
    `reason_code` 分布完全不同，混在一起会把卸载率算错。
    """
    meta = _meta()

    assert meta.shadow_forced_target is None
    assert meta.shadow_would_route is None


# --------------------------------------------------------------------------
# 装饰器：内层的判定照跑，输出的 plan 被换成 DIRECT + 固定目标
# --------------------------------------------------------------------------

def test_shadow_replaces_the_whole_plan_shape_not_just_the_target() -> None:
    """**只换 target 是不够的。**

    内层给的是 `GUARDED_SLM` + 守卫 + 指回大模型的 fallback。若装饰器只把
    `target` 换成大模型、留着重守卫的 `execution_mode`，那这次调用会被拖进
    守卫路径 —— 而「小模型也走大模型的路径」正是这组实验要保证的东西。

    所以四个字段必须一起换：`target` / `execution_mode` / `verifier_plan` /
    `fallback_plan`。
    """
    plan = _shadow_of(_offloading_inner()).decide(_profile(), (), _policy())

    assert plan.target.kind == ModelTargetKind.LARGE_MODEL
    assert plan.target.model_ref == LLM
    assert plan.execution_mode == ExecutionMode.DIRECT
    assert plan.verifier_plan.verifier_plan_id == "verifier-none"
    assert plan.fallback_plan is None


def test_shadow_records_what_the_router_would_have_done() -> None:
    """`shadow_would_route` = 内层原本的 `reason_code`。

    没有这个字段，「路由器本来会卸多少」就只能离线估 —— 而那个数正是
    对照组存在的理由。
    """
    plan = _shadow_of(_offloading_inner()).decide(_profile(), (), _policy())

    assert plan.decision_meta.reason_code == "shadow_forced_large"
    assert plan.decision_meta.shadow_would_route == "l2_add4_slm"
    assert plan.decision_meta.shadow_forced_target == "large"


def test_shadow_keeps_the_predictor_fields() -> None:
    """**预测器那几栏不许被动。**

    `classifier_p` / `classifier_p_gate` / `classifier_predictor` 是
    「M2 真的算了」的唯一证据。装饰器把它们一起覆盖掉的话，对照组就退化成
    了「没装预测器」，说服力全没。
    """
    plan = _shadow_of(_offloading_inner()).decide(_profile(), (), _policy())

    assert plan.decision_meta.classifier_predictor == "l2_add4_two_stage"
    assert plan.decision_meta.classifier_p == 0.9
    assert plan.decision_meta.classifier_p_gate == 0.9
    assert plan.decision_meta.classifier_tau_gate == 0.5381


def test_shadow_still_marks_a_span_the_inner_would_not_offload() -> None:
    """内层本来就走大模型时，plan 内容不变，但**仍要留痕**。

    不然「这批开了 shadow」这件事只出现在真正被改写的那些 span 上，
    汇总时按 `shadow_would_route is not None` 筛会漏掉大半。
    """
    plan = _shadow_of(_not_offloading_inner()).decide(_profile(), (), _policy())

    assert plan.target.kind == ModelTargetKind.LARGE_MODEL
    assert plan.execution_mode == ExecutionMode.DIRECT
    assert plan.decision_meta.reason_code == "shadow_forced_large"
    assert plan.decision_meta.shadow_would_route == "l2_add4_llm"


def test_shadow_rejects_an_unknown_forced_target() -> None:
    """`force` 只收 large/small。写错必须报错。

    静默收下一个拼错的字符串（比如 `Large`）会让装饰器把目标换成一个
    不存在的 kind —— 那会在执行期以一个不相干的错误冒出来。
    """
    with pytest.raises(ValueError, match="force"):
        _ShadowRouter(_not_offloading_inner(), LLM, SLM, force="Large")


# --------------------------------------------------------------------------
# 配置：runner.shadow_route
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
ADD4_ARTIFACT = REPO / "router/artifact/l2_add4_twostage_logit.json"

needs_artifact = pytest.mark.skipif(
    not ADD4_ARTIFACT.is_file(), reason="add4 artifact 不在（先跑 sf_export_add4.py）"
)


def _pilot(tmp_path: Path, *, shadow: dict[str, Any] | None) -> PilotRunner:
    """一份最小可构造的 runner。

    预测器**真的装上**（用真 artifact）—— shadow 的意义就是「预测器在、
    只是不生效」，用一个假的 lookup 测不出这条语义。
    """
    config = deepcopy(load_pilot_config())
    config.setdefault("runner", {})["m4_memory"] = {
        "enabled": False,
        "embedding_mode": "m1_online",
        "template_version": "span-embed-v2",
    }
    config["runner"]["l2_v4"] = {"enabled": False}
    config["runner"]["l2_add4"] = {
        "enabled": True,
        "model_path": str(ADD4_ARTIFACT),
    }
    if shadow is not None:
        config["runner"]["shadow_route"] = shadow
    return PilotRunner(
        config=config,
        mode="classifier_route",
        output_dir=tmp_path,
        gateway=FakeModelGateway(),
        model=FakeMiniSweModel(),
        env_factory=FakeAgentEnv,
        verifier=FakeVerifier(),
        evaluator=FakeEvaluator(),
        step_limit=3,
    )


@needs_artifact
def test_shadow_route_wraps_the_router(tmp_path: Path) -> None:
    runner = _pilot(tmp_path, shadow={"enabled": True, "force_target": "large"})

    assert isinstance(runner._make_router(), _ShadowRouter)


@needs_artifact
def test_shadow_route_disabled_leaves_the_router_unwrapped(tmp_path: Path) -> None:
    """`enabled: false` 时**完全不包装** —— 返回的是内层那个对象本身。

    包一层「什么都不做」的装饰器也能跑，但那样 `type()` 变了，
    别的测试按类型判断会失准。
    """
    runner = _pilot(tmp_path, shadow={"enabled": False, "force_target": "large"})

    assert not isinstance(runner._make_router(), _ShadowRouter)


@needs_artifact
def test_shadow_route_absent_leaves_the_router_unwrapped(tmp_path: Path) -> None:
    """没写这段配置 ⇒ 与「关着」完全同义。老配置不能因为缺键而变行为。"""
    runner = _pilot(tmp_path, shadow=None)

    assert not isinstance(runner._make_router(), _ShadowRouter)


@needs_artifact
@pytest.mark.parametrize("bad", ["Large", "big", "", 1, None])
def test_shadow_route_rejects_a_bad_force_target(tmp_path: Path, bad: Any) -> None:
    """**构造期**就要拦。留到 `decide()` 才炸的话，跑批会死在半路。"""
    with pytest.raises(ValueError, match="force_target"):
        _pilot(tmp_path, shadow={"enabled": True, "force_target": bad})

