"""``M4FailureMemoryVerifier``：在形状检查之上叠一层 M4 失败记忆。

它 **继承** ``BasicToolCallVerifier`` 而不是替换它 ——
形状/白名单检查是现成的、有用的，M4 是加在它**后面**的一道。

判定顺序是契约的一部分：

    1. 形状不过          → REJECT（**短路，不查 M4** —— 形状都不合法就不必查历史）
    2. 形状过了 → 查 M4  → 命中则 REJECT
    3. 查不到 / 没命中   → PASS
    4. M4 抛异常         → PASS（**fail-open** —— 守卫不能因为记忆库坏了而卡住路由）

``lookup`` 是注入的：verifier 只负责**顺序**，谁去查、查什么由调用方给。
这样 M4 的接线（span 向量、记忆库）不用搬进守卫里。
"""

from typing import Any

from traceoffload.contracts import (
    FailureAction,
    FinishReason,
    GuardOutcome,
    GuardRef,
    ModelResponse,
    ModelResponseToolCall,
    ParseStatus,
    PassAction,
    PassCondition,
    PassConditionKind,
    VerifierStage,
    VerifierStep,
    VerifierSubject,
)
from traceoffload.m3.guards import M4FailureMemoryVerifier, ToolPolicy
from traceoffload.interfaces import SpanIdentity


def _policy() -> ToolPolicy:
    return ToolPolicy.from_mapping(
        {
            "allowed_tools": ["query"],
            "tools": {
                "query": {
                    "required_argument_paths": ["sql"],
                    "non_empty_string_paths": ["sql"],
                    "allowed_argument_keys": ["sql"],
                }
            },
        }
    )


def _step() -> VerifierStep:
    return VerifierStep(
        step_id="pre",
        guard_ref=GuardRef(kind="guard", id="m4-failure-memory", version="v1"),
        stage=VerifierStage.PRE_COMMIT,
        subject=VerifierSubject.TOOL_CALLS,
        pass_condition=PassCondition(kind=PassConditionKind.TOOL_ARGUMENTS_ALLOWED),
        timeout_ms=10,
        on_pass=PassAction.CONTINUE,
        on_failure=FailureAction.REJECT_AND_FALLBACK,
    )


def _candidate(arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        response_id="response-1",
        model_attempt_id="attempt-1",
        tool_calls=(
            ModelResponseToolCall(
                tool_call_id="call-1",
                tool_name="query",
                arguments=arguments,
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        parse_status=ParseStatus.PARSED,
    )


def test_rejects_when_memory_reports_a_matching_failure() -> None:
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: ("neighbor-r1",),
    )

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1"}),
        model_attempt_id="attempt-1",
    )

    assert result.outcome == GuardOutcome.REJECT


def test_shape_failure_short_circuits_without_querying_memory() -> None:
    """形状不过就必须直接拒 —— **而且不能去查 M4**。"""

    queried: list[Any] = []
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: queried.append(candidate) or (),
    )

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1", "unexpected": True}),
        model_attempt_id="attempt-1",
    )

    assert result.outcome == GuardOutcome.REJECT
    assert queried == [], "形状已经不过了，不该再去查记忆库"


def test_passes_when_memory_reports_no_match() -> None:
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: (),
    )

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1"}),
        model_attempt_id="attempt-1",
    )

    assert result.outcome == GuardOutcome.PASS


def test_passes_when_memory_lookup_raises() -> None:
    """fail-open：记忆库坏了要**放行**，不能让路由卡住。"""

    def _offline(candidate: ModelResponse, identity: Any) -> tuple[Any, ...]:
        raise RuntimeError("memory store offline")

    verifier = M4FailureMemoryVerifier(tool_policy=_policy(), lookup=_offline)

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1"}),
        model_attempt_id="attempt-1",
    )

    assert result.outcome == GuardOutcome.PASS


def test_passes_when_no_lookup_is_configured() -> None:
    """没接 M4 时行为与纯形状检查一致 —— 这是「包一层」的退化保证。"""

    verifier = M4FailureMemoryVerifier(tool_policy=_policy())

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1"}),
        model_attempt_id="attempt-1",
    )

    assert result.outcome == GuardOutcome.PASS


def test_lookup_receives_the_span_identity() -> None:
    """守卫必须告诉 M4「现在是哪个 span」—— 否则闭包拿不到它的 52 维。

    ``ModelResponse`` 里**没有任何 span 标识**（只有 response_id / model_attempt_id
    / tool_calls / …），所以身份只能由 executor 从 ``ExecutionRequest`` 里取出来、
    显式传下来。这就是 `Verifier.check` 要加 `span_context` 的原因。
    """

    seen: list[Any] = []
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: seen.append(identity) or (),
    )
    identity = SpanIdentity(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
    )

    result = verifier.check(
        _step(),
        candidate=_candidate({"sql": "select 1"}),
        model_attempt_id="attempt-1",
        span_context=identity,
    )

    assert result.outcome == GuardOutcome.PASS
    assert seen == [identity]
