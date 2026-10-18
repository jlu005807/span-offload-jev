"""T2 计数接线：守卫那一段的可观测性（spec §3.2 / §7.3）。

为什么单独一个文件
------------------
这不是新功能，是**把已存在的判定过程变成可观测的**。行为一个字都不许变：
所有既有测试（判定顺序、fail-open、短路）必须原样通过。

没有它，`B1-no-guard` 和 `A5-full` 在报告里长得一模一样。
"""
from __future__ import annotations

from typing import Any

from traceoffload.m4.action_guard import lookup_failed_actions
from traceoffload.m4.memory import M4ClassSpec, M4Memory
from traceoffload.m4.store import SQLiteM4MemoryStore
from traceoffload.m4.span_features import action_signature
from traceoffload.mechanism_counters import MechanismCounters
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

VEC = (1.0, 0.0)
IDENTITY = SpanIdentity(
    workflow_run_id="workflow-1",
    episode_id="episode-1",
    logical_invocation_id="invocation-1",
)


# ─────────────────────────── 守卫查询侧 ───────────────────────────


def _memory_with(signature: str) -> M4Memory:
    m = M4Memory(classes=(M4ClassSpec(class_key=0),), top_k=5)
    m.observe("r1", VEC, _span(), failed_actions=(signature,))
    return m


def _span() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        route_plan=SimpleNamespace(
            workflow_run_id="workflow-1",
            episode_id="episode-1",
            logical_invocation_id="invocation-1",
            target=SimpleNamespace(kind="small_model"),
            verifier_plan=SimpleNamespace(steps=()),
        ),
        execution_summary=SimpleNamespace(status="success"),
    )


def _candidate_of(*codes: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        tool_calls=tuple(
            SimpleNamespace(
                tool_call_id=f"call-{i}",
                tool_name="execute_code_batch",
                arguments={"code": code},
            )
            for i, code in enumerate(codes)
        )
    )


def test_lookup_counts_each_tool_call_queried() -> None:
    """`m4_guard_lookups` 数的是**查了几次**（每 tool call 一次）。"""
    sig = action_signature("execute_code_batch", {"code": "boom()"})
    counters = MechanismCounters(keys=("m4_guard_lookups", "m4_guard_hits"), maps=())

    lookup_failed_actions(
        _candidate_of("boom()", "autre()", "encore()"),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with(sig),
        vector_key=lambda ident: "workflow-1|invocation-1",
        counters=counters,
    )

    assert counters.snapshot()["m4_guard_lookups"] == 3


def test_lookup_counts_hits() -> None:
    """`m4_guard_hits` 数的是**命中几次** —— 它才是「守卫真的拦下来了」的证据。"""
    sig = action_signature("execute_code_batch", {"code": "boom()"})
    counters = MechanismCounters(keys=("m4_guard_lookups", "m4_guard_hits"), maps=())

    lookup_failed_actions(
        _candidate_of("boom()", "safe()"),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with(sig),
        vector_key=lambda ident: "workflow-1|invocation-1",
        counters=counters,
    )

    snap = counters.snapshot()
    assert snap["m4_guard_lookups"] == 2
    assert snap["m4_guard_hits"] == 1


def test_lookup_without_identity_does_not_count_a_query() -> None:
    """身份是 None ⇒ **一个动作都没查**，不该记 lookup。

    否则「守卫没接线」和「守卫查了但没命中」看起来一样 —— 正是要区分的两件事。
    """
    counters = MechanismCounters(keys=("m4_guard_lookups", "m4_guard_hits"), maps=())

    lookup_failed_actions(
        _candidate_of("boom()"),
        None,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with(action_signature("execute_code_batch", {"code": "boom()"})),
        vector_key=lambda ident: "workflow-1|invocation-1",
        counters=counters,
    )

    assert counters.snapshot()["m4_guard_lookups"] == 0


def test_lookup_without_counters_still_works() -> None:
    """不接线时行为不变（组件默认 `counters=None`）。"""
    sig = action_signature("execute_code_batch", {"code": "boom()"})

    hits = lookup_failed_actions(
        _candidate_of("boom()"),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with(sig),
        vector_key=lambda ident: "workflow-1|invocation-1",
    )

    assert len(hits) == 1


# ─────────────────────────── 守卫判定侧 ───────────────────────────


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


def _resp(arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        response_id="response-1",
        model_attempt_id="attempt-1",
        tool_calls=(
            ModelResponseToolCall(
                tool_call_id="call-1", tool_name="query", arguments=arguments
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        parse_status=ParseStatus.PARSED,
    )


def _keys() -> MechanismCounters:
    return MechanismCounters(
        keys=("guard_checks", "guard_shape_rejects", "guard_memory_rejects"), maps=()
    )


def test_shape_reject_is_counted_as_shape_not_memory() -> None:
    """**这一条是整件事的重点。**

    `guard_shape_rejects` 和 `guard_memory_rejects` 必须分开数 ——
    它们答的是两个不同的问题：

    * 形状拒绝多 ⇒ 小模型输出格式差
    * 记忆拒绝多 ⇒ **M4 记忆真的在起作用**

    合成一个数，消融表里「M4 记忆这道值多少」就还是答不出来。
    """
    counters = _keys()
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: ("hit",),
        counters=counters,
    )

    result = verifier.check(
        _step(),
        candidate=_resp({"sql": "select 1", "unexpected": True}),  # 形状不过
        model_attempt_id="attempt-1",
    )

    snap = counters.snapshot()
    assert result.outcome == GuardOutcome.REJECT
    assert snap["guard_checks"] == 1
    assert snap["guard_shape_rejects"] == 1
    assert snap["guard_memory_rejects"] == 0


def test_memory_reject_is_counted_as_memory_not_shape() -> None:
    counters = _keys()
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: ("hit",),
        counters=counters,
    )

    result = verifier.check(
        _step(),
        candidate=_resp({"sql": "select 1"}),
        model_attempt_id="attempt-1",
    )

    snap = counters.snapshot()
    assert result.outcome == GuardOutcome.REJECT
    assert snap["guard_checks"] == 1
    assert snap["guard_shape_rejects"] == 0
    assert snap["guard_memory_rejects"] == 1


def test_pass_counts_the_check_but_no_reject() -> None:
    counters = _keys()
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(),
        lookup=lambda candidate, identity: (),
        counters=counters,
    )

    result = verifier.check(
        _step(), candidate=_resp({"sql": "select 1"}), model_attempt_id="attempt-1"
    )

    snap = counters.snapshot()
    assert result.outcome == GuardOutcome.PASS
    assert snap == {"guard_checks": 1, "guard_shape_rejects": 0, "guard_memory_rejects": 0}


def test_lookup_exception_passes_and_does_not_count_a_memory_reject() -> None:
    """fail-open 时**不能**记成记忆拒绝 —— 记忆库坏了不等于记忆拦下来了。"""

    def _boom(candidate: Any, identity: Any) -> Any:
        raise RuntimeError("记忆库坏了")

    counters = _keys()
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(), lookup=_boom, counters=counters
    )

    result = verifier.check(
        _step(), candidate=_resp({"sql": "select 1"}), model_attempt_id="attempt-1"
    )

    assert result.outcome == GuardOutcome.PASS
    assert counters.snapshot()["guard_memory_rejects"] == 0


def test_verifier_without_counters_still_works() -> None:
    verifier = M4FailureMemoryVerifier(
        tool_policy=_policy(), lookup=lambda candidate, identity: ("hit",)
    )

    result = verifier.check(
        _step(), candidate=_resp({"sql": "select 1"}), model_attempt_id="attempt-1"
    )

    assert result.outcome == GuardOutcome.REJECT


# ─────────────────────────── 库条数 ───────────────────────────


def test_store_can_count_records(tmp_path) -> None:
    """`m4_records_at_start` / `_at_end` 靠它。

    没有这两个数，「seed 没加载上」和「守卫就是不触发」看起来一样。
    """
    store = SQLiteM4MemoryStore(tmp_path / "m.sqlite3")

    assert store.count_records() == 0

    store.save_record(
        type(
            "_R",
            (),
            {
                "record_id": "r1",
                "workflow_run_id": "w",
                "episode_id": "e",
                "logical_invocation_id": "i",
                "class_key": 0,
                "embedding": VEC,
                "full_span_record": {},
                "local_success": False,
                "danger_tier": 1,
                "verifier_refs": (),
                "entered_at": 0.0,
                "last_access_at": 0.0,
                "pair_id": None,
                "switch_direction": None,
                "forced_class": False,
                "assignment_status": "no_candidate",
                "failed_actions": (),
            },
        )()
    )

    assert store.count_records() == 1
