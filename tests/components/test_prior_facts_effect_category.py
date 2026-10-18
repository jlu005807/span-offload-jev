"""`effect_category` 从 tool_finished 事件流到 `PriorToolOutcome` 的管道。

这条管道服务的是在线特征 ``prefix_n_A``（"前缀里成功且是写的工具执行数"）：
它在 L1 pre-check 时判定，**不在** proxy 的 output dict 里，所以由 adapter 在
``tool_finished`` 的 ``result`` 里补上，再由这里读进 prior facts。

三件事分开钉：
  1. 带类别 -> 读到；
  2. 不带类别 -> 留 ``None``（**不猜**成只读也不猜成写）；
  3. 报文里是坏值时（空串 / 非字符串）-> 同样留 ``None``，不崩。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from traceoffload.m3.integration import RuntimeEvent, _prior_tool_outcome


def _event(*, result=None, error_type=None, tool_call_id="call-1") -> RuntimeEvent:
    payload: dict = {"result": result if result is not None else {}}
    if error_type is not None:
        payload["error_type"] = error_type
        payload["error"] = "boom"
    return RuntimeEvent(
        event_id="evt-1",
        sequence=1,
        event_type="tool_finished",
        occurred_at=datetime.now(timezone.utc),
        workflow_run_id="wf-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        tool_call_id=tool_call_id,
        model_attempt_id="attempt-1",
        payload=payload,
    )


def test_prior_tool_outcome_reads_effect_category() -> None:
    outcome = _prior_tool_outcome(
        _event(result={"returncode": 0, "effect_category": "declared_write"})
    )
    assert outcome.effect_category == "declared_write"
    assert outcome.success is True


def test_prior_tool_outcome_leaves_category_none_when_absent() -> None:
    """缺类别时留 None —— 在线统计据此**不计入**，而不是猜成只读或写。"""

    outcome = _prior_tool_outcome(_event(result={"returncode": 0}))
    assert outcome.effect_category is None

    # 空串 / 非字符串同样视为缺失
    assert (
        _prior_tool_outcome(
            _event(result={"returncode": 0, "effect_category": ""})
        ).effect_category
        is None
    )
    assert (
        _prior_tool_outcome(
            _event(result={"returncode": 0, "effect_category": 123})
        ).effect_category
        is None
    )


def test_prior_tool_outcome_keeps_failure_semantics_with_category() -> None:
    """类别不影响成败判定：失败的写仍然 success=False（离线归 "F"，不计入 A）。"""

    outcome = _prior_tool_outcome(
        _event(result={"returncode": 1, "effect_category": "declared_write"})
    )
    assert outcome.success is False
    assert outcome.effect_category == "declared_write"

    errored = _prior_tool_outcome(
        _event(result={"returncode": 0, "effect_category": "read_only"}, error_type="Boom")
    )
    assert errored.success is False


# ---------------------------------------------------------------------------
# 整条 session 链路：record_tool_finished -> 事件 -> PriorToolOutcome -> prefix_n_A
#
# 前半段的单测只覆盖了 `_prior_tool_outcome` 这一个环节。这里把中间那一段
# （session 真的把 effect_category 落到事件里）也接上——否则"管道通不通"
# 全靠推断。真机上另有 P12 探针覆盖同样的分支。
# ---------------------------------------------------------------------------


def _session():
    from traceoffload.m3.integration import AdapterSession, AdapterSessionSpec
    from traceoffload.contracts import WorkflowGoal

    return AdapterSession(
        AdapterSessionSpec(
            workflow_run_id="workflow-l2",
            agent_id="appworld",
            agent_version=None,
            workflow_goal=WorkflowGoal(goal_text="complete the task"),
        )
    )


def _finish(session, *, tool_call_id, returncode, effect_category):
    run_id = session.record_tool_started({"tool_call_id": tool_call_id})
    session.record_tool_finished(
        run_id, {"returncode": returncode, "effect_category": effect_category}
    )


def _finished_events(session):
    """把 session 里的 ``tool_finished`` 事件还原成 ``RuntimeEvent``。"""

    out = []
    for raw in session.to_dict()["events"]:
        if raw["event_type"] != "tool_finished":
            continue
        out.append(
            RuntimeEvent(
                event_id=raw["event_id"],
                sequence=raw["sequence"],
                event_type=raw["event_type"],
                occurred_at=datetime.fromisoformat(raw["occurred_at"]),
                workflow_run_id=raw["workflow_run_id"],
                episode_id=raw["episode_id"],
                logical_invocation_id=raw["logical_invocation_id"],
                tool_call_id=raw["tool_call_id"],
                model_attempt_id=raw["model_attempt_id"],
                payload=raw["payload"],
            )
        )
    return out


def test_failed_write_block_is_excluded_from_prefix_n_A_end_to_end() -> None:
    """失败的写块**不计入** `prefix_n_A`；成功的计入。逐块对照。

    两条的 effect_category 都是 `declared_write`，所以差异**只**来自成败
    —— 这正是与离线 `tool_event_kind`（`ok and is_write` -> "A"）的那条分界。
    """
    from traceoffload.m1.appworld_features import count_prefix_write_ops

    session = _session()
    _finish(session, tool_call_id="c1", returncode=1, effect_category="declared_write")  # 失败写
    _finish(session, tool_call_id="c2", returncode=0, effect_category="declared_write")  # 成功写
    _finish(session, tool_call_id="c3", returncode=0, effect_category="read_only")       # 成功读

    outcomes = [_prior_tool_outcome(e) for e in _finished_events(session)]
    assert [o.effect_category for o in outcomes] == [
        "declared_write",
        "declared_write",
        "read_only",
    ], "effect_category 没能从 record_tool_finished 流到事件 payload"
    assert [o.success for o in outcomes] == [False, True, True]
    assert count_prefix_write_ops(SimpleNamespace(tool_outcomes=tuple(outcomes))) == 1.0


def test_session_drops_category_when_adapter_did_not_supply_it() -> None:
    """adapter 没补类别时，事件里就没有这个键 -> prior facts 留 None -> 不计入。

    宁可少算也不猜：把"没分类"当成只读会低估 prefix_n_A，当成写会高估，
    两者都会让 p 偏，而且**都不可见**。
    """
    from traceoffload.m1.appworld_features import count_prefix_write_ops

    session = _session()
    _finish(session, tool_call_id="c1", returncode=0, effect_category=None)

    outcomes = [_prior_tool_outcome(e) for e in _finished_events(session)]
    assert outcomes[0].effect_category is None
    assert outcomes[0].success is True          # 成败照样判得出
    assert count_prefix_write_ops(SimpleNamespace(tool_outcomes=tuple(outcomes))) == 0.0
