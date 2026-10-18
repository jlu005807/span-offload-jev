"""M4 动作守卫：``query_failed_action`` 的两级判定。

拦截条件是 **AND**，两个条件查的是不同的东西：

    · 相似度  —— span 的 52 维表示（场景像不像）
    · 动作签名 —— (工具名, 参数) 精确串（动作一不一样）

⇒ 命中 ⟺ 「在相似的场景下，做过一模一样的动作，**而且失败了**」。

四个分支各自一个测试，边界（相似度不足 / 动作没失败过 / 库为空）
都有独立用例 —— 这是 fail-open 的凭据：查不到就必须放行。
"""

from types import SimpleNamespace

from traceoffload.m4.memory import M4ClassSpec, M4Memory
from traceoffload.m4.span_features import action_signature


def _span(*, kind: str = "large_model"):
    """最小 Span 记录 —— ``observe`` 只要求它带 route_plan / execution_summary。"""

    return SimpleNamespace(
        route_plan=SimpleNamespace(
            workflow_run_id="workflow-1",
            episode_id="episode-1",
            logical_invocation_id="invocation-1",
            target=SimpleNamespace(kind=kind),
            verifier_plan=SimpleNamespace(steps=()),
        ),
        execution_summary=SimpleNamespace(status="success"),
    )


def _sig(tool: str, code: str) -> str:
    """测试用的动作签名 —— 与写入侧同一个函数，不自己拼格式。"""

    return action_signature(tool, {"code": code})


def _memory() -> M4Memory:
    return M4Memory(classes=(M4ClassSpec(class_key=0),), top_k=5)


def test_query_failed_action_returns_match_when_similar_and_action_failed() -> None:
    memory = _memory()
    memory.observe(
        "r1",
        (1.0, 0.0),
        _span(),
        failed_actions=(_sig("execute_code_batch", "boom()"),),
    )

    hits = memory.query_failed_action(
        (1.0, 0.0),
        _sig("execute_code_batch", "boom()"),
        similarity_threshold=0.8,
    )

    assert hits
    assert hits[0].record.record_id == "r1"


def test_query_failed_action_ignores_match_when_similarity_below_threshold() -> None:
    """AND 的左半边：动作一模一样，但场景不像 ⇒ 放行。"""

    memory = _memory()
    memory.observe(
        "r1",
        (1.0, 0.0),
        _span(),
        failed_actions=(_sig("execute_code_batch", "boom()"),),
    )

    hits = memory.query_failed_action(
        (0.0, 1.0),  # 与 (1.0, 0.0) 正交 ⇒ 相似度 0.0
        _sig("execute_code_batch", "boom()"),
        similarity_threshold=0.8,
    )

    assert hits == ()


def test_query_failed_action_ignores_match_when_action_never_failed() -> None:
    """AND 的右半边：场景一模一样，但那个动作没失败过 ⇒ 放行。"""

    memory = _memory()
    memory.observe(
        "r1",
        (1.0, 0.0),
        _span(),
        failed_actions=(_sig("execute_code_batch", "boom()"),),
    )

    hits = memory.query_failed_action(
        (1.0, 0.0),
        _sig("execute_code_batch", "ok()"),
        similarity_threshold=0.8,
    )

    assert hits == ()


def test_query_failed_action_returns_empty_for_empty_memory() -> None:
    """fail-open：库是空的就必须放行，不能因为没有历史而卡住。"""

    memory = _memory()

    hits = memory.query_failed_action(
        (1.0, 0.0),
        _sig("execute_code_batch", "boom()"),
        similarity_threshold=0.8,
    )

    assert hits == ()


def test_signatures_produced_by_the_writer_are_findable_by_the_reader() -> None:
    """写入侧与查询侧必须用**同一把钥匙**。

    这条是接口一致性测试：`failed_action_signatures` 产出的是**规范化字符串**，
    而 `query_failed_action` 如果按 `(工具, 参数)` 元组去比，就永远查不到 ——
    **而且不会报错**：守卫会安静地一直放行，日志里什么都看不出来。

    同理 `observe(failed_actions=...)` 收的是一个字符串的可迭代对象；
    如果实现里写成 `for tool, args in failed_actions`，一个字符串会被**逐字符解包**
    成 ('e','x') 这种垃圾，也不报错。
    """

    from traceoffload.m4.span_features import (
        action_signature,
        failed_action_signatures,
    )

    response = SimpleNamespace(
        tool_calls=(
            SimpleNamespace(
                tool_call_id="c1", tool_name="query", arguments={"sql": "boom"}
            ),
        )
    )
    results = (
        SimpleNamespace(
            tool_call_id="c1",
            tool_name="query",
            status="failed",
            returncode=1,
            error_type=None,
        ),
    )
    recorded = failed_action_signatures(response, results)
    assert recorded == frozenset({action_signature("query", {"sql": "boom"})})

    memory = _memory()
    memory.observe("r1", (1.0, 0.0), _span(), failed_actions=recorded)

    for signature in recorded:
        hits = memory.query_failed_action(
            (1.0, 0.0), signature, similarity_threshold=0.8
        )
        assert hits, f"写入侧记下的 {signature!r} 查询侧找不到"
        assert hits[0].record.record_id == "r1"
