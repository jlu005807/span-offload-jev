"""动作签名：M4 记忆与查询共用的那把钥匙。

为什么要有「签名」这一层，而不是直接用参数字典
----------------------------------------------
字典不能当集合元素（不可哈希），而且**键序不同的两个等值字典必须签出同一个名字** ——
否则同一个动作会因为 `json.dumps` 的键序差异被当成两个，命中率凭空掉一半。
所以签名必须是**规范化**的字符串。

签名**包含工具名**：`(工具, 参数)` 是一个整体。同一个参数交给不同工具是两回事。

为什么按 `tool_call_id` 连
-------------------------
参数在 `ModelResponse.tool_calls` 上，成败在 `ToolExecutionResult` 上，
**两张表都没有对方的字段**（`ToolExecutionResult` 里根本没有 `arguments`）。
唯一能对齐两者的是 `tool_call_id`。不按它连 = 把 A 动作的参数配到 B 动作的
成败上，而且**不会报错**。
"""

from types import SimpleNamespace

from traceoffload.m4.span_features import (
    action_signature,
    failed_action_signatures,
)
from traceoffload.contracts import ToolExecutionStatus


def test_action_signature_is_stable_under_argument_key_order() -> None:
    """同一个动作、键序不同 ⇒ 必须签出同一个名字。"""

    assert action_signature("query", {"sql": "select 1", "limit": 2}) == action_signature(
        "query", {"limit": 2, "sql": "select 1"}
    )


def test_action_signature_distinguishes_argument_values() -> None:
    assert action_signature("query", {"sql": "select 1"}) != action_signature(
        "query", {"sql": "select 2"}
    )


def test_action_signature_includes_the_tool_name() -> None:
    """同参数、不同工具 = 两个动作。"""

    assert action_signature("query", {"sql": "select 1"}) != action_signature(
        "exec", {"sql": "select 1"}
    )


# --------------------------------------------------------------------------- 连接
def _call(call_id: str, tool: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(tool_call_id=call_id, tool_name=tool, arguments=arguments)


def _result(
    call_id: str,
    tool: str,
    *,
    status: ToolExecutionStatus = ToolExecutionStatus.SUCCESS,
    returncode: int | None = 0,
    error_type: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call_id=call_id,
        tool_name=tool,
        status=status,
        returncode=returncode,
        error_type=error_type,
    )


def test_failed_action_signatures_pairs_arguments_with_the_matching_result() -> None:
    """参数在响应上、成败在结果上，靠 ``tool_call_id`` 对齐。

    两条调用都叫 ``query``、参数不同 —— 只有按 id 连才能把
    「b 失败了」配到 b 的参数上。按工具名连会把 a 的参数配错。
    """

    response = SimpleNamespace(
        tool_calls=(
            _call("c1", "query", {"sql": "a"}),
            _call("c2", "query", {"sql": "b"}),
        )
    )
    results = (
        _result("c1", "query"),
        _result("c2", "query", status=ToolExecutionStatus.FAILED, returncode=1),
    )

    assert failed_action_signatures(response, results) == frozenset(
        {action_signature("query", {"sql": "b"})}
    )


def test_failed_action_signatures_treats_nonzero_returncode_as_failure() -> None:
    """``status`` 说成功但返回码非零 ⇒ 仍然算失败（与 `compute_local_success` 同口径）。"""

    response = SimpleNamespace(tool_calls=(_call("c1", "query", {"sql": "a"}),))
    results = (_result("c1", "query", returncode=1),)

    assert failed_action_signatures(response, results) == frozenset(
        {action_signature("query", {"sql": "a"})}
    )


def test_failed_action_signatures_ignores_calls_that_never_ran() -> None:
    """响应里有、结果里没有 = **没执行**，不是失败。

    守卫拦下的动作正是这种情况 —— 它没跑，所以没有成败。
    把它当失败会把「拦下来的」写回库里，下次拦得更狠，形成自激。
    """

    response = SimpleNamespace(tool_calls=(_call("c1", "query", {"sql": "a"}),))

    assert failed_action_signatures(response, ()) == frozenset()


def test_failed_action_signatures_ignores_unknown_call_ids() -> None:
    """结果里有、响应里没有 = 对不上号，丢弃（宁可漏记，不可记错）。"""

    response = SimpleNamespace(tool_calls=(_call("c1", "query", {"sql": "a"}),))
    results = (_result("c9", "query", status=ToolExecutionStatus.FAILED, returncode=1),)

    assert failed_action_signatures(response, results) == frozenset()
