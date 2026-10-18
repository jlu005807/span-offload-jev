"""`_observe_m4_span` 的取响应逻辑：**失败时不能只看 `accepted_response`**。

Coordinator 的回调签名是
``span_observer(record, accepted_response: ModelResponse | None)`` ——
**只有被接受时那个参数才非 None**。

而失败（`verification_failed`）恰恰是最需要记的时候：那正是「这个动作
不该做」的证据。只看 `accepted_response` 会在每一次失败上记下空集 ——
守卫于是**永远学不到东西**，而日志里看不出任何异常（记空集不报错）。

候选响应一直躺在 ``record.execution_summary.model_attempts`` 里
（`role=small_model_candidate` 那条的 `response`），所以取不到顶层参数时
应当回头从那里取。
"""

from types import SimpleNamespace

from traceoffload.runner import _candidate_response_for_m4


def _call(call_id: str, code: str) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call_id=call_id,
        tool_name="execute_code_batch",
        arguments={"codes": [code]},
    )


def _attempt(role: str, *, calls=(), status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        status=status,
        response=SimpleNamespace(tool_calls=tuple(calls)),
    )


def test_prefers_the_response_passed_by_the_coordinator() -> None:
    """顶层给了响应就用它 —— 这是「被接受」的正常路径。"""

    passed = SimpleNamespace(tool_calls=(_call("c-passed", "passed()"),))
    record = SimpleNamespace(
        execution_summary=SimpleNamespace(
            model_attempts=(_attempt("small_model_candidate", calls=[_call("c-x", "x()")]),)
        )
    )

    assert _candidate_response_for_m4(record, passed) is passed


def test_falls_back_to_the_candidate_attempt_when_the_response_is_none() -> None:
    """**失败路径**：顶层是 None，必须回头从候选尝试里取。"""

    candidate = _call("c-1", "boom()")
    record = SimpleNamespace(
        execution_summary=SimpleNamespace(
            model_attempts=(
                _attempt("small_model_candidate", calls=[candidate]),
                _attempt("fallback", calls=[_call("c-2", "other()")]),
            )
        )
    )

    response = _candidate_response_for_m4(record, None)

    assert response is not None
    assert [c.tool_call_id for c in response.tool_calls] == ["c-1"]


def test_returns_none_when_there_is_no_candidate_attempt() -> None:
    """没有候选尝试 ⇒ None（调用方按「没有失败签名」处理）。"""

    record = SimpleNamespace(execution_summary=SimpleNamespace(model_attempts=()))

    assert _candidate_response_for_m4(record, None) is None


def test_tolerates_a_record_without_an_execution_summary() -> None:
    assert _candidate_response_for_m4(SimpleNamespace(), None) is None
