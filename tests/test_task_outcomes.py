from types import SimpleNamespace

from experiments.task_outcomes import observe_termination, finalize_task_outcome


CONTEXT_ERROR = ("Error code: 400 - This model's maximum context length is 262144 tokens. "
                 "Your prompt contains at least 253953 input tokens and 8192 output tokens.")


def test_swallowed_context_error_is_a_failed_task_and_preserves_consumption():
    messages = []
    agent = SimpleNamespace(logger=SimpleNamespace(
        show_message=lambda *args, **kwargs: messages.append((args, kwargs))))
    terminations = observe_termination(agent)
    agent.logger.show_message(role="agent", content="not an error")
    agent.logger.show_message(role="termination", content=CONTEXT_ERROR)
    row = {"task_success": False, "failure": None, "total_tokens": 123,
           "framework_model_calls_by_target": {"large_model": 18}}
    finalize_task_outcome(row, terminations)
    assert row["task_failure_reason"] == "context_length_exceeded"
    assert row["task_success"] is False and row["failure"] is None
    assert row["total_tokens"] == 123
    assert row["framework_model_calls_by_target"] == {"large_model": 18}
    assert len(messages) == 2  # Logger forwarding remains unchanged.


def test_task_quality_failure_is_distinct_from_framework_exception():
    row = {"task_success": False, "failure": None}
    finalize_task_outcome(row, [])
    assert row["task_failure_reason"] == "evaluation_failed"
    crash = {"task_success": False, "failure": "RuntimeError: initialization failed"}
    finalize_task_outcome(crash, [])
    assert crash["task_failure_reason"] == "runtime_error"


def test_success_has_no_failure_reason_and_termination_overrides_success():
    row = {"task_success": True, "appworld_success": True}
    finalize_task_outcome(row, [])
    assert row["task_failure_reason"] is None
    finalize_task_outcome(row, [CONTEXT_ERROR])
    assert row["task_success"] is False
    assert row["task_failure_reason"] == "context_length_exceeded"


def test_termination_is_captured_even_when_logger_is_silent():
    agent = SimpleNamespace(logger=SimpleNamespace(show_message=lambda *args, **kwargs: None))
    terminations = observe_termination(agent)
    agent.logger.show_message("termination", "Unusable model response")
    row = {"task_success": False}
    finalize_task_outcome(row, terminations)
    assert row["task_failure_reason"] == "agent_terminated"
