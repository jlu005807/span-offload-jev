"""Observe native Agent termination without changing prompts or execution."""
from functools import wraps


def observe_termination(agent):
    """Capture status.failed messages even when native verbose logging is off."""
    messages = []
    logger = getattr(agent, "logger", None)
    original = getattr(logger, "show_message", None)
    if not callable(original):
        return messages

    @wraps(original)
    def show_message(role, content, *args, **kwargs):
        if role == "termination":
            messages.append(str(content))
        return original(role, content, *args, **kwargs)

    logger.show_message = show_message
    return messages


def finalize_task_outcome(row, terminations=()):
    """Add a stable failure category; keep usage and raw evaluation counters."""
    errors = [str(row["failure"])] if row.get("failure") else []
    errors.extend(terminations)
    text = "\n".join(errors).lower()
    if any(marker in text for marker in (
        "maximum context length", "context_length_exceeded", "context window exceeded",
    )):
        reason = "context_length_exceeded"
    elif row.get("failure"):
        reason = "runtime_error"
    elif terminations:
        reason = "agent_terminated"
    elif row.get("task_success") is not True:
        reason = "evaluation_failed"
    else:
        reason = None
    row["task_failure_reason"] = reason
    if reason is not None:
        row["task_success"] = False
    return row
