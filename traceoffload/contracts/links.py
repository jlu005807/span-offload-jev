"""链接校验层：主链对象之间的关联校验函数。

Coordinator 在每次对象边界调用这些函数，防止 ID 漂移：
- validate_profile_link：M1 返回画像时
- validate_route_link：M2 返回路由方案时
- validate_execution_link：M3 返回执行结果时
- validate_outcome_link：延迟标签摄入时（由 outcomes 处理器包装）
"""

from __future__ import annotations

from .base import ContractError, InvocationEnvelope
from .capabilities import _validate_capability_binding
from .context import InvocationContext
from .execution import ExecutionRequest, ExecutionResult
from .outcomes import OutcomeEvent
from .profiling import RuntimeProfile
from .records import SpanRecord
from .routing import RoutePlan


def validate_profile_link(context: InvocationContext, profile: RuntimeProfile) -> None:
    """拒绝不是由该 invocation 产生的画像（Coordinator 在 M1 返回时调用）。"""

    _validate_same_invocation(context, profile)
    if context.request_ref != profile.request_ref:
        raise ContractError("runtime_profile must preserve invocation_context.request_ref")


def validate_route_link(profile: RuntimeProfile, route_plan: RoutePlan) -> None:
    """拒绝指向不同 episode 或 workflow 修订的路由方案。"""

    _validate_same_invocation(profile, route_plan)


def validate_execution_link(request: ExecutionRequest, result: ExecutionResult) -> None:
    """拒绝不回答其执行请求的结果。"""

    if request.route_plan.route_plan_id != result.route_plan_id:
        raise ContractError("execution_result.route_plan_id does not match execution_request")
    _validate_same_invocation(request.route_plan, result)


def validate_outcome_link(record: SpanRecord, event: OutcomeEvent) -> None:
    """拒绝其 ID 无法指到已记录 episode 的延迟标签。"""

    if event.workflow_run_id != record.route_plan.workflow_run_id:
        raise ContractError("outcome_event workflow_run_id does not match span_record")
    if event.episode_id is not None and event.episode_id != record.route_plan.episode_id:
        raise ContractError("outcome_event episode_id does not match span_record")
    if (
        event.logical_invocation_id is not None
        and event.logical_invocation_id != record.route_plan.logical_invocation_id
    ):
        raise ContractError("outcome_event logical_invocation_id does not match span_record")
    attempt_ids = {attempt.model_attempt_id for attempt in record.execution_summary.model_attempts}
    if event.model_attempt_id is not None and event.model_attempt_id not in attempt_ids:
        raise ContractError("outcome_event model_attempt_id does not match span_record")


def _validate_same_invocation(left: InvocationEnvelope, right: InvocationEnvelope) -> None:
    """五个关联字段逐一比较（链接校验的公共部分）。"""
    for field_name in (
        "workflow_run_id",
        "workflow_plan_ref",
        "workflow_plan_revision",
        "episode_id",
        "logical_invocation_id",
    ):
        if getattr(left, field_name) != getattr(right, field_name):
            raise ContractError(f"invocation association mismatch: {field_name}")


__all__ = [
    "validate_profile_link",
    "validate_route_link",
    "validate_execution_link",
    "validate_outcome_link",
]
