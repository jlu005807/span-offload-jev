"""记录层：Coordinator 生成的完整执行记录。

- ExecutionSummary / RuntimeProjection：一次 episode 的执行与画像记录
- SpanRecord：一次 episode 终态后的完整记录
- 旧的投影类型名保留为兼容别名
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import (
    CanonicalKey,
    ContractModel,
    ExecutionStatus,
    Identifier,
    MessageEnvelope,
    VersionedRef,
)
from .execution import GuardResult, ModelAttempt, StateOutcome, ToolExecutionResult
from .profiling import RuntimeProfile, UncertainField
from .recovery import RecoveryCheck, RecoveryDirective
from .requests import ModelRequest
from .routing import RoutePlan


class ExecutionSummary(ContractModel):
    """SpanRecord 内的完整执行摘要。"""

    route_plan_id: Identifier = Field(min_length=1, max_length=128)
    workflow_run_id: Identifier = Field(min_length=1, max_length=128)
    workflow_plan_ref: VersionedRef | None = None
    workflow_plan_revision: int | None = Field(default=None, ge=0)
    episode_id: Identifier = Field(min_length=1, max_length=128)
    logical_invocation_id: Identifier = Field(min_length=1, max_length=128)
    status: ExecutionStatus
    model_attempts: tuple[ModelAttempt, ...] = ()
    guard_results: tuple[GuardResult, ...] = ()
    tool_executions: tuple[ToolExecutionResult, ...] = ()
    state_outcome: StateOutcome
    accepted_model_attempt_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def accepted_attempt_is_present(self) -> ExecutionSummary:
        """被接受 attempt 必须存在于 attempts 列表。"""
        ids = {attempt.model_attempt_id for attempt in self.model_attempts}
        if self.accepted_model_attempt_id is not None and self.accepted_model_attempt_id not in ids:
            raise ValueError("summary accepted_model_attempt_id must reference an attempt")
        return self


class RuntimeProjection(ContractModel):
    """M1 profile 的学习投影：descriptor/family key + uncertain_fields。

    生产者：Coordinator（从 RuntimeProfile 投影）；消费者：M4（经验分桶）。
    ``route_family_key`` 用于跨候选安全隔离；它是画像索引，不替代同一
    SpanRecord 中保存的完整请求和执行事实。
    """

    descriptor_key: CanonicalKey = Field(min_length=1, max_length=512)
    route_family_key: CanonicalKey | None = None
    uncertain_fields: tuple[UncertainField, ...] = ()


class SpanRecord(MessageEnvelope):
    """一次 episode 执行终态后的完整记录（Coordinator 生成）。

    生产者：Coordinator（_build_span）；消费者：M4、replay、task_trace。
    能力引用仍用于授权，但记录本身保留完整 runtime profile、请求、模型响应和工具输出，
    便于在线调试与 M4 分析。
    """

    runtime_projection: RuntimeProjection
    route_plan: RoutePlan
    execution_summary: ExecutionSummary
    runtime_profile: RuntimeProfile | None = None
    model_request: ModelRequest | None = None
    # A fallback is a separate physical model call, therefore a separate span.
    # The relation is explicit rather than inferred from attempt roles.
    span_kind: Literal["normal", "fallback", "rollback_recovery"] = "normal"
    parent_episode_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    span_relation: Literal["fallback_of", "recovery_of"] | None = None
    recovery_directive: RecoveryDirective | None = None
    recovery_check: RecoveryCheck | None = None

    @model_validator(mode="after")
    def route_and_summary_match(self) -> SpanRecord:
        """route_plan 与 execution_summary 的 route_plan_id 及五个关联字段必须一致。"""
        plan = self.route_plan
        summary = self.execution_summary
        if plan.route_plan_id != summary.route_plan_id:
            raise ValueError("span_record route_plan and execution_summary IDs differ")
        for field_name in (
            "workflow_run_id",
            "workflow_plan_ref",
            "workflow_plan_revision",
            "episode_id",
            "logical_invocation_id",
        ):
            if getattr(plan, field_name) != getattr(summary, field_name):
                raise ValueError(f"span_record lineage mismatch: {field_name}")
        if (self.parent_episode_id is None) != (self.span_relation is None):
            raise ValueError("span parent_episode_id and span_relation must be set together")
        if self.span_kind == "fallback":
            if self.span_relation != "fallback_of":
                raise ValueError("fallback span requires fallback_of relation")
        elif self.span_kind == "rollback_recovery":
            if self.span_relation != "recovery_of":
                raise ValueError("rollback_recovery span requires recovery_of relation")
            if self.recovery_directive is None or self.recovery_check is None:
                raise ValueError("rollback_recovery span requires directive and check")
        elif self.span_relation is not None:
            raise ValueError("normal span cannot have a parent relation")
        if self.recovery_check is not None and self.recovery_directive is None:
            raise ValueError("recovery_check requires recovery_directive")
        return self



__all__ = [
    "ExecutionSummary",
    "RuntimeProjection",
    "SpanRecord",
]
