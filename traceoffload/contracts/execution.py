"""执行层：M3 的执行请求与终态结果。

- ExecutionRequest：Coordinator 发给 M3 的一次受控执行命令（含全部权限引用）
- ModelAttempt / GuardResult / StateOutcome：执行过程中产生的记录对象
- ExecutionResult：M3 的终态回执（成功必须有 accepted，失败禁止有）
"""

from __future__ import annotations

from pydantic import AwareDatetime, Field, model_validator

from .base import (
    ActualEffect,
    AttemptRole,
    AttemptStatus,
    CapabilityScope,
    CommitStatus,
    Component,
    ContractModel,
    ExecutionStatus,
    GuardOutcome,
    GuardRef,
    Identifier,
    InvocationEnvelope,
    JsonObject,
    MessageEnvelope,
    ModelRef,
    NonNegativeFloat,
    ProducerRef,
    ResumeStatus,
    RollbackStatus,
    ToolExecutionStatus,
    VerifierStage,
    VersionedRef,
)
from .capabilities import CapabilityRef, _validate_capability_binding
from .routing import ExecutionMode, RoutePlan
from .responses import ModelResponse


class ExecutionRequest(MessageEnvelope):
    """Coordinator 发给 M3 的一次受控执行命令。

    把不可变路由方案与仅 M3 可用的请求、fallback 请求、事务与校准权限绑定
    在一起，并携带截止时间与幂等键。构造时校验：
    - request_ref 绑定目标 = 路由目标
    - fallback_request_ref 绑定目标 = fallback 目标（有 fallback 计划时必需）
    - safety_calibration 版本 = 路由反馈版本（防反馈穿越）
    - 非 direct 模式必须有事务能力引用
    """

    execution_request_id: Identifier = Field(min_length=1, max_length=128)
    route_plan: RoutePlan
    request_ref: CapabilityRef
    fallback_request_ref: CapabilityRef | None = None
    step_transaction_ref: CapabilityRef | None = None
    safety_calibration_ref: VersionedRef
    deadline_at: AwareDatetime
    idempotency_key: Identifier = Field(min_length=1, max_length=256)
    # The Coordinator sets this only for a child fallback episode.  The route
    # remains DIRECT, but the trace can still distinguish a fallback LLM call
    # from a normal large-model direct call.
    attempt_role_override: AttemptRole | None = None

    @model_validator(mode="after")
    def validate_bindings(self) -> ExecutionRequest:
        """请求引用与路由目标一致、fallback 引用与 fallback 目标一致、校准版本与反馈版本一致。"""
        _validate_capability_binding(self.request_ref, self.route_plan, CapabilityScope.EXECUTION_READ)
        if self.request_ref.audience != Component.M3:
            raise ValueError("execution_request.request_ref must target m3")
        if self.request_ref.content_hash is None:
            raise ValueError("execution_request.request_ref requires content_hash")
        if self.request_ref.expires_at <= self.produced_at:
            raise ValueError("request_ref expires_at must be after produced_at")
        if self.request_ref.binding.target_model_ref != self.route_plan.target.model_ref:
            raise ValueError("request_ref target_model_ref must match route_plan target")
        if self.route_plan.fallback_plan is None:
            if self.fallback_request_ref is not None:
                raise ValueError("fallback_request_ref requires a route fallback_plan")
        else:
            if self.fallback_request_ref is None:
                raise ValueError("route fallback_plan requires fallback_request_ref")
            _validate_capability_binding(
                self.fallback_request_ref,
                self.route_plan,
                CapabilityScope.EXECUTION_READ,
            )
            if self.fallback_request_ref.audience != Component.M3:
                raise ValueError("fallback_request_ref must target m3")
            if self.fallback_request_ref.content_hash is None:
                raise ValueError("fallback_request_ref requires content_hash")
            if self.fallback_request_ref.expires_at <= self.produced_at:
                raise ValueError("fallback_request_ref expires before execution request")
            if (
                self.fallback_request_ref.binding.target_model_ref
                != self.route_plan.fallback_plan.target.model_ref
            ):
                raise ValueError("fallback_request_ref target must match route fallback target")
        if self.step_transaction_ref is not None:
            _validate_capability_binding(
                self.step_transaction_ref,
                self.route_plan,
                CapabilityScope.TRANSACTION,
            )
            if self.step_transaction_ref.audience != Component.M3:
                raise ValueError("step_transaction_ref must target m3")
        if self.safety_calibration_ref.version != self.route_plan.decision_meta.feedback_version:
            raise ValueError("safety_calibration_ref must match route feedback_version")
        if self.deadline_at <= self.produced_at:
            raise ValueError("deadline_at must be after produced_at")
        if self.route_plan.execution_mode != ExecutionMode.DIRECT and self.step_transaction_ref is None:
            raise ValueError("verified execution requires step_transaction_ref")
        if self.attempt_role_override is not None:
            if self.route_plan.execution_mode != ExecutionMode.DIRECT:
                raise ValueError("attempt_role_override is only valid for direct execution")
            if self.route_plan.target.kind.value != "large_model":
                raise ValueError("attempt_role_override requires a large_model target")
            if self.attempt_role_override not in (
                AttemptRole.LARGE_MODEL_DIRECT,
                AttemptRole.LARGE_MODEL_FALLBACK,
            ):
                raise ValueError("direct attempt_role_override must be a large-model role")
        return self


class ModelAttempt(ContractModel):
    """一次物理模型调用记录：每次 SLM/LLM direct/fallback/retry 都生成新 ID。

    生产者：M3；消费者：ExecutionResult、SpanRecord。
    当前在线 trace 保留完整的模型响应；费用/时延/token 统计以该对象为准，
    避免在多个对象重复写入。
    """

    model_attempt_id: Identifier = Field(min_length=1, max_length=128)
    role: AttemptRole
    model_ref: ModelRef
    status: AttemptStatus
    # Synthetic/manual records may not have a store reference; real M3 calls
    # populate it whenever a provider response was persisted.
    result_ref: VersionedRef | None = None
    response: ModelResponse | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    cost: NonNegativeFloat | None = Field(default=None, ge=0)


class GuardResult(ContractModel):
    """一次验证（guard）的实际结果；subject_model_attempt_id 指向被审查的尝试。

    生产者：M3（调用 Verifier 后记录）；消费者：M4（detection 统计）。
    """

    guard_ref: GuardRef
    stage: VerifierStage
    subject_model_attempt_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    subject_tool_call_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    outcome: GuardOutcome
    latency_ms: int = Field(ge=0)
    cost: NonNegativeFloat | None = Field(default=None, ge=0)


class ToolExecutionResult(ContractModel):
    """一个真实工具动作的完整结构化结果。"""

    transaction_id: Identifier = Field(min_length=1, max_length=128)
    tool_call_id: Identifier = Field(min_length=1, max_length=128)
    tool_name: Identifier = Field(min_length=1, max_length=128)
    status: ToolExecutionStatus
    effect: ActualEffect
    commit_status: CommitStatus
    returncode: int | None = None
    latency_ms: int = Field(default=0, ge=0)
    output_summary: str | None = Field(default=None, max_length=4096)
    raw_output: JsonObject | None = None
    result_ref: VersionedRef | None = None
    rollback_status: RollbackStatus = RollbackStatus.NOT_NEEDED
    resume_status: ResumeStatus = ResumeStatus.NOT_NEEDED
    error_type: Identifier | None = Field(default=None, max_length=128)
    # pre-check L1 判定的效应类别（六类；M4 分桶键的数据源头，阶段 1 接入）
    effect_category: Identifier | None = Field(default=None, max_length=128)
    # post-check 四结果（阶段 2）：命令结果观察，不参与回滚判定
    command_status: Identifier | None = Field(default=None, max_length=128)
    # 失败归因（阶段 2）：env_fault / guard_fault；None = 非我方层故障
    fault_class: Identifier | None = Field(default=None, max_length=128)
    # 写溯源（阶段 2）：{author, paths}，供 M4 正确性归因
    provenance: JsonObject | None = None

class StateOutcome(ContractModel):
    """本次执行对状态与恢复过程的终态摘要（事后事实，不是事前估计）。

    read-only skeleton 会记录中性或只读结果；写路径只有在 State Adapter 明确
    提供恢复能力时才可以把 rollback/resume 写为成功。
    """

    effect: ActualEffect
    commit_status: CommitStatus
    recovery_required: bool
    rollback_status: RollbackStatus
    resume_status: ResumeStatus
    checkpoint_session_id: Identifier | None = Field(default=None, max_length=128)
    lifecycle_state: Identifier | None = Field(default=None, max_length=128)
    handle_retained: bool = False
    fault_class: Identifier | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def recovery_requires_possible_write(self) -> StateOutcome:
        """只有已发生或可能发生写入且后提交失败时才需要恢复；pre-commit 拒绝不算恢复样本。"""
        lifecycle_recovery = self.handle_retained and self.fault_class == "env_fault"
        if (
            self.recovery_required
            and self.effect not in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
            and not lifecycle_recovery
        ):
            raise ValueError("recovery_required requires write or unknown effect")
        if not self.recovery_required and self.rollback_status in (
            RollbackStatus.NOT_ATTEMPTED,
            RollbackStatus.FAILED,
            RollbackStatus.PARTIAL,
            RollbackStatus.UNKNOWN,
        ):
            raise ValueError("rollback status implies recovery_required=true")
        return self


class ExecutionResult(InvocationEnvelope):
    """M3 的终态回执：成功必须有被接受 attempt 与 accepted_output_ref，失败禁止有。

    生产者：M3；消费者：Coordinator（投影进 span、解析 accepted_output_ref）。
    Agent 只接收 accepted attempt 或明确失败，绝不泄露未接受候选。
    """

    route_plan_id: Identifier = Field(min_length=1, max_length=128)
    status: ExecutionStatus
    model_attempts: tuple[ModelAttempt, ...] = ()
    guard_results: tuple[GuardResult, ...] = ()
    tool_executions: tuple[ToolExecutionResult, ...] = ()
    state_outcome: StateOutcome
    accepted_model_attempt_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    accepted_output_ref: CapabilityRef | None = None
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> ExecutionResult:
        """attempt ID 唯一、guard 只能引用已知 attempt、成功/失败的 accepted 字段互斥。"""
        attempt_ids = [attempt.model_attempt_id for attempt in self.model_attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("model_attempt_id values must be unique within execution_result")
        guard_attempt_ids = {
            result.subject_model_attempt_id
            for result in self.guard_results
            if result.subject_model_attempt_id is not None
        }
        if not guard_attempt_ids.issubset(set(attempt_ids)):
            raise ValueError("guard result references an unknown model_attempt_id")
        execution_ids = [result.tool_call_id for result in self.tool_executions]
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("tool_execution tool_call_id values must be unique")
        if self.status == ExecutionStatus.SUCCESS:
            if self.accepted_model_attempt_id is None or self.accepted_output_ref is None:
                raise ValueError("success requires accepted_model_attempt_id and accepted_output_ref")
            accepted = next(
                (attempt for attempt in self.model_attempts if attempt.model_attempt_id == self.accepted_model_attempt_id),
                None,
            )
            if accepted is None or accepted.status != AttemptStatus.SUCCESS:
                raise ValueError("accepted_model_attempt_id must reference a successful attempt")
            _validate_capability_binding(
                self.accepted_output_ref,
                self,
                CapabilityScope.ACCEPTED_OUTPUT,
            )
            if self.accepted_output_ref.audience != Component.AGENT_ADAPTER:
                raise ValueError("accepted_output_ref must target agent_adapter")
        elif self.accepted_model_attempt_id is not None or self.accepted_output_ref is not None:
            raise ValueError("non-success execution_result cannot expose accepted output")
        return self


__all__ = [
    "ExecutionRequest",
    "ModelAttempt",
    "GuardResult",
    "ToolExecutionResult",
    "StateOutcome",
    "ExecutionResult",
]
