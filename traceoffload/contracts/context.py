"""上下文层：workflow 执行前可观察事实与调用入口对象。

- 各嵌套上下文对象（目标/节点/下游/状态/输出契约）：描述一次调用的执行环境
- WorkflowContext：M1 形成 descriptor 的事实来源
- InvocationContext：一次逻辑 LM 调用进入 TraceOffload 的不可变入口对象（M1 的输入）
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .base import (
    AgentRole,
    CapabilityScope,
    Component,
    ContractModel,
    DownstreamDependency,
    EffectRisk,
    GuardRef,
    Identifier,
    InvocationEnvelope,
    OutputKind,
    RecoveryLevel,
    StateAccessMode,
    VersionedRef,
    VersionId,
)
from .capabilities import CapabilityRef, _validate_capability_binding
from .prior import PriorExecutionFacts


class WorkflowGoal(ContractModel):
    """workflow 目标：文本或引用二选一（画像与 trace 的语义锚点）。"""

    goal_text: str | None = None
    goal_ref: VersionedRef | None = None

    @model_validator(mode="after")
    def has_goal(self) -> WorkflowGoal:
        if self.goal_text is None and self.goal_ref is None:
            raise ValueError("workflow_goal requires goal_text or goal_ref")
        return self


class NodeContext(ContractModel):
    """Agent 节点上下文：节点 ID/类型/步序号（M1 画像与 trace 的节点定位）。"""

    node_id: Identifier = Field(min_length=1, max_length=128)
    node_type: Identifier = Field(min_length=1, max_length=128)
    step_index: int = Field(ge=0)
    parent_node_id: Identifier | None = Field(default=None, min_length=1, max_length=128)


class UpstreamContextRef(ContractModel):
    """上游上下文引用（当前调用依赖的先前产物）。"""

    source_type: Identifier = Field(min_length=1, max_length=64)
    ref: VersionedRef
    summary: str | None = None


class DownstreamContract(ContractModel):
    """下游消费契约：谁消费本 span 输出、依赖程度如何（M1 画像与风险判断依据）。"""

    consumer_role: AgentRole
    dependency_level: DownstreamDependency
    next_action: Identifier | None = Field(default=None, min_length=1, max_length=128)


class StateContext(ContractModel):
    """状态上下文：访问模式与预期影响（access_mode 为 none 时禁止 state_ref）。

    用途：M1 把它映射为 descriptor 的 effect_risk；M2 据此做 SLM 准入。
    """

    state_ref: VersionedRef | None = None
    access_mode: StateAccessMode
    expected_effect: EffectRisk

    @model_validator(mode="after")
    def state_reference_matches_access(self) -> StateContext:
        if self.access_mode == StateAccessMode.NONE and self.state_ref is not None:
            raise ValueError("state_ref must be null when access_mode is none")
        return self


class OutputContract(ContractModel):
    """输出契约：Agent 对该 span 输出的机器可检查要求。

    用途：M1 画像（output_kind 决定子任务分类）、M2 准入（输出契约是否有
    可用验证器）、M3 解析路径。tool_calls 输出必须有工具白名单。
    """

    contract_id: Identifier = Field(min_length=1, max_length=128)
    version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )
    output_kind: OutputKind
    schema_ref: VersionedRef | None = None
    required_fields: tuple[str, ...] = ()
    parser_ref: VersionedRef | None = None
    tool_allowlist: tuple[Identifier, ...] | None = None
    consumer_role: AgentRole | None = None

    @model_validator(mode="after")
    def tool_contract_matches_kind(self) -> OutputContract:
        """tool_calls 输出必须有工具白名单；白名单只对 tool_calls/mixed 有效。"""
        if self.output_kind == OutputKind.TOOL_CALLS and not self.tool_allowlist:
            raise ValueError("tool_calls output requires a non-empty tool_allowlist")
        if self.output_kind not in (OutputKind.TOOL_CALLS, OutputKind.MIXED):
            if self.tool_allowlist is not None:
                raise ValueError("tool_allowlist is only valid for tool_calls or mixed output")
        return self


class WorkflowContext(ContractModel):
    """执行前可观察的 workflow 事实集合：M1 形成 descriptor 的事实来源。

    生产者：Coordinator-owned AdapterSession（由 integration hook 的可观察事实组装）；
    消费者：M1（画像）。它不是路由结论，只描述「这次调用发生时的环境」。
    """

    context_version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )
    agent_id: Identifier = Field(min_length=1, max_length=128)
    agent_role: AgentRole
    agent_version: VersionId | None = Field(default=None, min_length=1, max_length=128)
    workflow_goal: WorkflowGoal
    node_context: NodeContext
    upstream_context_refs: tuple[UpstreamContextRef, ...] = ()
    downstream_contract: DownstreamContract
    state_context: StateContext
    available_tools: tuple[VersionedRef, ...] = ()
    output_contract: Literal["none"] | OutputContract = "none"
    verifier_catalog: tuple[GuardRef, ...] = ()
    recovery_capability: RecoveryLevel
    # ``rollback_recovery`` is not a route decision.  It is a factual marker
    # supplied by the Coordinator-owned RecoveryController so M1 can build a
    # distinct runtime descriptor before M2 is deliberately bypassed.
    recovery_mode: Literal["normal", "rollback_recovery"] = "normal"


class InvocationContext(InvocationEnvelope):
    """一次逻辑 LM 调用进入 TraceOffload 的不可变入口对象（M1 的唯一输入）。

    生产者：Coordinator（包装 request/workflow，并生成 PROFILE_READ 能力引用）；
    消费者：M1。request_ref 必须是 audience=M1 的 PROFILE_READ 引用，
    M1 只能凭它读取画像视图，不能调用模型、提交状态或访问工具。
    """

    request_ref: CapabilityRef
    workflow_context: WorkflowContext
    prior_snapshot_ref: VersionedRef | None = None
    # 当前 invocation 之前已经完成的历史执行事实；None = 尚未完成任何 episode。
    prior_execution_facts: PriorExecutionFacts | None = None
    caller_deadline_at: AwareDatetime

    @model_validator(mode="after")
    def validate_request_capability(self) -> InvocationContext:
        """校验能力绑定、受众、调用期限与引用过期时间。"""
        _validate_capability_binding(self.request_ref, self, CapabilityScope.PROFILE_READ)
        if self.request_ref.audience != Component.M1:
            raise ValueError("invocation_context.request_ref must target m1")
        if self.caller_deadline_at <= self.produced_at:
            raise ValueError("caller_deadline_at must be after produced_at")
        if self.request_ref.expires_at <= self.produced_at:
            raise ValueError("request_ref expires_at must be after produced_at")
        return self


__all__ = [
    "WorkflowGoal",
    "NodeContext",
    "UpstreamContextRef",
    "DownstreamContract",
    "StateContext",
    "OutputContract",
    "WorkflowContext",
    "InvocationContext",
]
