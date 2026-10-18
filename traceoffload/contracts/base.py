"""基础层：类型别名、全部枚举、契约模型基类与通用引用。

本模块是整个契约包的根基，不依赖任何其他契约子模块：
- 类型别名（Identifier/VersionId/...）：所有 ID 字段的约束
- 29 个枚举：契约中所有取值域（角色、风险、状态、模式等）
- ContractModel：所有契约对象的基类（不可变/严格/封闭）
- ProducerRef / VersionedRef / ModelRef / GuardRef：通用引用
- MessageEnvelope / InvocationEnvelope：消息与主链对象的信封基类
"""

from __future__ import annotations

from enum import StrEnum
from math import isclose
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

# 当前契约 schema 版本：所有消息都必须声明该版本，不兼容版本一律拒绝
CONTRACT_SCHEMA_VERSION = "1.0"

# 通用标识符：1-128 字符，字母/数字开头，只能含 [A-Za-z0-9._:-]
Identifier: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
]
# 版本标识：比 Identifier 额外允许 "+"
VersionId: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    ),
]
# 规范键（descriptor_key / experience_key）：非空白字符串
CanonicalKey: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=512, pattern=r"^\S+$"),
]
Probability: TypeAlias = float
NonNegativeFloat: TypeAlias = float
JsonObject: TypeAlias = dict[str, JsonValue]


class ContractError(ValueError):
    """两个本身合法的契约对象之间关联不合法时抛出（区别于字段校验的 ValidationError）。"""


# ============================================================
# 枚举定义（按用途分组）
# ============================================================

# --- 组件与角色 ---


class Component(StrEnum):
    """系统组件清单：每个消息有 producer，每个能力引用有 audience。"""

    AGENT_ADAPTER = "agent_adapter"
    COORDINATOR = "coordinator"
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"
    M4 = "m4"
    EVALUATOR = "evaluator"


class AgentRole(StrEnum):
    """Agent 节点角色。"""

    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    TOOL_CALLER = "tool_caller"
    FINAL_RESPONDER = "final_responder"
    CUSTOM = "custom"


# --- 消息 ---


class MessageRole(StrEnum):
    """消息角色：与 OpenAI 兼容格式一致，tool 角色承载工具执行结果。"""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# --- 安全与风险 ---


class EffectScope(StrEnum):
    """工具/状态的副作用范围（事前声明）。"""

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE_WRITE = "irreversible_write"
    UNKNOWN = "unknown"


class StateAccessMode(StrEnum):
    """当前 span 对状态的可能访问方式。"""

    NONE = "none"
    READ_ONLY = "read_only"
    TRANSACTIONAL_WRITE = "transactional_write"
    UNKNOWN = "unknown"


class EffectRisk(StrEnum):
    """事前状态影响风险（unknown 必须按高风险处理，禁止降级）。"""

    NONE = "none"
    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE_WRITE = "irreversible_write"
    UNKNOWN = "unknown"


class RecoveryLevel(StrEnum):
    """恢复能力：后两种（unavailable/unknown）是高风险，限制 SLM 准入。"""

    NOT_NEEDED = "not_needed"
    ROLLBACK_AND_RESUME = "rollback_and_resume"
    FALLBACK_WITHOUT_STATE_RECOVERY = "fallback_without_state_recovery"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class OffloadState(StrEnum):
    """M4 对任务族的卸载准入状态。

    UNKNOWN/SHADOW 表示证据不足，不能证明 SLM 安全；是否允许一次受控
    探索由 M2 的保守策略决定。QUARANTINED 始终强制 direct LLM。
    """

    UNKNOWN = "unknown"
    SHADOW = "shadow"
    ENABLED = "enabled"
    QUARANTINED = "quarantined"


class CapabilityScope(StrEnum):
    """能力引用的用途：画像只读 / 执行只读 / 事务 / 被接受的输出。"""

    PROFILE_READ = "profile_read"
    EXECUTION_READ = "execution_read"
    TRANSACTION = "transaction"
    ACCEPTED_OUTPUT = "accepted_output"


class UncertaintyReason(StrEnum):
    """不确定字段的原因：纯缺失、预测值或相互冲突。"""

    MISSING = "missing"
    PREDICTED = "predicted"
    CONFLICTING = "conflicting"


class DownstreamDependency(StrEnum):
    """当前输出对下游关键行为的依赖程度。"""

    NONE = "none"
    PARTIAL = "partial"
    KEY_FIELDS = "key_fields"
    FULL = "full"
    UNKNOWN = "unknown"


# --- 输出与验证 ---


class OutputKind(StrEnum):
    """输出契约的种类，决定 M1 画像与 M3 解析路径。"""

    TEXT = "text"
    JSON = "json"
    TOOL_CALLS = "tool_calls"
    CODE_PATCH = "code_patch"
    SQL = "sql"
    PLAN = "plan"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class VerifierStage(StrEnum):
    """验证阶段：pre_commit 在执行前；post_check 在真实执行后。"""

    PRE_COMMIT = "pre_commit"
    POST_CHECK = "post_check"
    RECOVERY = "recovery"


class VerifierSubject(StrEnum):
    """验证对象：候选输出、结构化输出、工具调用、状态差异、最终产物或恢复结果。"""

    CANDIDATE_OUTPUT = "candidate_output"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLS = "tool_calls"
    STATE_DELTA = "state_delta"
    FINAL_ARTIFACT = "final_artifact"
    RECOVERY_RESULT = "recovery_result"


class PassConditionKind(StrEnum):
    """验证通过条件类型。"""

    SCHEMA_VALID = "schema_valid"
    TOOL_ARGUMENTS_ALLOWED = "tool_arguments_allowed"
    TEST_PASS = "test_pass"
    POLICY_PASS = "policy_pass"
    SCORE_AT_LEAST = "score_at_least"
    CUSTOM = "custom"


class PassAction(StrEnum):
    """验证通过后的动作。"""

    CONTINUE = "continue"
    COMMIT = "commit"
    RETURN = "return"
    RESUME_CHECK = "resume_check"


class FailureAction(StrEnum):
    """验证失败后的动作。"""

    REJECT_AND_FALLBACK = "reject_and_fallback"
    ABORT = "abort"
    ROLLBACK_THEN_FALLBACK = "rollback_then_fallback"
    MARK_UNKNOWN_AND_ABORT = "mark_unknown_and_abort"


class ParseStatus(StrEnum):
    """解析状态：NOT_NEEDED 表示无需解析（纯文本）。"""

    NOT_NEEDED = "not_needed"
    PARSED = "parsed"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FinishReason(StrEnum):
    """模型停止原因。"""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


# --- 模型与执行 ---


class ModelTargetKind(StrEnum):
    """目标模型种类：只有大模型与小模型两类（固定模型对，不做模型池）。"""

    LARGE_MODEL = "large_model"
    SMALL_MODEL = "small_model"


class ExecutionMode(StrEnum):
    """合法执行模式（组合约束见 routing 模块的 _validate_route_shape）。"""

    DIRECT = "direct"
    GUARDED_SLM = "guarded_slm"


class AttemptRole(StrEnum):
    """物理模型调用（model attempt）的角色。"""

    SMALL_MODEL_CANDIDATE = "small_model_candidate"
    LARGE_MODEL_DIRECT = "large_model_direct"
    LARGE_MODEL_FALLBACK = "large_model_fallback"


class AttemptStatus(StrEnum):
    """模型调用本身的状态（不代表是否被 Agent 接受）。"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class GuardOutcome(StrEnum):
    """验证结果五态，不能压缩成布尔。"""

    PASS = "pass"
    REJECT = "reject"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ExecutionStatus(StrEnum):
    """M3 的终态全集：success 有可交付输出，其余均为明确失败。"""

    SUCCESS = "success"
    MODEL_FAILED = "model_failed"
    VERIFICATION_FAILED = "verification_failed"
    ROLLBACK_FAILED = "rollback_failed"
    RESUME_FAILED = "resume_failed"
    FALLBACK_FAILED = "fallback_failed"
    TIMEOUT = "timeout"


class ToolExecutionStatus(StrEnum):
    """单个工具调用的终态。

    ``ExecutionStatus`` 描述整个 M3 episode；这个枚举只描述某个工具动作，
    以便把真实执行、工具错误和恢复结果分别归因。
    """

    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ActualEffect(StrEnum):
    """实际发生的状态作用（事后记录，不是事前估计）。"""

    NONE = "none"
    READ_ONLY = "read_only"
    WRITE = "write"
    UNKNOWN = "unknown"


class CommitStatus(StrEnum):
    """提交状态。"""

    NOT_ATTEMPTED = "not_attempted"
    BLOCKED = "blocked"
    COMMITTED = "committed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class RollbackStatus(StrEnum):
    """回滚状态。"""

    NOT_NEEDED = "not_needed"
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ResumeStatus(StrEnum):
    """fallback 注入后的恢复检查状态。"""

    NOT_NEEDED = "not_needed"
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# --- 反馈 ---


class OutcomeScope(StrEnum):
    """延迟标签的作用域：candidate/downstream/episode_final/workflow_final/safety。"""

    CANDIDATE = "candidate"
    DOWNSTREAM = "downstream"
    EPISODE_FINAL = "episode_final"
    WORKFLOW_FINAL = "workflow_final"
    SAFETY = "safety"


class OutcomeLabel(StrEnum):
    """标签取值：正确性标签与安全标签的合法取值不同（见 OutcomeEvent 校验）。"""

    SUCCESS = "success"
    FAILURE = "failure"
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


# ============================================================
# 基础模型与通用引用
# ============================================================
# ConfigDict是 Pydantic 的类型，用于配置模型行为

class ContractModel(BaseModel):
    """所有契约对象的基类：不可变、严格类型、拒绝未知字段的封闭世界模型。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )


class ProducerRef(ContractModel):
    """消息生产者标识：哪个组件（及其版本）产生了该对象，用于审计与故障归因。"""

    component: Component
    version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )


class VersionedRef(ContractModel):
    """通用版本化引用：kind + id + version，一切「指向」的基础（被 ModelRef/GuardRef 继承）。"""

    kind: Identifier = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
    )
    id: Identifier = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )


class ModelRef(VersionedRef):
    """模型身份引用：kind 必须为 "model"，携带稳定模型标识与版本（路由与网关的模型定位依据）。"""

    @model_validator(mode="after")
    def kind_is_model(self) -> ModelRef:
        if self.kind != "model":
            raise ValueError("model_ref.kind must be 'model'")
        return self


class GuardRef(VersionedRef):
    """guard（验证器 / evaluator）的引用：路由计划与结果事件用它定位验证实现。"""

    pass


class MessageEnvelope(ContractModel):
    """可投递消息基类：schema 版本 + 消息 ID + 生产者 + 生产时间（所有消息对象的公共头）。"""

    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    message_id: Identifier = Field(min_length=1, max_length=128)
    producer_ref: ProducerRef
    produced_at: AwareDatetime


class InvocationEnvelope(MessageEnvelope):
    """主链对象基类：把五个关联 ID 固化进所有主对象，防止 ID 漂移。

    RuntimeProfile / RoutePlan / ExecutionResult 都继承它，
    因此它们天然携带 workflow/episode/invocation 归属。
    """

    workflow_run_id: Identifier = Field(min_length=1, max_length=128)
    workflow_plan_ref: VersionedRef | None = None
    workflow_plan_revision: int | None = Field(default=None, ge=0)
    episode_id: Identifier = Field(min_length=1, max_length=128)
    logical_invocation_id: Identifier = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def revision_requires_plan(self) -> InvocationEnvelope:
        """workflow_plan_ref 与 revision 必须成对出现。"""
        if (self.workflow_plan_ref is None) != (self.workflow_plan_revision is None):
            raise ValueError(
                "workflow_plan_ref and workflow_plan_revision must be both set or both null"
            )
        return self


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "Identifier",
    "VersionId",
    "CanonicalKey",
    "Probability",
    "NonNegativeFloat",
    "JsonObject",
    "ContractError",
    "Component",
    "MessageRole",
    "EffectScope",
    "AgentRole",
    "StateAccessMode",
    "OutputKind",
    "VerifierStage",
    "VerifierSubject",
    "PassConditionKind",
    "PassAction",
    "FailureAction",
    "ParseStatus",
    "FinishReason",
    "CapabilityScope",
    "ModelTargetKind",
    "ExecutionMode",
    "RecoveryLevel",
    "OffloadState",
    "AttemptRole",
    "AttemptStatus",
    "GuardOutcome",
    "ExecutionStatus",
    "ToolExecutionStatus",
    "ActualEffect",
    "CommitStatus",
    "RollbackStatus",
    "ResumeStatus",
    "UncertaintyReason",
    "DownstreamDependency",
    "EffectRisk",
    "OutcomeScope",
    "OutcomeLabel",
    "ContractModel",
    "ProducerRef",
    "VersionedRef",
    "ModelRef",
    "GuardRef",
    "MessageEnvelope",
    "InvocationEnvelope",
]
