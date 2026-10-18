"""能力层：受限引用（CapabilityRef）及其绑定。

这是全文的安全核心：所有受限访问都通过能力引用完成。
- CapabilityBinding：能力与 episode 的归属绑定（五个关联 ID + 目标模型）
- CapabilityRef：受限引用本体（scope/audience/过期时间/内容哈希/绑定）
- _validate_capability_binding：能力 scope 与关联字段必须与持有者一致
"""

from __future__ import annotations

from pydantic import AwareDatetime, Field, model_validator

from .base import (
    CapabilityScope,
    Component,
    ContractModel,
    Identifier,
    InvocationEnvelope,
    ModelRef,
    VersionedRef,
    VersionId,
)


class CapabilityBinding(ContractModel):
    """能力引用的归属绑定：把能力锁死在特定 workflow/episode/invocation 上。

    由 Coordinator 为每次调用生成；防止能力引用被跨 episode 复用。
    """

    workflow_run_id: Identifier = Field(min_length=1, max_length=128)
    episode_id: Identifier = Field(min_length=1, max_length=128)
    logical_invocation_id: Identifier = Field(min_length=1, max_length=128)
    workflow_plan_ref: VersionedRef | None = None
    workflow_plan_revision: int | None = Field(default=None, ge=0)
    target_model_ref: ModelRef | None = None

    @model_validator(mode="after")
    def revision_requires_plan(self) -> CapabilityBinding:
        """workflow_plan_ref 与 revision 必须成对出现。"""
        if (self.workflow_plan_ref is None) != (self.workflow_plan_revision is None):
            raise ValueError(
                "workflow_plan_ref and workflow_plan_revision must be both set or both null"
            )
        return self


class CapabilityRef(VersionedRef):
    """受限能力引用：谁（audience）在什么范围（scope）内、到何时（expires_at）可访问什么。

    每个接收方只能解析完成其职责所需的对象；这些引用是访问授权和 lineage 校验，
    在线 trace 仍保留完整事实；引用只负责访问授权：
    - PROFILE_READ（audience=M1）：画像视图
    - EXECUTION_READ（audience=M3）：执行视图（需内容哈希与目标模型绑定）
    - TRANSACTION（audience=M3）：事务提交权限
    - ACCEPTED_OUTPUT（audience=AGENT_ADAPTER）：Agent 可读的被接受输出

    由 Coordinator / M3 生成，RequestResponseStore 负责按引用解析对象。
    """

    scope: CapabilityScope
    audience: Component
    content_hash: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$",
    )
    tenant_ref: VersionedRef | None = None
    expires_at: AwareDatetime
    binding: CapabilityBinding

    @model_validator(mode="after")
    def expiry_is_valid(self) -> CapabilityRef:
        """过期时间必须是带时区的绝对时间。"""
        if self.expires_at.tzinfo is None:
            raise ValueError("capability expires_at must be timezone-aware")
        return self


def _validate_capability_binding(
    capability: CapabilityRef,
    owner: InvocationEnvelope,
    expected_scope: CapabilityScope,
) -> None:
    """能力引用的 scope 与五个关联字段必须与持有者一致（防止跨 episode 复用）。

    在 InvocationContext / RuntimeProfile / ExecutionRequest / ExecutionResult
    构造时由各对象自身的 validator 调用。
    """
    if capability.scope != expected_scope:
        raise ValueError(
            f"capability scope {capability.scope.value!r} does not match {expected_scope.value!r}"
        )
    binding = capability.binding
    for field_name in (
        "workflow_run_id",
        "episode_id",
        "logical_invocation_id",
        "workflow_plan_ref",
        "workflow_plan_revision",
    ):
        if getattr(binding, field_name) != getattr(owner, field_name):
            raise ValueError(f"capability binding mismatch: {field_name}")


__all__ = [
    "CapabilityBinding",
    "CapabilityRef",
    "_validate_capability_binding",
]
