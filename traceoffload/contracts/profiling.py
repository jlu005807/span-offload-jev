"""画像层：M1 产出的执行前画像及其构成对象。

- AllowedTargets：数据策略允许的模型集合（当前 pilot 恒为 unknown）
- UncertainField：画像中不可直接观测的字段记录
- SpanDescriptor：执行前画像的 8 个字段
- RuntimeProfile：M1 的完整产出（M2 的决策输入、SpanRecord 画像索引的来源）
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import (
    CanonicalKey,
    CapabilityScope,
    Component,
    ContractModel,
    DownstreamDependency,
    EffectRisk,
    GuardRef,
    Identifier,
    InvocationEnvelope,
    Probability,
    RecoveryLevel,
    UncertaintyReason,
    VersionId,
)
from .capabilities import CapabilityRef, _validate_capability_binding
from .context import OutputContract
from .routing import ModelTarget


class AllowedTargets(ContractModel):
    """数据策略允许接收当前请求的模型集合及其判定依据版本。

    生产者：数据策略（当前 pilot 不可观察，M1 恒写 "unknown"）；
    消费者：M2（SLM 准入检查）。unknown 时 M2 拒绝所有小模型路径。
    """

    policy_version: VersionId = Field(min_length=1, max_length=128)
    targets: tuple["ModelTarget", ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_targets(self) -> AllowedTargets:
        """同一 (kind, model id, version) 不允许重复。"""
        keys = [(target.kind, target.model_ref.id, target.model_ref.version) for target in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("allowed model targets must be unique")
        return self


class UncertainField(ContractModel):
    """descriptor 中非直接观测的字段：missing 可以无置信度，predicted/conflicting 必须有。

    用途：M1 记录画像缺失，M4 投影到 span_record 供离线分析。
    """

    field_path: Identifier = Field(min_length=1, max_length=256)
    reason: UncertaintyReason
    confidence: Probability | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def confidence_matches_reason(self) -> UncertainField:
        if self.reason != UncertaintyReason.MISSING and self.confidence is None:
            raise ValueError("predicted or conflicting fields require confidence")
        return self


class SpanDescriptor(ContractModel):
    """执行前画像的 8 个字段：只保留会改变准入、兼容性或安全执行方式的信息。

    生产者：M1；消费者：M2（准入）、M4（投影）。unknown 值必须显式，
    禁止降级为低风险。
    """

    subtask_type: Identifier | Literal["unknown"] = Field(min_length=1, max_length=128)
    estimated_input_tokens: int = Field(ge=0)
    output_contract: Literal["none"] | OutputContract = "none"
    allowed_targets: AllowedTargets | Literal["unknown"]
    downstream_dependency: DownstreamDependency
    effect_risk: EffectRisk
    available_verifiers: tuple[GuardRef, ...] = ()
    recovery_level: RecoveryLevel


class RuntimeProfile(InvocationEnvelope):
    """M1 产出的执行前画像：M2 的决策输入、M4 学习投影的来源。

    生产者：M1（build_profile）；消费者：M2（decide）、Coordinator（投影进
    SpanRecord）。descriptor_key 用于经验分桶，uncertain_fields 记录缺失。
    """

    request_ref: CapabilityRef
    descriptor_key: CanonicalKey = Field(min_length=1, max_length=512)
    descriptor: SpanDescriptor
    uncertain_fields: tuple[UncertainField, ...] = ()
    # 路由预测输入（阶段 4）：最后一条 tool 观察的文本特征四档分类
    # （test_feedback/error_trace/code_view/plain/unknown），M1 确定性正则计算。
    last_observation_class: Identifier | Literal["unknown"] = "unknown"
    # 在线分类器特征旁路：不在 Agent 上下文里，只给 M2/审计使用。
    online_features: dict[str, float] = Field(default_factory=dict)
    feature_version: VersionId | None = None

    @model_validator(mode="after")
    def validate_request_capability(self) -> RuntimeProfile:
        """request_ref 必须保持 M1 的 PROFILE_READ 绑定。"""
        _validate_capability_binding(self.request_ref, self, "profile_read")
        if self.request_ref.audience != Component.M1:
            raise ValueError("runtime_profile.request_ref must target m1")
        return self


__all__ = [
    "AllowedTargets",
    "UncertainField",
    "SpanDescriptor",
    "RuntimeProfile",
]
