"""反馈层：延迟结果标签与经验画像（M4 的数据）。

- Attribution：workflow_final 的 episode 归因
- OutcomeEvent：延迟结果标签（按 event_id 去重、按 revision 修订）
- ConfidenceInterval / ExperienceProfile：M4 聚合出的经验画像
- FeedbackSnapshot：M4 发布的不可变版本化快照（回灌 M1/M2/M3）
"""

from __future__ import annotations

from math import isclose

from pydantic import AwareDatetime, Field, model_validator

from .base import (
    CanonicalKey,
    ContractModel,
    GuardRef,
    Identifier,
    MessageEnvelope,
    NonNegativeFloat,
    OffloadState,
    OutcomeLabel,
    OutcomeScope,
    Probability,
    VersionId,
    VersionedRef,
)


class Attribution(ContractModel):
    """workflow_final 的 episode 归因：权重非负、同 workflow、总和为 1。

    生产者：evaluator；消费者：M4（校验）、outcomes 处理器（校验同一 workflow）。
    """

    episode_id: Identifier = Field(min_length=1, max_length=128)
    weight: Probability = Field(ge=0, le=1)
    method: Identifier = Field(min_length=1, max_length=128)


class OutcomeEvent(MessageEnvelope):
    """延迟结果标签：按 event_id 去重、按 revision 修订；scope 决定必需字段组合。

    生产者：evaluator/Agent（任务结束后）；消费者：OutcomeProcessor（校验）
    → EventStore（去重/修订）→ M4（聚合）。没有标签 = censored，不能当成失败。
    """

    event_id: Identifier = Field(min_length=1, max_length=128)
    workflow_run_id: Identifier = Field(min_length=1, max_length=128)
    episode_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    scope: OutcomeScope
    logical_invocation_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    model_attempt_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    label: OutcomeLabel
    evaluator_ref: GuardRef
    attribution: tuple[Attribution, ...] | None = None
    observed_at: AwareDatetime
    revision: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_scope_fields(self) -> OutcomeEvent:
        """workflow_final 必须有 invocation 或 attribution；candidate 必须有 attempt；标签取值受 scope 约束。"""
        if self.scope == OutcomeScope.WORKFLOW_FINAL:
            if self.logical_invocation_id is None and not self.attribution:
                raise ValueError(
                    "workflow_final without logical_invocation_id requires attribution"
                )
            if self.attribution is not None:
                if not self.attribution:
                    raise ValueError("attribution cannot be empty")
                total = sum(item.weight for item in self.attribution)
                if not isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("attribution weights must sum to 1")
        else:
            if self.episode_id is None or self.logical_invocation_id is None:
                raise ValueError("non-workflow outcome requires episode_id and logical_invocation_id")
            if self.attribution is not None:
                raise ValueError("attribution is only valid for workflow_final")
        if self.scope == OutcomeScope.CANDIDATE and self.model_attempt_id is None:
            raise ValueError("candidate outcome requires model_attempt_id")
        if self.scope == OutcomeScope.SAFETY:
            if self.label not in (OutcomeLabel.SAFE, OutcomeLabel.UNSAFE, OutcomeLabel.UNKNOWN):
                raise ValueError("safety outcome requires safe, unsafe, or unknown label")
        elif self.label not in (OutcomeLabel.SUCCESS, OutcomeLabel.FAILURE, OutcomeLabel.UNKNOWN):
            raise ValueError("correctness outcome requires success, failure, or unknown label")
        return self


class ConfidenceInterval(ContractModel):
    """置信区间：lower 不超过 upper（M4 用 Wilson 区间填充）。"""

    lower: float = Field(ge=0)
    upper: float = Field(ge=0)

    @model_validator(mode="after")
    def lower_not_above_upper(self) -> ConfidenceInterval:
        if self.lower > self.upper:
            raise ValueError("confidence interval lower must not exceed upper")
        return self


class ExperienceProfile(ContractModel):
    """M4 聚合的经验画像：按 experience_key 分桶的成功率/风险/成本/时延后验。

    生产者：M4（SnapshotPublisher）；消费者：M2（启发式证据门控）、
    runner（汇总）。support_count 不含重复投递、隔离冲突或不适用样本。
    ``offload_state`` 是 M4 的准入状态：``quarantined`` 始终阻断后续 SLM；
    ``unknown``/``shadow`` 表示尚无充分证据，是否探索由 M2 保守策略决定。
    """

    experience_key: CanonicalKey = Field(min_length=1, max_length=512)
    support_count: int = Field(ge=0)
    p_local_success: Probability = Field(ge=0, le=1)
    p_downstream_success: Probability = Field(ge=0, le=1)
    p_episode_success: Probability = Field(ge=0, le=1)
    p_fallback_trigger: Probability = Field(ge=0, le=1)
    p_recovery: Probability = Field(ge=0, le=1)
    p_verifier_detection: Probability = Field(ge=0, le=1)
    p_unsafe: Probability = Field(ge=0, le=1)
    expected_cost: NonNegativeFloat = Field(ge=0)
    # True = 该桶所有计入 expected_cost 的 model_attempt 都有非空 cost；
    # False = 至少一个 attempt.cost is None（价格源未接入），成本门应禁用。
    cost_observed: bool = True
    expected_latency_ms: NonNegativeFloat = Field(ge=0)
    confidence_bounds: dict[str, ConfidenceInterval] = Field(default_factory=dict)
    # 桶画像专用标记（阶段 4）：本桶尚无下游检查点样本（DOWNSTREAM 摘要
    # 尚未归因到该桶）。此类桶只许探针、不生产正选（§6.5.3）；候选级画像
    # 保持 False。
    downstream_labels_absent: bool = False
    # 桶画像归因字段（§6.2，报告口径）：fault_class 按类计数、非零返回码
    # 次数、fallback 次数与探针分报告（探针观测同池入账，另列报告）。
    fault_class_counts: dict[str, int] = Field(default_factory=dict)
    nonzero_returncode_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    probe_stats: dict[str, int] = Field(default_factory=dict)
    # 写溯源报告（§6.2）：作者（M_S/M_L/fallback）-> 路径列表。桶级聚合时
    # 保留 executor 记录的 {author, paths}，供下游归因与实验报告。
    write_provenance: dict[str, list[str]] = Field(default_factory=dict)
    stale: bool
    # The field names remain wire-compatible with the earlier shadow design;
    # the guarded pilot fills them only from real guarded episodes.
    offload_state: OffloadState = OffloadState.UNKNOWN
    shadow_support_count: int = Field(default=0, ge=0)
    shadow_match_count: int = Field(default=0, ge=0)
    shadow_mismatch_count: int = Field(default=0, ge=0)
    quarantine_reason: Identifier | None = None


class FamilySafetyProfile(ContractModel):
    """M4 task-family safety state shared by candidate route variants.

    Candidate ``ExperienceProfile`` objects contain quality/cost evidence for
    one exact route key.  This smaller profile carries only the cross-route
    safety state needed by M2 to quarantine a family after a delayed unsafe or
    recovery failure signal.
    """

    family_key: CanonicalKey = Field(min_length=1, max_length=512)
    guarded_support_count: int = Field(ge=0)
    guarded_match_count: int = Field(ge=0)
    guarded_mismatch_count: int = Field(ge=0)
    offload_state: OffloadState = OffloadState.UNKNOWN
    quarantine_reason: Identifier | None = None
    stale: bool = False

    @model_validator(mode="after")
    def counts_fit_support(self) -> FamilySafetyProfile:
        if self.guarded_match_count + self.guarded_mismatch_count > self.guarded_support_count:
            raise ValueError("family safety match/mismatch counts exceed support")
        return self


class FeedbackSnapshot(MessageEnvelope):
    """M4 发布的不可变版本化快照：经验画像 + 安全校准引用（只影响后续 episode）。

    生产者：M4（publish_snapshot）；消费者：wrapper 的 snapshot_provider
    → Coordinator → M2（经验）/ M3（校准）。已发布版本不可覆盖。
    """

    snapshot_version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )
    descriptor_prior_ref: VersionedRef | None = None
    experience_profiles: tuple[ExperienceProfile, ...] = ()
    family_safety_profiles: tuple[FamilySafetyProfile, ...] = ()
    # 六类效应桶画像（阶段 3）：按 tool_execution.effect_category 聚合，
    # experience_key = "bucket/<category>"；路由阶段 4 消费。
    bucket_profiles: tuple[ExperienceProfile, ...] = ()
    safety_calibration_ref: VersionedRef

    @model_validator(mode="after")
    def unique_experience_keys(self) -> FeedbackSnapshot:
        """快照内 experience_key 必须唯一。"""
        keys = [profile.experience_key for profile in self.experience_profiles]
        if len(keys) != len(set(keys)):
            raise ValueError("feedback_snapshot experience_key values must be unique")
        family_keys = [profile.family_key for profile in self.family_safety_profiles]
        if len(family_keys) != len(set(family_keys)):
            raise ValueError("feedback_snapshot family_key values must be unique")
        return self


__all__ = [
    "Attribution",
    "OutcomeEvent",
    "ConfidenceInterval",
    "ExperienceProfile",
    "FamilySafetyProfile",
    "FeedbackSnapshot",
]
