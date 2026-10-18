"""执行前可见的历史执行事实（M1/embedding 旁路输入）。

这些对象只由 AdapterSession 从**当前 lm_invocation 之前已经完成**的事件
投影得到，用于 M1 的语义/embedding 特征构建，绝不注入 Agent 消息历史。

- PriorVerificationRecord：最近完成 episode 的 verifier 结果
- PriorStateRecord：最近完成 episode 的 state observation
- PriorToolOutcome：历史 tool_finished 的成败摘要，用于选择最近成功轮次
- PriorExecutionFacts：上述事实的只读聚合
"""

from __future__ import annotations

from typing import Literal

from .base import (
    ActualEffect,
    CommitStatus,
    ContractModel,
    GuardOutcome,
    GuardRef,
    Identifier,
    RollbackStatus,
    ResumeStatus,
    VerifierStage,
)


class PriorVerificationRecord(ContractModel):
    """最近完成 episode 的一条 verifier / guard 结果。"""

    episode_id: Identifier
    logical_invocation_id: Identifier | None = None
    guard_ref: GuardRef
    stage: VerifierStage
    outcome: GuardOutcome
    subject_tool_call_id: Identifier | None = None


class PriorStateRecord(ContractModel):
    """最近完成 episode 的 state observation 摘要。"""

    episode_id: Identifier
    logical_invocation_id: Identifier | None = None
    effect: ActualEffect
    commit_status: CommitStatus
    recovery_required: bool
    rollback_status: RollbackStatus
    resume_status: ResumeStatus
    fault_class: Identifier | None = None


class PriorToolOutcome(ContractModel):
    """一次历史 tool_finished 的成败摘要。"""

    tool_call_id: Identifier
    episode_id: Identifier | None = None
    logical_invocation_id: Identifier | None = None
    # success = 没有 error/error_type，且 returncode 为 None 或 0
    success: bool
    returncode: int | None = None
    error_type: Identifier | None = None
    #: 事前 L1 判定的效应类别（六类）。供在线特征 ``prefix_n_A`` 统计
    #: 「前缀里成功且是写的工具调用数」—— 与离线口径
    #: （``classfierTrain/aw3_build.py`` 的 ``tool_event_kind``：
    #: ``ok and is_write`` → ``"A"``）对齐。
    #: None = 该事件没有携带类别（旧事件 / 未分类路径），统计时按**未知**处理，
    #: 不猜成只读也不猜成写。
    effect_category: Identifier | None = None


class PriorExecutionFacts(ContractModel):
    """最近一条已完成 episode 的历史事实 + 历史工具成败摘要。

    ``verifier_results`` / ``state_observation`` 只来自最近完成的 episode；
    direct 路径没有 verifier 时为空的 tuple，属于合法情况。
    ``tool_outcomes`` 覆盖当前 invocation 之前所有可观察的 tool_finished，
    供“最近成功轮次”窗口选择使用，不代表当前 episode 的未来结果。
    """

    episode_id: Identifier
    logical_invocation_id: Identifier | None = None
    span_kind: Literal["normal", "fallback", "rollback_recovery"] = "normal"
    parent_episode_id: Identifier | None = None
    verifier_results: tuple[PriorVerificationRecord, ...] = ()
    state_observation: PriorStateRecord | None = None
    tool_outcomes: tuple[PriorToolOutcome, ...] = ()


__all__ = [
    "PriorExecutionFacts",
    "PriorStateRecord",
    "PriorToolOutcome",
    "PriorVerificationRecord",
]
