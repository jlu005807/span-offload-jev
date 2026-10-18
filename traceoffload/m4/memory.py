"""M4 Span 级经验 Memory。

**只收录本地执行失败（`local_success is False`）的 Span。** 成功的 Span 一条都不进
—— M4 学的是「这个动作会失败」，不是「这个动作发生过」。

⚠️ 这里曾经是「只接收大小模型**切换点前后**的两个 Span」，那是 M4 当**决策器**
时留下的口径。决策角色被证据否掉之后配对逻辑没跟着删，于是两个方向都错：
失败的 Span 不在切换点上就不记（守卫学不到），成功的 Span 恰好在切换点上却被记
（记忆里混进没失败的）。实测 12 个任务 / 212 次调用只落库 56 条，且**按任务整块缺**。

M1 负责前半段 embedding；本模块在 Span 完成后补齐 full_span_record、
local_success、danger_tier，并通过 verifier 子集过滤 + top-5 平均距离分配
class_key。所有写入按类做 LRU，每类最多 max_records_per_class 条，
总容量 max_total 条。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import json
import math
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from time import monotonic, sleep
from typing import Any, Callable

from .span_features import guard_vector
from ..contracts import (
    ExecutionStatus,
    GuardOutcome,
    OutcomeLabel,
    OutcomeScope,
    RollbackStatus,
    ToolExecutionStatus,
)

DEFAULT_MAX_RECORDS_PER_CLASS = 200
DEFAULT_MAX_RECORDS_TOTAL = 2_000
DEFAULT_TOP_K = 5

#: 动作守卫的相似度下限。**与 `DEFAULT_TOP_K` 那类检索参数不同** —— 它不是
#: 「取前 K 个」，而是 AND 判定的左半边：低于它就一票否决，不看动作签名。
#: 0.8 是在 810 个 SLM span 上扫出来的操作点（τ=0.8 时命中 22/810、
#: 精确率 100%；τ 降到 0.5 会多捞 10 条命中但一条误拦都不少）。
DEFAULT_MEMORY_GUARD_THRESHOLD = 0.8
MAX_PENDING_OUTCOMES = 10_000
MAX_PENDING_EMBEDDINGS = 5_000
_TIE_EPSILON = 1e-9


class DangerTier(IntEnum):
    """M4 danger 分层：成功=0，本地出错分低/高两层。"""

    SUCCESS = 0
    LOW = 1
    HIGH = 2


@dataclass(frozen=True)
class M4ClassSpec:
    """一个 Span 类的固定定义。"""

    class_key: int
    verifier_set: frozenset[str] = frozenset()
    center_embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.class_key <= 33:
            raise ValueError("class_key must be in 0-33")
        if self.center_embedding is not None and not self.center_embedding:
            raise ValueError("center_embedding cannot be an empty tuple")


@dataclass(frozen=True)
class ClassAssignment:
    """一次 Span 的类分配审计结果。"""

    class_key: int
    status: str  # normal / tie / no_candidate / empty_verifier
    forced_class: bool
    score: float
    compatible_class_keys: tuple[int, ...]


@dataclass
class M4MemoryRecord:
    """M4 Memory 中的一条 Span 记录。

    除 Span 事实（class_key / local_success / danger_tier）外，本记录同时保存
    「决策字段」与「结局字段」，使 M4 能产出 ``决策 -> 结局`` 的配对账本：

    - 决策字段在 Span 完成时从 ``route_plan`` 投影，之后不再变化；
    - ``task_success`` 只在延迟 ``OutcomeEvent`` 到达后回填，**未知保持 None**；
    - ``label_scope`` 记录最近一次写入 ``task_success`` 的 scope，用于审计
      「任务级标签是否被下放给了单个 Span」。
    """

    record_id: str
    class_key: int
    embedding: tuple[float, ...]
    full_span_record: Any
    local_success: bool
    danger_tier: DangerTier
    verifier_refs: frozenset[str] = frozenset()
    #: 本 Span 里**本地执行失败**的动作签名集合，元素是
    #: ``m4_span_features.action_signature`` 产出的**规范化字符串**。
    #:
    #: ⚠️ 这是**动作级**事实，与 ``local_success``（Span 级）不同：一个 Span 可能
    #: 有多个工具调用，其中一个失败、其余成功 —— 那种 Span 的 ``local_success``
    #: 是 False，但只有**失败的那一个**动作才该进这个集合。
    #:
    #: ⚠️ **必须是字符串，不能是 ``(工具名, 参数)`` 元组。** 写入侧
    #: （``failed_action_signatures``）产出的就是字符串；两侧不一致会
    #: **永远查不到且不报错**。
    failed_actions: frozenset[str] = frozenset()
    workflow_run_id: str = "workflow-unknown"
    episode_id: str = "episode-unknown"
    logical_invocation_id: str = "invocation-unknown"
    entered_at: float = 0.0
    last_access_at: float = 0.0
    pair_id: str | None = None
    switch_direction: str | None = None
    forced_class: bool = False
    assignment_status: str = "normal"
    embedding_latency_ms: float = 0.0
    embedding_source: str = "unknown"
    #: 产出这条向量的**文本配方版本**（如 ``span-embed-v1``）。
    #: 与 ``embedding_source``（谁算的：m1_online / worker / record_fallback）正交 ——
    #: 换配方后老记录必须还能被解释，所以两者都记。
    template_version: str = "unknown"
    # --- 观测成本（结局到达后回填）-----------------------------------------
    # ⚠️ 与 ``span_*`` 的区别：那三个是**本 Span 自身**开销，这三个是**轨迹口径**。
    # 线上只跑一条臂，所以**算不出离线实验里的 Δ（卸载 vs 不卸载）** ——
    # 能实测的是「实际发生了多少」。检索「贵不贵」用的就是这一组。
    #: 从本 Span 起到轨迹结束的花费。**强依赖位置**，必须配 `position_ratio` 归一化
    #: 后再横比，直接比会得出「早的 span 都贵」这种假结论。
    suffix_cost: float | None = None
    #: 整条轨迹成本（同一 workflow 的每一行相同）。
    trajectory_cost: float | None = None
    #: 本 Span 的相对位置 k/n ∈ (0,1]，用于 `suffix_cost` 的位置归一化。
    position_ratio: float | None = None
    #: 该轨迹的 Span 总数（归一化分母的审计值）。
    trajectory_span_count: int | None = None
    #: 本 Span 的**绝对步号** k —— 位置归一化的分桶坐标。
    #: 不用 `position_ratio`（k/n）当坐标：M2 决策时 n 还不知道，
    #: 查询点算不出来、参照点算得出来，两边不可比。
    step_index: int | None = None
    #: 产出本条记录的**运行**标识。`record_id` 在「同一 task 重跑」时会重复，
    #: 检索必须靠这个排掉本轮自己（否则库里全是本轮的，n_neighbors 恒为 0）。
    experiment_id: str = "unknown"
    # --- 决策字段（Span 完成时投影，不可变） -------------------------------
    # ``action`` = "<target_kind>:<execution_mode>"，如 "small_model:guarded_slm"；
    # ``decision_reason`` = DecisionMeta.reason_code（如 classifier_slm）；
    # predictor_* 复用 DecisionMeta 的 classifier_* 槽位，L2 预测器接入后同名复用。
    action: str = "unknown"
    decision_reason: str = "unknown"
    policy_version: str = "unknown"
    predictor_p: float | None = None
    predictor_threshold: float | None = None
    predictor_version: str | None = None
    predictor_feature_version: str | None = None
    # 预测器输入的原始特征（RuntimeProfile.online_features 的 JSON 投影）。
    router_features_json: str | None = None
    # --- 结局字段（延迟回填） ---------------------------------------------
    # None = 未知（没有 evaluator 标签 = censored），**不是 False**。
    task_success: bool | None = None
    label_scope: str | None = None
    # --- 成本字段（本 Span 自身开销） -------------------------------------
    # ⚠️ 这是**本 Span** 的 token/cost，不是整条轨迹成本，也不是该 Span 之后的
    # 剩余段成本。任务级成本必须取「整条轨迹」口径（崩盘后重跑要重付前缀），
    # 两者跨度差 10 倍以上，混用会得到假结论；见 L2与M4联动设计_0922.md §3.2。
    span_input_tokens: int | None = None
    span_output_tokens: int | None = None
    span_cost: float | None = None
    span_latency_ms: int = 0


@dataclass(frozen=True)
class MemoryNeighbor:
    """KNN 查询返回的一条邻居。"""

    similarity: float
    record: M4MemoryRecord


def placeholder_class_specs(
    *,
    class_count: int = 5,
    dimension: int = 1024,
) -> tuple[M4ClassSpec, ...]:
    """先用 0..class_count-1 的 one-hot 类中心跑通，后续替换为离线 warm-start。"""

    if class_count < 1 or dimension < class_count:
        raise ValueError("class_count and dimension must be positive")
    classes = []
    for class_key in range(class_count):
        center = [0.0] * dimension
        center[class_key] = 1.0
        classes.append(
            M4ClassSpec(
                class_key=class_key,
                center_embedding=tuple(center),
            )
        )
    return tuple(classes)


@dataclass
class _PendingSeed:
    """M1 embedding 已到、完整 Span 尚未到。跨实验不淘汰。"""

    record_id: str
    embedding: tuple[float, ...]
    verifier_refs: frozenset[str] = frozenset()


class M4Memory:
    """无网络的 M4 内存工作集：分类、补齐、每类 LRU。"""

    def __init__(
        self,
        *,
        classes: Sequence[M4ClassSpec],
        max_records_per_class: int = DEFAULT_MAX_RECORDS_PER_CLASS,
        max_records_total: int = DEFAULT_MAX_RECORDS_TOTAL,
        top_k: int = DEFAULT_TOP_K,
        clock: Callable[[], float] | None = None,
        persistence: Any | None = None,
    ) -> None:
        if not classes:
            raise ValueError("at least one class is required")
        if len({spec.class_key for spec in classes}) != len(classes):
            raise ValueError("class_key values must be unique")
        if max_records_per_class < 1 or max_records_total < 1 or top_k < 1:
            raise ValueError("capacity and top_k must be positive")
        self._classes: tuple[M4ClassSpec, ...] = tuple(
            sorted(classes, key=lambda spec: spec.class_key)
        )
        self._class_by_key = {spec.class_key: spec for spec in self._classes}
        self._max_per_class = max_records_per_class
        self._max_total = max_records_total
        self._top_k = top_k
        self._clock = clock or monotonic
        self._lock = RLock()
        self._persistence = persistence
        self._records: dict[str, M4MemoryRecord] = {}
        self._records_by_class: dict[int, dict[str, M4MemoryRecord]] = {
            spec.class_key: {} for spec in self._classes
        }
        self._pending: dict[str, _PendingSeed] = {}
        self._outcomes_by_lineage: OrderedDict[
            tuple[str, str, str], list[Any]
        ] = OrderedDict()
        self._restore_from_persistence()

    # ------------------------------------------------------------------
    # pending / complete
    # ------------------------------------------------------------------
    def register_pending(
        self,
        record_id: str,
        embedding: Sequence[float],
        *,
        verifier_refs: Iterable[str] = (),
    ) -> None:
        """M1 先登记 embedding seed；pending 跨实验不淘汰。"""

        normalized = _normalize_embedding(embedding)
        with self._lock:
            seed = _PendingSeed(
                record_id=record_id,
                embedding=normalized,
                verifier_refs=frozenset(str(ref) for ref in verifier_refs),
            )
            self._pending[record_id] = seed
            if self._persistence is not None:
                self._persistence.save_pending(seed)

    def observe(
        self,
        record_id: str,
        embedding: Sequence[float],
        full_span_record: Any,
        *,
        verifier_refs: Iterable[str] | None = None,
        pair_id: str | None = None,
        switch_direction: str | None = None,
        embedding_latency_ms: float = 0.0,
        embedding_source: str = "unknown",
        template_version: str = "unknown",
        experiment_id: str = "unknown",
        failed_actions: Iterable[tuple[str, str]] | None = None,
    ) -> M4MemoryRecord:
        """便捷入口：登记 seed 后立刻补齐完整 Span。"""

        self.register_pending(record_id, embedding, verifier_refs=verifier_refs or ())
        return self.complete(
            record_id,
            full_span_record,
            verifier_refs=verifier_refs,
            pair_id=pair_id,
            switch_direction=switch_direction,
            embedding_latency_ms=embedding_latency_ms,
            embedding_source=embedding_source,
            template_version=template_version,
            experiment_id=experiment_id,
            failed_actions=failed_actions,
        )

    def complete(
        self,
        record_id: str,
        full_span_record: Any,
        *,
        verifier_refs: Iterable[str] | None = None,
        pair_id: str | None = None,
        switch_direction: str | None = None,
        embedding_latency_ms: float = 0.0,
        embedding_source: str = "unknown",
        template_version: str = "unknown",
        experiment_id: str = "unknown",
        failed_actions: Iterable[tuple[str, str]] | None = None,
    ) -> M4MemoryRecord:
        """Span 完成后补齐记录：分类 + local_success + danger + LRU 入库。"""

        with self._lock:
            seed = self._pending.pop(record_id, None)
            if seed is None:
                raise KeyError(f"unknown pending record_id: {record_id}")
            resolved_verifiers = (
                frozenset(str(ref) for ref in verifier_refs)
                if verifier_refs is not None
                else seed.verifier_refs
            )
            if not resolved_verifiers:
                resolved_verifiers = span_verifier_keys(full_span_record)
            assignment = self.assign_class(seed.embedding, resolved_verifiers)
            local_success = compute_local_success(full_span_record)
            workflow_run_id, episode_id, logical_invocation_id = _span_lineage(
                full_span_record
            )
            now = float(self._clock())
            decision_fields = compute_span_decision_fields(full_span_record)
            cost_fields = compute_span_cost_fields(full_span_record)
            record = M4MemoryRecord(
                record_id=record_id,
                class_key=assignment.class_key,
                embedding=seed.embedding,
                full_span_record=full_span_record,
                local_success=local_success,
                danger_tier=compute_danger_tier(
                    full_span_record,
                    local_success=local_success,
                ),
                verifier_refs=resolved_verifiers,
                workflow_run_id=workflow_run_id,
                episode_id=episode_id,
                logical_invocation_id=logical_invocation_id,
                entered_at=now,
                last_access_at=now,
                pair_id=pair_id,
                switch_direction=switch_direction,
                forced_class=assignment.forced_class,
                assignment_status=assignment.status,
                embedding_latency_ms=float(embedding_latency_ms),
                embedding_source=str(embedding_source),
                template_version=str(template_version),
                experiment_id=str(experiment_id),
                failed_actions=frozenset(
                    str(item) for item in (failed_actions or ())
                ),
                **decision_fields,
                **cost_fields,
            )
            self._insert_record(record)
            if self._persistence is not None:
                self._persistence.delete_pending(record_id)
                self._persistence.save_record(record)
            self._apply_stored_outcomes(record)
            return record

    # ------------------------------------------------------------------
    # delayed outcome labels
    # ------------------------------------------------------------------
    def apply_outcome(self, event: Any) -> bool:
        """把延迟 OutcomeEvent 回填到对应 Span；记录未入库时先缓存。

        返回 True 表示找到并更新了已入库记录。
        """

        lineage = _event_lineage(event)
        if lineage is None:
            return False
        with self._lock:
            # ── workflow 级事件：按 workflow_run_id 扇出 ──────────────────
            # 为什么必须单独走这条：`OutcomeScope.WORKFLOW_FINAL` 的事件在契约里
            # `episode_id` 恒为 None（见 contracts/outcomes.py:51 与
            # `validate_scope_fields`），而 `_event_lineage` 会把它补成
            # "episode-unknown"；已入库的 Span 记录带的是真实 episode_id。
            # ⇒ 按 lineage 匹配**永远匹配不上**，工作流级结局是死代码。
            #
            # 语义上本就该扇出：`attribution` 的 weight 表达的就是「这次 workflow
            # 结局如何分摊到各个 span」，所以每个 span 都该拿到同一个 label。
            if _enum_value(getattr(event, "scope", None)) == OutcomeScope.WORKFLOW_FINAL.value:
                workflow_run_id = getattr(event, "workflow_run_id", None)
                targets = (
                    [
                        record
                        for record in self._records.values()
                        if record.workflow_run_id == str(workflow_run_id)
                    ]
                    if workflow_run_id
                    else []
                )
                if not targets:
                    if (
                        lineage not in self._outcomes_by_lineage
                        and len(self._outcomes_by_lineage) >= MAX_PENDING_OUTCOMES
                    ):
                        self._outcomes_by_lineage.popitem(last=False)
                    self._outcomes_by_lineage.setdefault(lineage, []).append(event)
                    return False
                for record in targets:
                    self._apply_outcome_to_record(record, event)
                    if self._persistence is not None:
                        self._persistence.save_record(record)
                return True

            record = self._find_record_by_lineage(lineage)
            if record is None:
                if (
                    lineage not in self._outcomes_by_lineage
                    and len(self._outcomes_by_lineage) >= MAX_PENDING_OUTCOMES
                ):
                    self._outcomes_by_lineage.popitem(last=False)
                self._outcomes_by_lineage.setdefault(lineage, []).append(event)
                return False
            self._apply_outcome_to_record(record, event)
            if self._persistence is not None:
                self._persistence.save_record(record)
            return True

    def apply_observed_cost(
        self,
        workflow_run_id: str,
        per_span: Mapping[str, dict[str, float | int | None]],
    ) -> int:
        """回填轨迹口径的**观测成本**（轨迹跑完后调用）。返回更新的记录数。

        与 `apply_outcome` 分开：那个写的是**结局标签**（来自 evaluator），
        这个写的是**观测到的量**（来自 runner 的 span 记账）。两者同时到达、
        语义不同，所以两条路。

        按 `workflow_run_id` 匹配而不是 lineage —— 理由与 workflow_final 那次
        修复相同：一条轨迹的每个 span 都要拿到**自己那一段**的 suffix_cost，
        lineage 匹配一次只能命中一行。
        """
        updated = 0
        with self._lock:
            for record in self._records.values():
                if record.workflow_run_id != str(workflow_run_id):
                    continue
                metrics = per_span.get(record.record_id)
                if not metrics:
                    continue
                record.suffix_cost = _optional_float(metrics.get("suffix_cost"))
                record.trajectory_cost = _optional_float(metrics.get("trajectory_cost"))
                record.position_ratio = _optional_float(metrics.get("position_ratio"))
                record.trajectory_span_count = _optional_int(
                    metrics.get("trajectory_span_count")
                )
                record.step_index = _optional_int(metrics.get("step_index"))
                if self._persistence is not None:
                    self._persistence.save_record(record)
                updated += 1
        return updated

    def _apply_stored_outcomes(self, record: M4MemoryRecord) -> None:
        lineage = (
            record.workflow_run_id,
            record.episode_id,
            record.logical_invocation_id,
        )
        outcomes = self._outcomes_by_lineage.pop(lineage, ())
        if not outcomes:
            return
        for event in outcomes:
            self._apply_outcome_to_record(record, event)
        if self._persistence is not None:
            self._persistence.save_record(record)

    @staticmethod
    def _apply_outcome_to_record(record: M4MemoryRecord, event: Any) -> None:
        scope = _enum_value(getattr(event, "scope", None))
        label = _enum_value(getattr(event, "label", None))
        _apply_task_outcome(record, scope=scope, label=label)
        if scope == OutcomeScope.SAFETY.value and label == OutcomeLabel.UNSAFE.value:
            record.danger_tier = DangerTier.HIGH
            record.local_success = False
            return
        if scope == OutcomeScope.CANDIDATE.value:
            if label == OutcomeLabel.SUCCESS.value:
                if record.danger_tier != DangerTier.HIGH:
                    record.danger_tier = DangerTier.SUCCESS
            elif label == OutcomeLabel.FAILURE.value:
                record.local_success = False
                if record.danger_tier == DangerTier.SUCCESS:
                    record.danger_tier = DangerTier.LOW
            return
        if scope in (
            OutcomeScope.DOWNSTREAM.value,
            OutcomeScope.EPISODE_FINAL.value,
        ) and label == OutcomeLabel.FAILURE.value:
            record.local_success = False
            if record.danger_tier == DangerTier.SUCCESS:
                record.danger_tier = DangerTier.LOW

    def _find_record_by_lineage(
        self,
        lineage: tuple[str, str, str],
    ) -> M4MemoryRecord | None:
        for record in self._records.values():
            if (
                record.workflow_run_id,
                record.episode_id,
                record.logical_invocation_id,
            ) == lineage:
                return record
        return None

    def _restore_from_persistence(self) -> None:
        if self._persistence is None:
            return
        for payload in self._persistence.load_pending():
            seed = _pending_from_payload(payload)
            self._pending[seed.record_id] = seed
        # 恢复历史时**按自己的维度过滤**。为什么不让调用方记得传：
        # `load_records()` 这一次是由库内部发起的，调用方无从介入 ——
        # 忘了传就会把换向量空间之前的记录读回来，然后在 `_cosine_distance`
        # 上撞「维度不符」。那个异常是响的，但**在启动路径上崩**同样是事故。
        configured_dimension = next(
            (
                len(spec.center_embedding)
                for spec in self._classes
                if spec.center_embedding is not None
            ),
            None,
        )
        for payload in self._persistence.load_records(
            embedding_dimension=configured_dimension
        ):
            record = _record_from_payload(payload)
            if record.class_key not in self._records_by_class:
                continue
            if record.record_id in self._records:
                continue
            self._records[record.record_id] = record
            self._records_by_class[record.class_key][record.record_id] = record
        for class_key in tuple(self._records_by_class):
            self._evict_class_if_needed(class_key)
        self._evict_total_if_needed()

    # ------------------------------------------------------------------
    # class assignment
    # ------------------------------------------------------------------
    def assign_class(
        self,
        embedding: Sequence[float],
        verifier_refs: Iterable[str] = (),
    ) -> ClassAssignment:
        """verifier 过滤 + top-k 平均距离；平局/无候选落到第一个类。"""

        normalized = _normalize_embedding(embedding)
        refs = frozenset(str(ref) for ref in verifier_refs)
        with self._lock:
            if refs:
                compatible = [
                    spec for spec in self._classes if refs <= spec.verifier_set
                ]
                if not compatible:
                    return self._fallback_first_class(
                        status="no_candidate",
                        compatible=(),
                    )
                status = "normal"
            else:
                compatible = list(self._classes)
                status = "empty_verifier"
            if not compatible:
                return self._fallback_first_class(status="no_candidate", compatible=())

            scored = [
                (self._class_score(spec, normalized), spec.class_key, spec)
                for spec in compatible
            ]
            best_score = min(score for score, _, _ in scored)
            tied = [
                (score, class_key, spec)
                for score, class_key, spec in scored
                if score <= best_score + _TIE_EPSILON
            ]
            if len(tied) == 1:
                score, class_key, _ = tied[0]
                return ClassAssignment(
                    class_key=class_key,
                    status=status,
                    forced_class=False,
                    score=score,
                    compatible_class_keys=tuple(spec.class_key for spec in compatible),
                )
            return self._fallback_first_class(
                status="tie",
                compatible=tuple(spec.class_key for spec in compatible),
            )

    def _fallback_first_class(
        self,
        *,
        status: str,
        compatible: tuple[int, ...],
    ) -> ClassAssignment:
        first = self._classes[0]
        return ClassAssignment(
            class_key=first.class_key,
            status=status,
            forced_class=True,
            score=float("inf"),
            compatible_class_keys=compatible,
        )

    def _class_score(self, spec: M4ClassSpec, embedding: tuple[float, ...]) -> float:
        records = self._records_by_class[spec.class_key]
        if records:
            distances = sorted(
                _cosine_distance(embedding, record.embedding)
                for record in records.values()
            )[: self._top_k]
            return sum(distances) / len(distances)
        if spec.center_embedding is not None:
            center = _normalize_embedding(spec.center_embedding)
            return _cosine_distance(embedding, center)
        return float("inf")

    # ------------------------------------------------------------------
    # LRU
    # ------------------------------------------------------------------
    def touch(self, record_id: str) -> bool:
        """某条记录被 KNN/读取时刷新 LRU。"""

        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return False
            record.last_access_at = float(self._clock())
            return True

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        class_key: int | None = None,
        touch: bool = True,
        require_labeled: bool = False,
        min_suffix_cost: float | None = None,
        exclude_record_ids: Iterable[str] = (),
        exclude_experiment_id: str | None = None,
        exclude_workflow_run_id: str | None = None,
    ) -> tuple[MemoryNeighbor, ...]:
        """cosine kNN 查询；相似度低于阈值不返回。

        `require_labeled=True` 时**只返回已经带上 `suffix_cost` 的记录** ——
        没有成本标签的邻居对「贵不贵」不提供信息，混进来只会稀释信号。
        （它们不是「便宜」，是**未知**；见 `M4MemoryRecord` 的 censored 语义。）

        `exclude_experiment_id` + `exclude_workflow_run_id` **必须成对传当前轨迹** ——
        否则一条轨迹会检索到自己前面那些 Span（相似度天然极高），
        「跨 task 泛化」这个前提就失效了（离线实验里 `same_task_frac`
        就是专门监视这件事的）。单看任一个都会误排，见下面循环里的注释。
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized = _normalize_embedding(embedding)
        excluded = {str(value) for value in exclude_record_ids}
        with self._lock:
            if class_key is None:
                records = tuple(self._records.values())
            else:
                records = tuple(self._records_by_class.get(class_key, {}).values())
            scored = []
            for record in records:
                if require_labeled and getattr(record, "suffix_cost", None) is None:
                    continue
                if (
                    min_suffix_cost is not None
                    and getattr(record, "suffix_cost", None) is not None
                    and float(record.suffix_cost) < min_suffix_cost
                ):
                    continue
                if record.record_id in excluded:
                    continue
                # 排掉**本轨迹**自己的 Span：`(experiment_id, workflow_run_id)`
                # **两个都相等**才排。
                #   · 只看 workflow_run_id → 同一 task 重跑会撞（`workflow-{task_id}`）
                #   · 只看 experiment_id   → 一个进程跑多个 task 时会误排前一个 task
                # 两者配对才是轨迹的唯一标识。
                if (
                    exclude_experiment_id is not None
                    and exclude_workflow_run_id is not None
                    and str(getattr(record, "experiment_id", "unknown"))
                    == str(exclude_experiment_id)
                    and str(record.workflow_run_id) == str(exclude_workflow_run_id)
                ):
                    continue
                similarity = 1.0 - _cosine_distance(normalized, record.embedding)
                if similarity >= similarity_threshold:
                    scored.append((similarity, record))
            scored.sort(key=lambda item: (-item[0], item[1].record_id))
            neighbors = tuple(
                MemoryNeighbor(similarity=score, record=record)
                for score, record in scored[:top_k]
            )
            if touch and neighbors:
                now = float(self._clock())
                for neighbor in neighbors:
                    neighbor.record.last_access_at = now
            return neighbors

    def query_failed_action(
        self,
        embedding: Sequence[float],
        signature: str,
        *,
        similarity_threshold: float = DEFAULT_MEMORY_GUARD_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        touch: bool = True,
    ) -> tuple[MemoryNeighbor, ...]:
        """动作守卫的 **AND** 判定：相似度 ≥ 阈值 **且** 这个动作失败过。

        两个条件查的不是一回事，缺一不可：

        * ``embedding`` 比的是 span 的向量表示 —— 「**场景**像不像」；
        * ``signature`` 比的是 ``action_signature`` 出的规范化串 ——
          「**动作**一不一样」。

        ⚠️ ``signature`` 必须由 ``m4_span_features.action_signature`` 产出 ——
        与写入侧同一把钥匙。自己另拼一个格式会**永远查不到，而且不报错**。

        ⇒ 命中意味着「在相似的场景下，做过一模一样的动作，**而且失败了**」。

        **与 `query()` 的关系**：不复用它。`query()` 先按 top_k 截断再返回，
        而这里是先过滤动作再截断 —— 顺序反了会把「动作命中但相似度排在第 6 名」
        的记录丢掉，那是 AND 的右半边失效。相似度扫描逻辑保持逐行一致，
        改一处要改两处。
        """

        normalized = _normalize_embedding(embedding)
        wanted = str(signature)
        with self._lock:
            scored = []
            for record in self._records.values():
                if wanted not in record.failed_actions:
                    continue
                similarity = 1.0 - _cosine_distance(normalized, record.embedding)
                if similarity >= similarity_threshold:
                    scored.append((similarity, record))
            scored.sort(key=lambda item: (-item[0], item[1].record_id))
            neighbors = tuple(
                MemoryNeighbor(similarity=score, record=record)
                for score, record in scored[:top_k]
            )
            if touch and neighbors:
                now = float(self._clock())
                for neighbor in neighbors:
                    neighbor.record.last_access_at = now
            return neighbors

    def get(self, record_id: str) -> M4MemoryRecord | None:
        with self._lock:
            return self._records.get(record_id)

    def records_for_class(self, class_key: int) -> tuple[M4MemoryRecord, ...]:
        with self._lock:
            return tuple(self._records_by_class[class_key].values())

    def all_records(self) -> tuple[M4MemoryRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def pending_size(self) -> int:
        with self._lock:
            return len(self._pending)

    def _insert_record(self, record: M4MemoryRecord) -> None:
        if record.record_id in self._records:
            raise ValueError(f"duplicate record_id: {record.record_id}")
        self._records[record.record_id] = record
        self._records_by_class[record.class_key][record.record_id] = record
        self._evict_class_if_needed(record.class_key)
        self._evict_total_if_needed()

    def _evict_class_if_needed(self, class_key: int) -> None:
        records = self._records_by_class[class_key]
        while len(records) > self._max_per_class:
            self._evict_record(min(records.values(), key=_lru_key).record_id)

    def _evict_total_if_needed(self) -> None:
        while len(self._records) > self._max_total:
            self._evict_record(min(self._records.values(), key=_lru_key).record_id)

    def _evict_record(self, record_id: str) -> None:
        record = self._records.pop(record_id, None)
        if record is None:
            return
        self._records_by_class[record.class_key].pop(record_id, None)
        if self._persistence is not None:
            self._persistence.delete_record(record_id)


@dataclass(frozen=True)
class M4MemoryJob:
    """主链路 enqueue 给 M4MemoryWorker 的一条完整 Span 记录。

    ``embedding`` 和 ``embedding_text`` 至少提供一个；优先使用 embedding。
    没有 embedding 时由 worker 在后台调用 embedder + text_builder。
    """

    record_id: str
    full_span_record: Any
    embedding: tuple[float, ...] | None = None
    embedding_text: str | None = None
    verifier_refs: tuple[str, ...] = ()
    #: 本 Span 里**本地执行失败**的动作签名（``action_signature`` 出的规范化串）。
    #: 由主链路从「响应 + 执行结果」按 ``tool_call_id`` 连出来，
    #: 见 ``m4_span_features.failed_action_signatures``。
    failed_actions: tuple[str, ...] = ()
    template_version: str = "unknown"
    experiment_id: str = "unknown"


@dataclass(frozen=True)
class M4MemoryOutcomeJob:
    """延迟 OutcomeEvent 的异步回填 job。"""

    event: Any


@dataclass(frozen=True)
class M4MemoryObservedCostJob:
    """轨迹跑完后回填**观测成本**的异步 job。

    为什么单独一条路、而不是塞进 ``OutcomeEvent``：
    ``OutcomeEvent`` 是**结局标签**（成功/失败/安全），这里是**观测到的量**
    （这条轨迹实际烧了多少 token）。两者生命周期相同但语义不同 —— 塞进契约
    要改 schema 和全部生产者，而这条只会有一个生产者，单独走更窄、更好审。

    也更诚实：线上只跑一条臂，**算不出离线实验里的 Δ（卸载 vs 不卸载）**。
    这里存的是「实际发生了多少」，检索时用**位置归一化**后再比。
    """

    workflow_run_id: str
    #: record_id -> {"suffix_cost","trajectory_cost","position_ratio","trajectory_span_count"}
    per_span: dict[str, dict[str, float | int | None]]


_WorkerJob = M4MemoryJob | M4MemoryOutcomeJob | M4MemoryObservedCostJob


class M4MemoryWorker:
    """M4 Memory 的异步消费线程。

    主链路只做 ``enqueue()``（unbounded Queue，put_nowait，不阻塞）；
    worker 线程按 workflow 检测 large<->small 切换，只把切换前后的两个
    Span 补齐写入 M4Memory。相同模型的普通 Span 不进入 Memory，也不会触发
    embedding 计算。
    """

    def __init__(
        self,
        memory: M4Memory,
        *,
        embedder: Any | None = None,
        text_builder: Callable[[Any], str] | None = None,
        guard_projector: Callable[[Any, Sequence[float]], Any] | None = None,
        thread_name: str = "m4-memory",
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._text_builder = text_builder
        #: 嵌入 → 52 维守卫向量。**默认用真实现**（读 artifact 做 PCA）；
        #: 单元测试注入直通实现，就不必为每个用例造 1024 维假嵌入。
        #: 与 `embedder` / `text_builder` 同一个模式。
        self._guard_projector = guard_projector
        self._thread_name = thread_name
        self._queue: Queue[_WorkerJob | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._state_lock = Lock()
        self._pending_embeddings: dict[str, tuple[tuple[float, ...], float]] = {}
        self._pending_texts: dict[str, str] = {}
        self._received = 0
        self._stored = 0
        #: 入库判据是「本地执行失败」⇒ **成功的 span 一律跳过**。
        #: 计数是为了让「M4 到底吃掉了多少轨迹」可见 —— 这个数偏大不是 bug，
        #: 但它必须看得见，否则「为什么守卫什么都没学到」又要靠猜。
        self._skipped_success = 0
        self._errors = 0
        self._embedded = 0
        self._m1_embedding_registered = 0
        self._embedding_source_counts: dict[str, int] = {}

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, job: _WorkerJob) -> None:
        """非阻塞 enqueue；队列无界，正常路径不会等待 worker。"""

        if not isinstance(job, M4MemoryJob | M4MemoryOutcomeJob | M4MemoryObservedCostJob):
            raise TypeError(
                "job must be an M4MemoryJob, M4MemoryOutcomeJob or "
                "M4MemoryObservedCostJob"
            )
        with self._state_lock:
            self._received += 1
        self._queue.put_nowait(job)

    def enqueue_completed_span(
        self,
        record_id: str,
        full_span_record: Any,
        *,
        embedding: Sequence[float] | None = None,
        embedding_text: str | None = None,
        verifier_refs: Iterable[str] = (),
        failed_actions: Iterable[str] = (),
        template_version: str = "unknown",
        experiment_id: str = "unknown",
    ) -> None:
        """主链路 O(1) 调用：只 enqueue，不做 embedding。"""

        self.enqueue(
            M4MemoryJob(
                record_id=record_id,
                full_span_record=full_span_record,
                embedding=(
                    tuple(float(value) for value in embedding)
                    if embedding is not None
                    else None
                ),
                embedding_text=embedding_text,
                verifier_refs=tuple(str(ref) for ref in verifier_refs),
                failed_actions=tuple(str(item) for item in failed_actions),
                template_version=str(template_version),
                experiment_id=str(experiment_id),
            )
        )

    def enqueue_outcome(self, event: Any) -> None:
        self.enqueue(M4MemoryOutcomeJob(event=event))

    def enqueue_observed_cost(
        self,
        workflow_run_id: str,
        per_span: dict[str, dict[str, float | int | None]],
    ) -> None:
        """轨迹跑完后回填观测成本（suffix_cost 等）。见 `M4MemoryObservedCostJob`。"""

        if not per_span:
            return
        self.enqueue(
            M4MemoryObservedCostJob(
                workflow_run_id=str(workflow_run_id),
                per_span={str(k): dict(v) for k, v in per_span.items()},
            )
        )

    def register_pending_embedding(
        self,
        record_id: str,
        embedding: Sequence[float],
        *,
        latency_ms: float = 0.0,
    ) -> None:
        """M1 在线算好的 embedding 先挂在 record_id 上，Span 完成后再消费。"""

        normalized = _normalize_embedding(embedding)
        with self._state_lock:
            self._evict_oldest_if_needed(self._pending_embeddings)
            self._pending_embeddings[record_id] = (normalized, float(latency_ms))
            self._m1_embedding_registered += 1

    def register_pending_text(self, record_id: str, text: str) -> None:
        """M1 文本兜底：worker 在 Span 完成后再异步调用 embedder。"""

        if not isinstance(text, str) or not text.strip():
            return
        with self._state_lock:
            self._evict_oldest_if_needed(self._pending_texts)
            self._pending_texts[record_id] = text

    @staticmethod
    def _evict_oldest_if_needed(mapping: dict) -> None:
        if len(mapping) >= MAX_PENDING_EMBEDDINGS:
            mapping.pop(next(iter(mapping)))

    def process_one(self, *, timeout_seconds: float = 0.1) -> bool:
        """同步处理一条 job，主要供测试和 drain 场景使用。"""

        try:
            job = self._queue.get(timeout=timeout_seconds)
        except Empty:
            return False
        try:
            self._handle_job(job)
        except Exception:
            with self._state_lock:
                self._errors += 1
        finally:
            self._queue.task_done()
        return True

    def drain(self, *, timeout_seconds: float = 5.0) -> bool:
        """等待已 enqueue 的 job 全部处理完；不阻塞主链路。"""

        deadline = monotonic() + timeout_seconds
        while monotonic() < deadline:
            if self._queue.empty() and self._queue.unfinished_tasks == 0:
                return True
            sleep(0.005)
        return self._queue.empty() and self._queue.unfinished_tasks == 0

    def stop(self, *, timeout_seconds: float = 5.0) -> bool:
        """停止 worker 线程；调用前应先 drain，否则队列中 job 不会处理。"""

        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_seconds)
            return not thread.is_alive()
        return True

    def stats(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "received": self._received,
                "stored": self._stored,
                "skipped_success": self._skipped_success,
                "errors": self._errors,
                "embedded": self._embedded,
                "m1_embedding_registered": self._m1_embedding_registered,
                "embedding_source_counts": dict(self._embedding_source_counts),
                "pending_embeddings": len(self._pending_embeddings),
                "pending_texts": len(self._pending_texts),
                "queue_size": self._queue.qsize(),
                "thread_alive": self._thread is not None and self._thread.is_alive(),
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._handle_job(job)
            except Exception:
                with self._state_lock:
                    self._errors += 1
            finally:
                self._queue.task_done()

    def _handle_job(self, job: _WorkerJob) -> None:
        if isinstance(job, M4MemoryOutcomeJob):
            self._memory.apply_outcome(job.event)
            return
        if isinstance(job, M4MemoryObservedCostJob):
            self._memory.apply_observed_cost(job.workflow_run_id, job.per_span)
            return
        self._handle_span_job(job)

    def _handle_span_job(self, job: M4MemoryJob) -> None:
        """只收录**本地执行失败**的 span —— 与路由是否切换**无关**。

        旧口径是「路由目标发生切换时成对收录」（`previous_kind == current_kind`
        就两条都丢），那是 M4 当**决策器**时留下的 —— 那时它要学的是「切换那一刻
        发生了什么」。决策角色已被证据否掉，配对逻辑没跟着删，于是两个方向都错：

            local 失败的 span 不在切换点上 ⇒ **一条都不记** ⇒ 守卫永远学不到它
            成功的 span 正好在切换点上   ⇒ **记了** ⇒ 记忆里混进没失败的

        实测代价（12 个任务 / 212 次模型调用）：只落库 **56 条（26%）**，而且是
        按任务整块缺 —— 全程单模型的那些任务**一条都不记**。

        判据是 `_span_had_an_executed_failure`，**不是** `not compute_local_success`。
        后者会把两种「没失败」也算进来：

            准入被拒的 span          `status=verification_failed`、`tool_executions` **空**
                                     ⇒ 它证明的是「分析器认不出这段代码」，
                                       不是「这个动作会失败」
            post-check 拒但工具成功   `status=verification_failed`、工具 `SUCCESS`
                                     ⇒ 动作本身执行成功了，被拒的是最终产物

        这两种一旦入库，守卫就会基于**无关证据**拒动作 —— 而且拦得下。
        读失败和写失败都会让工具记录变成 FAILED，所以收紧不会漏掉任何一类。
        """

        keep = _span_had_an_executed_failure(job.full_span_record)
        if not keep:
            # 没有「真的执行过并且失败」的工具 ⇒ 不入库。
            with self._state_lock:
                self._skipped_success += 1
                self._pending_embeddings.pop(job.record_id, None)
                self._pending_texts.pop(job.record_id, None)
            return
        self._store_job(job)
        with self._state_lock:
            self._pending_embeddings.pop(job.record_id, None)
            self._pending_texts.pop(job.record_id, None)

    def _store_job(self, job: M4MemoryJob) -> None:
        if self._memory.get(job.record_id) is not None:
            self._memory.touch(job.record_id)
            return
        embedding, latency_ms, embedding_source = self._resolve_embedding(job)
        guard = self._guard_vector(job, embedding)
        if guard is None:
            # **宁可漏记，不可存错维度。** 拿不到在线特征就派生不出 52 维，
            # 存 1024 维进去会让守卫在某个深夜拿 52 维比 1024 维 ——
            # 那个 ValueError 只在「库里正好有带失败签名的记录」时才触发。
            return
        self._memory.observe(
            job.record_id,
            guard,
            job.full_span_record,
            verifier_refs=job.verifier_refs,
            # 配对语义（pair_id / switch_direction）随决策角色一起作废：
            # 入库判据现在是「本地执行失败」，不是「路由切换」。字段留在 schema 里
            # 是为了老库还能被解释，新记录一律 None。
            pair_id=None,
            switch_direction=None,
            embedding_latency_ms=latency_ms,
            embedding_source=embedding_source,
            template_version=job.template_version,
            experiment_id=job.experiment_id,
            failed_actions=job.failed_actions,
        )
        with self._state_lock:
            self._stored += 1
            self._embedding_source_counts[embedding_source] = (
                self._embedding_source_counts.get(embedding_source, 0) + 1
            )

    def _guard_vector(
        self, job: M4MemoryJob, embedding: Sequence[float]
    ) -> tuple[float, ...] | None:
        """1024 维嵌入 → **52 维守卫向量**（= 2 前缀标量 + PCA50，已标准化）。

        为什么在**这里**统一做，而不是让调用方传：M4 有两条写入路径 ——
        主路径（M1 在线算好，塞进 pending）和回退路径（嵌入失败，worker 用
        文本重新嵌入）。只有在这一层做，两条路才必然产出同一个维度。

        在线特征从 `full_span_record.runtime_profile.online_features` 取 ——
        `compute_span_decision_fields` 已经在用同一个来源，所以它一定在。
        """

        if self._guard_projector is not None:
            return self._guard_projector(job, embedding)
        profile = getattr(job.full_span_record, "runtime_profile", None)
        features = getattr(profile, "online_features", None)
        if not isinstance(features, Mapping) or not features:
            return None
        try:
            return guard_vector(embedding, features)
        except Exception:  # noqa: BLE001 - 维度不符 / 缺列 ⇒ 当作派生不出
            return None

    def _resolve_embedding(self, job: M4MemoryJob) -> tuple[tuple[float, ...], float, str]:
        if job.embedding is not None:
            return job.embedding, 0.0, "job_embedding"
        with self._state_lock:
            pending = self._pending_embeddings.pop(job.record_id, None)
        if pending is not None:
            return pending[0], pending[1], "m1_online"
        if self._embedder is None:
            raise RuntimeError("M4MemoryWorker has no embedder and job has no embedding")
        text = job.embedding_text
        source = "job_text"
        if text is None:
            with self._state_lock:
                text = self._pending_texts.pop(job.record_id, None)
            source = "pending_text"
        if text is None and self._text_builder is not None:
            text = self._text_builder(job.full_span_record)
            source = "fallback_text"
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("M4MemoryWorker could not build embedding text")
        result = self._embedder.embed_one(text)
        with self._state_lock:
            self._embedded += 1
        values = getattr(result, "values", None)
        if values is None:
            raise RuntimeError("embedder result must expose .values")
        latency_ms = float(getattr(result, "latency_ms", 0.0))
        return tuple(float(value) for value in values), latency_ms, source


# ----------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------
def _span_had_an_executed_failure(span_record: Any) -> bool:
    """本 Span 里**有没有一个真的执行过、并且失败了的工具调用**。

    这是 M4 收录的唯一判据（见 `M4MemoryWorker._handle_span_job`）。

    为什么不用 `not compute_local_success(...)`：那个是「本地这一段有没有出错」的
    口径，把**没执行**（准入被拒）和**执行成功了但产物被拒**（post-check REJECT）
    也算成 False。对 M4 来说那两种都不构成「这个动作会失败」的证据。

    `status` 已经蕴含 returncode 判据（`_execute_with` 里
    `FAILED iff returncode not in (None, 0)`），所以这里只认 status。
    """

    summary = getattr(span_record, "execution_summary", None)
    for tool in getattr(summary, "tool_executions", ()) or ():
        if getattr(tool, "status", None) == ToolExecutionStatus.FAILED:
            return True
    return False


def compute_local_success(span_record: Any) -> bool:
    """本地执行不出错 = true；出错 / 验证失败 / fallback = false。"""

    summary = getattr(span_record, "execution_summary", None)
    if summary is None:
        return False
    if getattr(summary, "status", None) != ExecutionStatus.SUCCESS:
        return False
    for guard in getattr(summary, "guard_results", ()):
        if getattr(guard, "outcome", None) != GuardOutcome.PASS:
            return False
    for tool in getattr(summary, "tool_executions", ()):
        if getattr(tool, "status", None) != ToolExecutionStatus.SUCCESS:
            return False
        if getattr(tool, "error_type", None):
            return False
        returncode = getattr(tool, "returncode", None)
        if returncode is not None and returncode != 0:
            return False
    return True


def compute_danger_tier(
    span_record: Any,
    *,
    local_success: bool | None = None,
) -> DangerTier:
    """成功=0；本地出错但已 containment=低；未 containment/recovery 失败=高。"""

    if local_success is None:
        local_success = compute_local_success(span_record)
    if local_success:
        return DangerTier.SUCCESS
    summary = getattr(span_record, "execution_summary", None)
    if summary is None:
        return DangerTier.HIGH
    state = getattr(summary, "state_outcome", None)
    recovery_failed = (
        getattr(summary, "status", None)
        in (ExecutionStatus.ROLLBACK_FAILED, ExecutionStatus.RESUME_FAILED, ExecutionStatus.FALLBACK_FAILED)
    )
    rollback_status = getattr(state, "rollback_status", None)
    uncontained_rollback = rollback_status in (
        RollbackStatus.FAILED,
        RollbackStatus.PARTIAL,
        RollbackStatus.UNKNOWN,
    )
    uncontained_recovery = bool(
        getattr(state, "recovery_required", False)
        and rollback_status != RollbackStatus.SUCCEEDED
    )
    effect_misclassification = any(
        str(getattr(tool, "error_type", "") or "") == "effect_misclassification"
        for tool in getattr(summary, "tool_executions", ())
    )
    recovery_check = getattr(span_record, "recovery_check", None)
    recovery_check_failed = bool(
        recovery_check is not None and not getattr(recovery_check, "passed", False)
    )
    if (
        recovery_failed
        or uncontained_rollback
        or uncontained_recovery
        or effect_misclassification
        or recovery_check_failed
    ):
        return DangerTier.HIGH
    return DangerTier.LOW


def compute_span_decision_fields(span_record: Any) -> dict[str, Any]:
    """从 ``route_plan`` 投影本次决策字段，供 M4 账本使用。

    只读已冻结的契约，不改路由行为。字段缺失时给保守默认值（"unknown" / None），
    绝不伪造一个看起来有效的决策。
    """

    route_plan = getattr(span_record, "route_plan", None)
    decision_meta = getattr(route_plan, "decision_meta", None)
    target = getattr(route_plan, "target", None)
    target_kind = _enum_value(getattr(target, "kind", None))
    execution_mode = _enum_value(getattr(route_plan, "execution_mode", None))
    if target_kind is None or execution_mode is None:
        action = "unknown"
    else:
        action = f"{target_kind}:{execution_mode}"
    profile = getattr(span_record, "runtime_profile", None)
    online_features = getattr(profile, "online_features", None)
    features_json: str | None = None
    if isinstance(online_features, Mapping) and online_features:
        features_json = json.dumps(
            {str(key): float(value) for key, value in online_features.items()},
            sort_keys=True,
            ensure_ascii=True,
        )
    return {
        "action": action,
        "decision_reason": str(getattr(decision_meta, "reason_code", None) or "unknown"),
        "policy_version": str(getattr(decision_meta, "policy_version", None) or "unknown"),
        "predictor_p": _optional_float(getattr(decision_meta, "classifier_p", None)),
        "predictor_threshold": _optional_float(
            getattr(decision_meta, "classifier_threshold", None)
        ),
        "predictor_version": _optional_str(
            getattr(decision_meta, "classifier_version", None)
        ),
        "predictor_feature_version": _optional_str(
            getattr(decision_meta, "classifier_feature_version", None)
        ),
        "router_features_json": features_json,
    }


def compute_span_cost_fields(span_record: Any) -> dict[str, Any]:
    """汇总本 Span 自身的 token / 费用 / 时延。

    ⚠️ 口径：这是**本 Span** 的开销（同一 Span 内的 retry 全部计入），
    **不是**整条轨迹的成本，也**不是**该 Span 之后的剩余段成本。
    任务级记账必须另取「整条轨迹」口径（A.18.2：崩盘后重跑要重付前缀）。
    """

    summary = getattr(span_record, "execution_summary", None)
    if summary is None:
        return {
            "span_input_tokens": None,
            "span_output_tokens": None,
            "span_cost": None,
            "span_latency_ms": 0,
        }
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    for attempt in getattr(summary, "model_attempts", ()) or ():
        attempt_input = getattr(attempt, "input_tokens", None)
        if attempt_input is not None:
            input_tokens = (input_tokens or 0) + int(attempt_input)
        attempt_output = getattr(attempt, "output_tokens", None)
        if attempt_output is not None:
            output_tokens = (output_tokens or 0) + int(attempt_output)
        attempt_cost = getattr(attempt, "cost", None)
        if attempt_cost is not None:
            cost = (cost or 0.0) + float(attempt_cost)
    latency_ms = int(getattr(summary, "latency_ms", 0) or 0)
    return {
        "span_input_tokens": input_tokens,
        "span_output_tokens": output_tokens,
        "span_cost": cost,
        "span_latency_ms": latency_ms,
    }


def _apply_task_outcome(record: M4MemoryRecord, *, scope: Any, label: Any) -> None:
    """把任务级结局写入 ``task_success``；未知一律保持 None。

    只有 ``episode_final`` / ``workflow_final`` 携带「任务最终是否成功」的语义。
    ``downstream`` 描述的是该 Span 之后的下游后果，``safety`` 描述安全性，
    两者都不是任务级结局，不能写进 ``task_success``。

    ``label_scope`` 记录最近一次写入所用的 scope，使「任务级标签是否被下放给
    单个 Span」可以被事后审计（本函数不改变 ``local_success`` 的既有口径）。
    """

    if scope not in (
        OutcomeScope.EPISODE_FINAL.value,
        OutcomeScope.WORKFLOW_FINAL.value,
    ):
        return
    record.label_scope = str(scope)
    if label == OutcomeLabel.SUCCESS.value:
        record.task_success = True
    elif label == OutcomeLabel.FAILURE.value:
        record.task_success = False
    else:
        # unknown / safe / unsafe 不构成任务级成功或失败，保持 censored。
        record.task_success = None


def span_verifier_keys(span_record: Any) -> frozenset[str]:
    """从 route_plan.verifier_plan 提取 Span 实际使用的 verifier key。"""

    route_plan = getattr(span_record, "route_plan", None)
    verifier_plan = getattr(route_plan, "verifier_plan", None)
    keys: set[str] = set()
    for step in getattr(verifier_plan, "steps", ()):
        guard_ref = getattr(step, "guard_ref", None)
        if guard_ref is not None:
            keys.add(verifier_key(guard_ref))
    return frozenset(keys)


def verifier_key(guard_ref: Any) -> str:
    """GuardRef -> kind:id:version 稳定键。"""

    return (
        f"{getattr(guard_ref, 'kind', 'guard')}:"
        f"{getattr(guard_ref, 'id', 'unknown')}:"
        f"{getattr(guard_ref, 'version', 'v1')}"
    )


def _lru_key(record: M4MemoryRecord) -> tuple[float, float, str]:
    return (record.last_access_at, record.entered_at, record.record_id)


def _normalize_embedding(embedding: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in embedding)
    if not values:
        raise ValueError("embedding cannot be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("embedding cannot have zero norm")
    return tuple(value / norm for value in values)


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    similarity = sum(a * b for a, b in zip(left, right, strict=True))
    return 1.0 - similarity


def _span_lineage(span_record: Any) -> tuple[str, str, str]:
    route_plan = getattr(span_record, "route_plan", None)
    workflow_run_id = getattr(route_plan, "workflow_run_id", None) or "workflow-unknown"
    episode_id = getattr(route_plan, "episode_id", None) or "episode-unknown"
    logical_invocation_id = (
        getattr(route_plan, "logical_invocation_id", None) or "invocation-unknown"
    )
    return str(workflow_run_id), str(episode_id), str(logical_invocation_id)


def _event_lineage(event: Any) -> tuple[str, str, str] | None:
    workflow_run_id = getattr(event, "workflow_run_id", None)
    if not workflow_run_id:
        return None
    episode_id = getattr(event, "episode_id", None) or "episode-unknown"
    logical_invocation_id = (
        getattr(event, "logical_invocation_id", None) or "invocation-unknown"
    )
    return str(workflow_run_id), str(episode_id), str(logical_invocation_id)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    """SQLite 用 NULL 表示「未知」，不能塌缩成 False。"""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none", "null", "unknown"):
            return None
        return normalized in ("1", "true", "yes")
    return bool(value)


def _pending_from_payload(payload: Any) -> _PendingSeed:
    if isinstance(payload, _PendingSeed):
        return payload
    return _PendingSeed(
        record_id=str(payload["record_id"]),
        embedding=tuple(float(value) for value in payload["embedding"]),
        verifier_refs=frozenset(str(ref) for ref in payload.get("verifier_refs", ())),
    )


def _record_from_payload(payload: Any) -> M4MemoryRecord:
    if isinstance(payload, M4MemoryRecord):
        return payload
    return M4MemoryRecord(
        record_id=str(payload["record_id"]),
        class_key=int(payload["class_key"]),
        embedding=tuple(float(value) for value in payload["embedding"]),
        full_span_record=payload.get("full_span_record"),
        local_success=bool(payload["local_success"]),
        danger_tier=DangerTier(int(payload["danger_tier"])),
        verifier_refs=frozenset(str(ref) for ref in payload.get("verifier_refs", ())),
        failed_actions=frozenset(
            str(item) for item in (payload.get("failed_actions") or ())
        ),
        workflow_run_id=str(payload.get("workflow_run_id", "workflow-unknown")),
        episode_id=str(payload.get("episode_id", "episode-unknown")),
        logical_invocation_id=str(
            payload.get("logical_invocation_id", "invocation-unknown")
        ),
        entered_at=float(payload.get("entered_at", 0.0)),
        last_access_at=float(payload.get("last_access_at", 0.0)),
        pair_id=payload.get("pair_id"),
        switch_direction=payload.get("switch_direction"),
        forced_class=bool(payload.get("forced_class", False)),
        assignment_status=str(payload.get("assignment_status", "normal")),
        embedding_latency_ms=float(payload.get("embedding_latency_ms", 0.0)),
        embedding_source=str(payload.get("embedding_source", "unknown")),
        template_version=str(payload.get("template_version", "unknown")),
        action=str(payload.get("action", "unknown")),
        decision_reason=str(payload.get("decision_reason", "unknown")),
        policy_version=str(payload.get("policy_version", "unknown")),
        predictor_p=_optional_float(payload.get("predictor_p")),
        predictor_threshold=_optional_float(payload.get("predictor_threshold")),
        predictor_version=_optional_str(payload.get("predictor_version")),
        predictor_feature_version=_optional_str(
            payload.get("predictor_feature_version")
        ),
        router_features_json=_optional_str(payload.get("router_features_json")),
        task_success=_optional_bool(payload.get("task_success")),
        label_scope=_optional_str(payload.get("label_scope")),
        span_input_tokens=_optional_int(payload.get("span_input_tokens")),
        span_output_tokens=_optional_int(payload.get("span_output_tokens")),
        span_cost=_optional_float(payload.get("span_cost")),
        span_latency_ms=int(payload.get("span_latency_ms", 0) or 0),
        suffix_cost=_optional_float(payload.get("suffix_cost")),
        trajectory_cost=_optional_float(payload.get("trajectory_cost")),
        position_ratio=_optional_float(payload.get("position_ratio")),
        trajectory_span_count=_optional_int(payload.get("trajectory_span_count")),
        step_index=_optional_int(payload.get("step_index")),
        experiment_id=str(payload.get("experiment_id", "unknown")),
    )


__all__ = [
    "ClassAssignment",
    "MemoryNeighbor",
    "M4MemoryJob",
    "M4MemoryOutcomeJob",
    "M4MemoryWorker",
    "DEFAULT_MAX_RECORDS_PER_CLASS",
    "DEFAULT_MAX_RECORDS_TOTAL",
    "DEFAULT_TOP_K",
    "DangerTier",
    "M4ClassSpec",
    "M4Memory",
    "M4MemoryRecord",
    "placeholder_class_specs",
    "compute_danger_tier",
    "compute_local_success",
    "compute_span_cost_fields",
    "compute_span_decision_fields",
    "span_verifier_keys",
    "verifier_key",
]
