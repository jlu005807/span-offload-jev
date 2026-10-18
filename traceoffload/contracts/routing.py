"""路由层：M2 的决策输入/输出对象与验证方案。

- 验证方案（VerifierStep/VerifierPlan 等）：M2 选择、M3 执行的验证定义
- ModelTarget / FallbackPlan / DecisionMeta：路由计划的组成部分
- RoutePlan：M2 的不可变输出（M3 的执行依据、M4 的记录对象）
- RouteOption / RoutePolicy：路由候选与决策约束
- _validate_route_shape：三种合法目标/执行模式组合的约束
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import (
    CanonicalKey,
    Component,
    VersionedRef,
    ContractModel,
    ExecutionMode,
    FailureAction,
    GuardRef,
    Identifier,
    InvocationEnvelope,
    JsonObject,
    ModelRef,
    ModelTargetKind,
    NonNegativeFloat,
    PassAction,
    PassConditionKind,
    Probability,
    RecoveryLevel,
    VerifierStage,
    VerifierSubject,
    VersionId,
)
from .outcomes import ExperienceProfile


class VerifierInputBinding(ContractModel):
    """验证器输入绑定：从哪个源（引用或路径）取被验证对象。"""

    source: Identifier = Field(min_length=1, max_length=128)
    ref_or_path: Identifier = Field(min_length=1, max_length=512)


class PassCondition(ContractModel):
    """验证通过条件：条件类型 + 可选规则引用与说明。"""

    kind: PassConditionKind
    rule_ref: VersionedRef | None = None
    description: str | None = None


class VerifierStep(ContractModel):
    """验证方案中的一步：验证什么、何时验证、通过/失败后做什么。

    生产者：M2（或 runner 的模式路由器）；消费者：M3（调用 Verifier）。
    """

    step_id: Identifier = Field(min_length=1, max_length=128)
    guard_ref: GuardRef
    stage: VerifierStage
    subject: VerifierSubject
    input_bindings: tuple[VerifierInputBinding, ...] = ()
    pass_condition: PassCondition
    threshold: float | JsonObject | None = None
    timeout_ms: int = Field(gt=0)
    on_pass: PassAction
    on_failure: FailureAction

    @model_validator(mode="after")
    def threshold_matches_condition(self) -> VerifierStep:
        """只有 score_at_least 允许 threshold。"""
        if self.pass_condition.kind == PassConditionKind.SCORE_AT_LEAST:
            if self.threshold is None:
                raise ValueError("score_at_least requires threshold")
        elif self.threshold is not None:
            raise ValueError("threshold is only valid for score_at_least")
        return self


class VerifierPlan(ContractModel):
    """不可变验证方案：步骤列表 + 总超时；空计划必须说明原因。

    大模型直达路径使用显式空计划（empty_plan_reason）；
    guarded_slm 模式必须同时包含 pre-commit 与 post-check 步骤。
    """

    verifier_plan_id: Identifier = Field(min_length=1, max_length=128)
    version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )
    steps: tuple[VerifierStep, ...] = ()
    overall_timeout_ms: int = Field(gt=0)
    empty_plan_reason: Identifier | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_steps(self) -> VerifierPlan:
        """step_id 唯一；空/非空计划与 empty_plan_reason 的一致性；步骤超时不超过总超时。"""
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("verifier step_id values must be unique")
        if self.steps and self.empty_plan_reason is not None:
            raise ValueError("empty_plan_reason must be null for a non-empty verifier plan")
        if not self.steps and self.empty_plan_reason is None:
            raise ValueError("an empty verifier plan requires empty_plan_reason")
        if any(step.timeout_ms > self.overall_timeout_ms for step in self.steps):
            raise ValueError("verifier step timeout cannot exceed overall_timeout_ms")
        return self


class ModelTarget(ContractModel):
    """决策结果的目标模型：种类 + 模型引用（路由计划的核心字段）。"""

    kind: ModelTargetKind
    model_ref: ModelRef


class FallbackPlan(ContractModel):
    """fallback 方案：目标必须是大模型（升级语义），并声明恢复模式。"""

    target: ModelTarget
    recovery_mode: RecoveryLevel

    @model_validator(mode="after")
    def fallback_must_use_large_model(self) -> FallbackPlan:
        if self.target.kind != ModelTargetKind.LARGE_MODEL:
            raise ValueError("fallback target must be a large_model")
        return self


class DecisionMeta(ContractModel):
    """决策审计元数据：策略版本、反馈版本、原因码、是否探索、选择概率。

    原因码是路由可解释性的关键（如 fallback_allowed_targets_unknown、
    utility_guarded_slm）。
    """

    policy_version: VersionId = Field(min_length=1, max_length=128)
    feedback_version: VersionId = Field(min_length=1, max_length=128)
    # ``feedback_version`` remains the safety-calibration version for wire
    # compatibility.  M2 also records the exact immutable M4 snapshot used
    # for evidence lookup so the two version domains cannot be conflated.
    feedback_snapshot_version: VersionId | None = None
    reason_code: Identifier = Field(min_length=1, max_length=128)
    is_exploration: bool
    selection_probability: Probability = Field(gt=0, le=1)
    # 阶段 4 桶路由审计字段（BucketRouter 写入；其余路由器保持 None）。
    # 预测（last_observation_class -> 效应类别）与 M4 按 pre-check 真实类别
    # 聚合分离（§6.5.4），混淆矩阵离线更新权重表。
    bucket_key: CanonicalKey | None = None
    effect_prediction: Identifier | None = None
    prediction_source: Identifier | None = None
    downstream_labels_absent: bool | None = None
    # §3 成本门审计：价格源未接入时 expected_cost 聚合为 0，成本门被显式
    # 禁用并记录在此；True 表示该决策未验证“成本不劣于 direct”。
    cost_gate_disabled: bool | None = None
    # demo 分类器路由审计：p / threshold / artifact 与特征版本。
    classifier_p: Probability | None = Field(default=None, ge=0, le=1)
    classifier_threshold: Probability | None = Field(default=None, ge=0, le=1)
    classifier_version: Identifier | None = Field(default=None, min_length=1, max_length=128)
    classifier_feature_version: VersionId | None = None
    classifier_missing_features: tuple[Identifier, ...] = ()
    #: 这一刀的 `classifier_p` 是**哪个预测器**出的分。
    #: `l2_v4` = 2 个前缀标量 + `V2_goal_recent` 的 PCA(50) 的 logit/MLP；
    #: `l2_add4_two_stage` = 116 维双 logit（门控 → `classifier_p_gate`，
    #: 经济 → `classifier_p`）；`l2_new_v1` = 旧的 2 标量 artifact。**必须记**
    #: —— 换预测器会改变路由行为，不记就无法把两次实验的 `classifier_p` 放在一起比。
    classifier_predictor: Identifier | None = Field(default=None, min_length=1, max_length=64)
    #: **两阶段预测器**才有的门控那一段。
    #:
    #: `classifier_p` / `classifier_threshold` 保持**单值**语义不变（汇总脚本按它们
    #: 聚合），两阶段下它们记的是**经济那一段**；门控那一段单独放这里。
    #: 为什么必须两个都记：判定是「门控 **且** 经济」，只看一个 `p` 分不出
    #: 「全没卸载」是门控挡的还是经济挡的 —— 这两种情况的调参方向相反。
    classifier_p_gate: Probability | None = Field(default=None, ge=0, le=1)
    #: 门控那一段的 τ（与 `classifier_threshold` 是经济那一段的 τ 相对）。
    classifier_tau_gate: Probability | None = Field(default=None, ge=0, le=1)
    # --- shadow_route：预测器照常算，但决定不生效 ---------------------------
    #
    # 用于「接入 TraceOffload 但不生效」这个对照组：预测器**真的跑了**（所以
    # `classifier_p` / `classifier_p_gate` 照常有值），但路由被强制到一个固定目标。
    #
    #   开 shadow  →  `reason_code = shadow_forced_large`
    #                 `shadow_forced_target = "large"`
    #                 `shadow_would_route = <原本的 reason_code>`
    #   没开       →  两个字段都是 None
    #
    # ⚠️ **必须记「原本会怎样」**（`shadow_would_route`），不能只记「被强制了」。
    #    否则「路由器本来会卸多少」只能靠离线估，而不是这条真实轨迹上的数 ——
    #    而这个数正是对照组存在的理由。
    #
    # ⚠️ 两个字段**默认为 None 且不能给别的默认值**：汇总脚本靠
    #    `shadow_forced_target is not None` 判断「这一批开了 shadow」。
    #    没开的批次里出现非 None 会让两批的 `reason_code` 分布混在一起。
    shadow_forced_target: Identifier | None = None
    shadow_would_route: Identifier | None = None
    # --- 第二道门：M4 经济门审计 -------------------------------------------
    # 与 classifier_* 并列：第一门管「会不会坏」，这些管「会不会贵」。
    # `m4_gate_passed` 为 False 表示**是这道门否掉的**（reason_code 会是
    # `m4_economic_gate`），不是预测器否的 —— 两者必须能分开归因。
    m4_gate_passed: bool | None = None
    m4_threshold: Probability | None = Field(default=None, ge=0, le=1)
    #: 参与投票的邻居数（0 = 库里还没有带成本标签的历史，**fail-open 放行**）。
    m4_n_neighbors: int | None = Field(default=None, ge=0)
    #: 邻居中「比同位置基线贵」的相似度加权占比。
    m4_expensive_ratio: Probability | None = Field(default=None, ge=0, le=1)
    #: 邻居 `suffix_cost / 位置基线` 的加权均值（可 >1，故不设上界）。
    m4_mean_ratio: NonNegativeFloat | None = None
    #: 位置基线来源：`binned` / `global` / `none`。`none` 说明库太空。
    m4_baseline: Identifier | None = Field(default=None, min_length=1, max_length=64)
    #: **必须记**：出问题要知道是哪个历史 Span 把它带偏的。
    #: 用 `CanonicalKey` 而不是 `Identifier`：M4 的 `record_id` 形如
    #: `workflow-024c982_1|invocation-8`，含 `|`，而 `Identifier` 的 pattern
    #: 不允许（实测 4 条邻居全部校验失败）。
    m4_topk_ids: tuple[CanonicalKey, ...] = ()


def _validate_route_shape(
    target: ModelTarget,
    execution_mode: ExecutionMode,
    verifier_plan: VerifierPlan,
    fallback_plan: FallbackPlan | None,
) -> None:
    """合法目标/执行模式组合的约束：

    - large_model + direct（大模型直达或安全降级）
    - small_model + direct（兼容旧只读计划；新路由器不再生成）
    - small_model + guarded_slm（真实执行 + post-check）
    """
    if target.kind == ModelTargetKind.LARGE_MODEL and execution_mode != ExecutionMode.DIRECT:
        raise ValueError("large_model routes only support direct execution")
    if target.kind == ModelTargetKind.LARGE_MODEL and fallback_plan is not None:
        raise ValueError("large_model direct routes cannot define a fallback plan")
    if execution_mode == ExecutionMode.GUARDED_SLM and target.kind != ModelTargetKind.SMALL_MODEL:
        raise ValueError("guarded_slm routes require a small_model target")
    if execution_mode == ExecutionMode.DIRECT and verifier_plan.steps:
        raise ValueError("direct execution cannot contain verifier steps")
    if execution_mode == ExecutionMode.GUARDED_SLM:
        if not verifier_plan.steps:
            raise ValueError("guarded_slm requires a non-empty verifier plan")
        if not any(step.stage == VerifierStage.PRE_COMMIT for step in verifier_plan.steps):
            raise ValueError("guarded_slm requires a pre_commit verifier step")
        post_steps = [
            step for step in verifier_plan.steps if step.stage == VerifierStage.POST_CHECK
        ]
        if len(post_steps) != 1:
            raise ValueError("guarded_slm requires exactly one post-check")


class RoutePlan(InvocationEnvelope):
    """M2 的不可变执行方案：目标模型、执行模式、验证方案、fallback、超时与决策元数据。

    生产者：M2（decide）或 runner 的模式路由器；
    消费者：M3（执行）、M4（记录进 span_record）。构造时强制合法模式组合。
    """

    route_plan_id: Identifier = Field(min_length=1, max_length=128)
    target: ModelTarget
    execution_mode: ExecutionMode
    verifier_plan: VerifierPlan
    fallback_plan: FallbackPlan | None = None
    timeout_ms: int = Field(gt=0)
    # Task-family key shared by direct, guarded and fallback episodes so M4
    # can quarantine the family after a delayed SLM failure.
    experience_key: CanonicalKey = Field(min_length=1, max_length=512)
    # New routes use candidate-specific ``experience_key`` values and retain
    # the shared family key separately.  ``None`` is accepted for legacy
    # plans; M4 falls back to ``experience_key`` in that case.
    route_family_key: CanonicalKey | None = None
    # Effective whitelist used by pre/post verification; absent on legacy
    # plans that only carried a guard reference.
    tool_policy_ref: VersionedRef | None = None
    decision_meta: DecisionMeta

    @model_validator(mode="after")
    def validate_route(self) -> RoutePlan:
        _validate_route_shape(self.target, self.execution_mode, self.verifier_plan, self.fallback_plan)
        return self


class Budget(ContractModel):
    """当前调用或实验可接受的成本与时延上限（限制候选集合，不是事后指标）。"""

    max_cost: NonNegativeFloat = Field(ge=0)
    max_latency_ms: NonNegativeFloat = Field(ge=0)


class UtilityWeights(ContractModel):
    """M2 效用函数的成本/时延/风险权重。"""

    cost: NonNegativeFloat = Field(ge=0)
    latency: NonNegativeFloat = Field(ge=0)
    risk: NonNegativeFloat = Field(ge=0)
    # Optional weights used by the evidence-driven heuristic router.  Existing
    # pilot configs only provide cost/latency/risk and retain these defaults.
    success: NonNegativeFloat = Field(default=1.0, ge=0)
    fallback: NonNegativeFloat = Field(default=0.5, ge=0)
    overhead: NonNegativeFloat = Field(default=0.1, ge=0)


class RouteOption(ContractModel):
    """路由候选集合项：目标/模式/验证方案/fallback/可用性/经验画像。

    生产者：Coordinator（未来）；消费者：M2。available=false 的项
    必须保留原因但不得被选择。
    """

    option_id: Identifier = Field(min_length=1, max_length=128)
    target: ModelTarget
    execution_mode: ExecutionMode
    verifier_plan: VerifierPlan
    fallback_plan: FallbackPlan | None = None
    available: bool
    experience_profile: ExperienceProfile | None = None

    @model_validator(mode="after")
    def validate_route(self) -> RouteOption:
        """候选也必须满足合法执行模式组合；经验画像必须带 key。"""
        _validate_route_shape(self.target, self.execution_mode, self.verifier_plan, self.fallback_plan)
        if self.experience_profile is not None and not self.experience_profile.experience_key:
            raise ValueError("experience profile must have an experience_key")
        return self


class RoutePolicy(ContractModel):
    """本次决策的硬约束与效用权重（版本化，不包含运行时画像）。

    生产者：config（runner.policy/routing 段）；消费者：Coordinator → M2。
    """

    policy_version: VersionId = Field(min_length=1, max_length=128)
    correctness_lower_bound: Probability = Field(ge=0, le=1)
    risk_upper_bound: Probability = Field(ge=0, le=1)
    budget: Budget
    utility_weights: UtilityWeights
    exploration_constraints: JsonObject
    conservative_prior: JsonObject
    min_utility_gain: NonNegativeFloat = Field(default=0.0, ge=0)
    # Declarative rule bundles are kept as JSON objects in v1 so old callers
    # remain wire-compatible.  The loader validates their supported shape;
    # the router never evaluates arbitrary code from these mappings.
    admission_constraints: JsonObject = Field(default_factory=dict)
    evidence_constraints: JsonObject = Field(default_factory=dict)

__all__ = [
    "VerifierInputBinding",
    "PassCondition",
    "VerifierStep",
    "VerifierPlan",
    "ModelTarget",
    "FallbackPlan",
    "DecisionMeta",
    "RoutePlan",
    "Budget",
    "UtilityWeights",
    "RouteOption",
    "RoutePolicy",
]
