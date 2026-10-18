"""M2 · 路由器实现。

四种模式
--------
    llm_only         全部走大模型（基线）
    slm_only         全部走小模型（基线）
    random_route     按固定概率随机（对照臂）
    classifier_route 116 维两阶段预测器：p_gate >= tau_gate 且 p_net >= tau_net 才卸载

外加一层装饰器 _ShadowRouter：预测器照常算、照常记录，但**决定不生效**。
shadow 臂就是靠它把「路由能力」和「路由后果」拆开。

为什么小模型目标一律是 GUARDED_SLM
----------------------------------
RoutePlan 的构造期校验要求 guarded_slm 必须有 pre-commit + 恰好一个 post-check。
以前这里写的是 DIRECT，于是守卫一次都没被调用过 —— 这是本系统最早的一个 bug。
"""

from __future__ import annotations

from ..contracts import (
    Component,
    ProducerRef,
    ExecutionMode,
    FallbackPlan,
    GuardRef,
    ModelRef,
    ModelTarget,
    ModelTargetKind,
    RoutePlan,
    RoutePolicy,
    RuntimeProfile,
    VerifierPlan,
    VerifierStep,
    VerifierStage,
    VerifierSubject,
    PassCondition,
    PassConditionKind,
    PassAction,
    FailureAction,
    FeedbackSnapshot,
    RecoveryLevel,
)

_PRE_COMMIT_STAGE = VerifierStage.PRE_COMMIT


class AdmissionFacts:
    """路由前的准入事实。字段不可得时为 None（调用方应 fail-closed 到 direct）。"""

    __slots__ = (
        "deadline_remaining_ms",
        "steps_remaining",
        "probe_budget_remaining",
        "sandbox_backend",
        "supports_transactional_write",
        "registered_tools",
    )

    def __init__(self, *, deadline_remaining_ms=None, steps_remaining=None,
                 probe_budget_remaining=None, sandbox_backend=None,
                 supports_transactional_write=None, registered_tools=None) -> None:
        self.deadline_remaining_ms = deadline_remaining_ms
        self.steps_remaining = steps_remaining
        self.probe_budget_remaining = probe_budget_remaining
        self.sandbox_backend = sandbox_backend
        self.supports_transactional_write = supports_transactional_write
        self.registered_tools = registered_tools


def _empty_verifier() -> VerifierPlan:
    return VerifierPlan(
        verifier_plan_id="verifier-none",
        version="v1",
        overall_timeout_ms=1000,
        empty_plan_reason="large_model_direct_no_guard",
    )


def _guarded_slm_plan(
    guard: GuardRef,
    *,
    pre_check_timeout_ms: int = 500,
    post_check_timeout_ms: int = 500,
    overall_timeout_ms: int = 1000,
) -> VerifierPlan:
    if min(pre_check_timeout_ms, post_check_timeout_ms, overall_timeout_ms) < 1:
        raise ValueError("guard timeout values must be positive")
    if overall_timeout_ms < max(pre_check_timeout_ms, post_check_timeout_ms):
        raise ValueError("guard overall timeout cannot be below a step timeout")
    return VerifierPlan(
        verifier_plan_id="verifier-guarded-tool-call",
        version="v1",
        steps=(
            VerifierStep(
                step_id="step-1",
                guard_ref=guard,
                stage=_PRE_COMMIT_STAGE,
                subject=VerifierSubject.CANDIDATE_OUTPUT,
                pass_condition=PassCondition(kind=PassConditionKind.SCHEMA_VALID),
                timeout_ms=pre_check_timeout_ms,
                on_pass=PassAction.CONTINUE,
                on_failure=FailureAction.REJECT_AND_FALLBACK,
            ),
            VerifierStep(
                step_id="step-post",
                guard_ref=GuardRef(kind=guard.kind, id=f"{guard.id}-post", version=guard.version),
                stage=VerifierStage.POST_CHECK,
                subject=VerifierSubject.FINAL_ARTIFACT,
                pass_condition=PassCondition(kind=PassConditionKind.POLICY_PASS),
                timeout_ms=post_check_timeout_ms,
                on_pass=PassAction.RETURN,
                on_failure=FailureAction.ROLLBACK_THEN_FALLBACK,
            ),
        ),
        overall_timeout_ms=overall_timeout_ms,
    )

class _ModeRouter:
    """Shared plan construction for the ablation routers."""

    def __init__(self, large: ModelRef, small: ModelRef) -> None:
        self._large = large
        self._small = small
        self._counter = 0

    def _plan(
        self,
        profile: RuntimeProfile,
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None,
        *,
        target: ModelTarget,
        reason: str,
        verifier_plan: VerifierPlan | None = None,
        fallback_plan: FallbackPlan | None = None,
        execution_mode: ExecutionMode = ExecutionMode.DIRECT,
        decision_meta_extra: Mapping[str, object] | None = None,
    ) -> RoutePlan:
        self._counter += 1
        decision_meta: dict[str, object] = {
            "policy_version": policy.policy_version,
            # 反馈快照已移除（发布器不在本仓库）；这一栏保留为常量，便于历史对照。
            "feedback_version": "feedback-none",
            "reason_code": reason,
            "is_exploration": False,
            "selection_probability": 1.0,
        }
        if decision_meta_extra:
            decision_meta.update(decision_meta_extra)
        return RoutePlan(
            schema_version=profile.schema_version,
            message_id=f"message-route-{profile.episode_id}",
            producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
            produced_at=profile.produced_at,
            workflow_run_id=profile.workflow_run_id,
            workflow_plan_ref=profile.workflow_plan_ref,
            workflow_plan_revision=profile.workflow_plan_revision,
            episode_id=profile.episode_id,
            logical_invocation_id=profile.logical_invocation_id,
            route_plan_id=f"route-{profile.logical_invocation_id}-{self._counter}",
            target=target,
            execution_mode=execution_mode,
            verifier_plan=verifier_plan or _empty_verifier(),
            fallback_plan=fallback_plan,
            timeout_ms=int(policy.budget.max_latency_ms),
            experience_key=profile.descriptor_key,
            decision_meta=decision_meta,
        )

    def _small_target(self) -> ModelTarget:
        return ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=self._small)

    def _large_target(self) -> ModelTarget:
        return ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=self._large)


class _AlwaysLargeRouter(_ModeRouter):
    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence,
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan:
        return self._plan(
            profile,
            policy,
            snapshot,
            target=self._large_target(),
            reason="mode_llm_only",
        )


class _AlwaysSmallRouter(_ModeRouter):
    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence,
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan:
        return self._plan(
            profile,
            policy,
            snapshot,
            target=self._small_target(),
            reason="mode_slm_only",
        )


class _RandomRouter(_ModeRouter):
    """Route each direct model call to large/small by an independent coin flip.

    The flip is drawn from a per-router ``random.Random`` seeded from the config
    (``runner.random_route.seed``), so a run is reproducible while the sequence
    of targets does not depend on the task, the span content, or any feedback.
    """

    def __init__(
        self,
        large: ModelRef,
        small: ModelRef,
        *,
        probability_large: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(large, small)
        if isinstance(probability_large, bool) or not isinstance(
            probability_large, (int, float)
        ):
            raise ValueError("runner.random_route.probability_large must be a number")
        probability = float(probability_large)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "runner.random_route.probability_large must be within [0.0, 1.0]"
            )
        self._probability_large = probability
        self._seed = seed
        self._random = random.Random(seed)
        self._draws: list[str] = []

    @property
    def draws(self) -> tuple[str, ...]:
        return tuple(self._draws)

    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence,
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan:
        is_large = self._random.random() < self._probability_large
        target_name = "large" if is_large else "small"
        self._draws.append(target_name)
        # ``reason_code`` carries the whole audit trail for this router: the
        # draw index is the position in ``_draws`` and the seed/probability are
        # reconstructible from the frozen config, so no extra ``DecisionMeta``
        # fields (which are contract-validated) are needed.
        return self._plan(
            profile,
            policy,
            snapshot,
            target=self._large_target() if is_large else self._small_target(),
            reason=f"mode_random_route_{target_name}",
        )


class _ClassifierRouter(_ModeRouter):
    """116 维两阶段预测器：p_gate >= tau_gate 且 p_net >= tau_net 才卸载。

    执行模式是契约的一部分
    ----------------------
      * SLM 目标 -> GUARDED_SLM：先过形状守卫 + M4 失败记忆，被拒则回退大模型。
        **以前这里是 DIRECT，守卫一次都没被调用过。**
      * LLM 目标 -> DIRECT：一次调用，成功即接受，不过守卫。

    拿不到向量时**直接走大模型**，不退回旧 artifact —— 那是另一个模型，
    p 的分布和 τ 的语义都不同，中途换模型等于实验换了变量。
    """

    def __init__(
        self,
        large: ModelRef,
        small: ModelRef,
        *,
        lookup: Any | None = None,
        guard: GuardRef | None = None,
    ) -> None:
        super().__init__(large, small)
        #: 返回 Add4Decision（**不是**标量 p）——两个 τ 的与塞不进单个概率。
        self._lookup = lookup
        self._guard = guard or GuardRef(kind="guard", id="basic-tool-call", version="v1")

    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence[Any] | None = None,
        policy: RoutePolicy | None = None,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan:
        decision = self._lookup(profile) if self._lookup is not None else None
        if decision is None:
            return self._plan(
                profile,
                policy,
                snapshot,
                target=self._large_target(),
                reason="l2_add4_vector_unavailable",
                decision_meta_extra={
                    "classifier_p": None,
                    "classifier_predictor": "l2_add4_unavailable",
                    "classifier_feature_version": profile.feature_version,
                },
            )
        return self._route_with_two_stage(profile, policy, snapshot, decision)

    def _route_with_two_stage(
        self,
        profile: RuntimeProfile,
        policy: RoutePolicy | None,
        snapshot: FeedbackSnapshot | None,
        decision: Any,
    ) -> RoutePlan:
        """把已经做好的两阶段判定翻译成路由（不重新判）。"""
        use_small = bool(decision.offload)
        extra: dict[str, Any] = {
            # 汇总脚本按 classifier_p / classifier_threshold 聚合，语义保留「经济那一段」；
            # 门控那一段单独记在两个新字段里。
            "classifier_p": round(float(decision.p_net), 6),
            "classifier_threshold": float(decision.tau_net),
            "classifier_p_gate": round(float(decision.p_gate), 6),
            "classifier_tau_gate": float(decision.tau_gate),
            "classifier_version": "l2_add4_two_stage",
            "classifier_feature_version": profile.feature_version,
            "classifier_predictor": "l2_add4_two_stage",
        }
        if use_small:
            return self._plan(
                profile,
                policy,
                snapshot,
                target=self._small_target(),
                reason="l2_add4_slm",
                execution_mode=ExecutionMode.GUARDED_SLM,
                verifier_plan=_guarded_slm_plan(self._guard),
                fallback_plan=FallbackPlan(
                    target=self._large_target(),
                    recovery_mode=RecoveryLevel.NOT_NEEDED,
                ),
                decision_meta_extra=extra,
            )
        return self._plan(
            profile,
            policy,
            snapshot,
            target=self._large_target(),
            reason="l2_add4_llm",
            decision_meta_extra=extra,
        )


class _ShadowRouter:
    """包住**任何** router：它照常决定，但目标被强制 —— **决策不生效**。

    为什么做成装饰器，而不是某个预测器的一个参数
    ------------------------------------------
    「不生效」是对**路由输出**（`RoutePlan`）的操作，不是对预测器的操作。
    放在这一层，换预测器（v4 ↔ add4）、换预测器的**数量**、换 mode，这个类
    **一行都不用改** —— 只要那个 router 产 `RoutePlan` 就成立。

    为什么换的是**整个 plan 形状**，不只是 `target`
    ----------------------------------------------
    内层可能给 `GUARDED_SLM` + 守卫计划 + 指回大模型的 fallback。只换
    `target` 的话 `execution_mode` 还是 `GUARDED_SLM`，这次调用会被拖进守卫
    路径 —— 而「小模型也走大模型那条路」正是用它的那组实验要保证的东西。
    所以四个字段一起换：`target` / `execution_mode` / `verifier_plan` /
    `fallback_plan`。

    ⚠️ **`decision_meta` 里的预测器字段一个都不许动**（`classifier_p` /
    `classifier_p_gate` / `classifier_predictor` …）。那是「M2 真的算了」的
    唯一证据；覆盖掉就等于对照组退化成「没装预测器」，说服力全没。
    只改 `reason_code`，并把**原本的** `reason_code` 记进 `shadow_would_route`
    —— 否则「路由器本来会卸多少」只能离线估，而不是这条真实轨迹上的数。

    ⚠️ 强制目标一律走 `DIRECT`。本类的语义是「把这次调用当作**那个目标**的
    直接调用」，不是「换个目标但保留原来的执行模式」。
    """

    def __init__(
        self, inner: Any, large: ModelRef, small: ModelRef, *, force: str
    ) -> None:
        if force not in ("large", "small"):
            # 静默收下拼错的字符串（`Large`）会在执行期以一个不相干的错误冒出来。
            raise ValueError(
                f"shadow_route.force_target 只能是 'large' 或 'small'，收到 {force!r}"
            )
        self._inner = inner
        self._large = large
        self._small = small
        self._force = force

    def _forced_target(self) -> ModelTarget:
        if self._force == "large":
            return ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=self._large)
        return ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=self._small)

    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence[Any],
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan:
        plan = self._inner.decide(
            profile, options, policy, snapshot, admission_facts=admission_facts
        )
        return plan.model_copy(
            update={
                "target": self._forced_target(),
                "execution_mode": ExecutionMode.DIRECT,
                "verifier_plan": _empty_verifier(),
                "fallback_plan": None,
                # 只动这三个键，其余（预测器那几栏）原样透传。
                "decision_meta": plan.decision_meta.model_copy(
                    update={
                        "reason_code": f"shadow_forced_{self._force}",
                        "shadow_forced_target": self._force,
                        "shadow_would_route": plan.decision_meta.reason_code,
                    }
                ),
            }
        )


