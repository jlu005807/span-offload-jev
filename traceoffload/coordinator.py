"""在线控制面：每个被拦截的 LM span 编排为一个 episode。

Coordinator 拥有 Profiler -> Router -> Executor 的同步链，校验每个对象边界，
保持关联 ID 一致，每个 episode 发布一条完整 span 记录，并且只把解析后的
被接受输出交给 Agent。能力引用仍用于授权，不用于删减在线 trace 字段。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Literal

from pydantic import AwareDatetime

from .contracts import (
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    Component,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ExecutionSummary,
    ActualEffect,
    AttemptRole,
    CommitStatus,
    FeedbackSnapshot,
    InvocationContext,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseToolCall,
    ModelTarget,
    ModelTargetKind,
    ProducerRef,
    RouteOption,
    RoutePlan,
    RoutePolicy,
    RuntimeProfile,
    RuntimeProjection,
    SpanRecord,
    ToolExecutionResult,
    ToolExecutionStatus,
    RollbackStatus,
    RecoveryCheck,
    RecoveryDirective,
    ResumeStatus,
    VersionedRef,
    VerifierPlan,
    WorkflowContext,
    validate_execution_link,
    validate_profile_link,
    validate_route_link,
)
from .m2.routers import AdmissionFacts
from .mechanism_counters import MechanismCounters
from .interfaces import (
    Executor,
    ProbeBudgetLedger,
    Profiler,
    Router,
    ToolAction,
    ToolProxyResult,
)
from .storage import EventHistoryStore, MemoryProbeLedger, RequestResponseStore
from .m3.integration import AdapterSession, AdapterSessionSpec, IntegrationHub
# [removed] 依赖已删除：components.router


class CoordinatorError(RuntimeError):
    """episode 无法安全启动或完成时抛出。"""


@dataclass(frozen=True)
class EpisodeOutcome:
    """Agent Adapter 从一个 episode 可以消费的内容。

    ``response`` 只在成功时设置为被接受响应；完整的候选、fallback 和审计事实
    保留在该 episode 的 SpanRecord 中，不会作为 Agent 的兼容返回值暴露。
    """

    episode_id: str
    logical_invocation_id: str
    status: ExecutionStatus
    response: ModelResponse | None
    model_ref: ModelRef
    # Cost of the accepted physical model attempt.  The adapter uses this for
    # the same per-query accounting that DefaultAgent applies to native model
    # responses.
    cost: float | None = None


def _accepted_attempt_cost(result: ExecutionResult) -> float | None:
    accepted_id = result.accepted_model_attempt_id
    if accepted_id is None:
        return None
    for attempt in result.model_attempts:
        if attempt.model_attempt_id == accepted_id:
            return attempt.cost
    return None


@dataclass
class _PendingEpisode:
    """一个已交付模型响应、等待 Agent 工具结果的 Coordinator 状态。"""

    profile: RuntimeProfile
    route: RoutePlan
    source_request: ModelRequest
    request: ModelRequest
    workflow: WorkflowContext
    caller_deadline_at: AwareDatetime
    snapshot: FeedbackSnapshot | None
    execution_request: ExecutionRequest
    execution_result: ExecutionResult
    response: ModelResponse
    span_kind: Literal["normal", "fallback", "rollback_recovery"] = "normal"
    parent_episode_id: str | None = None
    span_relation: Literal["fallback_of", "recovery_of"] | None = None
    recovery_directive: RecoveryDirective | None = None
    tool_results: dict[str, ToolProxyResult] = field(default_factory=dict)


@dataclass
class _RecoveryLock:
    """Coordinator-local synchronous state for a blocked workflow."""

    directive: RecoveryDirective
    source_transaction_ids: tuple[str, ...]


@dataclass
class _WorkflowLoopState:
    """Cross-episode health state for a stalled small-model loop.

    The state is deliberately based on completed tool facts and accepted model
    outputs, rather than command names or parser-specific heuristics.  Two
    identical successful read-only action batches with no intervening state
    effect arm a one-shot direct-large handoff for the next episode.  The same
    reasoning content repeated across episodes (while commands keep shifting
    inside the same read-only region) arms a persistent direct-large lock that
    is only released by a state-changing episode, because one large-model turn
    may not itself perform the missing write.
    """

    last_action_fingerprint: str | None = None
    repeat_count: int = 0
    action_summary: str | None = None
    last_reasoning_fingerprint: str | None = None
    reasoning_repeat_count: int = 0
    reasoning_summary: str | None = None
    loop_kind: Literal["command", "reasoning"] = "command"
    force_direct: bool = False


@dataclass(frozen=True)
class _LoopDirective:
    """Instruction consumed by the next normal workflow episode.

    Command loops keep the historical one-shot semantics: the state is cleared
    before the large-model episode.  Reasoning fixed-point loops are persistent
    until a write/unknown/failed episode proves the workspace or control flow
    changed.
    """

    repeat_count: int
    action_summary: str
    loop_kind: Literal["command", "reasoning"] = "command"


class RecoveryController:
    """Owns workflow recovery locks; M4 only learns from their recorded facts.

    A lock is created only after an SLM post-check failure whose rollback
    failed.  While it exists, Coordinator bypasses normal routing and forces
    a large-model DIRECT recovery span.  ``resume_check`` is intentionally a
    separate operation from the empty verifier plan of that direct route.
    """

    def __init__(
        self,
        *,
        recovery_checker: Callable[[Sequence[str]], ResumeStatus] | None = None,
        recovery_attempt_limit: int = 3,
    ) -> None:
        if recovery_attempt_limit < 1:
            raise ValueError("recovery_attempt_limit must be at least 1")
        self._recovery_checker = recovery_checker
        self._recovery_attempt_limit = recovery_attempt_limit
        self._locks: dict[str, _RecoveryLock] = {}
        self._counter = 0

    def directive(self, workflow_run_id: str) -> RecoveryDirective | None:
        lock = self._locks.get(workflow_run_id)
        return None if lock is None else lock.directive

    def transaction_ids(self, workflow_run_id: str) -> tuple[str, ...]:
        """Return source transactions whose sandbox state is held by this lock."""

        lock = self._locks.get(workflow_run_id)
        return () if lock is None else lock.source_transaction_ids

    def activate(
        self,
        workflow_run_id: str,
        *,
        route: RoutePlan,
        result: ExecutionResult,
        caller_deadline_at: AwareDatetime,
    ) -> RecoveryDirective:
        fallback = route.fallback_plan
        if fallback is None:
            raise CoordinatorError(
                "rollback failure cannot enter recovery without a large-model fallback target"
            )
        self._counter += 1
        directive = RecoveryDirective(
            recovery_id=f"recovery-{self._counter}",
            source_episode_id=route.episode_id,
            source_logical_invocation_id=route.logical_invocation_id,
            source_route_plan_id=route.route_plan_id,
            target=fallback.target,
            failed_tool_results=result.tool_executions,
            state_outcome=result.state_outcome,
            recovery_attempt_count=0,
            recovery_attempt_limit=self._recovery_attempt_limit,
            deadline_at=caller_deadline_at,
        )
        transaction_ids = tuple(
            dict.fromkeys(
                item.transaction_id
                for item in result.tool_executions
                if item.effect in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
            )
        )
        self._locks[workflow_run_id] = _RecoveryLock(
            directive=directive,
            source_transaction_ids=transaction_ids,
        )
        return directive

    def exhausted_reason(
        self,
        workflow_run_id: str,
        *,
        now: AwareDatetime,
    ) -> str | None:
        """Return the terminal reason without silently dropping a recovery lock."""

        lock = self._locks.get(workflow_run_id)
        if lock is None:
            return None
        if now >= lock.directive.deadline_at:
            return "recovery_deadline_exhausted"
        if lock.directive.recovery_attempt_count >= lock.directive.recovery_attempt_limit:
            return "recovery_attempt_budget_exhausted"
        return None

    def clear(self, workflow_run_id: str) -> None:
        """Release a lock only after Coordinator has recorded its terminal reason."""

        self._locks.pop(workflow_run_id, None)

    def complete_attempt(
        self,
        workflow_run_id: str,
        tool_results: Sequence[ToolExecutionResult],
        *,
        now: AwareDatetime,
    ) -> RecoveryCheck | None:
        lock = self._locks.get(workflow_run_id)
        if lock is None:
            return None
        transaction_ids = tuple(
            dict.fromkeys(
                (*lock.source_transaction_ids, *(item.transaction_id for item in tool_results))
            )
        )
        if now >= lock.directive.deadline_at:
            status = ResumeStatus.UNKNOWN
            reason = "recovery_deadline_exhausted"
        else:
            try:
                status = (
                    self._recovery_checker(transaction_ids)
                    if self._recovery_checker is not None
                    else ResumeStatus.UNKNOWN
                )
            except Exception:
                status = ResumeStatus.UNKNOWN
            reason = None if status == ResumeStatus.SUCCEEDED else "recovery_check_failed"
        attempt_index = lock.directive.recovery_attempt_count + 1
        if (
            status != ResumeStatus.SUCCEEDED
            and attempt_index >= lock.directive.recovery_attempt_limit
        ):
            reason = "recovery_attempt_budget_exhausted"
        check = RecoveryCheck(
            recovery_id=lock.directive.recovery_id,
            attempt_index=attempt_index,
            checked_transaction_ids=transaction_ids,
            status=status,
            passed=status == ResumeStatus.SUCCEEDED,
            reason=reason,
        )
        if check.passed:
            self._locks.pop(workflow_run_id, None)
        else:
            self._locks[workflow_run_id] = _RecoveryLock(
                directive=lock.directive.model_copy(
                    update={
                        "recovery_attempt_count": check.attempt_index,
                    }
                ),
                source_transaction_ids=lock.source_transaction_ids,
            )
        return check


def _utcnow() -> AwareDatetime:
    return datetime.now(timezone.utc)


def _content_hash(request: ModelRequest) -> str:
    """执行引用的内容哈希：请求 JSON 的 sha256，防止执行请求被替换。"""
    return "sha256:" + sha256(request.model_dump_json().encode()).hexdigest()


def _jsonable(value: object) -> object:
    """保留完整工具结果，同时确保 SpanRecord 可序列化。"""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


def _raw_output(output: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _jsonable(value) for key, value in output.items()}


def _actions_from_message(message: Mapping[str, object]) -> tuple[ToolAction, ...]:
    """从 mini-swe-agent 的 assistant message 读取标准化 action 列表。"""

    extra = message.get("extra")
    raw_actions = extra.get("actions", []) if isinstance(extra, Mapping) else []
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        return ()
    actions: list[ToolAction] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            continue
        call_id = raw.get("tool_call_id")
        command = raw.get("command")
        if not isinstance(call_id, str):
            call_id = f"tool-call-{index + 1}"
        arguments = {
            str(key): value
            for key, value in raw.items()
            if key not in {"tool_call_id"}
        }
        actions.append(
            ToolAction(
                tool_call_id=call_id,
                tool_name=str(raw.get("tool_name", "bash")),
                arguments=arguments,
                cwd=str(raw.get("cwd", "")),
            )
        )
    return tuple(actions)


def _proxy_result_from_output(
    action: ToolAction,
    output: Mapping[str, object],
    pending: _PendingEpisode,
) -> ToolProxyResult:
    """兼容 adapter 未回传结构化 Proxy 结果。

    The observation is deliberately marked ``confirmed=False``.  It can keep
    a direct route moving, but a guarded route must not treat framework output
    as proof that the authorized primary action ran.
    """

    raw_code = output.get("returncode")
    try:
        returncode = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        returncode = -1
    status = (
        ToolExecutionStatus.SUCCESS
        if returncode in (None, 0)
        else ToolExecutionStatus.FAILED
    )
    effect = ActualEffect.UNKNOWN
    commit = CommitStatus.UNKNOWN
    summary_value = output.get("output")
    output_length = len(summary_value) if isinstance(summary_value, str) else len(str(summary_value or ""))
    summary = f"output_chars={output_length}; returncode={returncode!s}"
    record = ToolExecutionResult(
        transaction_id=f"transaction-{pending.execution_request.execution_request_id}-{action.tool_call_id}",
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        status=status,
        effect=effect,
        commit_status=commit,
        returncode=returncode,
        output_summary=summary,
        raw_output=_raw_output(output),
        result_ref=VersionedRef(
            kind="tool_result",
            id=f"observed-{action.tool_call_id}",
            version="v1",
        ),
        error_type=("tool-nonzero" if status != ToolExecutionStatus.SUCCESS else None),
    )
    return ToolProxyResult(record=record, output=dict(output), confirmed=False)


def _missing_action_result(
    action: ToolAction,
    pending: _PendingEpisode,
) -> ToolProxyResult:
    """将 adapter 未交回的已授权 tool call 显式标记为失败。"""

    record = ToolExecutionResult(
        transaction_id=f"transaction-{pending.execution_request.execution_request_id}-{action.tool_call_id}",
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        status=ToolExecutionStatus.REJECTED,
        effect=ActualEffect.UNKNOWN,
        commit_status=CommitStatus.BLOCKED,
        returncode=-1,
        output_summary="action_missing_from_adapter",
        raw_output={
            "output": "",
            "returncode": -1,
            "exception_info": "action missing from adapter",
        },
        result_ref=VersionedRef(
            kind="tool_result",
            id=f"missing-{action.tool_call_id}",
            version="v1",
        ),
        error_type="action-missing",
    )
    return ToolProxyResult(
        record=record,
        output={
            "output": "",
            "returncode": -1,
            "exception_info": "action missing from adapter",
        },
        confirmed=False,
    )


class Coordinator:
    """在线编排中心：组装 M1-M4，但不知道它们的实现细节（只依赖 Protocol）。"""

    def __init__(
        self,
        *,
        profiler: Profiler,
        router: Router,
        executor: Executor,
        store: RequestResponseStore,
        events: EventHistoryStore,
        policy: RoutePolicy,
        options: Sequence[RouteOption] = (),
        clock: Callable[[], AwareDatetime] | None = None,
        span_observer: Callable[[SpanRecord, ModelResponse | None], None] | None = None,
        profile_observer: (
            Callable[[InvocationContext, ModelRequest, RuntimeProfile], None] | None
        ) = None,
        integration_hub: IntegrationHub | None = None,
        admission_gate: object | None = None,
        recovery_attempt_limit: int = 3,
        step_limit: int | None = None,
        probe_budget_per_workflow: int | None = None,
        probe_budget_total: int | None = None,
        probe_ledger: ProbeBudgetLedger | None = None,
        experiment_id: str | None = None,
        sandbox_backend: str | None = None,
        supports_transactional_write: bool | None = None,
        registered_tools: tuple[str, ...] | None = None,
        large_model: ModelRef | None = None,
        reasoning_loop_repeat_limit: int = 2,
        counters: MechanismCounters | None = None,
    ) -> None:
        self._profiler = profiler
        self._router = router
        self._executor = executor
        self._store = store
        self._events = events
        self._policy = policy
        self._options = tuple(options)
        self._step_limit = step_limit
        self._probe_budget_per_workflow = probe_budget_per_workflow
        self._probe_budget_total = probe_budget_total
        self._probe_ledger = probe_ledger or MemoryProbeLedger()
        self._experiment_id = experiment_id or "default"
        self._sandbox_backend = sandbox_backend
        self._supports_transactional_write = supports_transactional_write
        self._registered_tools = registered_tools
        self._large_model = large_model
        if reasoning_loop_repeat_limit < 2:
            raise ValueError("reasoning_loop_repeat_limit must be at least 2")
        self._reasoning_loop_repeat_limit = reasoning_loop_repeat_limit
        #: 机制计数（T2）。fallback 的次数与原因记在这里 —— 之前这个计数位
        #: 只声明没人加，summary 里永远是 {}。
        self._counters = counters if counters is not None else MechanismCounters.null()
        self._clock = clock or _utcnow
        self._span_observer = span_observer
        self._profile_observer = profile_observer
        self._integration_hub = integration_hub or IntegrationHub(clock=self._clock)
        self._admission_gate = admission_gate
        self._episode_counter = 0
        self._pending: dict[str, _PendingEpisode] = {}
        self._response_costs: dict[tuple[str, str], float | None] = {}
        self._workflow_loop_state: dict[str, _WorkflowLoopState] = {}
        # Thread safety for concurrent tool result reporting
        from threading import Lock
        self._pending_lock = Lock()
        checker = getattr(executor, "execute_recovery_check", None)
        self._recovery = RecoveryController(
            recovery_checker=checker if callable(checker) else None,
            recovery_attempt_limit=recovery_attempt_limit,
        )

    @property
    def integration_hub(self) -> IntegrationHub:
        """Coordinator-owned framework integration boundary."""

        return self._integration_hub

    @property
    def large_model_ref(self) -> ModelRef | None:
        """Configured safety/fallback model, exposed for adapter provenance."""

        return self._large_model

    def open_adapter_session(self, spec: AdapterSessionSpec) -> AdapterSession:
        """Open or retrieve the framework-neutral session for one workflow."""

        return self._integration_hub.open_session(spec)

    def recovery_directive(self, workflow_run_id: str) -> RecoveryDirective | None:
        """Return an active recovery lock for the adapter's next LM request."""

        return self._recovery.directive(workflow_run_id)

    def accepted_response_cost(
        self,
        workflow_run_id: str,
        response: ModelResponse,
    ) -> float | None:
        """Return the cost recorded for an accepted response delivered to Agent."""

        return self._response_costs.get((workflow_run_id, response.response_id))

    def tool_transaction_id(self, workflow_run_id: str, tool_call_id: str) -> str:
        """Bind an Agent tool callback to its currently pending M3 span.

        Security: Only generate transaction IDs from active pending episodes.
        If no pending episode exists, this indicates a delayed or out-of-order callback.
        """
        pending = self._pending.get(workflow_run_id)
        if pending is not None:
            return f"transaction-{pending.execution_request.execution_request_id}-{tool_call_id}"

        # Warning: no pending episode - this may be a delayed callback
        # Generate a workflow-level ID but log the issue
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"tool_transaction_id called for workflow {workflow_run_id} with no pending episode. "
            f"This may indicate a delayed or out-of-order tool callback."
        )
        return f"transaction-{workflow_run_id}-{tool_call_id}"

    def close_workflow(self, workflow_run_id: str) -> None:
        """Release an unresolved recovery lock when its Agent workflow exits.

        Ensures complete cleanup including:
        - Recovery locks
        - Adapter resources (Docker containers, etc.)
        """

        # Health state is workflow-scoped and must not leak into a later run
        # that happens to reuse the same adapter object.
        self._workflow_loop_state.pop(workflow_run_id, None)
        directive = self._recovery.directive(workflow_run_id)
        if directive is not None:
            transaction_ids = self._recovery.transaction_ids(workflow_run_id)
            self._recovery.clear(workflow_run_id)
            self._finalize_recovery(transaction_ids)
            self._set_terminal_error(
                workflow_run_id,
                (
                    "workflow ended with an unresolved recovery lock "
                    f"(source episode {directive.source_episode_id})"
                ),
            )

        # Clean up adapter resources (Docker containers, etc.)
        session = self._integration_hub.session(workflow_run_id)
        if session is not None:
            cleanup = getattr(session, 'cleanup_resources', None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to cleanup resources for workflow {workflow_run_id}: {error}"
                    )

    def orchestrate_episode(
        self,
        request: ModelRequest,
        workflow: WorkflowContext,
        *,
        workflow_run_id: str,
        workflow_plan_ref: VersionedRef | None = None,
        workflow_plan_revision: int | None = None,
        caller_deadline_at: AwareDatetime,
        snapshot: FeedbackSnapshot | None = None,
        _forced_target: ModelTarget | None = None,
        _forced_reason: str | None = None,
        _attempt_role_override: AttemptRole | None = None,
        _span_kind: Literal["normal", "fallback", "rollback_recovery"] = "normal",
        _parent_episode_id: str | None = None,
        _span_relation: Literal["fallback_of", "recovery_of"] | None = None,
        _recovery_directive: RecoveryDirective | None = None,
    ) -> EpisodeOutcome:
        """编排一个 episode：M1 画像 → M2 路由 → M3 执行 → 记录 span → 返回接受输出。"""
        if (workflow_plan_ref is None) != (workflow_plan_revision is None):
            raise CoordinatorError(
                "workflow_plan_ref and workflow_plan_revision must be both set or both null"
            )
        produced_at = self._clock()
        recovery_directive = _recovery_directive or self._recovery.directive(workflow_run_id)
        if recovery_directive is not None:
            exhaustion = self._recovery.exhausted_reason(
                workflow_run_id,
                now=produced_at,
            )
            if exhaustion is not None:
                transaction_ids = self._recovery.transaction_ids(workflow_run_id)
                self._recovery.clear(workflow_run_id)
                self._finalize_recovery(transaction_ids)
                message = (
                    f"workflow recovery terminated: {exhaustion} "
                    f"(source episode {recovery_directive.source_episode_id})"
                )
                self._set_terminal_error(workflow_run_id, message)
                raise CoordinatorError(message)
            workflow = workflow.model_copy(update={"recovery_mode": "rollback_recovery"})
            caller_deadline_at = min(caller_deadline_at, recovery_directive.deadline_at)
        self._episode_counter += 1
        episode_id = f"episode-{self._episode_counter}"
        logical_invocation_id = f"invocation-{self._episode_counter}"
        if caller_deadline_at <= produced_at:
            raise CoordinatorError("caller deadline has already passed")

        # A loop lock is only applied to ordinary routing. It changes the
        # selected target, never the Agent-owned message history.
        loop_directive = None
        if _forced_target is None and recovery_directive is None:
            loop_directive = self._take_loop_directive(workflow_run_id)

        # --- 第 1 步：为 M1 构造 PROFILE_READ 能力与 InvocationContext ---
        binding = CapabilityBinding(
            workflow_run_id=workflow_run_id,
            episode_id=episode_id,
            logical_invocation_id=logical_invocation_id,
            workflow_plan_ref=workflow_plan_ref,
            workflow_plan_revision=workflow_plan_revision,
        )
        profile_ref = CapabilityRef(
            kind="request",
            id=f"request-{episode_id}",
            version="1.0",
            scope=CapabilityScope.PROFILE_READ,
            audience=Component.M1,
            expires_at=caller_deadline_at,
            binding=binding,
        )
        session = self._integration_hub.session(workflow_run_id)
        prior_execution_facts = (
            session.snapshot_prior_execution_facts() if session is not None else None
        )
        context = InvocationContext(
            schema_version="1.0",
            message_id=f"message-{episode_id}",
            producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
            produced_at=produced_at,
            workflow_run_id=workflow_run_id,
            workflow_plan_ref=workflow_plan_ref,
            workflow_plan_revision=workflow_plan_revision,
            episode_id=episode_id,
            logical_invocation_id=logical_invocation_id,
            request_ref=profile_ref,
            workflow_context=workflow,
            prior_execution_facts=prior_execution_facts,
            caller_deadline_at=caller_deadline_at,
        )
        # The agent history is the sole context source. M1, M3, and a possible
        # fallback consume this one immutable request; routing only selects a
        # model target and never projects target-specific context.
        canonical_request = request
        self._store.save_request(profile_ref, canonical_request)

        # --- 第 2 步：M1 画像 + 边界校验 ---
        profile = self._profiler.build_profile(context, canonical_request)
        validate_profile_link(context, profile)
        if self._profile_observer is not None:
            self._profile_observer(context, canonical_request, profile)

        # --- 第 3 步：loop/recovery lock > 路由前准入 > M2 决策 ---
        if _forced_target is not None:
            route = self._direct_route(
                profile,
                snapshot,
                target=_forced_target,
                reason=_forced_reason or "coordinator_forced_direct",
            )
        elif recovery_directive is not None:
            # 锁住的 workflow 不进入 M2 的取模/探索逻辑，强制大模型 direct。
            route = self._direct_route(
                profile,
                snapshot,
                target=recovery_directive.target,
                reason="workflow_recovery_lock",
            )
        elif loop_directive is not None:
            # Do not ask M2 to rediscover a health decision that Coordinator
            # already proved from completed tool results.
            route = self._direct_route(
                profile,
                snapshot,
                target=self._large_target(),
                reason=(
                    "repeated_reasoning_loop"
                    if loop_directive.loop_kind == "reasoning"
                    else "repeated_command_loop"
                ),
            )
        else:
            preflight_reason = (
                self._admission_gate.reject_reason(profile, self._policy)
                if self._admission_gate is not None
                else None
            )
            if preflight_reason is not None:
                route = self._admission_gate.direct_plan(
                    profile,
                    self._policy,
                    snapshot,
                    reason=preflight_reason,
                )
            else:
                route = self._router.decide(
                    profile,
                    self._options,
                    self._policy,
                    snapshot,
                    admission_facts=self._admission_facts(
                        workflow, caller_deadline_at, workflow_run_id
                    ),
                )

        # 探针预算原子扣减（A7）：台账失败仍 fail-closed 到 large/direct。
        if route.decision_meta.is_exploration and (
            self._probe_budget_per_workflow is not None
            or self._probe_budget_total is not None
        ):
            try:
                consumed = self._probe_ledger.try_consume(
                    self._experiment_id,
                    workflow_run_id,
                    route.route_plan_id,
                    route.decision_meta.reason_code,
                    workflow_limit=self._probe_budget_per_workflow,
                    total_limit=self._probe_budget_total,
                )
            except Exception:
                consumed = False
            if not consumed:
                route = self._direct_route(
                    profile,
                    snapshot,
                    target=self._large_target_for(route),
                    reason="probe_budget_exhausted",
                )
        validate_route_link(profile, route)

        execution_model_request = canonical_request

        # --- 第 4 步：为 M3 构造 EXECUTION_READ 能力（含内容哈希与目标模型绑定）---
        execution_ref = CapabilityRef(
            kind="request",
            id=f"request-exec-{episode_id}",
            version="1.0",
            scope=CapabilityScope.EXECUTION_READ,
            audience=Component.M3,
            content_hash=_content_hash(execution_model_request),
            expires_at=caller_deadline_at,
            binding=binding.model_copy(
                update={"target_model_ref": route.target.model_ref}
            ),
        )
        self._store.save_request(execution_ref, execution_model_request)

        # --- 第 5 步：有 fallback 计划时，为 fallback 目标单独构造执行引用 ---
        fallback_request_ref = None
        if route.fallback_plan is not None:
            fallback_request_ref = CapabilityRef(
                kind="request",
                id=f"request-fallback-{episode_id}",
                version="1.0",
                scope=CapabilityScope.EXECUTION_READ,
                audience=Component.M3,
                content_hash=_content_hash(canonical_request),
                expires_at=caller_deadline_at,
                binding=binding.model_copy(
                    update={"target_model_ref": route.fallback_plan.target.model_ref}
                ),
            )
            self._store.save_request(fallback_request_ref, canonical_request)

        # --- 第 6 步：非 direct 模式必须提供事务能力引用（契约要求）---
        step_transaction_ref = None
        if route.execution_mode != ExecutionMode.DIRECT:
            step_transaction_ref = CapabilityRef(
                kind="transaction",
                id=f"transaction-{episode_id}",
                version="1.0",
                scope=CapabilityScope.TRANSACTION,
                audience=Component.M3,
                expires_at=caller_deadline_at,
                binding=binding,
            )

        # --- 第 7 步：构造 ExecutionRequest（deadline 取调用方期限与路由预算的较早者）---
        execution_request = ExecutionRequest(
            schema_version="1.0",
            message_id=f"message-exec-{episode_id}",
            producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
            produced_at=produced_at,
            execution_request_id=f"execution-{episode_id}",
            route_plan=route,
            request_ref=execution_ref,
            fallback_request_ref=fallback_request_ref,
            step_transaction_ref=step_transaction_ref,
            safety_calibration_ref=VersionedRef(
                kind="calibration",
                id="safety",
                version=route.decision_meta.feedback_version,
            ),
            deadline_at=min(
                caller_deadline_at,
                produced_at + timedelta(milliseconds=route.timeout_ms),
            ),
            idempotency_key=f"idempotency-{episode_id}",
            attempt_role_override=_attempt_role_override,
        )

        # --- 第 8 步：M3 执行 + 边界校验 + 交付 accepted 输出 ---
        result = self._executor.execute_step(execution_request)
        validate_execution_link(execution_request, result)
        span_kind = (
            _span_kind
            if _span_kind != "normal" or recovery_directive is None
            else "rollback_recovery"
        )
        parent_episode_id = (
            _parent_episode_id
            if _parent_episode_id is not None
            else (
                recovery_directive.source_episode_id
                if recovery_directive is not None
                else None
            )
        )
        span_relation = (
            _span_relation
            if _span_relation is not None
            else ("recovery_of" if recovery_directive is not None else None)
        )

        if result.status != ExecutionStatus.SUCCESS:
            source_directive = recovery_directive
            if (
                (
                    result.status == ExecutionStatus.ROLLBACK_FAILED
                    or result.state_outcome.recovery_required
                )
                and route.execution_mode == ExecutionMode.GUARDED_SLM
            ):
                try:
                    source_directive = self._recovery.activate(
                        workflow_run_id,
                        route=route,
                        result=result,
                        caller_deadline_at=caller_deadline_at,
                    )
                except CoordinatorError:
                    source_directive = None
            recovery_check = self._complete_recovery_if_needed(
                workflow_run_id,
                recovery_directive,
                result.tool_executions,
            )
            span_record = self._build_span(
                execution_model_request,
                profile,
                route,
                result,
                span_kind=span_kind,
                parent_episode_id=parent_episode_id,
                span_relation=span_relation,
                recovery_directive=source_directive,
                recovery_check=recovery_check,
            )
            self._events.append_span_record(span_record)
            self._record_loop_outcome(workflow_run_id, route, response=None, result=result)
            self._observe_span(span_record, None)
            self._terminate_exhausted_recovery(workflow_run_id)
            if self._should_fallback(route, result):
                return self._start_fallback_episode(
                    source_request=canonical_request,
                    source_workflow=workflow,
                    source_route=route,
                    source_result=result,
                    workflow_run_id=workflow_run_id,
                    workflow_plan_ref=workflow_plan_ref,
                    workflow_plan_revision=workflow_plan_revision,
                    caller_deadline_at=caller_deadline_at,
                    snapshot=snapshot,
                )
            return EpisodeOutcome(
                episode_id=episode_id,
                logical_invocation_id=logical_invocation_id,
                status=result.status,
                response=None,
                model_ref=route.target.model_ref,
                cost=_accepted_attempt_cost(result),
            )
        accepted_ref = result.accepted_output_ref
        if accepted_ref is None:
            raise CoordinatorError("success result is missing accepted_output_ref")
        response = self._store.resolve_accepted_output(
            accepted_ref,
            Component.AGENT_ADAPTER,
        )
        accepted_cost = _accepted_attempt_cost(result)
        self._response_costs[(workflow_run_id, response.response_id)] = accepted_cost
        pending = _PendingEpisode(
            profile=profile,
            route=route,
            source_request=canonical_request,
            request=execution_model_request,
            workflow=workflow,
            caller_deadline_at=caller_deadline_at,
            snapshot=snapshot,
            execution_request=execution_request,
            execution_result=result,
            response=response,
            span_kind=span_kind,
            parent_episode_id=parent_episode_id,
            span_relation=span_relation,
            recovery_directive=recovery_directive,
        )
        if response.tool_calls and callable(getattr(self._executor, "execute_tool_phase", None)):
            self._pending[workflow_run_id] = pending
            # 先登记模型响应和工具绑定；终态 SpanRecord 等工具阶段完成后
            # 再发布。尤其是 recovery span，此时还没有真实 RecoveryCheck，
            # 因此不能构造临时的终态记录。
            self._integration_hub.record_pending_span(route, result, response)
        else:
            recovery_check = self._complete_recovery_if_needed(
                workflow_run_id,
                recovery_directive,
                result.tool_executions,
            )
            span_record = self._build_span(
                execution_model_request,
                profile,
                route,
                result,
                span_kind=pending.span_kind,
                parent_episode_id=pending.parent_episode_id,
                span_relation=pending.span_relation,
                recovery_directive=recovery_directive,
                recovery_check=recovery_check,
            )
            self._events.append_span_record(span_record)
            self._record_loop_outcome(workflow_run_id, route, response=response, result=result)
            self._observe_span(span_record, response)
            self._terminate_exhausted_recovery(workflow_run_id)
        return EpisodeOutcome(
            episode_id=episode_id,
            logical_invocation_id=logical_invocation_id,
            status=result.status,
            response=response,
            model_ref=route.target.model_ref,
            cost=accepted_cost,
        )

    def record_tool_proxy_result(
        self,
        workflow_run_id: str,
        result: ToolProxyResult,
    ) -> None:
        """由 Agent adapter 在真实 Environment 返回后上报工具事务结果。

        Thread-safe: Uses lock to prevent race conditions in concurrent scenarios.
        """

        with self._pending_lock:
            pending = self._pending.get(workflow_run_id)
            if pending is None:
                return
            pending.tool_results[result.record.tool_call_id] = result

    def complete_tool_phase(
        self,
        workflow_run_id: str,
        message: Mapping[str, object],
        outputs: Sequence[Mapping[str, object]],
    ) -> ModelResponse | None:
        """在 Agent 格式化 tool observation 时闭合 pending LM Span。

        返回值是需要注入 Agent 下一轮的 fallback 响应；普通成功路径返回
        ``None``。调用方不需要知道 M3 的恢复或能力引用细节。
        """

        pending = self._pending.get(workflow_run_id)
        if pending is None:
            # Check for terminal error to avoid silent failure
            session = self._integration_hub.session(workflow_run_id)
            if session and hasattr(session, 'terminal_error') and session.terminal_error:
                raise CoordinatorError(
                    f"Cannot complete tool phase: workflow terminated with error: {session.terminal_error}"
                )
            return None
        observed = self._collect_tool_results(pending, message, outputs)
        execute_tool_phase = getattr(self._executor, "execute_tool_phase", None)
        if not callable(execute_tool_phase):
            result = pending.execution_result
        else:
            result = execute_tool_phase(
                pending.execution_request,
                pending.execution_result,
                pending.response,
                tuple(observed),
            )
        validate_execution_link(pending.execution_request, result)
        self._pending.pop(workflow_run_id, None)
        if (
            (
                result.status == ExecutionStatus.ROLLBACK_FAILED
                or result.state_outcome.recovery_required
            )
            and pending.route.execution_mode == ExecutionMode.GUARDED_SLM
        ):
            try:
                directive = self._recovery.activate(
                    workflow_run_id,
                    route=pending.route,
                    result=result,
                    caller_deadline_at=pending.caller_deadline_at,
                )
            except CoordinatorError:
                directive = None
            self._publish_pending(
                pending,
                result,
                None,
                recovery_directive=directive,
            )
            if directive is None:
                self._set_terminal_error(
                    workflow_run_id,
                    f"episode {pending.route.episode_id} rollback failed without recovery target",
                )
            # No recovery response is queued here.  Agent first receives the
            # failed tool observation; its next query is forced by the lock.
            return None

        accepted_response = self._accepted_response(result)
        recovery_check = self._complete_recovery_if_needed(
            workflow_run_id,
            pending.recovery_directive,
            result.tool_executions,
        )
        self._publish_pending(
            pending,
            result,
            accepted_response,
            recovery_check=recovery_check,
        )
        self._terminate_exhausted_recovery(workflow_run_id)
        if self._should_fallback(pending.route, result):
            outcome = self._start_fallback_episode(
                source_request=pending.source_request,
                source_workflow=pending.workflow,
                source_route=pending.route,
                source_result=result,
                workflow_run_id=workflow_run_id,
                workflow_plan_ref=pending.route.workflow_plan_ref,
                workflow_plan_revision=pending.route.workflow_plan_revision,
                caller_deadline_at=pending.caller_deadline_at,
                snapshot=pending.snapshot,
            )
            if outcome.response is None:
                self._set_terminal_error(
                    workflow_run_id,
                    f"fallback episode {outcome.episode_id} ended with status {outcome.status.value}",
                )
            return outcome.response
        if result.status != ExecutionStatus.SUCCESS:
            self._set_terminal_error(
                workflow_run_id,
                f"episode {pending.route.episode_id} completed with status {result.status.value}",
            )
        return None

    def _collect_tool_results(
        self,
        pending: _PendingEpisode,
        message: Mapping[str, object],
        outputs: Sequence[Mapping[str, object]],
    ) -> list[ToolProxyResult]:
        actions = _actions_from_message(message)
        observed: list[ToolProxyResult] = []
        seen_call_ids: set[str] = set()
        for index, action in enumerate(actions):
            call_id = action.tool_call_id
            seen_call_ids.add(call_id)
            item = pending.tool_results.get(call_id)
            if item is None:
                output = outputs[index] if index < len(outputs) else {}
                item = _proxy_result_from_output(action, output, pending)
            observed.append(item)
        for call in pending.response.tool_calls:
            if call.tool_call_id not in seen_call_ids:
                observed.append(
                    _missing_action_result(
                        ToolAction(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            arguments=call.arguments,
                        ),
                        pending,
                    )
                )
        # Preserve proxy observations that the framework omitted from its
        # assistant message.  Dropping an extra execution could otherwise let
        # M3 commit a planned write without accounting for the unexpected one.
        consumed_call_ids = {item.record.tool_call_id for item in observed}
        observed.extend(
            item
            for call_id, item in pending.tool_results.items()
            if call_id not in consumed_call_ids
        )
        return observed

    def _should_fallback(self, route: RoutePlan, result: ExecutionResult) -> bool:
        """Only SLM verified failures create an independent LLM fallback span."""

        return (
            route.execution_mode == ExecutionMode.GUARDED_SLM
            and route.fallback_plan is not None
            and result.status
            in (
                ExecutionStatus.MODEL_FAILED,
                ExecutionStatus.TIMEOUT,
                ExecutionStatus.VERIFICATION_FAILED,
            )
            and not result.state_outcome.recovery_required
        )

    def _start_fallback_episode(
        self,
        *,
        source_request: ModelRequest,
        source_workflow: WorkflowContext,
        source_route: RoutePlan,
        source_result: ExecutionResult,
        workflow_run_id: str,
        workflow_plan_ref: VersionedRef | None,
        workflow_plan_revision: int | None,
        caller_deadline_at: AwareDatetime,
        snapshot: FeedbackSnapshot | None,
    ) -> EpisodeOutcome:
        """Create an independent LLM fallback child span after SLM verification failure.

        IMPORTANT: Fallback episodes are ALWAYS forced to use:
        - Large model (LLM)
        - DIRECT execution mode (no guards, no further fallback)
        - AttemptRole.LARGE_MODEL_FALLBACK

        This prevents infinite fallback recursion: a fallback episode cannot itself
        trigger another fallback, as DIRECT mode is not subject to verification failure.
        """
        fallback = source_route.fallback_plan
        if fallback is None:
            raise CoordinatorError("attempted fallback without a fallback plan")

        # Assertion: fallback target must be large model to prevent recursion
        assert fallback.target.kind == ModelTargetKind.LARGE_MODEL, (
            "Fallback target must be LARGE_MODEL to prevent infinite recursion"
        )
        # The fallback receives the unchanged Agent history. Route and
        # execution facts remain in the trace instead of becoming a synthetic
        # model-visible handoff message.
        child_request = source_request
        self._counters.bump_map("fallback_by_reason", "slm_fallback")
        child_workflow = source_workflow.model_copy(update={"recovery_mode": "normal"})
        session = self._integration_hub.session(workflow_run_id)
        internal = getattr(session, "begin_internal_lm_invocation", None)
        if callable(internal):
            internal(child_request, child_workflow, reason="slm_fallback")
        return self.orchestrate_episode(
            child_request,
            child_workflow,
            workflow_run_id=workflow_run_id,
            workflow_plan_ref=workflow_plan_ref,
            workflow_plan_revision=workflow_plan_revision,
            caller_deadline_at=caller_deadline_at,
            snapshot=snapshot,
            _forced_target=fallback.target,
            _forced_reason="slm_fallback",
            _attempt_role_override=AttemptRole.LARGE_MODEL_FALLBACK,
            _span_kind="fallback",
            _parent_episode_id=source_route.episode_id,
            _span_relation="fallback_of",
        )

    def _take_loop_directive(self, workflow_run_id: str) -> _LoopDirective | None:
        """Consume a loop lock for the next ordinary episode.

        Command loops keep the historical one-shot semantics: the state is
        cleared before the large-model episode so one fresh guarded episode can
        start a new observation window.  Reasoning fixed-point loops stay armed
        until ``_record_loop_outcome`` observes a state-changing episode; a
        single large-model read-only turn is not sufficient to prove the stall
        is gone.
        """

        state = self._workflow_loop_state.get(workflow_run_id)
        if state is None or not state.force_direct:
            return None
        if state.loop_kind == "reasoning":
            return _LoopDirective(
                repeat_count=max(2, state.reasoning_repeat_count),
                action_summary=state.reasoning_summary or "(reasoning details unavailable)",
                loop_kind="reasoning",
            )
        # Command loop: remove the state before executing the large-model
        # episode. A direct episode that emits no tools must still clear the
        # old loop completely; a later guarded episode may start fresh.
        self._workflow_loop_state.pop(workflow_run_id, None)
        return _LoopDirective(
            repeat_count=max(2, state.repeat_count),
            action_summary=state.action_summary or "(action details unavailable)",
            loop_kind="command",
        )

    def _record_loop_outcome(
        self,
        workflow_run_id: str,
        route: RoutePlan,
        *,
        response: ModelResponse | None,
        result: ExecutionResult,
    ) -> None:
        """Update loop health from one completed episode's actual tool facts.

        This is intentionally generic: the same canonical tool-call sequence
        works for bash and any future registered tool. A write, unknown effect,
        failed execution, or non-tool response resets the observation window,
        because it proves the workspace or control flow changed.

        In addition to exact action repeats, a repeated reasoning fingerprint
        (normalized whitespace) with shifting read-only actions arms a
        persistent ``reasoning`` lock.  This targets the observed qwen
        fixed-point loop: CoT frozen while commands micro-vary and zero writes
        occur.
        """

        state = self._workflow_loop_state.get(workflow_run_id)
        if route.target.kind != ModelTargetKind.SMALL_MODEL:
            # A non-small episode only observes/clears an active lock; it never
            # starts a new observation window.
            if state is not None and state.force_direct and state.loop_kind == "reasoning":
                if self._state_changing_episode(response, result):
                    self._workflow_loop_state.pop(workflow_run_id, None)
                # else: keep the reasoning lock armed until a write/unknown/
                # failed/non-tool episode proves the stall is broken.
            else:
                self._workflow_loop_state.pop(workflow_run_id, None)
            return

        if (
            response is None
            or not response.tool_calls
            or result.status != ExecutionStatus.SUCCESS
            or result.state_outcome.effect not in (ActualEffect.NONE, ActualEffect.READ_ONLY)
            or result.state_outcome.recovery_required
            or result.state_outcome.fault_class is not None
        ):
            self._workflow_loop_state.pop(workflow_run_id, None)
            return

        calls = response.tool_calls
        by_call_id = {item.tool_call_id: item for item in result.tool_executions}
        if set(by_call_id) != {call.tool_call_id for call in calls} or any(
            item.status != ToolExecutionStatus.SUCCESS
            or item.effect not in (ActualEffect.NONE, ActualEffect.READ_ONLY)
            or item.fault_class is not None
            for item in by_call_id.values()
        ):
            self._workflow_loop_state.pop(workflow_run_id, None)
            return

        action_fingerprint, action_summary = self._loop_action_identity(calls)
        reasoning_fingerprint, reasoning_summary = self._loop_reasoning_identity(
            response.reasoning_content
        )
        state = self._workflow_loop_state.setdefault(workflow_run_id, _WorkflowLoopState())

        if state.last_action_fingerprint == action_fingerprint:
            state.repeat_count += 1
        else:
            state.last_action_fingerprint = action_fingerprint
            state.repeat_count = 1
        state.action_summary = action_summary

        if reasoning_fingerprint is None:
            state.last_reasoning_fingerprint = None
            state.reasoning_repeat_count = 0
            state.reasoning_summary = None
        else:
            if state.last_reasoning_fingerprint == reasoning_fingerprint:
                state.reasoning_repeat_count += 1
            else:
                state.last_reasoning_fingerprint = reasoning_fingerprint
                state.reasoning_repeat_count = 1
            state.reasoning_summary = reasoning_summary

        if state.repeat_count >= 2:
            state.force_direct = True
            state.loop_kind = "command"
        elif state.reasoning_repeat_count >= self._reasoning_loop_repeat_limit:
            state.force_direct = True
            state.loop_kind = "reasoning"

    @staticmethod
    def _loop_action_identity(
        calls: Sequence[ModelResponseToolCall],
    ) -> tuple[str, str]:
        """Return a call-ID-independent, deterministic action identity."""

        actions = [
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
            for call in calls
        ]
        material = json.dumps(
            actions,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        summary = material[:768]
        if len(material) > len(summary):
            summary = summary.rstrip() + "..."
        return sha256(material.encode("utf-8")).hexdigest(), summary

    @staticmethod
    def _loop_reasoning_identity(
        reasoning_content: str | None,
    ) -> tuple[str | None, str | None]:
        """Return a deterministic identity for provider-native reasoning.

        Only exact whitespace-normalized content is considered a fixed point.
        Empty/non-thinking responses do not participate in reasoning-loop
        detection and reset the reasoning observation window.
        """

        if not reasoning_content or not reasoning_content.strip():
            return None, None
        normalized = " ".join(reasoning_content.split())
        summary = normalized[:768]
        if len(normalized) > len(summary):
            summary = summary.rstrip() + "..."
        return sha256(normalized.encode("utf-8")).hexdigest(), summary

    @staticmethod
    def _state_changing_episode(
        response: ModelResponse | None,
        result: ExecutionResult,
    ) -> bool:
        """Whether an episode is strong enough to clear a reasoning lock."""

        if response is None or not response.tool_calls:
            return True
        if result.status != ExecutionStatus.SUCCESS:
            return True
        if (
            result.state_outcome.effect in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
            or result.state_outcome.recovery_required
            or result.state_outcome.fault_class is not None
        ):
            return True
        return any(
            item.effect in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
            or item.status != ToolExecutionStatus.SUCCESS
            or item.fault_class is not None
            for item in result.tool_executions
        )

    def _admission_facts(
        self,
        workflow: WorkflowContext,
        caller_deadline_at: AwareDatetime,
        workflow_run_id: str,
    ) -> AdmissionFacts:
        """组装路由前可得的预算事实（准入 A4/A5/探针预算的输入）。

        事实不可得时对应字段为 None；guarded 准入将其视为未知并
        fail-closed 到 large/direct，不把缺失事实默认为安全。
        """
        step_index = workflow.node_context.step_index
        steps_remaining = (
            self._step_limit - step_index if self._step_limit is not None else None
        )
        deadline_remaining_ms = int(
            (caller_deadline_at - self._clock()).total_seconds() * 1000
        )
        # 探针预算（A7）：每 workflow 上限与全实验上限取更紧者；消耗由
        # 持久化台账累计（跨进程重启有效），不再用 Coordinator 内存字典。
        # 台账读取异常一律 fail-closed：剩余置 0，路由器不探索 → direct。
        probe_remaining = None
        limits: list[int] = []
        try:
            if self._probe_budget_per_workflow is not None:
                limits.append(
                    self._probe_budget_per_workflow
                    - self._probe_ledger.workflow_used(self._experiment_id, workflow_run_id)
                )
            if self._probe_budget_total is not None:
                limits.append(
                    self._probe_budget_total - self._probe_ledger.total_used(self._experiment_id)
                )
        except Exception:
            limits = [0]
        if limits:
            probe_remaining = max(0, min(limits))
        return AdmissionFacts(
            deadline_remaining_ms=deadline_remaining_ms,
            steps_remaining=steps_remaining,
            probe_budget_remaining=probe_remaining,
            sandbox_backend=self._sandbox_backend,
            supports_transactional_write=self._supports_transactional_write,
            registered_tools=self._registered_tools,
        )

    def _direct_route(
        self,
        profile: RuntimeProfile,
        snapshot: FeedbackSnapshot | None,
        *,
        target: ModelTarget,
        reason: str,
    ) -> RoutePlan:
        feedback_version = (
            snapshot.safety_calibration_ref.version if snapshot is not None else "feedback-none"
        )
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
            route_plan_id=(
                "route-v2-"
                + sha256(
                    f"{profile.workflow_run_id}|{profile.logical_invocation_id}|{reason}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:40]
            ),
            target=target,
            execution_mode=ExecutionMode.DIRECT,
            verifier_plan=VerifierPlan(
                verifier_plan_id="verifier-none",
                version="v1",
                overall_timeout_ms=1_000,
                empty_plan_reason="large_model_direct_no_guard",
            ),
            timeout_ms=max(1, int(self._policy.budget.max_latency_ms)),
            experience_key=profile.descriptor_key,
            decision_meta={
                "policy_version": self._policy.policy_version,
                "feedback_version": feedback_version,
                "reason_code": reason,
                "is_exploration": False,
                "selection_probability": 1.0,
            },
        )

    def _large_target_for(self, route: RoutePlan) -> ModelTarget:
        fallback = route.fallback_plan
        if fallback is not None and fallback.target.kind == ModelTargetKind.LARGE_MODEL:
            return fallback.target
        return self._large_target()

    def _large_target(self) -> ModelTarget:
        """Build the configured large-model target for Coordinator locks."""

        if self._large_model is None:
            raise CoordinatorError(
                "cannot fail closed to large/direct without a configured large model"
            )
        return ModelTarget(
            kind=ModelTargetKind.LARGE_MODEL,
            model_ref=self._large_model,
        )

    def _complete_recovery_if_needed(
        self,
        workflow_run_id: str,
        directive: RecoveryDirective | None,
        tool_results: Sequence[ToolExecutionResult],
    ) -> RecoveryCheck | None:
        if directive is None:
            return None
        check = self._recovery.complete_attempt(
            workflow_run_id,
            tool_results,
            now=self._clock(),
        )
        if check.passed or check.reason in {
            "recovery_deadline_exhausted",
            "recovery_attempt_budget_exhausted",
        }:
            self._finalize_recovery(check.checked_transaction_ids)
        return check

    def _finalize_recovery(self, transaction_ids: Sequence[str]) -> None:
        finalize = getattr(self._executor, "finalize_recovery", None)
        if callable(finalize):
            finalize(transaction_ids)

    def _terminate_exhausted_recovery(self, workflow_run_id: str) -> None:
        """End a recovery loop after its last recorded attempt, never silently spin."""

        directive = self._recovery.directive(workflow_run_id)
        if directive is None:
            return
        reason = self._recovery.exhausted_reason(workflow_run_id, now=self._clock())
        if reason is None:
            return
        transaction_ids = self._recovery.transaction_ids(workflow_run_id)
        self._recovery.clear(workflow_run_id)
        self._finalize_recovery(transaction_ids)
        self._set_terminal_error(
            workflow_run_id,
            (
                f"workflow recovery terminated: {reason} "
                f"(source episode {directive.source_episode_id})"
            ),
        )

    def _accepted_response(self, result: ExecutionResult) -> ModelResponse | None:
        if result.status != ExecutionStatus.SUCCESS or result.accepted_output_ref is None:
            return None
        return self._store.resolve_accepted_output(
            result.accepted_output_ref,
            Component.AGENT_ADAPTER,
        )

    def _publish_pending(
        self,
        pending: _PendingEpisode,
        result: ExecutionResult,
        accepted_response: ModelResponse | None,
        *,
        recovery_directive: RecoveryDirective | None = None,
        recovery_check: RecoveryCheck | None = None,
    ) -> None:
        record = self._build_span(
            pending.request,
            pending.profile,
            pending.route,
            result,
            span_kind=pending.span_kind,
            parent_episode_id=pending.parent_episode_id,
            span_relation=pending.span_relation,
            recovery_directive=(
                recovery_directive
                if recovery_directive is not None
                else pending.recovery_directive
            ),
            recovery_check=recovery_check,
        )
        self._events.append_span_record(record)
        self._record_loop_outcome(
            pending.route.workflow_run_id,
            pending.route,
            response=pending.response,
            result=result,
        )
        self._observe_span(record, accepted_response)

    def _set_terminal_error(self, workflow_run_id: str, error: str) -> None:
        session = self._integration_hub.session(workflow_run_id)
        if session is not None:
            session.set_terminal_error(error)

    def flush_pending(self, workflow_run_id: str) -> None:
        """在 Agent 异常/步数耗尽时关闭未完成 span，保证 trace 不丢节点。"""

        pending = self._pending.pop(workflow_run_id, None)
        if pending is None:
            return
        execute_tool_phase = getattr(self._executor, "execute_tool_phase", None)
        if (
            pending.route.execution_mode == ExecutionMode.GUARDED_SLM
            and callable(execute_tool_phase)
        ):
            # Treat an Agent that exits before returning observations as a
            # missing-tool failure.  This closes the real checkpoint through
            # the same post-check/rollback path, including workflow_lock when
            # restoration cannot be proven.
            observed = self._collect_tool_results(pending, {}, ())
            result = execute_tool_phase(
                pending.execution_request,
                pending.execution_result,
                pending.response,
                tuple(observed),
            )
        elif pending.tool_results and callable(execute_tool_phase):
            result = execute_tool_phase(
                pending.execution_request,
                pending.execution_result,
                pending.response,
                tuple(pending.tool_results.values()),
            )
        else:
            abort = getattr(self._executor, "abort_tool_phase", None)
            if callable(abort):
                abort(pending.execution_request)
            result = pending.execution_result.model_copy(
                update={
                    "status": ExecutionStatus.TIMEOUT,
                    "accepted_model_attempt_id": None,
                    "accepted_output_ref": None,
                }
            )
        if (
            (
                result.status == ExecutionStatus.ROLLBACK_FAILED
                or result.state_outcome.recovery_required
            )
            and pending.route.execution_mode == ExecutionMode.GUARDED_SLM
        ):
            try:
                directive = self._recovery.activate(
                    workflow_run_id,
                    route=pending.route,
                    result=result,
                    caller_deadline_at=pending.caller_deadline_at,
                )
            except CoordinatorError:
                directive = None
            self._publish_pending(
                pending,
                result,
                None,
                recovery_directive=directive,
            )
            if directive is None:
                self._set_terminal_error(
                    workflow_run_id,
                    f"episode {pending.route.episode_id} rollback failed without recovery target",
                )
            return

        accepted_response = self._accepted_response(result)
        recovery_check = self._complete_recovery_if_needed(
            workflow_run_id,
            pending.recovery_directive,
            result.tool_executions,
        )
        self._publish_pending(
            pending,
            result,
            accepted_response,
            recovery_check=recovery_check,
        )
        self._terminate_exhausted_recovery(workflow_run_id)
        if result.status != ExecutionStatus.SUCCESS:
            self._set_terminal_error(
                workflow_run_id,
                f"episode {pending.route.episode_id} closed with status {result.status.value}",
            )

    def _observe_span(
        self,
        record: SpanRecord,
        accepted_response: ModelResponse | None,
    ) -> None:
        """episode 终态后回调 span_observer（如 TaskTraceRecorder），失败也回调。"""
        self._integration_hub.record_span(record, accepted_response)
        if self._span_observer is not None:
            self._span_observer(record, accepted_response)

    def _build_span(
        self,
        request: ModelRequest,
        profile: RuntimeProfile,
        route: RoutePlan,
        result: ExecutionResult,
        *,
        span_kind: Literal["normal", "fallback", "rollback_recovery"] = "normal",
        parent_episode_id: str | None = None,
        span_relation: Literal["fallback_of", "recovery_of"] | None = None,
        recovery_directive: RecoveryDirective | None = None,
        recovery_check: RecoveryCheck | None = None,
    ) -> SpanRecord:
        """从 request/profile/route/result 构造完整 SpanRecord。"""
        summary = ExecutionSummary(
            route_plan_id=result.route_plan_id,
            workflow_run_id=result.workflow_run_id,
            workflow_plan_ref=result.workflow_plan_ref,
            workflow_plan_revision=result.workflow_plan_revision,
            episode_id=result.episode_id,
            logical_invocation_id=result.logical_invocation_id,
            status=result.status,
            model_attempts=result.model_attempts,
            guard_results=result.guard_results,
            tool_executions=result.tool_executions,
            state_outcome=result.state_outcome,
            accepted_model_attempt_id=result.accepted_model_attempt_id,
            latency_ms=result.latency_ms,
        )
        return SpanRecord(
            schema_version="1.0",
            # EventStore is shared across workflows; include workflow lineage
            # so identical episode counters cannot be treated as duplicates.
            message_id=f"message-span-{result.workflow_run_id}-{result.episode_id}",
            producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
            produced_at=self._clock(),
            runtime_projection=RuntimeProjection(
                descriptor_key=profile.descriptor_key,
                route_family_key=route.route_family_key or route.experience_key,
                uncertain_fields=profile.uncertain_fields,
            ),
            route_plan=route,
            execution_summary=summary,
            runtime_profile=profile,
            model_request=request,
            span_kind=span_kind,
            parent_episode_id=parent_episode_id,
            span_relation=span_relation,
            recovery_directive=recovery_directive,
            recovery_check=recovery_check,
        )


__all__ = [
    "Coordinator",
    "CoordinatorError",
    "EpisodeOutcome",
]
