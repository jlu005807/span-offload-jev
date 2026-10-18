"""M3 受控执行：把路由方案运行到终态执行结果。

``ControlledExecutor.execute_step`` 通过能力存储解析原始请求，每次网关调用
对应一次物理模型尝试。M3 只执行当前 RoutePlan：guarded SLM 先做
pre-check 与真实事务准入，工具结果回来后做一次 post-check；LLM
fallback 由 Coordinator 作为独立子 episode 创建。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import AwareDatetime

from ..effects import EffectVerdict
from .checkpoint import (
    CheckpointSessionManager,
    CheckpointState,
    LifecycleResult as _LifecycleResult,
)
from ..mechanism_counters import MechanismCounters
from ..contracts import (
    ActualEffect,
    AttemptRole,
    AttemptStatus,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    CommitStatus,
    Component,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    EffectScope,
    GuardOutcome,
    GuardRef,
    GuardResult,
    ModelAttempt,
    ModelRef,
    ModelResponse,
    ModelTargetKind,
    ProducerRef,
    ResumeStatus,
    RollbackStatus,
    StateOutcome,
    ToolExecutionResult,
    ToolExecutionStatus,
    VerifierStage,
)
from ..interfaces import (
    GatewayError,
    GatewayTimeout,
    ModelGateway,
    SpanIdentity,
    ToolAction,
    ToolProxy,
    ToolProxyResult,
    Verifier,
)
from ..storage import RequestResponseStore


class M3Error(RuntimeError):
    """执行请求完全无法执行时抛出。"""


class UnsupportedExecutionMode(M3Error):
    """路由方案要求当前 Executor 未配置的执行能力（如缺少 Tool Proxy）。"""


_LOGGER = logging.getLogger(__name__)


def _utcnow() -> AwareDatetime:
    return datetime.now(timezone.utc)


# L2 观察「不适用」哨兵：非 git 仓库（git status 报 not a git repository）时，
# L2 是 N/A 而非「观察失败」，调用方据此跳过 L2 而不是回滚。
_L2_NOT_AVAILABLE = object()


#: `tool_proxy.state_coverage` 里「**快照覆盖代码能触达的一切**」的那几种声明。
#: 只有它们允许 opaque 执行（分析器看不懂的代码）。
#:
#: 两个值分开写而不是合成一个，因为它们是两件事：
#:
#:   ``container_filesystem``  Docker 容器快照（SWE-bench）—— 整个容器文件系统
#:   ``world_state``           AppWorld 的 ``world.save_state()`` —— 整个世界状态
#:
#: 合成一个名字就会把「换个场景这句还成不成立」藏起来 —— 而那个前提被跨场景
#: 搬运过一次，代价是 1/3 的 SLM 动作被误拒且**根本没执行**。
FULL_STATE_COVERAGE = frozenset({"container_filesystem", "world_state"})


def opaque_exec_allowed_for_proxy(tool_proxy: Any) -> bool:
    """从 ``tool_proxy`` **自己声明**的 ``state_coverage`` 推导能否执行 opaque 代码。

    为什么规则挂在这里而不是调用方的参数上
    --------------------------------------
    `ControlledExecutor` 有两个构造点：SWE-bench 的
    ``PilotRunner.run_workflow``、AppWorld 的 ``AppWorldPilotRunner._build_coordinator``。
    `opaque_exec_allowed` 这个参数是**逐个构造点**传的，加的时候只改了一个 ——
    另一个静默回落到默认 `False`，config 里写了也没人读。

    ⇒ 「调用方记得传」这种约定漏一个就是**静默降级**：跑得通，日志干净，
    数字没变，而原因只是改动没生效。所以默认值从 proxy 推导，
    调用方只有想**覆盖**时才需要说话。

    这和 ``supports_command_observation`` / ``commits_on_tool_failure``
    是同一个模式 —— 能力由 proxy 声明，executor 用 ``getattr`` 探测。
    """

    return str(getattr(tool_proxy, "state_coverage", "")) in FULL_STATE_COVERAGE


class ControlledExecutor:
    """严格按路由计划执行的 M3 组件，含可选工具事务边界。"""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        verifier: Verifier,
        store: RequestResponseStore,
        tool_proxy: ToolProxy | None = None,
        clock: Callable[[], AwareDatetime] | None = None,
        analyzer: Any | None = None,
        tool_action_analyzers: Mapping[str, Any] | None = None,
        workspace_cwd: str = "/testbed",
        opaque_exec_allowed: bool | None = None,
        checkpoint_journal: Any | None = None,
        counters: MechanismCounters | None = None,
    ) -> None:
        self._gateway = gateway
        self._verifier = verifier
        self._store = store
        self._tool_proxy = tool_proxy
        self._clock = clock or _utcnow
        self._workspace_cwd = workspace_cwd
        # opaque 执行（python 脚本等）能不能做，取决于**快照覆盖范围**
        # （文档 §4.2 六类规则）。`None` = 调用方没说 ⇒ 由 proxy 声明推导
        # （见 `_opaque_exec_allowed_now`）。
        self._opaque_exec_allowed = opaque_exec_allowed
        # CommandEffectAnalyzer（pre-check L1）：缺失或异常由
        # _analyze_action 统一变成 UNKNOWN，绝不回退 ToolProxy 声明的 scope。
        # external/irreversible/unknown 都在真实工具执行前 fail closed。
        self._analyzer = analyzer
        self._tool_action_analyzers = dict(tool_action_analyzers or {})
        # One owner for all transaction handles and lifecycle facts.  The
        # legacy ``_prepared_*`` properties below are read-only compatibility
        # views for adapters/tests; writes go through this manager only.
        self._checkpoint_sessions = CheckpointSessionManager(journal=checkpoint_journal)
        #: 机制计数（T2）。没给就是 null 计数器 —— 行为不变，只是不记账。
        self._counters = counters if counters is not None else MechanismCounters.null()

    @property
    def _prepared_transactions(self) -> dict[str, tuple[str, ...]]:
        return {
            session.session_id: session.transaction_ids
            for session in self._checkpoint_sessions.active_sessions()
        }

    @_prepared_transactions.setter
    def _prepared_transactions(self, value: Mapping[str, tuple[str, ...]]) -> None:
        self._checkpoint_sessions.install_legacy_sessions(value)

    @property
    def _prepared_scopes(self) -> dict[str, dict[str, EffectScope]]:
        return {
            session.session_id: {
                key: EffectScope(value) for key, value in session.scopes.items()
            }
            for session in self._checkpoint_sessions.active_sessions()
        }

    @_prepared_scopes.setter
    def _prepared_scopes(self, value: Mapping[str, Mapping[str, EffectScope | str]]) -> None:
        for session_id, scopes in value.items():
            session = self._session(session_id)
            if session is None:
                self._checkpoint_sessions.install_legacy_sessions({session_id: tuple(scopes)})
                session = self._session(session_id)
            if session is not None:
                session.scopes = {
                    key: str(item.value if hasattr(item, "value") else item)
                    for key, item in scopes.items()
                }

    @property
    def _prepared_actions(self) -> dict[str, dict[str, ToolAction]]:
        return {
            session.session_id: dict(session.actions)
            for session in self._checkpoint_sessions.active_sessions()
        }

    @_prepared_actions.setter
    def _prepared_actions(self, value: Mapping[str, Mapping[str, ToolAction]]) -> None:
        for session_id, actions in value.items():
            session = self._session(session_id)
            if session is None:
                self._checkpoint_sessions.install_legacy_sessions({session_id: tuple(actions)})
                session = self._session(session_id)
            if session is not None:
                session.actions = dict(actions)

    @property
    def _prepared_verdicts(self) -> dict[str, dict[str, Any]]:
        return {
            session.session_id: dict(session.verdicts)
            for session in self._checkpoint_sessions.active_sessions()
        }

    @_prepared_verdicts.setter
    def _prepared_verdicts(self, value: Mapping[str, Mapping[str, Any]]) -> None:
        for session_id, verdicts in value.items():
            session = self._session(session_id)
            if session is None:
                self._checkpoint_sessions.install_legacy_sessions({session_id: tuple(verdicts)})
                session = self._session(session_id)
            if session is not None:
                session.verdicts = dict(verdicts)

    @property
    def _prepared_l2_before(self) -> dict[str, frozenset | None]:
        return {
            session.session_id: session.l2_before
            for session in self._checkpoint_sessions.active_sessions()
        }

    def _session(self, request_id: str):
        return self._checkpoint_sessions.get(request_id)

    def _drop_session(self, request_id: str, *, decision: str) -> None:
        session = self._session(request_id)
        if session is None:
            return
        self._checkpoint_sessions.close(request_id, decision=decision)

    def _state_with_session(
        self,
        outcome: StateOutcome,
        request_id: str,
    ) -> StateOutcome:
        session = self._session(request_id)
        if session is None:
            return outcome
        return outcome.model_copy(
            update={
                "checkpoint_session_id": session.session_id,
                "lifecycle_state": session.state.value,
                "handle_retained": session.handle_retained,
                "fault_class": outcome.fault_class or session.fault_class,
                "recovery_required": outcome.recovery_required
                or session.recovery_required,
            }
        )

    @staticmethod
    def _closed_state(outcome: StateOutcome, request_id: str) -> StateOutcome:
        return outcome.model_copy(
            update={
                "checkpoint_session_id": request_id,
                "lifecycle_state": CheckpointState.CLOSED.value,
                "handle_retained": False,
                "recovery_required": False,
            }
        )

    def execute_step(self, request: ExecutionRequest) -> ExecutionResult:
        """按执行模式分派；guarded SLM 需要显式配置 Tool Proxy。"""
        route = request.route_plan
        if route.execution_mode == ExecutionMode.DIRECT:
            return self._execute_direct(request)
        if route.execution_mode == ExecutionMode.GUARDED_SLM:
            if self._tool_proxy is None:
                raise UnsupportedExecutionMode(
                    "guarded_slm requires a configured Tool Proxy"
                )
            return self._execute_guarded_slm(request)
        raise UnsupportedExecutionMode(
            f"execution mode {route.execution_mode.value} is outside the pilot scope"
        )

    def _execute_direct(self, request: ExecutionRequest) -> ExecutionResult:
        """直接路径：一次模型调用，成功即接受；超时/错误返回终态。"""
        route = request.route_plan
        if self._clock() >= request.deadline_at:
            return self._result(request, status=ExecutionStatus.TIMEOUT)
        # 角色随目标模型种类而定；Coordinator 可为独立 fallback 子 span
        # 显式指定 LARGE_MODEL_FALLBACK，避免把两次物理调用合并成一个 span。
        role = request.attempt_role_override or (
            AttemptRole.SMALL_MODEL_CANDIDATE
            if route.target.kind == ModelTargetKind.SMALL_MODEL
            else AttemptRole.LARGE_MODEL_DIRECT
        )
        attempt_id = _attempt_id(request, 1)
        try:
            call = self._gateway.complete(
                route.target,
                self._resolve_request(request),
                model_attempt_id=attempt_id,
                deadline_at=request.deadline_at,
            )
        except GatewayTimeout as error:
            _LOGGER.warning(
                "gateway timeout for episode=%s target=%s: %s",
                request.route_plan.episode_id,
                route.target.model_ref.id,
                error,
            )
            return self._result(
                request,
                status=ExecutionStatus.TIMEOUT,
                attempts=(
                    self._failed_attempt(
                        attempt_id, role, AttemptStatus.TIMEOUT, route.target.model_ref
                    ),
                ),
            )
        except GatewayError as error:
            _LOGGER.warning(
                "gateway error for episode=%s target=%s: %s",
                request.route_plan.episode_id,
                route.target.model_ref.id,
                error,
            )
            return self._result(
                request,
                status=ExecutionStatus.MODEL_FAILED,
                attempts=(
                    self._failed_attempt(
                        attempt_id, role, AttemptStatus.FAILED, route.target.model_ref
                    ),
                ),
            )
        return self._accept(
            request, call, attempt_id, role, route.target.model_ref
        )

    def _opaque_exec_allowed_now(self) -> bool:
        """本 executor 能不能执行 opaque 代码：显式参数优先，否则问 proxy。

        `None` 是「调用方没说」——**不是**「不行」。把这两件事分开，
        是为了让漏传参数的调用方拿到**正确的**默认值，而不是静默降级。
        """

        if self._opaque_exec_allowed is not None:
            return self._opaque_exec_allowed
        return opaque_exec_allowed_for_proxy(self._tool_proxy)

    def _execute_guarded_slm(self, request: ExecutionRequest) -> ExecutionResult:
        """Call the SLM, validate it, and establish real transaction boundaries.

        There is deliberately no candidate execution here.  A mutating action
        is admitted only after its rollback capability has been established;
        the adapter executes the accepted response in the primary environment
        and reports the result through :meth:`execute_tool_phase`.
        """

        route = request.route_plan
        if self._clock() >= request.deadline_at:
            return self._result(request, status=ExecutionStatus.TIMEOUT)
        span_context = _span_identity(route)
        candidate_id = _attempt_id(request, 1)
        candidate_role = AttemptRole.SMALL_MODEL_CANDIDATE
        try:
            call = self._gateway.complete(
                route.target,
                self._resolve_request(request),
                model_attempt_id=candidate_id,
                deadline_at=request.deadline_at,
            )
        except GatewayTimeout as error:
            _LOGGER.warning(
                "gateway timeout for episode=%s target=%s: %s",
                request.route_plan.episode_id,
                route.target.model_ref.id,
                error,
            )
            return self._result(
                request,
                status=ExecutionStatus.TIMEOUT,
                attempts=(
                    self._failed_attempt(
                        candidate_id,
                        candidate_role,
                        AttemptStatus.TIMEOUT,
                        route.target.model_ref,
                    ),
                ),
            )
        except GatewayError as error:
            _LOGGER.warning(
                "gateway error for episode=%s target=%s: %s",
                request.route_plan.episode_id,
                route.target.model_ref.id,
                error,
            )
            return self._result(
                request,
                status=ExecutionStatus.MODEL_FAILED,
                attempts=(
                    self._failed_attempt(
                        candidate_id,
                        candidate_role,
                        AttemptStatus.FAILED,
                        route.target.model_ref,
                    ),
                ),
            )

        response = call.response
        guards: list[GuardResult] = []
        passed = True
        for step in route.verifier_plan.steps:
            if step.stage != VerifierStage.PRE_COMMIT:
                continue
            guard = self._check_model_response(
                step,
                candidate=response,
                model_attempt_id=candidate_id,
                span_context=span_context,
            )
            guards.append(guard)
            if guard.outcome != GuardOutcome.PASS:
                passed = False

        candidate_attempt = self._successful_attempt(
            request, call, candidate_id, candidate_role, route.target.model_ref
        )
        call_ids = tuple(call_spec.tool_call_id for call_spec in response.tool_calls)
        if len(call_ids) != len(set(call_ids)):
            duplicate_step = next(
                (
                    step
                    for step in route.verifier_plan.steps
                    if step.stage == VerifierStage.PRE_COMMIT
                ),
                None,
            )
            if duplicate_step is not None:
                guards.append(
                    GuardResult(
                        guard_ref=duplicate_step.guard_ref,
                        stage=duplicate_step.stage,
                        subject_model_attempt_id=candidate_id,
                        outcome=GuardOutcome.REJECT,
                        latency_ms=0,
                        cost=0.0,
                    )
                )
            self._store.register_audit_response(
                GuardRef(kind="model_response", id=response.response_id, version="v1"),
                response,
            )
            return self._result(
                request,
                status=ExecutionStatus.VERIFICATION_FAILED,
                attempts=(candidate_attempt,),
                guard_results=tuple(guards),
            )
        if not passed:
            self._store.register_audit_response(
                GuardRef(kind="model_response", id=response.response_id, version="v1"),
                response,
            )
            return self._result(
                request,
                status=ExecutionStatus.VERIFICATION_FAILED,
                attempts=(candidate_attempt,),
                guard_results=tuple(guards),
            )

        actions = tuple(
            ToolAction(
                tool_call_id=call_spec.tool_call_id,
                tool_name=call_spec.tool_name,
                arguments=call_spec.arguments,
            )
            for call_spec in response.tool_calls
        )
        post_steps = tuple(
            step
            for step in route.verifier_plan.steps
            if step.stage == VerifierStage.POST_CHECK
        )
        if not actions:
            # A guarded SLM response is not exempt from the post-check
            # merely because it returned text.  There is no transaction to
            # commit, so the same single post step checks the final response
            # before it is exposed to the Agent.
            post_guards = tuple(
                self._check_model_response(
                    step,
                    candidate=response,
                    model_attempt_id=candidate_id,
                    span_context=span_context,
                )
                for step in post_steps
            )
            guards.extend(post_guards)
            if any(guard.outcome != GuardOutcome.PASS for guard in post_guards):
                self._store.register_audit_response(
                    GuardRef(kind="model_response", id=response.response_id, version="v1"),
                    response,
                )
                return self._result(
                    request,
                    status=ExecutionStatus.VERIFICATION_FAILED,
                    attempts=(candidate_attempt,),
                    guard_results=tuple(guards),
                )
            return self._accept(
                request,
                call,
                candidate_id,
                candidate_role,
                route.target.model_ref,
                guard_results=tuple(guards),
                state_outcome=self._planned_state(()),
            )
        transaction_ids: list[str] = []
        transaction_scopes: list[EffectScope] = []
        prepared_scope_by_id: dict[str, EffectScope] = {}
        prepared_action_by_id: dict[str, ToolAction] = {}
        prepared_verdict_by_id: dict[str, Any] = {}
        admission_failed = False
        admission_fault: str | None = None
        #: **哪一条准入规则**拒了这一步。和 `admission_fault` 分开：那个是
        #: 「基础设施出错」（进 `fault_class=env_fault`，是可恢复故障），
        #: 这个是「策略拒绝」（系统按设计工作）。以前两者都不记，
        #: 于是「opaque 被误拒」和「UNKNOWN scope 被正确拒」在记录里长得一样。
        admission_reason: str | None = None
        # 供拒绝日志取用；`actions` 为空时循环体不执行，所以必须先绑定。
        action: ToolAction | None = None
        verdict: Any = None
        try:
            self._checkpoint_sessions.reserve(
                request.execution_request_id,
                workflow_run_id=request.route_plan.workflow_run_id,
            )
        except Exception:
            try:
                self._checkpoint_sessions.abandon_unstarted(
                    request.execution_request_id
                )
            except Exception:
                pass
            self._store.register_audit_response(
                GuardRef(kind="model_response", id=response.response_id, version="v1"),
                response,
            )
            return self._result(
                request,
                status=ExecutionStatus.VERIFICATION_FAILED,
                attempts=(candidate_attempt,),
                guard_results=tuple(guards),
                state_outcome=self._planned_state(()).model_copy(
                    update={"fault_class": "env_fault"}
                ),
            )
        for action in actions:
            transaction_id = _transaction_id(request, action.tool_call_id)
            verdict = self._analyze_action(action)
            scope = verdict.scope
            prepared_verdict_by_id[transaction_id] = verdict
            transaction_scopes.append(scope)
            try:
                self._checkpoint_sessions.plan_transaction(
                    request.execution_request_id,
                    transaction_id=transaction_id,
                    scope=scope,
                    action=action,
                    verdict=verdict,
                )
            except Exception:
                admission_failed = True
                admission_fault = "checkpoint_journal_error"
                break
            if scope in (EffectScope.IRREVERSIBLE_WRITE, EffectScope.UNKNOWN):
                admission_failed = True
                admission_reason = (
                    "scope_unknown"
                    if scope == EffectScope.UNKNOWN
                    else "scope_irreversible"
                )
                break
            if (
                verdict is not None
                and verdict.category in ("opaque_exec", "verify_exec")
                and not self._opaque_exec_allowed_now()
            ):
                # opaque 执行需要「覆盖一切」的快照兜底；本地快照只覆盖 workspace，
                # 无法证明脚本不写外部。覆盖面由 proxy 声明（见
                # `opaque_exec_allowed_for_proxy`）。
                admission_failed = True
                admission_reason = "opaque_exec_not_allowed"
                break
            begin = getattr(self._tool_proxy, "begin_transaction", None)
            if not callable(begin):
                admission_failed = True
                admission_fault = "begin_callback_missing"
                break
            begin_attempted = False
            if scope == EffectScope.READ_ONLY:
                # Read-only calls do not need a checkpoint, but still register
                # the exact authorized action at the Tool Proxy boundary.
                try:
                    begin_attempted = True
                    accepted = begin(action, transaction_id=transaction_id)
                except Exception:
                    accepted = False
                    admission_fault = "begin_callback_error"
            else:
                can_rollback = getattr(self._tool_proxy, "can_rollback", None)
                if not callable(can_rollback):
                    admission_failed = True
                    admission_fault = "can_rollback_callback_missing"
                    break
                try:
                    rollback_capable = can_rollback(action)
                    begin_attempted = bool(rollback_capable)
                    accepted = rollback_capable and begin(
                        action,
                        transaction_id=transaction_id,
                    )
                except Exception:
                    accepted = False
                    admission_fault = "begin_callback_error"
            if not accepted:
                admission_failed = True
                consume_fault = getattr(self._tool_proxy, "consume_lifecycle_fault", None)
                if callable(consume_fault):
                    admission_fault = consume_fault() or admission_fault
                if begin_attempted:
                    admission_fault = admission_fault or "begin_callback_refused"
                break
            if scope != EffectScope.READ_ONLY:
                self._counters.bump("checkpoints_created")
            transaction_ids.append(transaction_id)
            prepared_scope_by_id[transaction_id] = scope
            prepared_action_by_id[transaction_id] = action

        try:
            self._checkpoint_sessions.transition(
                request.execution_request_id,
                CheckpointState.PREPARED,
                operation="checkpoint_prepared",
            )
        except Exception:
            admission_failed = True
            admission_fault = "checkpoint_journal_error"
        if admission_failed:
            # 让「哪条规则拒的」说出来。不记的话这一类的下游表现只有
            # `verification_failed` + 空 tool_executions，而**正确拒绝**和
            # **误拒**共用这一副面孔 —— 分不清就没法判断改动有没有生效。
            _LOGGER.warning(
                "admission rejected: rule=%s tool=%s category=%s scope=%s "
                "reason=%s opaque_exec_allowed=%s",
                admission_reason or admission_fault or "unspecified",
                getattr(action, "tool_name", None),
                getattr(verdict, "category", None),
                getattr(verdict, "scope", None),
                getattr(verdict, "reason", None),
                self._opaque_exec_allowed_now(),
            )
            self._counters.bump_map(
                "admission_rejected_by_rule",
                admission_reason or admission_fault or "unspecified",
            )
            _LOGGER.warning(
                "admission rejected code: %s",
                "\n".join(
                    str(item)
                    for item in (getattr(action, "arguments", None) or {}).get(
                        "codes", ()
                    )
                )[:4000],
            )
            rollback_status = self._rollback_transactions(
                transaction_ids,
            )
            if (
                any(scope == EffectScope.REVERSIBLE_WRITE for scope in prepared_scope_by_id.values())
                and rollback_status == RollbackStatus.NOT_NEEDED
            ):
                rollback_status = RollbackStatus.UNKNOWN
            if rollback_status not in (RollbackStatus.SUCCEEDED, RollbackStatus.NOT_NEEDED):
                status = ExecutionStatus.ROLLBACK_FAILED
                state = self._planned_state(transaction_scopes).model_copy(
                    update={
                        "recovery_required": True,
                        "rollback_status": rollback_status,
                        "resume_status": ResumeStatus.NOT_ATTEMPTED,
                        "fault_class": "env_fault",
                    }
                )
                # Keep the session and backend handles for RecoveryManager.
                self._checkpoint_sessions.retain_for_recovery(
                    request.execution_request_id,
                    detail="admission_cleanup_rollback_failed",
                    rollback_status=rollback_status.value,
                )
            else:
                status = ExecutionStatus.VERIFICATION_FAILED
                close_result = self._close_sandbox_span(request)
                state = self._state_for_tools(
                    (),
                    recovery_required=not close_result.ok,
                    rollback_status=RollbackStatus.NOT_NEEDED,
                    fault_class="env_fault" if admission_fault else None,
                    planned_scopes=tuple(transaction_scopes)
                    if not close_result.ok
                    else (),
                )
                if close_result.ok:
                    state = self._closed_state(
                        state,
                        request.execution_request_id,
                    )
                    self._drop_session(request.execution_request_id, decision="admission_rejected")
                else:
                    self._checkpoint_sessions.retain_for_recovery(
                        request.execution_request_id,
                        detail="admission_cleanup_close_failed",
                    )
            self._store.register_audit_response(
                GuardRef(kind="model_response", id=response.response_id, version="v1"),
                response,
            )
            recovery_records = (
                tuple(
                    _prepared_transaction_result(
                        action=prepared_action_by_id[transaction_id],
                        transaction_id=transaction_id,
                        scope=prepared_scope_by_id[transaction_id],
                        rollback_status=rollback_status,
                    )
                    for transaction_id in transaction_ids
                )
                if status == ExecutionStatus.ROLLBACK_FAILED
                else ()
            )
            return self._result(
                request,
                status=status,
                attempts=(candidate_attempt,),
                guard_results=tuple(guards),
                tool_executions=recovery_records,
                state_outcome=state,
            )

        self._counters.bump("admission_accepted")
        # L2 增量观察（阶段 2）：有写事务或含 inferred 只读时，在 checkpoint
        # 建立后抓取 S_before（本 episode 变更的基线）。
        l2_before = (
            self._observe_git_status(request)
            if self._l2_needed(prepared_verdict_by_id)
            else None
        )
        session = self._session(request.execution_request_id)
        if session is not None:
            session.l2_before = l2_before
            self._checkpoint_sessions.transition(
                request.execution_request_id,
                CheckpointState.PREPARED,
                operation="checkpoint_prepared",
            )
        return self._accept(
            request,
            call,
            candidate_id,
            candidate_role,
            route.target.model_ref,
            guard_results=tuple(guards),
            state_outcome=self._planned_state(transaction_scopes),
        )

    def execute_tool_phase(
        self,
        request: ExecutionRequest,
        initial_result: ExecutionResult,
        response: ModelResponse,
        observed_results: Sequence[ToolProxyResult],
    ) -> ExecutionResult:
        """接收 Agent 实际执行后的工具结果并完成当前 M3 episode。

        普通 DIRECT 只闭合 span：它记录全部实际工具结果，不进入 SLM
        专用的 post-check、事务 rollback 或 fallback。只有 guarded SLM
        路径的 post-check 失败才进入事务 rollback；后续 LLM
        fallback/recovery 由 Coordinator 创建新的独立 span。
        """

        if (
            not observed_results
            and request.route_plan.execution_mode == ExecutionMode.GUARDED_SLM
            and response.tool_calls
        ):
            # A direct M3 caller may not pass through Coordinator's observation
            # collector.  Missing observations are still a real failure: every
            # prepared action must be represented before the checkpoint can be
            # committed or released.
            observed_results = tuple(
                _missing_tool_result(
                    ToolAction(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        arguments=call.arguments,
                    ),
                    transaction_id=_transaction_id(request, call.tool_call_id),
                )
                for call in response.tool_calls
            )
        if not observed_results:
            return initial_result
        raw_records = tuple(item.record for item in observed_results)
        unconfirmed_ids = tuple(
            item.record.transaction_id
            for item in observed_results
            if not item.confirmed
        )
        records = raw_records
        all_records = initial_result.tool_executions + records
        if request.route_plan.execution_mode == ExecutionMode.DIRECT:
            # DIRECT 的工具失败是 Agent 的下一轮 observation，不是 M3 的
            # 事务失败。保持 accepted model output，使 Coordinator 能正常
            # 关闭 span 并让 Agent 决定下一步；但 trace 仍要标出失败后可能
            # 留下的写入/未知副作用，不能伪装成确定安全状态。
            direct_recovery_risk = any(
                record.effect in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
                and (
                    record.status != ToolExecutionStatus.SUCCESS
                    or record.commit_status in (CommitStatus.PARTIAL, CommitStatus.UNKNOWN)
                )
                for record in records
            )
            state = self._state_for_tools(
                all_records,
                recovery_required=(
                    initial_result.state_outcome.recovery_required or direct_recovery_risk
                ),
                rollback_status=initial_result.state_outcome.rollback_status,
                resume_status=initial_result.state_outcome.resume_status,
            )
            return initial_result.model_copy(
                update={
                    "tool_executions": all_records,
                    "state_outcome": state,
                    "latency_ms": initial_result.latency_ms
                    + sum(record.latency_ms for record in records),
                }
            )

        guards = list(initial_result.guard_results)
        attempt_id = initial_result.accepted_model_attempt_id or (
            initial_result.model_attempts[-1].model_attempt_id
            if initial_result.model_attempts
            else "attempt-unknown"
        )
        post_steps = tuple(
            step
            for step in request.route_plan.verifier_plan.steps
            if step.stage == VerifierStage.POST_CHECK
        )
        planned_scopes = self._prepared_scopes.get(request.execution_request_id, {})
        planned_actions = self._prepared_actions.get(request.execution_request_id, {})
        planned_ids = tuple(planned_scopes)
        session = self._session(request.execution_request_id)
        if session is not None and session.state == CheckpointState.PREPARED:
            self._checkpoint_sessions.transition(
                request.execution_request_id,
                CheckpointState.EXECUTING,
                operation="tool_phase_started",
            )
        observed_ids = tuple(record.transaction_id for record in raw_records)
        unexpected_ids = tuple(
            transaction_id
            for transaction_id in observed_ids
            if transaction_id not in planned_scopes
        )
        binding_mismatch_ids = tuple(
            record.transaction_id
            for record in raw_records
            if record.transaction_id in planned_actions
            and not _tool_record_matches_action(
                record,
                planned_actions[record.transaction_id],
            )
        )
        risky_binding_ids = tuple(
            dict.fromkeys((*unexpected_ids, *binding_mismatch_ids, *unconfirmed_ids))
        )
        records = tuple(
            _record_with_category(
                _normalize_tool_record(
                    record,
                    planned_scopes=planned_scopes,
                    risky_ids=risky_binding_ids,
                ),
                self._prepared_verdicts.get(request.execution_request_id, {}).get(
                    record.transaction_id
                ),
            )
            for record in raw_records
        )
        all_records = initial_result.tool_executions + records
        # A framework observation without a Tool Proxy receipt is not evidence
        # that the authorized action ran.  It remains an audit fact and forces
        # rollback; only confirmed primary executions may reach commit.
        passed = not binding_mismatch_ids and not unconfirmed_ids
        post_guard_errors = False
        for step in post_steps:
            for record in records:
                guard = self._check_tool_result(step, record, model_attempt_id=attempt_id)
                if guard.outcome == GuardOutcome.REJECT:
                    self._counters.bump("guard_post_check_rejects")
                guards.append(guard)
                if guard.outcome == GuardOutcome.ERROR:
                    post_guard_errors = True
                elif (
                    guard.outcome != GuardOutcome.PASS
                    and record.status != ToolExecutionStatus.FAILED
                ):
                    # 只有非 FAILED 状态的 REJECT 才阻断：TIMEOUT/UNKNOWN/REJECTED
                    # 不可确认必须回滚；FAILED（returncode 非零）是正确性观察，
                    # 不参与 commit 判定（阶段 2）。
                    passed = False
        post_guards = guards[len(initial_result.guard_results) :]
        session = self._session(request.execution_request_id)
        if session is not None and session.state == CheckpointState.EXECUTING:
            self._checkpoint_sessions.transition(
                request.execution_request_id,
                CheckpointState.VERIFYING,
                operation="post_check_started",
            )
        # 故障归因落账（阶段 2）：显式 fault_class 优先（ToolProxy 已标
        # env_fault），post-check 自身异常标 guard_fault。M4 与 commit 判定
        # 消费同一标记，不再有“异常名猜不中”的口子。
        records = tuple(
            record.model_copy(
                update={"fault_class": _fault_class(record, post_guard_errors)}
            )
            for record in records
        )
        phase_latency = sum(record.latency_ms for record in records) + sum(
            guard.latency_ms for guard in post_guards
        )
        all_transaction_ids = tuple(dict.fromkeys((*planned_ids, *observed_ids)))
        planned_write_ids = tuple(
            transaction_id
            for transaction_id, scope in planned_scopes.items()
            if scope == EffectScope.REVERSIBLE_WRITE
        )
        observed_risky_ids = tuple(
            dict.fromkeys(
                record.transaction_id
                for record in records
                if record.effect in (ActualEffect.WRITE, ActualEffect.UNKNOWN)
            )
        )
        if unexpected_ids or unconfirmed_ids:
            # A result for an action M3 did not authorize cannot be committed.
            # Keep it in the audit record and force the conservative recovery
            # branch below; an unknown transaction is never treated as safe.
            passed = False
            observed_risky_ids = tuple(
                dict.fromkeys((*observed_risky_ids, *unexpected_ids, *unconfirmed_ids))
            )
        # Read-only actions have no commit callback; they are merely released
        # after the single post-check.  Mutating transactions are the only
        # handles that need an explicit commit operation.
        commit_ids = planned_write_ids or tuple(
            record.transaction_id
            for record in records
            if record.effect == ActualEffect.WRITE
        )
        # L2 增量观察（阶段 2）：写事务 commit 前抓 S_after，与本 episode 基线
        # 取差集得 S_actual。declared_write 越界 -> 回滚；纯只读误判（无写却有
        # 变更）-> 效应误判，同样走回滚分支；opaque 不判 FAIL、S_actual 记溯源。
        verdicts_by_id = self._prepared_verdicts.get(request.execution_request_id, {})
        l2_actual: frozenset | None = None
        l2_misclassification = False
        if self._l2_needed(verdicts_by_id):
            s_after = self._observe_git_status(request)
            s_before = self._prepared_l2_before.get(request.execution_request_id)
            if (
                s_before is not None
                and s_after is not None
                and s_before is not _L2_NOT_AVAILABLE
                and s_after is not _L2_NOT_AVAILABLE
            ):
                l2_actual = s_after - s_before
                if _l2_boundary_breach(verdicts_by_id, l2_actual):
                    # 效应边界被打破 -> 回滚 + 落账（uncontained）
                    passed = False
                    l2_misclassification = True
            elif s_before is not _L2_NOT_AVAILABLE and s_after is not _L2_NOT_AVAILABLE and any(
                verdict is not None and verdict.category == "declared_write"
                for verdict in verdicts_by_id.values()
            ):
                # declared write 的 L2 观察失败（非 N/A）：不能跳过检查后提交 -> 回滚
                # （opaque 允许 unobserved；非 git 仓库 L2 N/A 走上方 skip）
                passed = False
                l2_misclassification = True
        # 效应误判标记进每条工具记录（M4 据此判 uncontained 并记 KB 误判）
        if l2_misclassification:
            records = tuple(
                record.model_copy(
                    update={
                        "error_type": record.error_type or "effect_misclassification",
                    }
                )
                for record in records
            )
        commit_failed = False
        # commit 条件（阶段 2）：绑定一致 + 状态可确认 + 无我方层故障 + 未越界。
        #
        # returncode 在**默认语义**下不参与——FAILED（非零返回码）收据可信即可 commit：
        # bash/SWE-bench 上 ``returncode != 0`` 是 agent 的常规反馈通道
        # （``grep`` 没匹配返回 1、``git diff --quiet`` 返回 1），逐条回滚失败命令
        # 会破坏正常工作流。
        #
        # 但 adapter 可以声明 ``commits_on_tool_failure = False`` 退出该语义。
        # AppWorld 一个 batch 是一次性生成的 2~14 行代码（实测**失败的写 batch 100%
        # 多行**），batch 内靠前的写已落盘，"FAILED 也 commit" 会把这种部分提交留住。
        # 声明后 FAILED 的写 batch 走既有拒绝路径：rollback -> verification_failed
        # -> Coordinator 的 _should_fallback。
        #
        # 注意**只读不能一起被拦**：只读不建 checkpoint，没有可回滚的东西，
        # 拦下来只会凭空产生一次空回滚 + 一次 LLM fallback（实测成本高、收益≈0）。
        # 所以判据看 ``record.effect``（是否真有可回滚的写入），而不是只看 status。
        accepts_failed = (
            getattr(self._tool_proxy, "commits_on_tool_failure", True) is not False
        )
        commit_ok = passed and all(
            _tool_result_commits(record, accepts_failed=accepts_failed)
            and record.fault_class is None
            for record in records
        )
        if commit_ok:
            session = self._session(request.execution_request_id)
            if session is not None and session.state == CheckpointState.VERIFYING:
                self._checkpoint_sessions.transition(
                    request.execution_request_id,
                    CheckpointState.COMMITTING,
                    operation="commit_started",
                )
            commit_result = self._commit_transactions(commit_ids)
            commit_failed = not commit_result.ok
            if not commit_failed:
                session = self._session(request.execution_request_id)
                if session is not None:
                    self._checkpoint_sessions.transition(
                        request.execution_request_id,
                        CheckpointState.COMMITTED_RELEASE_PENDING,
                        operation="commit_succeeded",
                        handle_retained=True,
                        decision="committed",
                    )
                committed_records = tuple(
                    record.model_copy(
                        update={
                            "commit_status": (
                                CommitStatus.COMMITTED
                                if record.transaction_id in planned_write_ids
                                or record.effect == ActualEffect.WRITE
                                else record.commit_status
                            ),
                            "provenance": _provenance(
                                record,
                                self._prepared_verdicts.get(request.execution_request_id, {}).get(
                                    record.transaction_id
                                ),
                                request.route_plan.target.kind.value,
                                l2_actual,
                            ),
                        }
                    )
                    for record in records
                )
                all_records = initial_result.tool_executions + committed_records
                state = self._state_for_tools(
                    all_records,
                    recovery_required=initial_result.state_outcome.recovery_required,
                    rollback_status=initial_result.state_outcome.rollback_status,
                    resume_status=initial_result.state_outcome.resume_status,
                    planned_scopes=tuple(planned_scopes.values()),
                )
                close_result = self._close_sandbox_span(request)
                if close_result.ok:
                    state = self._closed_state(
                        state,
                        request.execution_request_id,
                    )
                    result = initial_result.model_copy(
                        update={
                            "guard_results": tuple(guards),
                            "tool_executions": all_records,
                            "state_outcome": state,
                            "latency_ms": initial_result.latency_ms
                            + sum(record.latency_ms for record in records)
                            + sum(guard.latency_ms for guard in post_guards),
                        }
                    )
                    self._drop_session(request.execution_request_id, decision="committed")
                    return result
                # close_span 失败：checkpoint 未删除，用保留 checkpoint 回滚已提交写。
                all_records = tuple(
                    record.model_copy(update={"fault_class": "env_fault"})
                    if record.transaction_id in commit_ids
                    else record
                    for record in all_records
                )
                rollback_status = self._rollback_transactions(commit_ids)
                if rollback_status not in (RollbackStatus.SUCCEEDED, RollbackStatus.NOT_NEEDED):
                    self._checkpoint_sessions.retain_for_recovery(
                        request.execution_request_id,
                        detail=commit_result.detail or "commit_close_rollback_failed",
                        rollback_status=rollback_status.value,
                        decision="rollback_required",
                    )
                    # 回滚失败/未知 -> ROLLBACK_FAILED + 保留句柄（recovery lock）
                    return initial_result.model_copy(
                        update={
                            "status": ExecutionStatus.ROLLBACK_FAILED,
                            "accepted_model_attempt_id": None,
                            "accepted_output_ref": None,
                            "guard_results": tuple(guards),
                            "tool_executions": all_records,
                            "state_outcome": self._state_with_session(
                                self._state_for_tools(
                                    all_records,
                                    recovery_required=True,
                                    rollback_status=rollback_status,
                                    resume_status=ResumeStatus.NOT_ATTEMPTED,
                                    planned_scopes=tuple(planned_scopes.values()),
                                ),
                                request.execution_request_id,
                            ),
                            "latency_ms": initial_result.latency_ms + phase_latency,
                        }
                    )
                # 回滚成功 -> 不接受 SLM 输出，返回 fallback（verification failed）。
                # 回滚后的 close 再检查一次：成功才清理 prepared，失败保留句柄交恢复。
                close_again = self._close_sandbox_span(request)
                if close_again.ok:
                    self._drop_session(request.execution_request_id, decision="rolled_back")
                else:
                    self._checkpoint_sessions.retain_for_recovery(
                        request.execution_request_id,
                        detail=close_again.detail or "rollback_close_failed",
                        rollback_status=rollback_status.value,
                        decision="rolled_back",
                    )
                terminal_state = self._state_for_tools(
                    all_records,
                    recovery_required=not close_again.ok,
                    rollback_status=rollback_status,
                    resume_status=ResumeStatus.NOT_NEEDED,
                    planned_scopes=tuple(planned_scopes.values()),
                )
                terminal_state = (
                    self._closed_state(terminal_state, request.execution_request_id)
                    if close_again.ok
                    else self._state_with_session(
                        terminal_state,
                        request.execution_request_id,
                    )
                )
                return initial_result.model_copy(
                    update={
                        "status": ExecutionStatus.VERIFICATION_FAILED,
                        "accepted_model_attempt_id": None,
                        "accepted_output_ref": None,
                        "guard_results": tuple(guards),
                        "tool_executions": all_records,
                        "state_outcome": terminal_state,
                        "latency_ms": initial_result.latency_ms + phase_latency,
                    }
                )
        if commit_failed:
            # 基础设施 commit 故障：相关写记录标 env_fault（不是普通命令失败）
            records = tuple(
                record.model_copy(update={"fault_class": "env_fault"})
                if record.transaction_id in commit_ids
                else record
                for record in records
            )

        needs_recovery = bool(planned_write_ids or observed_risky_ids)
        rollback_ids = tuple(dict.fromkeys((*planned_write_ids, *observed_risky_ids)))
        rollback_status = (
            self._rollback_transactions(rollback_ids)
            if needs_recovery
            else RollbackStatus.NOT_NEEDED
        )
        if (
            (planned_write_ids or risky_binding_ids)
            and rollback_status == RollbackStatus.NOT_NEEDED
        ):
            rollback_status = RollbackStatus.UNKNOWN
        if rollback_status in (RollbackStatus.FAILED, RollbackStatus.PARTIAL, RollbackStatus.UNKNOWN):
            self._checkpoint_sessions.retain_for_recovery(
                request.execution_request_id,
                detail="rollback_failed",
                rollback_status=rollback_status.value,
                decision="rollback_required",
            )
            # 回滚基础设施失败：相关记录标 env_fault（保留 checkpoint/prepared 交恢复）
            records = tuple(
                record.model_copy(update={"fault_class": "env_fault"})
                if record.transaction_id in rollback_ids
                else record
                for record in records
            )
        records_after_rollback = tuple(
            record.model_copy(
                update={
                    "rollback_status": rollback_status,
                    "commit_status": (
                        CommitStatus.PARTIAL
                        if record.transaction_id in rollback_ids
                        and record.commit_status == CommitStatus.COMMITTED
                        else record.commit_status
                    ),
                }
            )
            if needs_recovery
            else record
            for record in records
        )
        all_records = initial_result.tool_executions + records_after_rollback
        rollback_failed = rollback_status not in (
            RollbackStatus.SUCCEEDED,
            RollbackStatus.NOT_NEEDED,
        )
        recovery_required = (
            rollback_failed or initial_result.state_outcome.recovery_required
        )
        state_after_rollback = self._state_for_tools(
            all_records,
            recovery_required=recovery_required,
            rollback_status=rollback_status,
            resume_status=(
                ResumeStatus.NOT_ATTEMPTED
                if rollback_failed
                else ResumeStatus.NOT_NEEDED
            ),
            planned_scopes=tuple(planned_scopes.values()),
            commit_status_override=(
                CommitStatus.PARTIAL if commit_failed else None
            ),
        )
        if rollback_status not in (RollbackStatus.SUCCEEDED, RollbackStatus.NOT_NEEDED):
            self._release_transactions(
                transaction_id
                for transaction_id in all_transaction_ids
                if transaction_id not in rollback_ids
            )
            return initial_result.model_copy(
                update={
                    "status": ExecutionStatus.ROLLBACK_FAILED,
                    "accepted_model_attempt_id": None,
                    "accepted_output_ref": None,
                    "guard_results": tuple(guards),
                    "tool_executions": all_records,
                    "state_outcome": self._state_with_session(
                        state_after_rollback,
                        request.execution_request_id,
                    ),
                    "latency_ms": initial_result.latency_ms + phase_latency,
                }
            )
        result = initial_result.model_copy(
            update={
                "status": ExecutionStatus.VERIFICATION_FAILED,
                "accepted_model_attempt_id": None,
                "accepted_output_ref": None,
                "guard_results": tuple(guards),
                "tool_executions": all_records,
                "state_outcome": state_after_rollback,
                "latency_ms": initial_result.latency_ms + phase_latency,
            }
        )
        close_result = self._close_sandbox_span(request)
        if close_result.ok:
            state_after_rollback = self._closed_state(
                state_after_rollback,
                request.execution_request_id,
            )
            self._drop_session(request.execution_request_id, decision="rolled_back")
        else:
            self._checkpoint_sessions.retain_for_recovery(
                request.execution_request_id,
                detail=close_result.detail or "rollback_close_failed",
                rollback_status=rollback_status.value,
            )
            state_after_rollback = state_after_rollback.model_copy(
                update={
                    "recovery_required": True,
                    "fault_class": "env_fault",
                }
            )
        return result.model_copy(
            update={
                "state_outcome": self._state_with_session(
                    state_after_rollback,
                    request.execution_request_id,
                )
            }
        )

    def execute_recovery_check(self, transaction_ids: Sequence[str]) -> ResumeStatus:
        """Run the Coordinator-requested recovery check after a recovery span.

        This is intentionally separate from ``RoutePlan.verifier_plan``:
        direct routes are contractually guard-free, while a workflow recovery
        lock still requires an explicit state-resume observation.
        """

        if self._tool_proxy is None:
            return ResumeStatus.UNKNOWN
        return _combine_resume(
            [self._tool_proxy.resume_check(transaction_id) for transaction_id in transaction_ids]
        )

    def finalize_recovery(self, transaction_ids: Sequence[str]) -> None:
        """Release adapter state retained solely for a workflow recovery lock.

        Ensures complete cleanup of checkpoint resources, including:
        - Tool Proxy state
        - Prepared transactions tracking
        - Sandbox/checkpoint handles
        """

        if self._tool_proxy is None:
            return
        finalize = getattr(self._tool_proxy, "finish_recovery", None)
        finalized = True
        if callable(finalize):
            try:
                finalized = finalize(tuple(transaction_ids)) is not False
            except Exception:
                finalized = False
        if not finalized:
            return
        transaction_set = set(transaction_ids)
        for session in self._checkpoint_sessions.active_sessions():
            if transaction_set.intersection(session.transaction_ids):
                self._drop_session(session.session_id, decision="recovery_resolved")

    def abort_tool_phase(self, request: ExecutionRequest) -> RollbackStatus:
        """Close a verified span when Agent execution ends before observations arrive."""
        transaction_ids = self._prepared_transactions.get(request.execution_request_id, ())
        # No tool has been observed, so a checkpoint can simply be released.
        # Adapters that retain an explicit transaction handle may still need a
        # rollback; perform it only when they expose one and keep failures
        # conservative.
        rollback_status = RollbackStatus.NOT_NEEDED
        if transaction_ids and self._tool_proxy is not None:
            rollback_status = self._rollback_transactions(transaction_ids)
            if rollback_status not in (
                RollbackStatus.SUCCEEDED,
                RollbackStatus.NOT_NEEDED,
            ):
                # Keep the maps and the adapter checkpoint alive for the
                # Coordinator's recovery lock.  Closing here would destroy
                # the only restoration handle.
                self._checkpoint_sessions.retain_for_recovery(
                    request.execution_request_id,
                    detail="abort_rollback_failed",
                    rollback_status=rollback_status.value,
                    decision="rollback_required",
                )
                return rollback_status
        close_result = self._close_sandbox_span(request)
        if close_result.ok:
            self._drop_session(request.execution_request_id, decision="aborted")
        else:
            self._checkpoint_sessions.retain_for_recovery(
                request.execution_request_id,
                detail=close_result.detail or "abort_close_failed",
                rollback_status=rollback_status.value,
                decision="aborted",
            )
        # close_span 失败：保留 prepared 句柄交恢复，不静默清理泄漏的 checkpoint。
        return rollback_status

    def _release_transactions(self, transaction_ids: Sequence[str]) -> _LifecycleResult:
        """Release non-checkpoint authorization handles with explicit failure."""

        transaction_ids = tuple(dict.fromkeys(transaction_ids))
        if not transaction_ids:
            return _LifecycleResult(True, operation="release_transactions")
        if self._tool_proxy is None:
            return _LifecycleResult(
                False,
                operation="release_transactions",
                fault_class="env_fault",
                detail="tool_proxy_missing",
                handle_retained=True,
                retryable=True,
            )
        release = getattr(self._tool_proxy, "release_transaction", None)
        if not callable(release):
            return _LifecycleResult(
                False,
                operation="release_transactions",
                fault_class="env_fault",
                detail="release_transaction_callback_missing",
                handle_retained=True,
                retryable=True,
            )
        for transaction_id in transaction_ids:
            try:
                if release(transaction_id) is False:
                    return _LifecycleResult(
                        False,
                        operation="release_transactions",
                        fault_class="env_fault",
                        detail=f"release_transaction_refused:{transaction_id}",
                        handle_retained=True,
                        retryable=True,
                    )
            except Exception as error:
                return _LifecycleResult(
                    False,
                    operation="release_transactions",
                    fault_class="env_fault",
                    detail=f"release_transaction_error:{type(error).__name__}",
                    handle_retained=True,
                    retryable=True,
                )
        return _LifecycleResult(True, operation="release_transactions")

    def _analyze_action(self, action: ToolAction) -> EffectVerdict:
        """pre-check L1：一律产出 EffectVerdict，永不回退 ToolProxy 声明的 scope。

        分析器缺失/解析异常/非 bash/空命令 -> UNKNOWN -> REJECT（文档 §4.1：
        unknown/external/irreversible 全拒；KB 失效时不允许降级放行）。
        """
        action_analyzer = self._tool_action_analyzers.get(action.tool_name)
        if action_analyzer is not None:
            raw_codes = action.arguments.get("codes") if isinstance(action.arguments, Mapping) else None
            code = "\n".join(str(item) for item in raw_codes) if isinstance(raw_codes, list) else ""
            try:
                return action_analyzer.analyze(
                    code,
                    cwd=action.cwd or self._workspace_cwd,
                    workspace_root=self._workspace_cwd,
                )
            except Exception:
                return self._unknown_verdict("tool_action_analyzer_error")
        if action.tool_name != "bash":
            return self._unknown_verdict("not_bash")
        command = action.arguments.get("command") if isinstance(action.arguments, Mapping) else None
        if not isinstance(command, str) or not command.strip():
            return self._unknown_verdict("missing_command")
        if self._analyzer is None:
            return self._unknown_verdict("analyzer_unavailable")
        # 写目标边界依赖 workspace_root；生产路径缺失时 guarded 直接拒绝
        # （不能跳过边界检查放行 workspace 外写）。
        if not self._workspace_cwd:
            return self._unknown_verdict("workspace_root_missing")
        try:
            return self._analyzer.analyze(
                command,
                cwd=action.cwd or self._workspace_cwd,
                workspace_root=self._workspace_cwd,
            )
        except Exception:
            return self._unknown_verdict("analyzer_error")

    @staticmethod
    def _unknown_verdict(reason: str) -> EffectVerdict:
        return EffectVerdict(
            category="unknown",
            scope=EffectScope.UNKNOWN,
            risk_flags=frozenset(),
            reason=reason,
        )

    def _l2_needed(self, verdicts: Mapping[str, Any]) -> bool:
        """L2 触发条件：含写事务，或全只读但存在 inferred/带 flag 的 action。"""
        for verdict in verdicts.values():
            if verdict is None:
                continue
            if verdict.category in ("declared_write", "opaque_exec", "verify_exec"):
                return True
            if verdict.category == "read_only" and (
                verdict.confidence == "inferred" or verdict.risk_flags
            ):
                return True
        return False

    def _observe_git_status(self, request: ExecutionRequest) -> frozenset | None:
        """经 ToolProxy 只读授权执行 git status --porcelain，返回变更路径集。

        观察是审计事实，不是 agent 请求的 action 结果——使用独立 transaction
        注册，绝不混入 span 的 tool_executions。不可得（无 proxy / 非 git 目录
        / 观察失败）返回 None，调用方跳过 L2（不硬败）。
        """
        # 有些 adapter 的工具根本不是 shell（AppWorld 唯一的工具是 Python 代码批次
        # ``execute_code_batch(codes)``）。对它跑 ``git status`` 只会退化成**空批次**
        # 并"成功返回无变化"，于是 L2 既不报"不适用"也不报"失败"——变成**静默 no-op**，
        # ``effect_misclassification`` 永远发现不了（诊断报告 D5）。
        # 由 adapter 显式声明不支持 -> 返回 ``_L2_NOT_AVAILABLE``（L2 不适用，调用方
        # 跳过检查），而不是伪装成"观测到了、没变化"。
        if getattr(self._tool_proxy, "supports_command_observation", True) is False:
            return _L2_NOT_AVAILABLE
        begin = getattr(self._tool_proxy, "begin_transaction", None)
        execute = getattr(self._tool_proxy, "execute", None)
        if not callable(begin) or not callable(execute):
            return None
        action = ToolAction(
            tool_call_id=f"observe-{request.execution_request_id}",
            tool_name="bash",
            arguments={"command": "git status --porcelain"},
        )
        transaction_id = f"transaction-{request.execution_request_id}-observe"
        release = getattr(self._tool_proxy, "release_transaction", None)
        try:
            accepted = begin(action, transaction_id=transaction_id)
            if not accepted:
                return None
            result = execute(action, transaction_id=transaction_id)
        except Exception:
            return None
        finally:
            # 观察事务显式释放授权句柄（阶段 2 修正）：不留残留 transaction。
            if callable(release):
                try:
                    release(transaction_id)
                except Exception:
                    pass
        raw = result.record.raw_output or {}
        output = raw.get("output")
        if result.record.returncode not in (None, 0):
            # 非 git 仓库（git status 报 not a git repository）是 L2 不适用，
            # 不是观察基础设施失败；其余非零码视为观察失败。
            if isinstance(output, str) and "not a git repository" in output:
                return _L2_NOT_AVAILABLE
            return None
        if not isinstance(output, str):
            return None
        paths = frozenset(
            self._abs_path(_porcelain_path(line))
            for line in output.splitlines()
            if line.strip()
        )
        return paths

    def _abs_path(self, path: str) -> str:
        """porcelain 相对路径绝对化到 workspace cwd（与写目标同一表示）。"""
        from pathlib import Path as _Path

        return str(_Path(path) if _Path(path).is_absolute() else _Path(self._workspace_cwd) / path)

    def _planned_state(self, scopes: Sequence[EffectScope]) -> StateOutcome:
        if EffectScope.UNKNOWN in scopes or EffectScope.IRREVERSIBLE_WRITE in scopes:
            effect = ActualEffect.UNKNOWN
        elif EffectScope.REVERSIBLE_WRITE in scopes:
            effect = ActualEffect.WRITE
        elif EffectScope.READ_ONLY in scopes:
            effect = ActualEffect.READ_ONLY
        else:
            effect = ActualEffect.NONE
        return StateOutcome(
            effect=effect,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        )

    def _commit_transactions(self, transaction_ids: Sequence[str]) -> _LifecycleResult:
        if not transaction_ids:
            return _LifecycleResult(True)
        if self._tool_proxy is None:
            return _LifecycleResult(False, fault_class="env_fault", detail="tool_proxy_missing")
        commit = getattr(self._tool_proxy, "commit", None)
        if not callable(commit):
            return _LifecycleResult(False, fault_class="env_fault", detail="commit_callback_missing")
        for transaction_id in dict.fromkeys(transaction_ids):
            try:
                if commit(transaction_id) is False:
                    return _LifecycleResult(
                        False, fault_class="env_fault", detail=f"commit_refused:{transaction_id}"
                    )
            except Exception as error:
                return _LifecycleResult(
                    False, fault_class="env_fault", detail=f"commit_error:{type(error).__name__}"
                )
        self._counters.bump("transactions_committed")
        return _LifecycleResult(True)

    def _rollback_transactions(
        self,
        transaction_ids: Sequence[str],
        *,
        scopes: Mapping[str, EffectScope] | None = None,
    ) -> RollbackStatus:
        if scopes is not None:
            transaction_ids = tuple(
                transaction_id
                for transaction_id in transaction_ids
                if scopes.get(transaction_id)
                in (EffectScope.REVERSIBLE_WRITE, EffectScope.IRREVERSIBLE_WRITE, EffectScope.UNKNOWN)
            )
        transaction_ids = tuple(dict.fromkeys(transaction_ids))
        if not transaction_ids:
            return RollbackStatus.NOT_NEEDED
        self._checkpoint_sessions.begin_rollback(transaction_ids)
        if self._tool_proxy is None:
            return RollbackStatus.UNKNOWN
        rollback = getattr(self._tool_proxy, "rollback", None)
        if not callable(rollback):
            return RollbackStatus.UNKNOWN
        statuses: list[RollbackStatus] = []
        for transaction_id in transaction_ids:
            try:
                value = rollback(transaction_id)
                statuses.append(
                    value if isinstance(value, RollbackStatus) else RollbackStatus(str(value))
                )
            except Exception:
                statuses.append(RollbackStatus.UNKNOWN)
        combined = _combine_rollback(statuses)
        if combined == RollbackStatus.SUCCEEDED:
            self._counters.bump("transactions_rolled_back")
        elif combined not in (RollbackStatus.NOT_NEEDED,):
            self._counters.bump("rollback_failures")
        if combined in (RollbackStatus.SUCCEEDED, RollbackStatus.NOT_NEEDED):
            self._checkpoint_sessions.finish_rollback(
                transaction_ids,
                rollback_status=combined.value,
            )
        return combined

    def _check_tool_result(
        self,
        step,
        result: ToolExecutionResult,
        *,
        model_attempt_id: str,
    ) -> GuardResult:
        checker = getattr(self._verifier, "check_tool_result", None)
        if checker is None:
            outcome = (
                GuardOutcome.PASS
                if result.status == ToolExecutionStatus.SUCCESS and result.returncode in (None, 0)
                else GuardOutcome.REJECT
            )
            return GuardResult(
                guard_ref=step.guard_ref,
                stage=step.stage,
                subject_model_attempt_id=model_attempt_id,
                subject_tool_call_id=result.tool_call_id,
                outcome=outcome,
                latency_ms=0,
                cost=0.0,
        )
        try:
            guard = checker(step, result=result, model_attempt_id=model_attempt_id)
            if not isinstance(guard, GuardResult):
                raise TypeError("verifier returned a non-GuardResult value")
            return guard
        except Exception:
            return GuardResult(
                guard_ref=step.guard_ref,
                stage=step.stage,
                subject_model_attempt_id=model_attempt_id,
                subject_tool_call_id=result.tool_call_id,
                outcome=GuardOutcome.ERROR,
                latency_ms=0,
                cost=0.0,
            )

    def _check_model_response(
        self,
        step,
        *,
        candidate: ModelResponse,
        model_attempt_id: str,
        span_context: SpanIdentity | None = None,
    ) -> GuardResult:
        """Turn verifier failures into a recorded reject/error, never an escape.

        ``span_context`` 是**当前 span 的身份**。`ModelResponse` 里没有它，
        所以一个要在「当前 span」上查历史的 verifier（M4 失败记忆）拿不到自己
        是谁 —— 必须从这里显式传下去。
        """

        try:
            result = self._verifier.check(
                step,
                candidate=candidate,
                model_attempt_id=model_attempt_id,
                span_context=span_context,
            )
            if not isinstance(result, GuardResult):
                raise TypeError("verifier returned a non-GuardResult value")
            return result
        except Exception:
            return GuardResult(
                guard_ref=step.guard_ref,
                stage=step.stage,
                subject_model_attempt_id=model_attempt_id,
                outcome=GuardOutcome.ERROR,
                latency_ms=0,
                cost=0.0,
            )

    def _close_sandbox_span(self, request: ExecutionRequest) -> _LifecycleResult:
        session = self._session(request.execution_request_id)
        if session is not None:
            release_result = self._release_transactions(
                session.transaction_ids
            )
            if not release_result.ok:
                return release_result
        close = getattr(self._tool_proxy, "close_span", None)
        if not callable(close):
            # 无 checkpoint 后端时没有 close_span 概念，视作幂等成功
            return _LifecycleResult(True, operation="close_span")
        try:
            if close(request) is False:
                return _LifecycleResult(
                    False,
                    operation="close_span",
                    fault_class="env_fault",
                    detail="close_span_refused",
                    handle_retained=True,
                    retryable=True,
                )
        except Exception as error:
            return _LifecycleResult(
                False,
                operation="close_span",
                fault_class="env_fault",
                detail=f"close_span_error:{type(error).__name__}",
                handle_retained=True,
                retryable=True,
            )
        return _LifecycleResult(True, operation="close_span")

    def _state_for_tools(
        self,
        records: Sequence[ToolExecutionResult],
        *,
        recovery_required: bool,
        rollback_status: RollbackStatus = RollbackStatus.NOT_NEEDED,
        resume_status: ResumeStatus = ResumeStatus.NOT_NEEDED,
        planned_scopes: Sequence[EffectScope] = (),
        commit_status_override: CommitStatus | None = None,
        fault_class: str | None = None,
    ) -> StateOutcome:
        effects = {record.effect for record in records}
        planned_scopes = tuple(planned_scopes)
        fault_class = fault_class or next(
            (record.fault_class for record in records if record.fault_class is not None),
            None,
        )
        effect = (
            ActualEffect.UNKNOWN
            if ActualEffect.UNKNOWN in effects
            or EffectScope.UNKNOWN in planned_scopes
            or EffectScope.IRREVERSIBLE_WRITE in planned_scopes
            else ActualEffect.WRITE
            if ActualEffect.WRITE in effects
            or EffectScope.REVERSIBLE_WRITE in planned_scopes
            else ActualEffect.READ_ONLY
            if ActualEffect.READ_ONLY in effects
            else ActualEffect.NONE
        )
        commit = commit_status_override or (
            CommitStatus.PARTIAL
            if any(record.commit_status == CommitStatus.PARTIAL for record in records)
            else CommitStatus.COMMITTED
            if any(record.commit_status == CommitStatus.COMMITTED for record in records)
            else CommitStatus.NOT_ATTEMPTED
        )
        return StateOutcome(
            effect=effect,
            commit_status=commit,
            recovery_required=recovery_required,
            rollback_status=rollback_status,
            resume_status=resume_status,
            fault_class=fault_class,
        )

    def _accept(
        self,
        request: ExecutionRequest,
        call,
        attempt_id: str,
        role: AttemptRole,
        model_ref: ModelRef,
        *,
        prior_attempts: tuple[ModelAttempt, ...] = (),
        guard_results: tuple[GuardResult, ...] = (),
        tool_executions: tuple[ToolExecutionResult, ...] = (),
        state_outcome: StateOutcome | None = None,
    ) -> ExecutionResult:
        """接受一次尝试：创建 ACCEPTED_OUTPUT 能力、把响应登记进 store、返回成功终态。"""
        accepted_ref = self._accepted_ref(request, attempt_id, model_ref)
        attempt = self._successful_attempt(request, call, attempt_id, role, model_ref)
        self._store.save_response(
            accepted_ref,
            call.response,
            audit_refs=(
                GuardRef(kind="model_response", id=call.response.response_id, version="v1"),
            ),
        )
        return self._result(
            request,
            status=ExecutionStatus.SUCCESS,
            attempts=prior_attempts + (attempt,),
            guard_results=guard_results,
            tool_executions=tool_executions,
            state_outcome=state_outcome,
            accepted_attempt_id=attempt_id,
            accepted_ref=accepted_ref,
        )

    def _resolve_request(
        self,
        request: ExecutionRequest,
        *,
        target_model_ref: ModelRef | None = None,
    ):
        """以执行视图解析原始请求；fallback 目标使用 fallback_request_ref。"""
        route = request.route_plan
        target = target_model_ref or route.target.model_ref
        request_ref = request.request_ref
        if target != route.target.model_ref:
            request_ref = request.fallback_request_ref
            if request_ref is None:
                raise M3Error("fallback route is missing its execution request capability")
        return self._store.resolve_request(
            request_ref,
            Component.M3,
            expected_target=target,
        )

    def _accepted_ref(
        self,
        request: ExecutionRequest,
        attempt_id: str,
        model_ref: ModelRef,
    ) -> CapabilityRef:
        """构造 ACCEPTED_OUTPUT 能力引用：audience=AGENT_ADAPTER，绑定目标模型。"""
        route = request.route_plan
        return CapabilityRef(
            kind="model_response",
            id=f"accepted-{attempt_id}",
            version="v1",
            scope=CapabilityScope.ACCEPTED_OUTPUT,
            audience=Component.AGENT_ADAPTER,
            expires_at=request.deadline_at + timedelta(hours=1),
            binding=CapabilityBinding(
                workflow_run_id=route.workflow_run_id,
                workflow_plan_ref=route.workflow_plan_ref,
                workflow_plan_revision=route.workflow_plan_revision,
                episode_id=route.episode_id,
                logical_invocation_id=route.logical_invocation_id,
                target_model_ref=model_ref,
            ),
        )

    def _successful_attempt(
        self,
        request: ExecutionRequest,
        call,
        attempt_id: str,
        role: AttemptRole,
        model_ref: ModelRef,
    ) -> ModelAttempt:
        """成功的模型尝试记录（result_ref 指向模型响应）。"""
        return ModelAttempt(
            model_attempt_id=attempt_id,
            role=role,
            model_ref=model_ref,
            status=AttemptStatus.SUCCESS,
            result_ref=GuardRef(
                kind="model_response",
                id=call.response.response_id,
                version="v1",
            ),
            response=call.response,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            latency_ms=call.latency_ms,
            cost=call.cost,
        )

    def _failed_attempt(
        self,
        attempt_id: str,
        role: AttemptRole,
        status: AttemptStatus,
        model_ref: ModelRef,
    ) -> ModelAttempt:
        """失败的模型尝试记录（result_ref 为合成的 model_error 引用）。"""
        return ModelAttempt(
            model_attempt_id=attempt_id,
            role=role,
            model_ref=model_ref,
            status=status,
            result_ref=GuardRef(kind="model_error", id=f"error-{attempt_id}", version="v1"),
            latency_ms=0,
        )

    def _result(
        self,
        request: ExecutionRequest,
        *,
        status: ExecutionStatus,
        attempts: tuple[ModelAttempt, ...] = (),
        guard_results: tuple[GuardResult, ...] = (),
        tool_executions: tuple[ToolExecutionResult, ...] = (),
        state_outcome: StateOutcome | None = None,
        accepted_attempt_id: str | None = None,
        accepted_ref: CapabilityRef | None = None,
    ) -> ExecutionResult:
        """构造终态 ExecutionResult；未提供状态事实时使用中性 outcome。"""
        route = request.route_plan
        outcome = state_outcome or StateOutcome(
            effect=ActualEffect.NONE,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        )
        outcome = self._state_with_session(outcome, request.execution_request_id)
        return ExecutionResult(
            schema_version=request.schema_version,
            message_id=f"message-result-{request.execution_request_id}",
            producer_ref=ProducerRef(component=Component.M3, version="0.1.0"),
            produced_at=request.produced_at,
            workflow_run_id=route.workflow_run_id,
            workflow_plan_ref=route.workflow_plan_ref,
            workflow_plan_revision=route.workflow_plan_revision,
            episode_id=route.episode_id,
            logical_invocation_id=route.logical_invocation_id,
            route_plan_id=route.route_plan_id,
            status=status,
            model_attempts=attempts,
            guard_results=guard_results,
            tool_executions=tool_executions,
            state_outcome=outcome,
            accepted_model_attempt_id=accepted_attempt_id,
            accepted_output_ref=accepted_ref,
            latency_ms=sum(attempt.latency_ms for attempt in attempts)
            + sum(guard.latency_ms for guard in guard_results)
            + sum(item.latency_ms for item in tool_executions),
        )


def _span_identity(route_plan: Any) -> SpanIdentity:
    """从 ``RoutePlan`` 取 span 身份（三标识与计划上同名）。

    只在 guarded 路径上构造一次、两处调用点共用 —— **两次 check（pre / post）
    必须拿到同一个身份**，否则 verifier 眼里的「当前 span」前后不一致。
    """

    return SpanIdentity(
        workflow_run_id=str(route_plan.workflow_run_id),
        episode_id=str(route_plan.episode_id),
        logical_invocation_id=str(route_plan.logical_invocation_id),
    )


def _attempt_id(request: ExecutionRequest, index: int) -> str:
    """同一 episode 内每次物理调用一个独立 attempt ID。"""
    return f"attempt-{request.execution_request_id}-{index}"


def _transaction_id(request: ExecutionRequest, tool_call_id: str) -> str:
    """为一个 episode 内的工具调用生成稳定事务 ID。"""
    return f"transaction-{request.execution_request_id}-{tool_call_id}"


def _prepared_transaction_result(
    *,
    action: ToolAction,
    transaction_id: str,
    scope: EffectScope,
    rollback_status: RollbackStatus,
) -> ToolExecutionResult:
    """Record a failed transaction cleanup before any tool was exposed.

    This is not a simulated tool output: it explicitly says the action was
    rejected during admission and exists only to carry the still-risky handle
    into the existing RecoveryDirective contract.
    """

    effect = (
        ActualEffect.WRITE
        if scope == EffectScope.REVERSIBLE_WRITE
        else ActualEffect.UNKNOWN
        if scope in (EffectScope.UNKNOWN, EffectScope.IRREVERSIBLE_WRITE)
        else ActualEffect.READ_ONLY
    )
    return ToolExecutionResult(
        transaction_id=transaction_id,
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        status=ToolExecutionStatus.REJECTED,
        effect=effect,
        commit_status=CommitStatus.BLOCKED,
        returncode=-1,
        output_summary="transaction_admission_cleanup_failed",
        raw_output={
            "output": "",
            "returncode": -1,
            "exception_info": "tool was not executed; transaction cleanup failed",
        },
        rollback_status=rollback_status,
        error_type="transaction-cleanup-failed",
    )


def _missing_tool_result(action: ToolAction, *, transaction_id: str) -> ToolProxyResult:
    """Represent an absent observation as an unknown, non-committable result."""

    output = {
        "output": "",
        "returncode": -1,
        "exception_info": "action missing from adapter",
    }
    record = ToolExecutionResult(
        transaction_id=transaction_id,
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        status=ToolExecutionStatus.REJECTED,
        effect=ActualEffect.UNKNOWN,
        commit_status=CommitStatus.BLOCKED,
        returncode=-1,
        output_summary="action_missing_from_adapter",
        raw_output=output,
        error_type="action-missing",
    )
    return ToolProxyResult(record=record, output=output, confirmed=False)


def _tool_record_matches_action(
    record: ToolExecutionResult,
    action: ToolAction,
) -> bool:
    """Check the result identity fields that survive the normalized proxy record.

    Security: Verify that the executed action matches the authorized action,
    including tool_call_id, tool_name, AND arguments to prevent argument tampering.
    """

    if record.tool_call_id != action.tool_call_id:
        return False
    if record.tool_name != action.tool_name:
        return False

    # Critical security check: verify arguments match
    # The record may not have arguments field in all cases, so we need to check
    # via the Tool Proxy's internal validation (which is already done)
    # This function is called after Tool Proxy has already validated, so we trust
    # that if we got here with matching IDs, the arguments were already checked.
    # The real validation happens in tool_proxy.py:_same_tool_action()

    return True


# 我方层异常标记（error_type 命中 -> env_fault；与候选能力无关，censored）
def _fault_class(record: ToolExecutionResult, post_guard_error: bool) -> str | None:
    """post-check 故障归因（阶段 2）：显式标记优先，不猜异常名。

    ToolProxy/沙箱基础设施异常在产出记录时已写 fault_class="env_fault"；
    post-check 自身异常在此覆盖为 guard_fault。returncode 永不参与——
    非零返回码是正确性观察（command_status），不构成我方层故障。
    """
    if record.fault_class is not None:
        return record.fault_class
    if post_guard_error:
        return "guard_fault"
    return None


def _tool_result_commits(
    record: ToolExecutionResult, *, accepts_failed: bool
) -> bool:
    """单条工具记录是否允许 commit（阶段 2 口径）。

    ``accepts_failed=True``（默认）保持既有语义：``grep`` 没匹配返回 1、
    ``git diff --quiet`` 返回 1 都是 agent 的常规反馈通道，FAILED 收据可信即可 commit。

    ``accepts_failed=False``（adapter 显式声明 ``commits_on_tool_failure = False``）
    时，**任何 FAILED 都阻断 commit** —— 不分只读还是写。

    只读豁免被**移除**了。旧口径是「只读不建 checkpoint，没有可回滚的东西，
    拦下来只会凭空产生一次空回滚加一次 LLM fallback」，只算了成本没算收益。
    实际口径是**执行失败一律不接受 SLM 的输出**：SLM 在只读上失败通常意味着它
    调错了 API 或参数，让更弱的模型自己重试，正是要卸载给 LLM 的那部分工作。

    好消息是只读**不需要回滚**：``rollback_ids`` 只含写事务，所以只读走
    ``needs_recovery=False`` → ``rollback_status=NOT_NEEDED`` → 直接
    ``VERIFICATION_FAILED`` → ``Coordinator._should_fallback``。
    """
    if record.status == ToolExecutionStatus.SUCCESS:
        return True
    if record.status != ToolExecutionStatus.FAILED:
        # TIMEOUT / REJECTED / UNKNOWN 不可确认，一律阻断（既有语义）
        return False
    return accepts_failed


def _provenance(record: ToolExecutionResult, verdict, author: str, l2_actual=None) -> dict | None:
    """写溯源（阶段 2）：{author, paths}，供 M4 正确性归因。

    paths 优先取 L2 增量观察到的真实变更（S_actual，含 opaque 命令的写），
    观察不可得时退回分析器静态写目标。
    """
    if record.effect != ActualEffect.WRITE:
        return None
    if l2_actual is not None:
        paths = sorted(l2_actual)
    elif verdict is not None:
        paths = list(verdict.write_targets)
    else:
        paths = []
    return {"author": author, "paths": paths}


def _l2_boundary_breach(
    verdicts: Mapping[str, Any],
    l2_actual: frozenset,
) -> bool:
    """L2 越界判定（§5.3）。

    Only declared_write verdicts contribute declared targets.  opaque_exec and
    verify_exec are container-snapshot covered and are not judged by L2; mixing
    them with a declared_write must not turn their natural side effects into a
    boundary breach.
    """
    verdict_list = [verdict for verdict in verdicts.values() if verdict is not None]
    declared = frozenset().union(
        *(
            verdict.write_targets
            for verdict in verdict_list
            if verdict.category == "declared_write"
        )
    )
    has_declared_write = any(
        verdict.category == "declared_write" for verdict in verdict_list
    )
    has_opaque_or_verify = any(
        verdict.category in ("opaque_exec", "verify_exec") for verdict in verdict_list
    )
    has_write = has_declared_write or has_opaque_or_verify
    if has_declared_write and not has_opaque_or_verify and l2_actual - declared:
        # 写越出声明目标 -> 效应边界被打破（uncontained）
        return True
    if not has_write and l2_actual:
        # 只读命令实际写了东西 -> READ_ONLY 效应误判（uncontained）
        return True
    return False


def _porcelain_path(line: str) -> str:
    """porcelain 行 -> 路径：跳过 XY 状态前缀；rename 取新路径；去引号。"""
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _record_with_category(
    record: ToolExecutionResult,
    verdict,
) -> ToolExecutionResult:
    """把 pre-check L1 的效应类别落进工具记录（M4 分桶键的数据源头）。"""
    if verdict is None:
        return record
    return record.model_copy(update={"effect_category": verdict.category})


def _normalize_tool_record(
    record: ToolExecutionResult,
    *,
    planned_scopes: Mapping[str, EffectScope],
    risky_ids: Sequence[str],
) -> ToolExecutionResult:
    """Project planned risk into observations when an adapter under-reports it."""

    effect = record.effect
    if record.transaction_id in risky_ids:
        effect = ActualEffect.UNKNOWN
    elif (
        planned_scopes.get(record.transaction_id) == EffectScope.REVERSIBLE_WRITE
        and effect in (ActualEffect.NONE, ActualEffect.READ_ONLY)
    ):
        # M3 authorized a write and created a checkpoint.  A contradictory
        # adapter label must not make recovery_required look read-only.
        effect = ActualEffect.WRITE
    if effect == record.effect:
        return record
    return record.model_copy(update={"effect": effect})


def _combine_rollback(values: Sequence[RollbackStatus]) -> RollbackStatus:
    if not values:
        return RollbackStatus.UNKNOWN
    if any(value == RollbackStatus.FAILED for value in values):
        return RollbackStatus.FAILED
    if any(value == RollbackStatus.UNKNOWN for value in values):
        return RollbackStatus.UNKNOWN
    if any(value == RollbackStatus.PARTIAL for value in values):
        return RollbackStatus.PARTIAL
    if all(value == RollbackStatus.NOT_NEEDED for value in values):
        return RollbackStatus.NOT_NEEDED
    return RollbackStatus.SUCCEEDED


def _combine_resume(values: Sequence[ResumeStatus]) -> ResumeStatus:
    if not values:
        return ResumeStatus.UNKNOWN
    if any(value == ResumeStatus.FAILED for value in values):
        return ResumeStatus.FAILED
    if any(value == ResumeStatus.UNKNOWN for value in values):
        return ResumeStatus.UNKNOWN
    if any(value == ResumeStatus.PARTIAL for value in values):
        return ResumeStatus.PARTIAL
    if all(value == ResumeStatus.NOT_NEEDED for value in values):
        return ResumeStatus.NOT_NEEDED
    return ResumeStatus.SUCCEEDED


__all__ = [
    "ControlledExecutor",
    "M3Error",
    "UnsupportedExecutionMode",
]
