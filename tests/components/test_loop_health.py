"""Coordinator loop-health tests for exact-action and frozen-reasoning locks.

The command-loop lock is one-shot; the reasoning fixed-point lock persists until
a state-changing (write/unknown/failed/non-tool) episode clears it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from traceoffload.contracts import (
    ActualEffect,
    AttemptRole,
    AttemptStatus,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    CommitStatus,
    Component,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FallbackPlan,
    FinishReason,
    GuardRef,
    MessageRole,
    ModelAttempt,
    ModelMessage,
    ModelParameters,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseToolCall,
    ModelTarget,
    ModelTargetKind,
    ParseStatus,
    FailureAction,
    PassAction,
    PassCondition,
    PassConditionKind,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    RollbackStatus,
    RoutePlan,
    RoutePolicy,
    ResumeStatus,
    StateOutcome,
    ToolExecutionResult,
    ToolExecutionStatus,
    VerifierPlan,
    VerifierStage,
    VerifierStep,
    VerifierSubject,
)
from traceoffload.coordinator import Coordinator, _LoopDirective
from traceoffload.fakes import FakeExecutor, FakeProfiler, FakeRouter
from traceoffload.storage import EventStore, RequestResponseStore

NOW = datetime.now(timezone.utc)
LARGE = ModelRef(kind="model", id="large", version="v1")
SMALL = ModelRef(kind="model", id="qwen", version="v1")


def _request() -> ModelRequest:
    return ModelRequest(
        request_version="1.0",
        messages=(ModelMessage(role=MessageRole.USER, content="fix the bug"),),
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            data_policy_ref={"kind": "policy", "id": "default", "version": "v1"}
        ),
    )


def _policy() -> "RoutePolicy":
    return RoutePolicy(
        policy_version="policy-1",
        correctness_lower_bound=0.9,
        risk_upper_bound=0.05,
        budget={"max_cost": 10.0, "max_latency_ms": 5000.0},
        utility_weights={"cost": 1.0, "latency": 1.0, "risk": 1.0},
        exploration_constraints={},
        conservative_prior={},
    )


def _coordinator() -> Coordinator:
    store = RequestResponseStore(clock=lambda: NOW)
    return Coordinator(
        profiler=FakeProfiler(),
        router=FakeRouter(large_model=LARGE),
        executor=FakeExecutor(store=store),
        store=store,
        events=EventStore(),
        policy=_policy(),
        clock=lambda: NOW,
        large_model=LARGE,
    )


def _small_guarded_route(episode_id: str = "episode-1") -> RoutePlan:
    return RoutePlan(
        schema_version="1.0",
        message_id=f"message-route-{episode_id}",
        producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        episode_id=episode_id,
        logical_invocation_id=f"invocation-{episode_id}",
        route_plan_id=f"route-{episode_id}",
        target=ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=SMALL),
        execution_mode=ExecutionMode.GUARDED_SLM,
        verifier_plan=VerifierPlan(
            verifier_plan_id=f"verifier-{episode_id}",
            version="v1",
            steps=(
                VerifierStep(
                    step_id="step-pre",
                    guard_ref=GuardRef(kind="guard", id="schema", version="v1"),
                    stage=VerifierStage.PRE_COMMIT,
                    subject=VerifierSubject.CANDIDATE_OUTPUT,
                    pass_condition=PassCondition(kind=PassConditionKind.SCHEMA_VALID),
                    timeout_ms=500,
                    on_pass=PassAction.CONTINUE,
                    on_failure=FailureAction.REJECT_AND_FALLBACK,
                ),
                VerifierStep(
                    step_id="step-post",
                    guard_ref=GuardRef(kind="guard", id="schema-post", version="v1"),
                    stage=VerifierStage.POST_CHECK,
                    subject=VerifierSubject.FINAL_ARTIFACT,
                    pass_condition=PassCondition(kind=PassConditionKind.POLICY_PASS),
                    timeout_ms=500,
                    on_pass=PassAction.RETURN,
                    on_failure=FailureAction.ROLLBACK_THEN_FALLBACK,
                ),
            ),
            overall_timeout_ms=1000,
        ),
        fallback_plan=FallbackPlan(
            target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LARGE),
            recovery_mode=RecoveryLevel.NOT_NEEDED,
        ),
        timeout_ms=5000,
        experience_key="test/family",
        decision_meta={
            "policy_version": "policy-1",
            "feedback_version": "feedback-none",
            "reason_code": "utility_guarded_slm",
            "is_exploration": False,
            "selection_probability": 1.0,
        },
    )


def _large_direct_route(episode_id: str = "episode-large") -> RoutePlan:
    return RoutePlan(
        schema_version="1.0",
        message_id=f"message-route-{episode_id}",
        producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        episode_id=episode_id,
        logical_invocation_id=f"invocation-{episode_id}",
        route_plan_id=f"route-{episode_id}",
        target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LARGE),
        execution_mode=ExecutionMode.DIRECT,
        verifier_plan=VerifierPlan(
            verifier_plan_id=f"verifier-none-{episode_id}",
            version="v1",
            overall_timeout_ms=1000,
            empty_plan_reason="large_model_direct_no_guard",
        ),
        timeout_ms=5000,
        experience_key="test/family",
        decision_meta={
            "policy_version": "policy-1",
            "feedback_version": "feedback-none",
            "reason_code": "coordinator_forced_direct",
            "is_exploration": False,
            "selection_probability": 1.0,
        },
    )


def _tool_execution(tool_call_id: str, *, effect: ActualEffect = ActualEffect.READ_ONLY) -> ToolExecutionResult:
    return ToolExecutionResult(
        transaction_id=f"transaction-{tool_call_id}",
        tool_call_id=tool_call_id,
        tool_name="bash",
        status=ToolExecutionStatus.SUCCESS,
        effect=effect,
        commit_status=CommitStatus.NOT_ATTEMPTED,
        rollback_status=RollbackStatus.NOT_NEEDED,
        resume_status=ResumeStatus.NOT_NEEDED,
    )


def _success_result(
    episode_id: str,
    route_plan_id: str,
    *,
    tool_executions: tuple[ToolExecutionResult, ...] = (),
    state_outcome: StateOutcome | None = None,
    accepted_model_attempt_id: str | None = None,
    accepted_output_ref: CapabilityRef | None = None,
) -> ExecutionResult:
    attempt_id = accepted_model_attempt_id or f"attempt-{episode_id}"
    response_id = f"response-{episode_id}"
    if accepted_output_ref is None:
        accepted_output_ref = CapabilityRef(
            kind="model_response",
            id=f"accepted-{episode_id}",
            version="v1",
            scope=CapabilityScope.ACCEPTED_OUTPUT,
            audience=Component.AGENT_ADAPTER,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(
                workflow_run_id="workflow-1",
                episode_id=episode_id,
                logical_invocation_id=f"invocation-{episode_id}",
            ),
        )
    attempt = ModelAttempt(
        model_attempt_id=attempt_id,
        role=AttemptRole.SMALL_MODEL_CANDIDATE,
        model_ref=SMALL,
        status=AttemptStatus.SUCCESS,
        response=ModelResponse(
            response_id=response_id,
            model_attempt_id=attempt_id,
            output_text="ok",
            finish_reason=FinishReason.TOOL_CALLS,
            parse_status=ParseStatus.NOT_NEEDED,
        ),
        latency_ms=1,
    )
    return ExecutionResult(
        schema_version="1.0",
        message_id=f"message-result-{episode_id}",
        producer_ref=ProducerRef(component=Component.M3, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        episode_id=episode_id,
        logical_invocation_id=f"invocation-{episode_id}",
        route_plan_id=route_plan_id,
        status=ExecutionStatus.SUCCESS,
        model_attempts=(attempt,),
        tool_executions=tool_executions,
        state_outcome=state_outcome
        or StateOutcome(
            effect=ActualEffect.READ_ONLY,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        ),
        accepted_model_attempt_id=attempt_id,
        accepted_output_ref=accepted_output_ref,
        latency_ms=1,
    )


def _response(reasoning: str | None, command: str, *, tool_call_id: str = "tc-1") -> ModelResponse:
    return ModelResponse(
        response_id=f"response-{tool_call_id}",
        model_attempt_id=f"attempt-{tool_call_id}",
        output_text=None,
        reasoning_content=reasoning,
        tool_calls=(
            ModelResponseToolCall(
                tool_call_id=tool_call_id,
                tool_name="bash",
                arguments={"command": command},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        parse_status=ParseStatus.NOT_NEEDED,
    )


def _armed_reasoning_lock(coordinator: Coordinator) -> None:
    reasoning = "Let me look at the autodetector.py around line 1313 again."
    for index, command in enumerate(
        ("sed -n '811,848p' autodetector.py", "sed -n '849,905p' autodetector.py"),
        start=1,
    ):
        episode_id = f"episode-{index}"
        route = _small_guarded_route(episode_id)
        call_id = f"tc-{index}"
        result = _success_result(
            episode_id,
            route.route_plan_id,
            tool_executions=(_tool_execution(call_id),),
        )
        coordinator._record_loop_outcome(
            "workflow-1",
            route,
            response=_response(reasoning, command, tool_call_id=call_id),
            result=result,
        )


def test_repeated_reasoning_arms_persistent_lock() -> None:
    coordinator = _coordinator()

    _armed_reasoning_lock(coordinator)

    state = coordinator._workflow_loop_state["workflow-1"]
    assert state.force_direct is True
    assert state.loop_kind == "reasoning"
    assert state.reasoning_repeat_count == 2

    directive = coordinator._take_loop_directive("workflow-1")
    assert isinstance(directive, _LoopDirective)
    assert directive.loop_kind == "reasoning"
    assert "autodetector.py" in directive.action_summary
    # Persistent: the lock remains armed after being consumed.
    assert coordinator._workflow_loop_state["workflow-1"].force_direct is True
    second = coordinator._take_loop_directive("workflow-1")
    assert second is not None and second.loop_kind == "reasoning"


def test_command_loop_keeps_one_shot_semantics() -> None:
    coordinator = _coordinator()

    for index in range(2):
        episode_id = f"episode-{index}"
        route = _small_guarded_route(episode_id)
        call_id = f"tc-{index}"
        result = _success_result(
            episode_id,
            route.route_plan_id,
            tool_executions=(_tool_execution(call_id),),
        )
        coordinator._record_loop_outcome(
            "workflow-1",
            route,
            response=_response(None, "git status --short", tool_call_id=call_id),
            result=result,
        )

    state = coordinator._workflow_loop_state["workflow-1"]
    assert state.force_direct is True
    assert state.loop_kind == "command"

    directive = coordinator._take_loop_directive("workflow-1")
    assert directive is not None and directive.loop_kind == "command"
    assert "workflow-1" not in coordinator._workflow_loop_state


def test_reasoning_lock_clears_on_write_episode() -> None:
    coordinator = _coordinator()
    _armed_reasoning_lock(coordinator)

    route = _large_direct_route("episode-large")
    call_id = "tc-write"
    result = _success_result(
        "episode-large",
        route.route_plan_id,
        tool_executions=(_tool_execution(call_id, effect=ActualEffect.WRITE),),
        accepted_model_attempt_id=f"attempt-{call_id}",
        accepted_output_ref=CapabilityRef(
            kind="model_response",
            id=f"accepted-{call_id}",
            version="v1",
            scope=CapabilityScope.ACCEPTED_OUTPUT,
            audience=Component.AGENT_ADAPTER,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(
                workflow_run_id="workflow-1",
                episode_id="episode-large",
                logical_invocation_id="invocation-episode-large",
            ),
        ),
    )
    coordinator._record_loop_outcome(
        "workflow-1",
        route,
        response=_response(None, "git apply fix.patch", tool_call_id=call_id),
        result=result,
    )

    assert "workflow-1" not in coordinator._workflow_loop_state


def test_reasoning_lock_survives_read_only_large_episode() -> None:
    coordinator = _coordinator()
    _armed_reasoning_lock(coordinator)

    route = _large_direct_route("episode-large")
    call_id = "tc-read"
    result = _success_result(
        "episode-large",
        route.route_plan_id,
        tool_executions=(_tool_execution(call_id),),
        accepted_model_attempt_id=f"attempt-{call_id}",
        accepted_output_ref=CapabilityRef(
            kind="model_response",
            id=f"accepted-{call_id}",
            version="v1",
            scope=CapabilityScope.ACCEPTED_OUTPUT,
            audience=Component.AGENT_ADAPTER,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(
                workflow_run_id="workflow-1",
                episode_id="episode-large",
                logical_invocation_id="invocation-episode-large",
            ),
        ),
    )
    coordinator._record_loop_outcome(
        "workflow-1",
        route,
        response=_response(None, "git status --short", tool_call_id=call_id),
        result=result,
    )

    assert coordinator._workflow_loop_state["workflow-1"].force_direct is True
    assert coordinator._workflow_loop_state["workflow-1"].loop_kind == "reasoning"


def test_reasoning_lock_clears_on_failed_episode() -> None:
    coordinator = _coordinator()
    _armed_reasoning_lock(coordinator)

    route = _large_direct_route("episode-large")
    result = ExecutionResult(
        schema_version="1.0",
        message_id="message-result-failed",
        producer_ref=ProducerRef(component=Component.M3, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        episode_id="episode-large",
        logical_invocation_id="invocation-episode-large",
        route_plan_id=route.route_plan_id,
        status=ExecutionStatus.MODEL_FAILED,
        state_outcome=StateOutcome(
            effect=ActualEffect.NONE,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        ),
        latency_ms=1,
    )
    coordinator._record_loop_outcome("workflow-1", route, response=None, result=result)

    assert "workflow-1" not in coordinator._workflow_loop_state


def test_reasoning_loop_route_reason_is_recorded() -> None:
    """The next ordinary episode after a reasoning lock is forced direct with the right reason code."""
    coordinator = _coordinator()
    _armed_reasoning_lock(coordinator)

    outcome = coordinator.orchestrate_episode(
        _request(),
        _workflow(),
        workflow_run_id="workflow-1",
        caller_deadline_at=NOW + timedelta(minutes=5),
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    spans = coordinator._events.span_records()
    assert spans[-1].route_plan.decision_meta.reason_code == "repeated_reasoning_loop"
    assert spans[-1].route_plan.target.kind == ModelTargetKind.LARGE_MODEL


def _workflow():
    from traceoffload.contracts import (
        AgentRole,
        DownstreamContract,
        DownstreamDependency,
        EffectRisk,
        NodeContext,
        RecoveryLevel,
        StateAccessMode,
        StateContext,
        WorkflowContext,
        WorkflowGoal,
    )

    return WorkflowContext(
        context_version="1.0",
        agent_id="mini-swe-agent",
        agent_role=AgentRole.EXECUTOR,
        workflow_goal=WorkflowGoal(goal_text="repair the repository"),
        node_context=NodeContext(node_id="node-1", node_type="executor", step_index=0),
        downstream_contract=DownstreamContract(
            consumer_role=AgentRole.FINAL_RESPONDER,
            dependency_level=DownstreamDependency.FULL,
        ),
        state_context=StateContext(
            access_mode=StateAccessMode.NONE,
            expected_effect=EffectRisk.NONE,
        ),
        recovery_capability=RecoveryLevel.NOT_NEEDED,
    )
