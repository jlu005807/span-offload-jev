from datetime import datetime, timedelta, timezone

import pytest

from traceoffload.contracts import (
    AgentRole,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    ExecutionStatus,
    MessageRole,
    ModelMessage,
    ModelParameters,
    ModelRef,
    ModelRequest,
    NodeContext,
    RecoveryLevel,
    RequestScope,
    RoutePolicy,
    StateAccessMode,
    StateContext,
    WorkflowContext,
    WorkflowGoal,
)
from traceoffload.coordinator import Coordinator, CoordinatorError, EpisodeOutcome
from traceoffload.fakes import FakeExecutor, FakeProfiler, FakeRouter
from traceoffload.storage import EventStore, RequestResponseStore

NOW = datetime.now(timezone.utc)
LLM = ModelRef(kind="model", id="minimax", version="v1")


def _request() -> ModelRequest:
    return ModelRequest(
        request_version="1.0",
        messages=(ModelMessage(role=MessageRole.USER, content="hello"),),
        input_text="hello",
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            data_policy_ref={"kind": "policy", "id": "default", "version": "v1"}
        ),
    )


def _workflow() -> WorkflowContext:
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


def _policy(max_latency_ms: float = 5000.0) -> RoutePolicy:
    return RoutePolicy(
        policy_version="policy-1",
        correctness_lower_bound=0.9,
        risk_upper_bound=0.05,
        budget={"max_cost": 10.0, "max_latency_ms": max_latency_ms},
        utility_weights={"cost": 1.0, "latency": 1.0, "risk": 1.0},
        exploration_constraints={},
        conservative_prior={},
    )


class RecordingExecutor(FakeExecutor):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.last_request = None

    def execute_step(self, request):
        self.last_request = request
        return super().execute_step(request)


class SequenceExecutor(FakeExecutor):
    def __init__(
        self,
        statuses: list[ExecutionStatus],
        store: RequestResponseStore | None = None,
    ) -> None:
        super().__init__(store=store)
        self._queue = list(statuses)
        self._failer = FakeExecutor(status=ExecutionStatus.MODEL_FAILED, store=store)

    def execute_step(self, request):
        status = self._queue.pop(0)
        if status == ExecutionStatus.SUCCESS:
            return super().execute_step(request)
        return self._failer.execute_step(request)


def _coordinator(
    *,
    executor: FakeExecutor | None = None,
    policy: RoutePolicy | None = None,
    store: RequestResponseStore | None = None,
    router=None,
) -> tuple[Coordinator, RequestResponseStore, EventStore]:
    store = store or RequestResponseStore(clock=lambda: NOW)
    events = EventStore()
    executor = executor or FakeExecutor(store=store)
    coordinator = Coordinator(
        profiler=FakeProfiler(),
        router=router or FakeRouter(large_model=LLM),
        executor=executor,
        store=store,
        events=events,
        policy=policy or _policy(),
        clock=lambda: NOW,
    )
    return coordinator, store, events


def _run(coordinator: Coordinator) -> EpisodeOutcome:
    return coordinator.orchestrate_episode(
        _request(),
        _workflow(),
        workflow_run_id="workflow-1",
        caller_deadline_at=NOW + timedelta(minutes=5),
    )


def test_llm_only_episode_returns_accepted_response() -> None:
    coordinator, _, _ = _coordinator()

    outcome = _run(coordinator)

    assert outcome.status == ExecutionStatus.SUCCESS
    assert outcome.response is not None
    assert outcome.response.output_text == "fake-executed-output"
    assert outcome.response.raw_response_ref is not None


def test_each_episode_appends_exactly_one_span_record() -> None:
    coordinator, _, events = _coordinator()

    outcome = _run(coordinator)

    spans = events.span_records()
    assert len(spans) == 1
    assert spans[0].route_plan.episode_id == outcome.episode_id
    assert spans[0].route_plan.route_plan_id == spans[0].execution_summary.route_plan_id


def test_episode_ids_are_consistent_across_objects_and_unique() -> None:
    coordinator, _, events = _coordinator()

    first = _run(coordinator)
    second = _run(coordinator)

    assert first.episode_id != second.episode_id
    span1, span2 = events.span_records()
    assert span1.route_plan.episode_id == span1.execution_summary.episode_id == first.episode_id
    assert span1.route_plan.logical_invocation_id == first.logical_invocation_id
    assert span1.route_plan.workflow_run_id == "workflow-1"
    assert span2.route_plan.episode_id == second.episode_id


def test_failed_execution_returns_no_response_and_no_accepted_output() -> None:
    store = RequestResponseStore(clock=lambda: NOW)
    coordinator, _, events = _coordinator(
        executor=FakeExecutor(store=store, status=ExecutionStatus.MODEL_FAILED),
        store=store,
    )

    outcome = _run(coordinator)

    assert outcome.status == ExecutionStatus.MODEL_FAILED
    assert outcome.response is None
    span = events.span_records()[0]
    assert span.execution_summary.status == ExecutionStatus.MODEL_FAILED
    assert span.execution_summary.accepted_model_attempt_id is None
    assert span.execution_summary.model_attempts == ()


def test_passed_caller_deadline_is_rejected() -> None:
    coordinator, _, _ = _coordinator()

    with pytest.raises(CoordinatorError):
        coordinator.orchestrate_episode(
            _request(),
            _workflow(),
            workflow_run_id="workflow-1",
            caller_deadline_at=NOW,
        )


def test_workflow_plan_mismatch_is_rejected() -> None:
    coordinator, _, _ = _coordinator()

    with pytest.raises(CoordinatorError):
        coordinator.orchestrate_episode(
            _request(),
            _workflow(),
            workflow_run_id="workflow-1",
            workflow_plan_ref={"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
            workflow_plan_revision=None,
            caller_deadline_at=NOW + timedelta(minutes=5),
        )


def test_execution_deadline_is_capped_by_route_timeout() -> None:
    store = RequestResponseStore(clock=lambda: NOW)
    recording = RecordingExecutor(store=store)
    coordinator, _, _ = _coordinator(
        executor=recording,
        policy=_policy(max_latency_ms=100.0),
        store=store,
    )

    _run(coordinator)

    assert recording.last_request is not None
    assert recording.last_request.deadline_at <= NOW + timedelta(milliseconds=100)
    assert recording.last_request.idempotency_key.startswith("idempotency-")
    assert recording.last_request.safety_calibration_ref.version == "feedback-none"


def test_episodes_continue_after_failure() -> None:
    store = RequestResponseStore(clock=lambda: NOW)
    executor = SequenceExecutor(
        [ExecutionStatus.MODEL_FAILED, ExecutionStatus.SUCCESS],
        store=store,
    )
    coordinator, _, events = _coordinator(executor=executor, store=store)

    failed = _run(coordinator)
    succeeded = _run(coordinator)

    assert failed.status == ExecutionStatus.MODEL_FAILED
    assert succeeded.status == ExecutionStatus.SUCCESS
    assert len(events.span_records()) == 2


def test_guarded_route_gets_a_step_transaction_ref() -> None:
    from traceoffload.contracts import (
        Component,
        ExecutionMode,
        FailureAction,
        FallbackPlan,
        GuardRef,
        ModelRef,
        ModelTarget,
        ModelTargetKind,
        PassAction,
        PassCondition,
        PassConditionKind,
        ProducerRef,
        RecoveryLevel,
        RoutePlan,
        VerifierPlan,
        VerifierStage,
        VerifierStep,
        VerifierSubject,
    )

    qwen = ModelRef(kind="model", id="qwen", version="v1")
    minimax = ModelRef(kind="model", id="minimax", version="v1")
    route = RoutePlan(
        schema_version="1.0",
        message_id="message-route-1",
        producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        route_plan_id="route-1",
        target=ModelTarget(
            kind=ModelTargetKind.SMALL_MODEL,
            model_ref=qwen,
        ),
        execution_mode=ExecutionMode.GUARDED_SLM,
        verifier_plan=VerifierPlan(
            verifier_plan_id="verifier-guarded",
            version="v1",
            steps=(
                VerifierStep(
                    step_id="step-1",
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
            target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=minimax),
            recovery_mode=RecoveryLevel.NOT_NEEDED,
        ),
        timeout_ms=5000,
        experience_key="descriptor/v1|output:none|tools:none",
        decision_meta={
            "policy_version": "policy-1",
            "feedback_version": "feedback-1",
            "reason_code": "utility_guarded_slm",
            "is_exploration": False,
            "selection_probability": 1.0,
        },
    )

    class FixedRouter:
        def decide(self, profile, options, policy, snapshot, *, admission_facts=None):
            return route

    store = RequestResponseStore(clock=lambda: NOW)
    recording = RecordingExecutor(store=store)
    coordinator, _, _ = _coordinator(
        executor=recording,
        store=store,
        router=FixedRouter(),
    )

    _run(coordinator)

    assert recording.last_request is not None
    assert recording.last_request.step_transaction_ref is not None
    assert recording.last_request.step_transaction_ref.scope == "transaction"
    assert recording.last_request.step_transaction_ref.audience == "m3"
