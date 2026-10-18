from datetime import datetime, timedelta, timezone

from traceoffload.coordinator import Coordinator
from traceoffload.m3.integration import AdapterSession, AdapterSessionSpec
from traceoffload.m1.profiler import RuntimeProfiler
from traceoffload.m1.span_embedding import SpanEmbeddingTextBuilder
from traceoffload.fakes import FakeExecutor, FakeRouter
from traceoffload.storage import EventStore, RequestResponseStore
from traceoffload.contracts import (
    ActualEffect,
    AgentRole,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    CommitStatus,
    Component,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    GuardOutcome,
    GuardRef,
    InvocationContext,
    MessageRole,
    ModelMessage,
    ModelMessageToolCall,
    ModelParameters,
    ModelRef,
    ModelRequest,
    NodeContext,
    PriorExecutionFacts,
    PriorToolOutcome,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    ResumeStatus,
    RoutePolicy,
    RollbackStatus,
    StateAccessMode,
    StateContext,
    StateOutcome,
    VerifierStage,
    WorkflowContext,
    WorkflowGoal,
)

NOW = datetime.now(timezone.utc)


def _lineage() -> dict[str, object]:
    return {
        "workflow_run_id": "workflow-1",
        "workflow_plan_ref": {"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
        "workflow_plan_revision": 0,
        "episode_id": "episode-1",
        "logical_invocation_id": "invocation-1",
    }


def _context(prior: PriorExecutionFacts | None) -> InvocationContext:
    return InvocationContext(
        schema_version="1.0",
        message_id="message-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW,
        **_lineage(),
        request_ref=CapabilityRef(
            kind="request",
            id="request-1",
            version="v1",
            scope=CapabilityScope.PROFILE_READ,
            audience=Component.M1,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(**_lineage()),
        ),
        workflow_context=WorkflowContext(
            context_version="1.0",
            agent_id="mini-swe-agent",
            agent_role=AgentRole.EXECUTOR,
            workflow_goal=WorkflowGoal(goal_text="repair the repository"),
            node_context=NodeContext(
                node_id="node-1", node_type="react_loop", step_index=7
            ),
            downstream_contract=DownstreamContract(
                consumer_role=AgentRole.FINAL_RESPONDER,
                dependency_level=DownstreamDependency.FULL,
            ),
            state_context=StateContext(
                access_mode=StateAccessMode.TRANSACTIONAL_WRITE,
                expected_effect=EffectRisk.REVERSIBLE_WRITE,
            ),
            recovery_capability=RecoveryLevel.ROLLBACK_AND_RESUME,
        ),
        prior_execution_facts=prior,
        caller_deadline_at=NOW + timedelta(minutes=5),
    )


def _assistant_tool_call(call_id: str, command: str) -> ModelMessage:
    return ModelMessage(
        role=MessageRole.ASSISTANT,
        content=f"run {command}",
        tool_calls=(
            ModelMessageToolCall(
                tool_call_id=call_id,
                tool_name="bash",
                arguments={"command": command},
            ),
        ),
    )


def _request_with_rounds() -> ModelRequest:
    messages = [
        ModelMessage(role=MessageRole.USER, content="fix the bug"),
        _assistant_tool_call("c1", "cat a.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c1", tool_call_id="c1"),
        _assistant_tool_call("c2", "cat b.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c2", tool_call_id="c2"),
        _assistant_tool_call("c3", "cat c.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c3", tool_call_id="c3"),
        ModelMessage(role=MessageRole.USER, content="interleaved user note"),
        _assistant_tool_call("c4", "cat d.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c4", tool_call_id="c4"),
        _assistant_tool_call("c5", "cat e.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c5", tool_call_id="c5"),
        _assistant_tool_call("c6", "cat f.py"),
        ModelMessage(role=MessageRole.TOOL, content="FAIL c6", tool_call_id="c6"),
        _assistant_tool_call("c7", "cat g.py"),
        ModelMessage(role=MessageRole.TOOL, content="ok c7", tool_call_id="c7"),
    ]
    return ModelRequest(
        request_version="1.0",
        messages=tuple(messages),
        input_text="fix the bug",
        native_payload=[{"role": "user", "content": "fix the bug"}],
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            data_policy_ref={"kind": "policy", "id": "default", "version": "v1"}
        ),
    )


def _fact_for(call_id: str, success: bool) -> PriorToolOutcome:
    return PriorToolOutcome(
        tool_call_id=call_id,
        episode_id="episode-0",
        logical_invocation_id="invocation-0",
        success=success,
        returncode=0 if success else 1,
    )


def _prior_facts() -> PriorExecutionFacts:
    outcomes = [_fact_for(f"c{index}", True) for index in range(1, 6)]
    outcomes.append(_fact_for("c6", False))
    outcomes.append(_fact_for("c7", True))
    return PriorExecutionFacts(
        episode_id="episode-0",
        logical_invocation_id="invocation-0",
        verifier_results=(),
        state_observation=None,
        tool_outcomes=tuple(outcomes),
    )


def test_recent_window_keeps_failed_rounds_between_last_five_successes() -> None:
    request = _request_with_rounds()
    context = _context(_prior_facts())
    profile = RuntimeProfiler().build_profile(context, request)

    embedding_input = SpanEmbeddingTextBuilder().build(context, request, profile)

    assert embedding_input.prior_facts_present is True
    assert embedding_input.successful_round_count == 6
    assert embedding_input.selected_success_round_count == 5
    assert "c1" not in embedding_input.text
    assert "c2" in embedding_input.text
    assert "FAIL c6" in embedding_input.text
    assert "c7" in embedding_input.text
    assert "interleaved user note" in embedding_input.text


def test_missing_verifier_and_state_are_allowed() -> None:
    request = _request_with_rounds()
    prior = _prior_facts().model_copy(update={"state_observation": None})
    context = _context(prior)
    profile = RuntimeProfiler().build_profile(context, request)

    embedding_input = SpanEmbeddingTextBuilder().build(context, request, profile)

    assert "[PRIOR_VERIFICATION] none" in embedding_input.text
    assert "[PRIOR_STATE] none" in embedding_input.text


def test_builder_does_not_mutate_request_or_native_payload() -> None:
    request = _request_with_rounds()
    original_dump = request.model_dump(mode="json")
    original_native = request.native_payload
    context = _context(_prior_facts())
    profile = RuntimeProfiler().build_profile(context, request)

    SpanEmbeddingTextBuilder().build(context, request, profile)

    assert request.model_dump(mode="json") == original_dump
    assert request.native_payload == original_native


def _session() -> AdapterSession:
    return AdapterSession(
        AdapterSessionSpec(
            workflow_run_id="workflow-1",
            agent_id="mini-swe-agent",
            agent_version=None,
            workflow_goal=WorkflowGoal(goal_text="repair the repository"),
        )
    )


def test_adapter_session_snapshot_projects_latest_completed_episode() -> None:
    session = _session()
    guard = GuardOutcome.PASS
    guard_result = {
        "guard_ref": {"kind": "guard", "id": "basic-tool-call", "version": "v1"},
        "stage": VerifierStage.POST_CHECK.value,
        "subject_model_attempt_id": "attempt-1",
        "outcome": guard.value,
        "latency_ms": 1,
        "cost": 0.0,
    }
    state_observation = StateOutcome(
        effect=ActualEffect.READ_ONLY,
        commit_status=CommitStatus.NOT_ATTEMPTED,
        recovery_required=False,
        rollback_status=RollbackStatus.NOT_NEEDED,
        resume_status=ResumeStatus.NOT_NEEDED,
    )
    session._emit(  # noqa: SLF001 - session event writer is intentionally internal
        "tool_finished",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        tool_call_id="c1",
        payload={"result": {"returncode": 0}},
    )
    session._emit(
        "tool_finished",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        tool_call_id="c2",
        payload={"result": {"returncode": 1}},
    )
    session._emit(
        "verifier_result",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        payload=guard_result,
    )
    session._emit(
        "state_observation",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        payload=state_observation.model_dump(mode="json"),
    )
    session._emit(
        "span_recorded",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        payload={"span_kind": "normal", "parent_episode_id": None},
    )

    facts = session.snapshot_prior_execution_facts()

    assert facts is not None
    assert facts.episode_id == "episode-1"
    assert facts.span_kind == "normal"
    assert len(facts.verifier_results) == 1
    assert facts.verifier_results[0].outcome == GuardOutcome.PASS
    assert facts.state_observation is not None
    assert facts.state_observation.effect == ActualEffect.READ_ONLY
    assert {item.tool_call_id: item.success for item in facts.tool_outcomes} == {
        "c1": True,
        "c2": False,
    }


def test_adapter_session_snapshot_returns_none_before_first_completed_span() -> None:
    assert _session().snapshot_prior_execution_facts() is None


class _RecordingProfiler(RuntimeProfiler):
    def __init__(self) -> None:
        self.context = None

    def build_profile(self, context, request):
        self.context = context
        return super().build_profile(context, request)


def _policy() -> RoutePolicy:
    return RoutePolicy(
        policy_version="policy-1",
        correctness_lower_bound=0.9,
        risk_upper_bound=0.05,
        budget={"max_cost": 10.0, "max_latency_ms": 5000.0},
        utility_weights={"cost": 1.0, "latency": 1.0, "risk": 1.0},
        exploration_constraints={},
        conservative_prior={},
    )


def test_coordinator_passes_prior_facts_into_m1_context() -> None:
    store = RequestResponseStore(clock=lambda: NOW)
    events = EventStore()
    profiler = _RecordingProfiler()
    coordinator = Coordinator(
        profiler=profiler,
        router=FakeRouter(large_model=ModelRef(kind="model", id="large", version="v1")),
        executor=FakeExecutor(store=store),
        store=store,
        events=events,
        policy=_policy(),
        clock=lambda: NOW,
    )
    session = coordinator.open_adapter_session(
        AdapterSessionSpec(
            workflow_run_id="workflow-1",
            agent_id="mini-swe-agent",
            agent_version=None,
            workflow_goal=WorkflowGoal(goal_text="repair the repository"),
        )
    )
    session._emit(  # noqa: SLF001 - build a completed prior episode for the test
        "span_recorded",
        episode_id="episode-0",
        logical_invocation_id="invocation-0",
        payload={"span_kind": "normal", "parent_episode_id": None},
    )

    coordinator.orchestrate_episode(
        _request_with_rounds(),
        _context(None).workflow_context,
        workflow_run_id="workflow-1",
        caller_deadline_at=NOW + timedelta(minutes=5),
    )

    assert profiler.context is not None
    assert profiler.context.prior_execution_facts is not None
    assert profiler.context.prior_execution_facts.episode_id == "episode-0"
