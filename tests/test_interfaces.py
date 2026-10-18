from datetime import datetime, timedelta, timezone

import pytest

from traceoffload.contracts import (
    AgentRole,
    AttemptRole,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    Component,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    EffectScope,
    ExecutionMode,
    ExecutionRequest,
    ExecutionStatus,
    FeedbackSnapshot,
    InvocationContext,
    ModelMessage,
    ModelParameters,
    ModelRef,
    ModelRequest,
    ModelTarget,
    ModelTargetKind,
    MessageRole,
    NodeContext,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    RouteOption,
    RoutePolicy,
    StateAccessMode,
    StateContext,
    VerifierPlan,
    WorkflowContext,
    WorkflowGoal,
    validate_execution_link,
    validate_profile_link,
    validate_route_link,
)
from traceoffload.fakes import FakeExecutor, FakeModelGateway, FakeProfiler, FakeRouter
from traceoffload.m3.integration import ToolRegistryEntry
from traceoffload.interfaces import GatewayError, GatewayTimeout

NOW = datetime.now(timezone.utc)
LLM = ModelRef(kind="model", id="minimax", version="v1")
SLM = ModelRef(kind="model", id="qwen", version="v1")


def test_read_only_tool_prefix_classifies_each_compound_shell_segment() -> None:
    entry = ToolRegistryEntry(
        tool_name="bash",
        default_effect_scope=EffectScope.REVERSIBLE_WRITE,
        read_only_command_prefixes=("pwd", "ls", "cat", "rg", "sed", "git diff"),
    )

    assert entry.effect_scope_for_action({"command": "pwd"}) == EffectScope.READ_ONLY
    assert entry.effect_scope_for_action({"command": "cat file.txt"}) == EffectScope.READ_ONLY
    assert entry.effect_scope_for_action({"command": "sed -n '1,10p' file.txt"}) == EffectScope.READ_ONLY
    assert (
        entry.effect_scope_for_action({"command": "pwd && ls -la"})
        == EffectScope.READ_ONLY
    )
    assert (
        entry.effect_scope_for_action({"command": "cat file.txt | rg 'needle'"})
        == EffectScope.READ_ONLY
    )
    assert (
        entry.effect_scope_for_action({"command": "git diff || pwd ; ls"})
        == EffectScope.READ_ONLY
    )
    assert (
        entry.effect_scope_for_action({"command": "pwd ; rm file.txt"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "pwd && rm file.txt"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "cat file.txt | tee copy.txt"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "cat file.txt > copy.txt"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "sed -i 's/a/b/' file.txt"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "git diff --output=change.patch"})
        == EffectScope.REVERSIBLE_WRITE
    )
    assert (
        entry.effect_scope_for_action({"command": "pwd &&"})
        == EffectScope.REVERSIBLE_WRITE
    )


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


def _context(
    expected_effect: EffectRisk = EffectRisk.NONE,
    recovery: RecoveryLevel = RecoveryLevel.NOT_NEEDED,
    dependency: DownstreamDependency = DownstreamDependency.FULL,
) -> InvocationContext:
    return InvocationContext(
        schema_version="1.0",
        message_id="message-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW,
        workflow_run_id="workflow-1",
        workflow_plan_ref={"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
        workflow_plan_revision=0,
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        request_ref=CapabilityRef(
            kind="request",
            id="request-1",
            version="v1",
            scope=CapabilityScope.PROFILE_READ,
            audience=Component.M1,
            expires_at=NOW + timedelta(hours=1),
            binding=CapabilityBinding(
                workflow_run_id="workflow-1",
                workflow_plan_ref={"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
                workflow_plan_revision=0,
                episode_id="episode-1",
                logical_invocation_id="invocation-1",
            ),
        ),
        workflow_context=WorkflowContext(
            context_version="1.0",
            agent_id="mini-swe-agent",
            agent_role=AgentRole.EXECUTOR,
            workflow_goal=WorkflowGoal(goal_text="repair the repository"),
            node_context=NodeContext(node_id="node-1", node_type="executor", step_index=0),
            downstream_contract=DownstreamContract(
                consumer_role=AgentRole.FINAL_RESPONDER,
                dependency_level=dependency,
            ),
            state_context=StateContext(access_mode=StateAccessMode.NONE, expected_effect=expected_effect),
            recovery_capability=recovery,
        ),
        caller_deadline_at=NOW + timedelta(minutes=5),
    )


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


def _snapshot() -> FeedbackSnapshot:
    return FeedbackSnapshot(
        schema_version="1.0",
        message_id="message-snapshot-1",
        producer_ref=ProducerRef(component=Component.M4, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=5),
        snapshot_version="snapshot-1",
        safety_calibration_ref={
            "kind": "calibration",
            "id": "safety",
            "version": "calibration-v1",
        },
    )


def _empty_verifier() -> VerifierPlan:
    return VerifierPlan(
        verifier_plan_id="verifier-none",
        version="v1",
        overall_timeout_ms=1000,
        empty_plan_reason="large_model_direct_no_guard",
    )


def _option(option_id: str, model_ref: ModelRef) -> RouteOption:
    return RouteOption(
        option_id=option_id,
        target=ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=model_ref),
        execution_mode=ExecutionMode.DIRECT,
        verifier_plan=_empty_verifier(),
        available=True,
    )


def _profile() -> object:
    return FakeProfiler().build_profile(_context(), _request())


def _execution_request(route) -> ExecutionRequest:
    request_ref = CapabilityRef(
        kind="request",
        id="request-exec",
        version="v1",
        scope=CapabilityScope.EXECUTION_READ,
        audience=Component.M3,
        content_hash="sha256:request-exec",
        expires_at=NOW + timedelta(hours=1),
        binding=CapabilityBinding(
            workflow_run_id=route.workflow_run_id,
            workflow_plan_ref=route.workflow_plan_ref,
            workflow_plan_revision=route.workflow_plan_revision,
            episode_id=route.episode_id,
            logical_invocation_id=route.logical_invocation_id,
            target_model_ref=route.target.model_ref,
        ),
    )
    return ExecutionRequest(
        schema_version="1.0",
        message_id="message-exec-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=1),
        execution_request_id="execution-1",
        route_plan=route,
        request_ref=request_ref,
        safety_calibration_ref={
            "kind": "calibration",
            "id": "safety",
            "version": route.decision_meta.feedback_version,
        },
        deadline_at=NOW + timedelta(minutes=1),
        idempotency_key="idempotency-1",
    )


class TestFakeModelGateway:
    def test_complete_is_deterministic_and_counts_calls(self) -> None:
        gateway = FakeModelGateway()
        target = ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LLM)

        first = gateway.complete(
            target, _request(), model_attempt_id="attempt-1", deadline_at=NOW
        )
        second = gateway.complete(
            target, _request(), model_attempt_id="attempt-1", deadline_at=NOW
        )

        assert first == second
        assert gateway.calls == 2
        assert first.response.model_attempt_id == "attempt-1"
        assert first.response.output_text == "fake-minimax"

    def test_metrics_and_cost_follow_target_kind(self) -> None:
        gateway = FakeModelGateway()
        large = gateway.complete(
            ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LLM),
            _request(),
            model_attempt_id="a",
            deadline_at=NOW,
        )
        small = gateway.complete(
            ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=SLM),
            _request(),
            model_attempt_id="b",
            deadline_at=NOW,
        )

        assert large.cost == 1.0
        assert small.cost == 0.1
        assert large.input_tokens == 10
        assert large.output_tokens == 20
        assert large.latency_ms == 5

    def test_configured_failure_is_raised(self) -> None:
        gateway = FakeModelGateway(fail_with=GatewayTimeout("provider timeout"))

        with pytest.raises(GatewayTimeout):
            gateway.complete(
                ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LLM),
                _request(),
                model_attempt_id="attempt-1",
                deadline_at=NOW,
            )

    def test_gateway_error_is_a_known_exception_type(self) -> None:
        assert issubclass(GatewayTimeout, GatewayError)


class TestFakeProfiler:
    def test_build_profile_is_traceable(self) -> None:
        context = _context()
        profile = FakeProfiler().build_profile(context, _request())

        validate_profile_link(context, profile)
        assert "|" in profile.descriptor_key

    def test_unknown_values_are_recorded_uncertain(self) -> None:
        context = _context(
            expected_effect=EffectRisk.UNKNOWN,
            recovery=RecoveryLevel.UNKNOWN,
            dependency=DownstreamDependency.UNKNOWN,
        )
        profile = FakeProfiler().build_profile(context, _request())

        paths = {field.field_path for field in profile.uncertain_fields}
        assert {"effect_risk", "recovery_level", "downstream_dependency", "allowed_targets"} <= paths

    def test_build_profile_is_deterministic(self) -> None:
        profiler = FakeProfiler()

        assert profiler.build_profile(_context(), _request()) == profiler.build_profile(
            _context(), _request()
        )


class TestFakeRouter:
    def test_decides_large_model_direct(self) -> None:
        profile = _profile()
        route = FakeRouter(large_model=LLM).decide(profile, (), _policy(), _snapshot())

        validate_route_link(profile, route)
        assert route.target.model_ref == LLM
        assert route.target.kind == ModelTargetKind.LARGE_MODEL
        assert route.execution_mode == ExecutionMode.DIRECT
        assert route.verifier_plan.steps == ()
        assert route.fallback_plan is None
        assert route.decision_meta.feedback_version == "calibration-v1"
        assert route.decision_meta.policy_version == "policy-1"
        assert route.experience_key == profile.descriptor_key

    def test_prefers_available_large_direct_option(self) -> None:
        profile = _profile()
        options = (
            _option("small-1", SLM),
            RouteOption(
                option_id="large-1",
                target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LLM),
                execution_mode=ExecutionMode.DIRECT,
                verifier_plan=_empty_verifier(),
                available=True,
            ),
        )

        route = FakeRouter(large_model=LLM).decide(profile, options, _policy(), None)

        assert route.target.model_ref == LLM

    def test_decide_is_deterministic(self) -> None:
        router = FakeRouter(large_model=LLM)
        profile = _profile()

        assert router.decide(profile, (), _policy(), None) == router.decide(
            profile, (), _policy(), None
        )


class TestFakeExecutor:
    def test_returns_terminal_success_result(self) -> None:
        profile = _profile()
        route = FakeRouter(large_model=LLM).decide(profile, (), _policy(), _snapshot())
        request = _execution_request(route)

        result = FakeExecutor().execute_step(request)

        validate_execution_link(request, result)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.accepted_output_ref is not None
        assert result.accepted_model_attempt_id == result.model_attempts[0].model_attempt_id
        assert result.model_attempts[0].role == AttemptRole.LARGE_MODEL_DIRECT


class TestFakePipeline:
    def test_fake_only_large_direct_pipeline_yields_execution_result(self) -> None:
        context = _context()
        request = _request()

        profile = FakeProfiler().build_profile(context, request)
        validate_profile_link(context, profile)

        route = FakeRouter(large_model=LLM).decide(profile, (), _policy(), _snapshot())
        validate_route_link(profile, route)

        execution_request = _execution_request(route)
        result = FakeExecutor().execute_step(execution_request)
        validate_execution_link(execution_request, result)

        assert result.route_plan_id == route.route_plan_id
        assert result.accepted_output_ref is not None
