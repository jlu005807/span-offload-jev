import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from traceoffload.contracts import (
    ActualEffect,
    AgentRole,
    AttemptRole,
    AttemptStatus,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    CommitStatus,
    Component,
    ContractError,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ExecutionSummary,
    GuardRef,
    InvocationContext,
    ModelAttempt,
    ModelMessage,
    ModelParameters,
    ModelRef,
    ModelRequest,
    ModelTarget,
    ModelTargetKind,
    MessageRole,
    NodeContext,
    OutcomeEvent,
    OutcomeLabel,
    OutcomeScope,
    PassCondition,
    PassConditionKind,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    ResumeStatus,
    RollbackStatus,
    RoutePlan,
    RuntimeProfile,
    SpanDescriptor,
    SpanRecord,
    StateContext,
    StateAccessMode,
    StateOutcome,
    UncertaintyReason,
    UncertainField,
    VerifierPlan,
    VerifierStage,
    VerifierStep,
    WorkflowContext,
    WorkflowGoal,
    validate_execution_link,
    validate_outcome_link,
    validate_profile_link,
)


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
PLAN_REF = {"kind": "workflow_plan", "id": "plan-1", "version": "v1"}


def test_execution_and_verifier_modes_have_no_compatibility_aliases() -> None:
    assert set(ExecutionMode.__members__) == {"DIRECT", "GUARDED_SLM"}
    assert set(VerifierStage.__members__) == {"PRE_COMMIT", "POST_CHECK", "RECOVERY"}


def _lineage() -> dict[str, object]:
    return {
        "workflow_run_id": "workflow-1",
        "workflow_plan_ref": PLAN_REF,
        "workflow_plan_revision": 0,
        "episode_id": "episode-1",
        "logical_invocation_id": "invocation-1",
    }


def _request_capability(
    scope: CapabilityScope,
    audience: Component,
    *,
    target: ModelRef | None = None,
) -> CapabilityRef:
    return CapabilityRef(
        kind="request",
        id="request-1",
        version="v1",
        scope=scope,
        audience=audience,
        content_hash="sha256:request-1" if scope == CapabilityScope.EXECUTION_READ else None,
        expires_at=NOW + timedelta(hours=1),
        binding=CapabilityBinding(**_lineage(), target_model_ref=target),
    )


def _workflow_context() -> WorkflowContext:
    return WorkflowContext(
        context_version="1.0",
        agent_id="mini-swe-agent",
        agent_role=AgentRole.EXECUTOR,
        agent_version="pilot-1",
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
        output_contract="none",
        recovery_capability=RecoveryLevel.NOT_NEEDED,
    )


def _invocation() -> InvocationContext:
    return InvocationContext(
        schema_version="1.0",
        message_id="message-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW,
        **_lineage(),
        request_ref=_request_capability(CapabilityScope.PROFILE_READ, Component.M1),
        workflow_context=_workflow_context(),
        caller_deadline_at=NOW + timedelta(minutes=5),
    )


def _empty_verifier() -> VerifierPlan:
    return VerifierPlan(
        verifier_plan_id="verifier-none",
        version="v1",
        overall_timeout_ms=1000,
        empty_plan_reason="large_model_direct_no_guard",
    )


def _route() -> RoutePlan:
    large_ref = ModelRef(kind="model", id="llm", version="v1")
    return RoutePlan(
        schema_version="1.0",
        message_id="message-route-1",
        producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=1),
        **_lineage(),
        route_plan_id="route-1",
        target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=large_ref),
        execution_mode=ExecutionMode.DIRECT,
        verifier_plan=_empty_verifier(),
        timeout_ms=1000,
        experience_key="descriptor/llm/direct",
        decision_meta={
            "policy_version": "policy-1",
            "feedback_version": "feedback-1",
            "reason_code": "policy_baseline",
            "is_exploration": False,
            "selection_probability": 1.0,
        },
    )


def _execution_result(route: RoutePlan) -> ExecutionResult:
    model_ref = route.target.model_ref
    attempt = ModelAttempt(
        model_attempt_id="attempt-1",
        role=AttemptRole.LARGE_MODEL_DIRECT,
        model_ref=model_ref,
        status=AttemptStatus.SUCCESS,
        result_ref=GuardRef(kind="model_response", id="response-1", version="v1"),
        input_tokens=10,
        output_tokens=20,
        latency_ms=50,
        cost=0.01,
    )
    accepted_output = CapabilityRef(
        kind="model_response",
        id="accepted-1",
        version="v1",
        scope=CapabilityScope.ACCEPTED_OUTPUT,
        audience=Component.AGENT_ADAPTER,
        expires_at=NOW + timedelta(hours=1),
        binding=CapabilityBinding(**_lineage(), target_model_ref=model_ref),
    )
    return ExecutionResult(
        schema_version="1.0",
        message_id="message-result-1",
        producer_ref=ProducerRef(component=Component.M3, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=2),
        **_lineage(),
        route_plan_id=route.route_plan_id,
        status=ExecutionStatus.SUCCESS,
        model_attempts=(attempt,),
        state_outcome=StateOutcome(
            effect=ActualEffect.NONE,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        ),
        accepted_model_attempt_id=attempt.model_attempt_id,
        accepted_output_ref=accepted_output,
        latency_ms=60,
    )


def test_contracts_round_trip_as_json() -> None:
    invocation = _invocation()
    payload = invocation.model_dump(mode="json")

    assert payload["schema_version"] == "1.0"
    assert InvocationContext.model_validate_json(json.dumps(payload)) == invocation

    request = ModelRequest(
        request_version="1.0",
        messages=(ModelMessage(role=MessageRole.USER, content="hello"),),
        input_text="hello",
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            data_policy_ref={"kind": "policy", "id": "default", "version": "v1"}
        ),
    )
    assert ModelRequest.model_validate_json(request.model_dump_json()) == request


def test_unsupported_version_and_unknown_fields_are_rejected() -> None:
    payload = _invocation().model_dump(mode="json")

    payload["schema_version"] = "9.0"
    with pytest.raises(ValidationError):
        InvocationContext.model_validate_json(json.dumps(payload))

    payload = _invocation().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        InvocationContext.model_validate_json(json.dumps(payload))


def test_illegal_ids_and_field_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GuardRef(kind="guard", id="contains space", version="v1")

    with pytest.raises(ValidationError):
        WorkflowGoal()

    with pytest.raises(ValidationError):
        UncertainField(field_path="descriptor.x", reason=UncertaintyReason.PREDICTED)

    with pytest.raises(ValidationError):
        VerifierStep(
            step_id="step-1",
            guard_ref=GuardRef(kind="guard", id="schema", version="v1"),
            stage=VerifierStage.PRE_COMMIT,
            subject="candidate_output",
            pass_condition=PassCondition(kind=PassConditionKind.SCHEMA_VALID),
            threshold=0.5,
            timeout_ms=100,
            on_pass="continue",
            on_failure="abort",
        )


def test_removed_pre_commit_only_execution_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExecutionMode("verify_before_commit")


def test_association_ids_are_validated_at_each_boundary() -> None:
    invocation = _invocation()
    profile = RuntimeProfile(
        schema_version="1.0",
        message_id="message-profile-1",
        producer_ref=ProducerRef(component=Component.M1, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=1),
        **_lineage(),
        request_ref=invocation.request_ref,
        descriptor_key="descriptor/v1",
        descriptor=SpanDescriptor(
            subtask_type="generation",
            estimated_input_tokens=10,
            output_contract="none",
            allowed_targets="unknown",
            downstream_dependency=DownstreamDependency.FULL,
            effect_risk=EffectRisk.NONE,
            recovery_level=RecoveryLevel.NOT_NEEDED,
        ),
        uncertain_fields=(
            UncertainField(field_path="effect_risk", reason=UncertaintyReason.MISSING),
        ),
    )
    validate_profile_link(invocation, profile)
    with pytest.raises(ContractError):
        validate_profile_link(
            invocation,
            profile.model_copy(update={"episode_id": "episode-other"}),
        )

    route = _route()
    with pytest.raises(ContractError):
        validate_profile_link(
            invocation,
            profile.model_copy(update={"logical_invocation_id": "other"}),
        )

    result = _execution_result(route)
    request_ref = _request_capability(
        CapabilityScope.EXECUTION_READ,
        Component.M3,
        target=route.target.model_ref,
    )
    execution_request = ExecutionRequest(
        schema_version="1.0",
        message_id="message-request-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=1),
        execution_request_id="execution-1",
        route_plan=route,
        request_ref=request_ref,
        safety_calibration_ref={"kind": "calibration", "id": "safety", "version": "feedback-1"},
        deadline_at=NOW + timedelta(minutes=1),
        idempotency_key="idempotency-1",
    )
    validate_execution_link(execution_request, result)
    with pytest.raises(ContractError):
        validate_execution_link(
            execution_request,
            result.model_copy(update={"route_plan_id": "route-other"}),
        )

    summary = ExecutionSummary(
        route_plan_id=route.route_plan_id,
        **_lineage(),
        status=result.status,
        model_attempts=(
            {
                "model_attempt_id": "attempt-1",
                "role": AttemptRole.LARGE_MODEL_DIRECT,
                "model_ref": route.target.model_ref,
                "status": AttemptStatus.SUCCESS,
                "input_tokens": 10,
                "output_tokens": 20,
                "latency_ms": 50,
                "cost": 0.01,
            },
        ),
        state_outcome=result.state_outcome,
        accepted_model_attempt_id="attempt-1",
        latency_ms=60,
    )
    record = SpanRecord(
        schema_version="1.0",
        message_id="message-span-1",
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=3),
        runtime_projection={"descriptor_key": profile.descriptor_key},
        route_plan=route,
        execution_summary=summary,
    )
    event = OutcomeEvent(
        schema_version="1.0",
        message_id="message-event-1",
        producer_ref=ProducerRef(component=Component.EVALUATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=4),
        event_id="event-1",
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        scope=OutcomeScope.CANDIDATE,
        logical_invocation_id="invocation-1",
        model_attempt_id="attempt-1",
        label=OutcomeLabel.SUCCESS,
        evaluator_ref=GuardRef(kind="evaluator", id="swebench", version="v1"),
        observed_at=NOW + timedelta(seconds=4),
        revision=1,
    )
    validate_outcome_link(record, event)
    with pytest.raises(ContractError):
        validate_outcome_link(record, event.model_copy(update={"model_attempt_id": "missing"}))
