"""Deterministic, memory-only fakes for the TraceOffload component seam.

The fakes never touch the network, Docker or any model provider.  They exist
so that the Coordinator, adapters and M1-M4 can be developed and verified
against the contract types before the real workload is reachable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from pydantic import AwareDatetime

from .contracts import (
    ActualEffect,
    AttemptRole,
    AttemptStatus,
    Attribution,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    CommitStatus,
    Component,
    DownstreamDependency,
    EffectRisk,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    FinishReason,
    GuardOutcome,
    GuardRef,
    GuardResult,
    InvocationContext,
    ModelAttempt,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseToolCall,
    ModelTarget,
    ModelTargetKind,
    OutcomeEvent,
    OutcomeLabel,
    OutcomeScope,
    OutputContract,
    ParseStatus,
    ProducerRef,
    RecoveryLevel,
    ResumeStatus,
    RollbackStatus,
    RoutePlan,
    RouteOption,
    RoutePolicy,
    RuntimeProfile,
    SpanDescriptor,
    SpanRecord,
    StateOutcome,
    ToolExecutionResult,
    ToolExecutionStatus,
    UncertaintyReason,
    UncertainField,
    VerifierPlan,
    VersionedRef,
    WorkflowContext,
)
from .interfaces import ModelCallResult
from .storage import RequestResponseStore


class FakeVerifier:
    """Deterministic verifier: outcome per guard id, default pass."""

    def __init__(
        self,
        *,
        outcomes: dict[str, GuardOutcome] | None = None,
        default: GuardOutcome = GuardOutcome.PASS,
    ) -> None:
        self._outcomes = outcomes or {}
        self._default = default
        self.calls = 0
        #: 每次 `check` 收到的 span 身份（按调用顺序）。**测试探针** ——
        #: executor 有没有把「现在是哪个 span」传下来，只能从这里看。
        self.span_contexts: list[Any] = []

    def check(
        self,
        step,
        *,
        candidate: ModelResponse,
        model_attempt_id: str,
        span_context: Any = None,
    ) -> GuardResult:
        self.calls += 1
        self.span_contexts.append(span_context)
        outcome = self._outcomes.get(step.guard_ref.id, self._default)
        return GuardResult(
            guard_ref=step.guard_ref,
            stage=step.stage,
            subject_model_attempt_id=model_attempt_id,
            outcome=outcome,
            latency_ms=1,
            cost=0.0,
        )

    def check_tool_result(
        self,
        step,
        *,
        result: ToolExecutionResult,
        model_attempt_id: str,
    ) -> GuardResult:
        self.calls += 1
        configured = self._outcomes.get(step.guard_ref.id)
        outcome = configured or (
            self._default
            if self._default != GuardOutcome.PASS
            else GuardOutcome.PASS
            if result.status == ToolExecutionStatus.SUCCESS and result.returncode in (None, 0)
            else GuardOutcome.REJECT
        )
        return GuardResult(
            guard_ref=step.guard_ref,
            stage=step.stage,
            subject_model_attempt_id=model_attempt_id,
            subject_tool_call_id=result.tool_call_id,
            outcome=outcome,
            latency_ms=1,
            cost=0.0,
        )


class FakeEvaluator:
    """Synthetic, deterministic evaluator for the fake workload.

    Labels follow the execution status: accepted, successful executions are
    labelled success; rejected or failed executions are labelled failure.
    """

    def __init__(
        self,
        *,
        observed_at: AwareDatetime | None = None,
        evaluator_id: str = "fake-evaluator",
    ) -> None:
        self._observed_at = observed_at
        self._evaluator_id = evaluator_id

    def _now(self) -> AwareDatetime:
        return self._observed_at or datetime.now(timezone.utc)

    def evaluate_span(
        self,
        record: SpanRecord,
        *,
        accepted_attempt_id: str | None,
    ) -> tuple[OutcomeEvent, ...]:
        route = record.route_plan
        summary = record.execution_summary
        success = summary.status == ExecutionStatus.SUCCESS
        label = OutcomeLabel.SUCCESS if success else OutcomeLabel.FAILURE
        if accepted_attempt_id is None and summary.model_attempts:
            accepted_attempt_id = summary.model_attempts[0].model_attempt_id
        now = self._now()
        common = {
            "schema_version": "1.0",
            "producer_ref": ProducerRef(component=Component.EVALUATOR, version="0.1.0"),
            "produced_at": now,
            "workflow_run_id": route.workflow_run_id,
            "episode_id": route.episode_id,
            "logical_invocation_id": route.logical_invocation_id,
            "evaluator_ref": GuardRef(
                kind="evaluator", id=self._evaluator_id, version="v1"
            ),
            "observed_at": now,
            "revision": 1,
        }
        events = [
            OutcomeEvent(
                **common,
                message_id=f"message-candidate-{route.route_plan_id}",
                event_id=f"candidate-{route.route_plan_id}",
                scope=OutcomeScope.CANDIDATE,
                model_attempt_id=accepted_attempt_id,
                label=label,
            ),
            OutcomeEvent(
                **common,
                message_id=f"message-downstream-{route.route_plan_id}",
                event_id=f"downstream-{route.route_plan_id}",
                scope=OutcomeScope.DOWNSTREAM,
                label=label,
            ),
            OutcomeEvent(
                **common,
                message_id=f"message-episode_final-{route.route_plan_id}",
                event_id=f"episode_final-{route.route_plan_id}",
                scope=OutcomeScope.EPISODE_FINAL,
                label=label,
            ),
        ]
        return tuple(events)

    def evaluate_workflow_final(
        self,
        records: Sequence[SpanRecord],
        *,
        attribution: Sequence[Attribution] | None = None,
    ) -> OutcomeEvent:
        if attribution is None and len(records) != 1:
            raise ValueError("multi-episode workflow_final requires attribution")
        first = records[0]
        route = first.route_plan
        success = all(
            record.execution_summary.status == ExecutionStatus.SUCCESS
            for record in records
        )
        now = self._now()
        return OutcomeEvent(
            schema_version="1.0",
            message_id=f"message-workflow_final-{route.workflow_run_id}",
            producer_ref=ProducerRef(component=Component.EVALUATOR, version="0.1.0"),
            produced_at=now,
            event_id=f"workflow_final-{route.workflow_run_id}",
            workflow_run_id=route.workflow_run_id,
            episode_id=None,
            scope=OutcomeScope.WORKFLOW_FINAL,
            logical_invocation_id=(
                None if attribution is not None else route.logical_invocation_id
            ),
            label=OutcomeLabel.SUCCESS if success else OutcomeLabel.FAILURE,
            evaluator_ref=GuardRef(
                kind="evaluator", id=self._evaluator_id, version="v1"
            ),
            attribution=tuple(attribution) if attribution is not None else None,
            observed_at=now,
            revision=1,
        )


class FakeMiniSweModel:
    """mini-swe-agent Model-protocol stand-in that must never be queried directly."""

    config = None

    def query(self, messages, **kwargs):
        raise AssertionError("underlying model must not be queried directly")

    def format_message(self, **kwargs):
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {
                "role": "tool",
                "content": "".join(output.get("output", "") for output in outputs),
                "extra": {"tool_call_id": action.get("tool_call_id", "") for action in outputs},
            }
        ]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FakeAgentEnv:
    """mini-swe-agent Environment stand-in; records executed actions."""

    config = None

    def __init__(self):
        self.executed = []

    def execute(self, action, cwd=""):
        self.executed.append(action)
        return {
            "returncode": 0,
            "output": action.get("command", ""),
            "exception_info": None,
        }

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


def _empty_verifier() -> VerifierPlan:
    return VerifierPlan(
        verifier_plan_id="verifier-none",
        version="v1",
        overall_timeout_ms=1000,
        empty_plan_reason="large_model_direct_no_guard",
    )


def _neutral_state() -> StateOutcome:
    return StateOutcome(
        effect=ActualEffect.NONE,
        commit_status=CommitStatus.NOT_ATTEMPTED,
        recovery_required=False,
        rollback_status=RollbackStatus.NOT_NEEDED,
        resume_status=ResumeStatus.NOT_NEEDED,
    )


class FakeModelGateway:
    """Returns a fixed-shape completion for any target, deterministically."""

    def __init__(
        self,
        *,
        cost_map: dict[ModelTargetKind, float] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
        latency_ms: int = 5,
        fail_with: Exception | None = None,
        failures: dict[ModelTargetKind, Exception] | None = None,
    ) -> None:
        self._cost_map = cost_map or {
            ModelTargetKind.LARGE_MODEL: 1.0,
            ModelTargetKind.SMALL_MODEL: 0.1,
        }
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms
        self._fail_with = fail_with
        self._failures = failures or {}
        self.calls = 0

    def complete(
        self,
        target: ModelTarget,
        request: ModelRequest,
        *,
        model_attempt_id: str,
        deadline_at,
    ) -> ModelCallResult:
        self.calls += 1
        failure = self._failures.get(target.kind) or self._fail_with
        if failure is not None:
            raise failure
        response = ModelResponse(
            response_id=f"response-{model_attempt_id}",
            model_attempt_id=model_attempt_id,
            output_text=f"fake-{target.model_ref.id}",
            finish_reason=FinishReason.STOP,
            parse_status=ParseStatus.NOT_NEEDED,
            raw_response_ref=VersionedRef(
                kind="raw", id=f"raw-{model_attempt_id}", version="v1"
            ),
        )
        return ModelCallResult(
            response=response,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=self._latency_ms,
            cost=self._cost_map[target.kind],
        )


def _estimate_input_tokens(request: ModelRequest) -> int:
    return sum(len(message.content or "") for message in request.messages) + len(
        request.input_text or ""
    )


def _descriptor_key(workflow: WorkflowContext, descriptor: SpanDescriptor) -> str:
    output_version = (
        descriptor.output_contract.version
        if isinstance(descriptor.output_contract, OutputContract)
        else "none"
    )
    tool_version = "none"
    if workflow.available_tools:
        tool_version = ",".join(
            f"{ref.kind}:{ref.version}" for ref in workflow.available_tools
        )
    return f"fake-desc/schema:v1|output:{output_version}|tools:{tool_version}"


def _uncertain_fields(descriptor: SpanDescriptor) -> tuple[UncertainField, ...]:
    fields: list[UncertainField] = []
    if descriptor.allowed_targets == "unknown":
        fields.append(
            UncertainField(field_path="allowed_targets", reason=UncertaintyReason.MISSING)
        )
    if descriptor.downstream_dependency == DownstreamDependency.UNKNOWN:
        fields.append(
            UncertainField(
                field_path="downstream_dependency", reason=UncertaintyReason.MISSING
            )
        )
    if descriptor.effect_risk == EffectRisk.UNKNOWN:
        fields.append(
            UncertainField(field_path="effect_risk", reason=UncertaintyReason.MISSING)
        )
    if descriptor.recovery_level == RecoveryLevel.UNKNOWN:
        fields.append(
            UncertainField(field_path="recovery_level", reason=UncertaintyReason.MISSING)
        )
    return tuple(fields)


class FakeProfiler:
    """Mirrors the workflow context into a traceable runtime profile."""

    def build_profile(
        self,
        context: InvocationContext,
        request: ModelRequest,
    ) -> RuntimeProfile:
        workflow = context.workflow_context
        descriptor = SpanDescriptor(
            subtask_type="fake_generation",
            estimated_input_tokens=_estimate_input_tokens(request),
            output_contract=workflow.output_contract,
            allowed_targets="unknown",
            downstream_dependency=workflow.downstream_contract.dependency_level,
            effect_risk=workflow.state_context.expected_effect,
            available_verifiers=workflow.verifier_catalog,
            recovery_level=workflow.recovery_capability,
        )
        return RuntimeProfile(
            schema_version=context.schema_version,
            message_id=f"message-profile-{context.episode_id}",
            producer_ref=ProducerRef(component=Component.M1, version="0.1.0"),
            produced_at=context.produced_at,
            workflow_run_id=context.workflow_run_id,
            workflow_plan_ref=context.workflow_plan_ref,
            workflow_plan_revision=context.workflow_plan_revision,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            request_ref=context.request_ref,
            descriptor_key=_descriptor_key(workflow, descriptor),
            descriptor=descriptor,
            uncertain_fields=_uncertain_fields(descriptor),
        )


class FakeRouter:
    """Fixed large-model direct routing; never picks a small-model path."""

    def __init__(self, *, large_model: ModelRef) -> None:
        self._large_model = large_model

    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence[RouteOption],
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None = None,
        *,
        admission_facts=None,
    ) -> RoutePlan:
        selected = next(
            (
                option
                for option in options
                if option.available
                and option.target.kind == ModelTargetKind.LARGE_MODEL
                and option.execution_mode == ExecutionMode.DIRECT
            ),
            None,
        )
        if selected is not None:
            target = selected.target
            verifier_plan = selected.verifier_plan
        else:
            target = ModelTarget(
                kind=ModelTargetKind.LARGE_MODEL,
                model_ref=self._large_model,
            )
            verifier_plan = _empty_verifier()
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
            route_plan_id=f"route-{profile.logical_invocation_id}",
            target=target,
            execution_mode=ExecutionMode.DIRECT,
            verifier_plan=verifier_plan,
            timeout_ms=int(policy.budget.max_latency_ms),
            experience_key=profile.descriptor_key,
            decision_meta={
                "policy_version": policy.policy_version,
                "feedback_version": (
                    snapshot.safety_calibration_ref.version
                    if snapshot is not None
                    else "feedback-none"
                ),
                "reason_code": "fake_large_direct",
                "is_exploration": False,
                "selection_probability": 1.0,
            },
        )


class FakeExecutor:
    """Executes any direct route plan as a single large-model attempt.

    With ``status != success`` the executor returns a terminal failure result
    with no attempts and no accepted output.  When a ``store`` is provided the
    accepted response is registered there so that the Coordinator can resolve
    ``accepted_output_ref`` for the Agent, mirroring what real M3 does.
    """

    def __init__(
        self,
        *,
        output_text: str = "fake-executed-output",
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        store: RequestResponseStore | None = None,
        with_tool_call: bool = False,
    ) -> None:
        self._output_text = output_text
        self._status = status
        self._store = store
        self._with_tool_call = with_tool_call

    def execute_step(self, request: ExecutionRequest) -> ExecutionResult:
        route = request.route_plan
        attempt_id = f"attempt-{request.execution_request_id}"
        response_id = f"response-{attempt_id}"
        model_ref = route.target.model_ref
        accepted_ref = CapabilityRef(
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
        if self._status != ExecutionStatus.SUCCESS:
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
                status=self._status,
                state_outcome=_neutral_state(),
                latency_ms=0,
            )
        result_ref = GuardRef(kind="model_response", id=response_id, version="v1")
        tool_calls = ()
        if self._with_tool_call:
            tool_calls = (
                ModelResponseToolCall(
                    tool_call_id="tc-1",
                    tool_name="bash",
                    arguments={"command": "echo hello"},
                ),
            )
        response = ModelResponse(
            response_id=response_id,
            model_attempt_id=attempt_id,
            output_text=self._output_text,
            tool_calls=tool_calls,
            finish_reason=FinishReason.STOP,
            parse_status=ParseStatus.NOT_NEEDED,
            raw_response_ref=VersionedRef(kind="raw", id=f"raw-{response_id}", version="v1"),
        )
        attempt = ModelAttempt(
            model_attempt_id=attempt_id,
            role=AttemptRole.LARGE_MODEL_DIRECT,
            model_ref=model_ref,
            status=AttemptStatus.SUCCESS,
            result_ref=result_ref,
            response=response,
            input_tokens=10,
            output_tokens=20,
            latency_ms=5,
            cost=0.01,
        )
        if self._store is not None:
            self._store.save_response(accepted_ref, response, audit_refs=(result_ref,))
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
            status=ExecutionStatus.SUCCESS,
            model_attempts=(attempt,),
            state_outcome=_neutral_state(),
            accepted_model_attempt_id=attempt_id,
            accepted_output_ref=accepted_ref,
            latency_ms=5,
        )


__all__ = [
    "FakeAgentEnv",
    "FakeEvaluator",
    "FakeExecutor",
    "FakeMiniSweModel",
    "FakeModelGateway",
    "FakeProfiler",
    "FakeRouter",
    "FakeVerifier",
]
