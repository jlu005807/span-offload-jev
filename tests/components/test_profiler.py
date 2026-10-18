from datetime import datetime, timedelta, timezone

from traceoffload.contracts import (
    AgentRole,
    CapabilityBinding,
    CapabilityRef,
    CapabilityScope,
    Component,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    EffectScope,
    GuardRef,
    InvocationContext,
    MessageRole,
    ModelMessage,
    ModelParameters,
    ModelRequest,
    NodeContext,
    OutputContract,
    OutputKind,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    StateAccessMode,
    StateContext,
    ToolDefinition,
    UncertaintyReason,
    VersionedRef,
    WorkflowContext,
    WorkflowGoal,
    validate_profile_link,
)
from traceoffload.m1.profiler import DESCRIPTOR_SCHEMA_VERSION, RuntimeProfiler

NOW = datetime.now(timezone.utc)


def _lineage() -> dict[str, object]:
    return {
        "workflow_run_id": "workflow-1",
        "workflow_plan_ref": {"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
        "workflow_plan_revision": 0,
        "episode_id": "episode-1",
        "logical_invocation_id": "invocation-1",
    }


def _context(
    *,
    expected_effect: EffectRisk = EffectRisk.NONE,
    recovery: RecoveryLevel = RecoveryLevel.NOT_NEEDED,
    dependency: DownstreamDependency = DownstreamDependency.FULL,
    output_contract: "OutputContract | str" = "none",
    available_tools: tuple[VersionedRef, ...] = (),
    verifier_catalog: tuple[GuardRef, ...] = (),
) -> InvocationContext:
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
            node_context=NodeContext(node_id="node-1", node_type="executor", step_index=0),
            downstream_contract=DownstreamContract(
                consumer_role=AgentRole.FINAL_RESPONDER,
                dependency_level=dependency,
            ),
            state_context=StateContext(
                access_mode=StateAccessMode.NONE,
                expected_effect=expected_effect,
            ),
            output_contract=output_contract,
            verifier_catalog=verifier_catalog,
            available_tools=available_tools,
            recovery_capability=recovery,
        ),
        caller_deadline_at=NOW + timedelta(minutes=5),
    )


def _request(*, tool_definitions: tuple[ToolDefinition, ...] = ()) -> ModelRequest:
    return ModelRequest(
        request_version="1.0",
        messages=(
            ModelMessage(role=MessageRole.SYSTEM, content="You are an agent."),
            ModelMessage(role=MessageRole.USER, content="Fix the bug."),
        ),
        input_text="Fix the bug.",
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            data_policy_ref=VersionedRef(kind="policy", id="default", version="v1")
        ),
        tool_definitions=tool_definitions,
    )


def _bash_tool(version: str = "v3") -> ToolDefinition:
    return ToolDefinition(
        tool_name="bash",
        schema_ref=VersionedRef(kind="tool_schema", id="bash", version=version),
        effect_scope=EffectScope.READ_ONLY,
    )


def _profile(context: InvocationContext, request: ModelRequest):
    return RuntimeProfiler().build_profile(context, request)


class TestRuntimeProfiler:
    def test_read_only_input_produces_traceable_profile(self) -> None:
        context = _context()
        request = _request()

        profile = _profile(context, request)

        validate_profile_link(context, profile)
        assert profile.descriptor.estimated_input_tokens > 0
        assert profile.descriptor.output_contract == "none"
        assert profile.descriptor.allowed_targets == "unknown"
        assert profile.descriptor.downstream_dependency == DownstreamDependency.FULL
        assert profile.descriptor.effect_risk == EffectRisk.NONE
        assert profile.descriptor.recovery_level == RecoveryLevel.NOT_NEEDED
        assert profile.descriptor.available_verifiers == ()

    def test_reversible_write_risk_is_preserved(self) -> None:
        profile = _profile(
            _context(expected_effect=EffectRisk.REVERSIBLE_WRITE),
            _request(),
        )

        assert profile.descriptor.effect_risk == EffectRisk.REVERSIBLE_WRITE
        paths = {field.field_path for field in profile.uncertain_fields}
        assert "descriptor.effect_risk" not in paths

    def test_unknown_risk_stays_unknown(self) -> None:
        profile = _profile(
            _context(expected_effect=EffectRisk.UNKNOWN),
            _request(),
        )

        assert profile.descriptor.effect_risk == EffectRisk.UNKNOWN
        paths = {field.field_path for field in profile.uncertain_fields}
        assert "descriptor.effect_risk" in paths
        reason = next(
            field for field in profile.uncertain_fields if field.field_path == "descriptor.effect_risk"
        )
        assert reason.reason == UncertaintyReason.MISSING

    def test_missing_fields_are_recorded_uncertain(self) -> None:
        profile = _profile(
            _context(
                dependency=DownstreamDependency.UNKNOWN,
                recovery=RecoveryLevel.UNKNOWN,
            ),
            _request(),
        )

        assert profile.descriptor.downstream_dependency == DownstreamDependency.UNKNOWN
        assert profile.descriptor.recovery_level == RecoveryLevel.UNKNOWN
        paths = {field.field_path for field in profile.uncertain_fields}
        assert "descriptor.downstream_dependency" in paths
        assert "descriptor.recovery_level" in paths

    def test_all_unknown_input_never_downgrades_risk(self) -> None:
        profile = _profile(
            _context(
                expected_effect=EffectRisk.UNKNOWN,
                recovery=RecoveryLevel.UNKNOWN,
                dependency=DownstreamDependency.UNKNOWN,
            ),
            _request(),
        )

        assert profile.descriptor.effect_risk == EffectRisk.UNKNOWN
        assert profile.descriptor.recovery_level == RecoveryLevel.UNKNOWN
        assert profile.descriptor.downstream_dependency == DownstreamDependency.UNKNOWN

    def test_allowed_targets_are_unknown_when_policy_is_not_observable(self) -> None:
        profile = _profile(_context(), _request())

        assert profile.descriptor.allowed_targets == "unknown"
        assert {"descriptor.allowed_targets"} <= {
            field.field_path for field in profile.uncertain_fields
        }

    def test_descriptor_key_contains_schema_output_and_tool_versions(self) -> None:
        contract = OutputContract(
            contract_id="patch",
            version="v2",
            output_kind=OutputKind.CODE_PATCH,
        )
        context = _context(
            output_contract=contract,
            available_tools=(VersionedRef(kind="tool_schema", id="bash", version="v3"),),
        )
        request = _request(tool_definitions=(_bash_tool(),))

        profile = _profile(context, request)

        assert profile.descriptor_key == (
            f"descriptor/v{DESCRIPTOR_SCHEMA_VERSION}|subtask:tool_argument_generation|"
            "output:v2|tools:tool_schema:v3|risk:none|downstream:full|"
            "verifiers:none|recovery:not_needed"
        )

    def test_output_contract_version_changes_the_key(self) -> None:
        v1 = OutputContract(
            contract_id="patch",
            version="v1",
            output_kind=OutputKind.CODE_PATCH,
        )
        v2 = v1.model_copy(update={"version": "v2"})

        key_v1 = _profile(_context(output_contract=v1), _request()).descriptor_key
        key_v2 = _profile(_context(output_contract=v2), _request()).descriptor_key

        assert key_v1 != key_v2

    def test_estimated_input_tokens_are_derived_from_request(self) -> None:
        request = _request()

        profile = _profile(_context(), request)

        total_chars = sum(len(message.content) for message in request.messages) + len(
            request.input_text or ""
        )
        assert profile.descriptor.estimated_input_tokens == total_chars // 4

    def test_subtask_classification_follows_tools_and_contract(self) -> None:
        tool_profile = _profile(_context(), _request(tool_definitions=(_bash_tool(),)))
        assert tool_profile.descriptor.subtask_type == "tool_argument_generation"

        observed_tool = _request(tool_definitions=(_bash_tool(),)).model_copy(
            update={
                "messages": (
                    ModelMessage(role=MessageRole.USER, content="Run a check."),
                    ModelMessage(role=MessageRole.TOOL, content="<returncode>0</returncode>"),
                )
            }
        )
        interpreted = _profile(_context(), observed_tool)
        assert interpreted.descriptor.subtask_type == "tool_result_interpretation"

        json_contract = OutputContract(
            contract_id="out",
            version="v1",
            output_kind=OutputKind.JSON,
        )
        structured = _profile(_context(output_contract=json_contract), _request())
        assert structured.descriptor.subtask_type == "structured_generation"

        unclassified = _profile(_context(), _request())
        assert unclassified.descriptor.subtask_type == "unknown"
        assert {"descriptor.subtask_type"} <= {
            field.field_path for field in unclassified.uncertain_fields
        }

    def test_build_profile_is_pure_and_deterministic(self) -> None:
        context = _context()
        request = _request()

        first = _profile(context, request)
        second = _profile(context, request)

        assert first == second

    def test_last_observation_classification(self) -> None:
        context = _context()

        def observed(content: str) -> ModelRequest:
            return _request().model_copy(
                update={
                    "messages": (
                        ModelMessage(role=MessageRole.USER, content="Run a check."),
                        ModelMessage(role=MessageRole.TOOL, content=content),
                    )
                }
            )

        assert _profile(context, observed("Ran 2 tests: 2 passed, 0 failed")).last_observation_class == "test_feedback"
        assert _profile(context, observed("2 passed")).last_observation_class == "test_feedback"
        assert _profile(context, observed("FAILED")).last_observation_class == "test_feedback"
        assert _profile(context, observed("Traceback (most recent call last): boom")).last_observation_class == "error_trace"
        assert _profile(context, observed("Error: division by zero")).last_observation_class == "error_trace"
        assert _profile(context, observed("open /testbed/foo.py\n\ndef main():\n    pass")).last_observation_class == "code_view"
        assert _profile(context, observed("random tool output")).last_observation_class == "plain"
        assert _profile(context, observed("")).last_observation_class == "plain"
        # 无工具消息 -> unknown；优先级：test_feedback > error_trace > code_view
        assert _profile(context, _request()).last_observation_class == "unknown"
        assert _profile(context, observed("ERROR: 1 passed, 3 failed")).last_observation_class == "test_feedback"
        assert _profile(context, observed("open /testbed/foo.py; raise Error: bad")).last_observation_class == "error_trace"
