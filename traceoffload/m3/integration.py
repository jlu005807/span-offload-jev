"""Coordinator-owned runtime integration sessions.

The Coordinator owns an :class:`IntegrationHub` rather than embedding a
particular agent framework.  Framework adapters publish normalized facts to an
``AdapterSession`` for one workflow: model invocations, tool execution,
verifier results, state observations, and workflow completion.  M1 consumes
only the pre-invocation ``WorkflowContext`` assembled by the session; events
that happen afterwards are retained for the next span and for trace output.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from ..clock import real_time
from ..contracts import (
    AgentRole,
    DownstreamContract,
    DownstreamDependency,
    EffectRisk,
    EffectScope,
    ExecutionResult,
    GuardRef,
    GuardResult,
    PriorExecutionFacts,
    PriorStateRecord,
    PriorToolOutcome,
    PriorVerificationRecord,
    ModelRequest,
    ModelResponse,
    NodeContext,
    OutputContract,
    RecoveryLevel,
    RoutePlan,
    StateAccessMode,
    StateContext,
    StateOutcome,
    ToolDefinition,
    UpstreamContextRef,
    VersionedRef,
    WorkflowContext,
    WorkflowGoal,
)
from ..contracts.records import SpanRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _jsonable(value: Any) -> Any:
    """Convert framework-owned values to a JSON-safe trace projection."""

    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _effect_risk(scope: EffectScope) -> EffectRisk:
    """Project a declared tool scope into the pre-routing state-risk enum."""

    return {
        EffectScope.READ_ONLY: EffectRisk.READ_ONLY,
        EffectScope.REVERSIBLE_WRITE: EffectRisk.REVERSIBLE_WRITE,
        EffectScope.IRREVERSIBLE_WRITE: EffectRisk.IRREVERSIBLE_WRITE,
        EffectScope.UNKNOWN: EffectRisk.UNKNOWN,
    }[scope]


_SHELL_PUNCTUATION = frozenset(";&|<>()")
_READ_ONLY_SEPARATORS = frozenset({"&&", "||", ";", "|"})


def _command_segments(command: str) -> tuple[tuple[str, ...], ...] | None:
    """Split a shell expression at safe control operators.

    Redirections, grouping, background execution, substitutions, and malformed
    expressions remain fail-closed. Ambiguous quoted or escaped punctuation
    may also be rejected conservatively because ``shlex`` removes quote metadata.
    """

    if "\n" in command or "\r" in command or "`" in command or "$(" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens:
        return None

    segments: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token in _READ_ONLY_SEPARATORS:
            if not current:
                return None
            segments.append(tuple(current))
            current = []
        elif token and set(token) <= _SHELL_PUNCTUATION:
            return None
        else:
            current.append(token)
    if not current:
        return None
    segments.append(tuple(current))
    return tuple(segments)


def _command_tokens_match(command_tokens: tuple[str, ...], prefix: str) -> bool:
    """Match one parsed command segment against a configured prefix."""

    try:
        prefix_tokens = shlex.split(prefix, posix=True)
    except ValueError:
        return False
    if not prefix_tokens or command_tokens[: len(prefix_tokens)] != tuple(prefix_tokens):
        return False
    if command_tokens[0] == "sed" and any(
        token == "--in-place"
        or token.startswith("--in-place=")
        or (
            token.startswith("-")
            and not token.startswith("--")
            and "i" in token[1:]
        )
        for token in command_tokens[1:]
    ):
        return False
    if any(
        token == "--output" or token.startswith("--output=")
        for token in command_tokens[1:]
    ):
        return False
    return True


def _command_is_read_only(command: str, prefixes: tuple[str, ...]) -> bool:
    """Require every command in a compound shell expression to be allowlisted."""

    segments = _command_segments(command)
    return bool(
        segments
        and prefixes
        and all(
            any(_command_tokens_match(segment, prefix) for prefix in prefixes)
            for segment in segments
        )
    )


@dataclass(frozen=True)
class ToolRegistryEntry:
    """Framework-neutral tool metadata supplied by an adapter or user config.

    ``default_effect_scope`` is deliberately conservative.  A generic shell
    command is not assumed to be read-only merely because it is named ``bash``.
    """

    tool_name: str
    interface: str = "unknown"
    input_schema: Mapping[str, Any] | None = None
    return_schema: Mapping[str, Any] | None = None
    permissions: tuple[str, ...] = ()
    default_effect_scope: EffectScope = EffectScope.UNKNOWN
    read_only_command_prefixes: tuple[str, ...] = ()
    verifier_command_prefixes: tuple[str, ...] = ()
    version: str = "v1"
    description: str | None = None

    def effect_scope_for_action(self, action: Mapping[str, Any]) -> EffectScope:
        command = action.get("command")
        if isinstance(command, str) and _command_is_read_only(
            command,
            self.read_only_command_prefixes,
        ):
            return EffectScope.READ_ONLY
        return self.default_effect_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "interface": self.interface,
            "input_schema": _jsonable(self.input_schema),
            "return_schema": _jsonable(self.return_schema),
            "permissions": list(self.permissions),
            "default_effect_scope": self.default_effect_scope.value,
            "read_only_command_prefixes": list(self.read_only_command_prefixes),
            "verifier_command_prefixes": list(self.verifier_command_prefixes),
            "version": self.version,
            "description": self.description,
        }


@dataclass(frozen=True)
class AdapterSessionSpec:
    """Static, pre-routing facts used to open one Coordinator adapter session."""

    workflow_run_id: str
    agent_id: str
    agent_version: str | None
    workflow_goal: WorkflowGoal
    tool_definitions: tuple[ToolDefinition, ...] = ()
    output_contract: Literal["none"] | OutputContract = "none"
    verifier_catalog: tuple[GuardRef, ...] = ()
    state_access_mode: StateAccessMode = StateAccessMode.NONE
    expected_effect: EffectRisk = EffectRisk.NONE
    recovery_capability: RecoveryLevel = RecoveryLevel.NOT_NEEDED
    downstream_dependency: DownstreamDependency = DownstreamDependency.FULL
    agent_role: AgentRole = AgentRole.EXECUTOR
    node_id: str = "root"
    node_type: str = "agent"
    parent_node_id: str | None = None
    framework_metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_registry: tuple[ToolRegistryEntry, ...] = ()


@dataclass
class RuntimeEvent:
    """One normalized adapter event retained in a session-local trace."""

    event_id: str
    sequence: int
    event_type: str
    occurred_at: datetime
    workflow_run_id: str
    episode_id: str | None
    logical_invocation_id: str | None
    tool_call_id: str | None
    model_attempt_id: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "workflow_run_id": self.workflow_run_id,
            "episode_id": self.episode_id,
            "logical_invocation_id": self.logical_invocation_id,
            "tool_call_id": self.tool_call_id,
            "model_attempt_id": self.model_attempt_id,
            "payload": _jsonable(self.payload),
        }


class AdapterSession:
    """Coordinator-owned collection and context assembly for one workflow."""

    def __init__(
        self,
        spec: AdapterSessionSpec,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._spec = spec
        self._clock = clock or _utcnow
        self._lock = RLock()
        self._events: list[RuntimeEvent] = []
        self._events_by_id: dict[str, RuntimeEvent] = {}
        self._pending_invocations: list[str] = []
        self._opened_spans: set[str] = set()
        self._response_attempts: dict[str, str | None] = {}
        self._recovery_responses: list[tuple[ModelResponse, float | None]] = []
        self._terminal_error: str | None = None
        self._tool_bindings: dict[str, dict[str, str | None]] = {}
        self._tool_runs: dict[str, tuple[float, RuntimeEvent]] = {}
        self._event_counter = 0
        self._span_count = 0
        self._finished = False
        self._metadata = _jsonable(dict(spec.framework_metadata))
        self._frame: dict[str, Any] = {
            "node_id": spec.node_id,
            "node_type": spec.node_type,
            "parent_node_id": spec.parent_node_id,
            "step_index": 0,
        }
        self._registry = {entry.tool_name: entry for entry in spec.tool_registry}
        for definition in spec.tool_definitions:
            self._registry.setdefault(
                definition.tool_name,
                ToolRegistryEntry(
                    tool_name=definition.tool_name,
                    default_effect_scope=definition.effect_scope,
                    version=definition.schema_ref.version,
                    description=definition.description,
                ),
            )
        # M1 prior-facts 索引：事件写入时增量维护，避免每次 M1 扫描全事件流。
        self._latest_completed_span_event: RuntimeEvent | None = None
        self._prior_verification_by_episode: dict[str, list[PriorVerificationRecord]] = {}
        self._prior_state_by_episode: dict[str, PriorStateRecord] = {}
        self._prior_tool_outcomes: dict[str, PriorToolOutcome] = {}
        self._prior_tool_order: list[str] = []
        self._emit(
            "workflow_started",
            payload={
                "agent_id": spec.agent_id,
                "agent_version": spec.agent_version,
                "workflow_goal": spec.workflow_goal.model_dump(mode="json"),
                "framework_metadata": self._metadata,
                "tool_registry": [entry.to_dict() for entry in self._registry.values()],
            },
        )

    @property
    def workflow_run_id(self) -> str:
        return self._spec.workflow_run_id

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def update_framework_metadata(
        self,
        metadata: Mapping[str, Any] | None = None,
        *,
        node_id: str | None = None,
        node_type: str | None = None,
        parent_node_id: str | None = None,
        step_index: int | None = None,
    ) -> None:
        """Record newly observable Agent facts before a model invocation."""

        with self._lock:
            if metadata:
                self._metadata.update(_jsonable(dict(metadata)))
            if node_id is not None:
                self._frame["node_id"] = node_id
            if node_type is not None:
                self._frame["node_type"] = node_type
            if parent_node_id is not None:
                self._frame["parent_node_id"] = parent_node_id
            if step_index is not None:
                self._frame["step_index"] = step_index
            self._emit(
                "framework_metadata",
                payload={"metadata": self._metadata, "frame": dict(self._frame)},
            )

    def begin_lm_invocation(
        self,
        request: ModelRequest,
        *,
        recovery_mode: Literal["normal", "rollback_recovery"] = "normal",
    ) -> WorkflowContext:
        """Capture pre-routing facts and build the factual ``WorkflowContext``.

        The returned context contains only information known before the current
        model is routed.  Tool, verifier, and state events from a completed
        prior span are represented as upstream references.
        """

        with self._lock:
            step_index = self._frame.get("step_index")
            if not isinstance(step_index, int) or step_index < 0:
                step_index = self._span_count
            context = WorkflowContext(
                context_version="1.0",
                agent_id=self._spec.agent_id,
                agent_role=self._spec.agent_role,
                agent_version=self._spec.agent_version,
                workflow_goal=self._spec.workflow_goal,
                node_context=NodeContext(
                    node_id=str(self._frame["node_id"]),
                    node_type=(
                        "rollback_recovery"
                        if recovery_mode == "rollback_recovery"
                        else str(self._frame["node_type"])
                    ),
                    step_index=step_index,
                    parent_node_id=self._frame.get("parent_node_id"),
                ),
                upstream_context_refs=self._upstream_context_refs(),
                downstream_contract=DownstreamContract(
                    consumer_role=AgentRole.FINAL_RESPONDER,
                    dependency_level=self._spec.downstream_dependency,
                ),
                state_context=StateContext(
                    access_mode=self._spec.state_access_mode,
                    expected_effect=self._spec.expected_effect,
                ),
                available_tools=tuple(
                    definition.schema_ref for definition in self._spec.tool_definitions
                ),
                output_contract=self._spec.output_contract,
                verifier_catalog=self._spec.verifier_catalog,
                recovery_capability=self._spec.recovery_capability,
                recovery_mode=recovery_mode,
            )
            self._span_count += 1
            event = self._emit(
                "lm_invocation",
                payload={
                    "span_ordinal": self._span_count,
                    "request": request.model_dump(mode="json"),
                    "workflow_context": context.model_dump(mode="json"),
                    "framework_metadata": self._metadata,
                },
            )
            self._pending_invocations.append(event.event_id)
            return context

    def begin_internal_lm_invocation(
        self,
        request: ModelRequest,
        workflow: WorkflowContext,
        *,
        reason: str,
    ) -> None:
        """Record a Coordinator-triggered fallback as a real LM invocation.

        The fallback is not a new Agent query, but it is still a physical
        model call and must own its own integration event and eventual span.
        """

        with self._lock:
            self._span_count += 1
            event = self._emit(
                "lm_invocation",
                payload={
                    "span_ordinal": self._span_count,
                    "request": request.model_dump(mode="json"),
                    "workflow_context": workflow.model_dump(mode="json"),
                    "framework_metadata": self._metadata,
                    "internally_triggered": True,
                    "reason": reason,
                },
            )
            self._pending_invocations.append(event.event_id)

    def record_span(self, record: SpanRecord, response: ModelResponse | None) -> None:
        """Attach Coordinator/M3 terminal facts to the oldest pending span."""

        if record.route_plan.workflow_run_id != self.workflow_run_id:
            return
        with self._lock:
            opened = record.route_plan.episode_id in self._opened_spans
            invocation_event = (
                None
                if opened
                else self._events_by_id.get(self._pending_invocations.pop(0))
                if self._pending_invocations
                else None
            )
            if invocation_event is not None:
                invocation_event.episode_id = record.route_plan.episode_id
                invocation_event.logical_invocation_id = record.route_plan.logical_invocation_id

            summary = record.execution_summary
            accepted_attempt_id = summary.accepted_model_attempt_id
            self._emit(
                "span_recorded",
                episode_id=record.route_plan.episode_id,
                logical_invocation_id=record.route_plan.logical_invocation_id,
                model_attempt_id=accepted_attempt_id,
                payload={
                    "route": record.route_plan.model_dump(mode="json"),
                    "runtime_projection": record.runtime_projection.model_dump(mode="json"),
                    "model_request": (
                        record.model_request.model_dump(mode="json")
                        if record.model_request is not None
                        else None
                    ),
                    "execution_summary": summary.model_dump(mode="json"),
                    "span_kind": record.span_kind,
                    "parent_episode_id": record.parent_episode_id,
                    "span_relation": record.span_relation,
                },
            )
            for guard in summary.guard_results:
                self._emit(
                    "verifier_result",
                    episode_id=record.route_plan.episode_id,
                    logical_invocation_id=record.route_plan.logical_invocation_id,
                    model_attempt_id=guard.subject_model_attempt_id,
                    payload=guard.model_dump(mode="json"),
                )
            self._emit(
                "state_observation",
                episode_id=record.route_plan.episode_id,
                logical_invocation_id=record.route_plan.logical_invocation_id,
                model_attempt_id=accepted_attempt_id,
                payload=summary.state_outcome.model_dump(mode="json"),
            )
            if response is None:
                self._opened_spans.discard(record.route_plan.episode_id)
                return
            if self._response_attempts.get(record.route_plan.episode_id) != accepted_attempt_id:
                self._emit(
                    "model_response",
                    episode_id=record.route_plan.episode_id,
                    logical_invocation_id=record.route_plan.logical_invocation_id,
                    model_attempt_id=accepted_attempt_id,
                    payload=response.model_dump(mode="json"),
                )
            for call in response.tool_calls:
                self._tool_bindings[call.tool_call_id] = {
                    "tool_name": call.tool_name,
                    "episode_id": record.route_plan.episode_id,
                    "logical_invocation_id": record.route_plan.logical_invocation_id,
                    "model_attempt_id": accepted_attempt_id,
                }
            self._opened_spans.discard(record.route_plan.episode_id)

    def record_span_opened(self, record: SpanRecord, response: ModelResponse) -> None:
        """登记已交付模型响应但尚未完成工具事务的旧 API。"""

        self.record_pending_span(
            record.route_plan,
            record.execution_summary,
            response,
        )

    def record_pending_span(
        self,
        route: RoutePlan,
        result: ExecutionResult,
        response: ModelResponse,
    ) -> None:
        """登记未闭合的工具阶段，不构造临时终态 ``SpanRecord``。

        Recovery spans must carry a real ``RecoveryCheck`` in their final
        record.  Keeping this open-event API at the route/result level lets the
        Coordinator bind tool calls before that check exists.
        """

        if route.workflow_run_id != self.workflow_run_id:
            return
        with self._lock:
            if self._pending_invocations:
                invocation_event = self._events_by_id.get(self._pending_invocations.pop(0))
                if invocation_event is not None:
                    invocation_event.episode_id = route.episode_id
                    invocation_event.logical_invocation_id = route.logical_invocation_id
            self._opened_spans.add(route.episode_id)
            self._emit(
                "span_opened",
                episode_id=route.episode_id,
                logical_invocation_id=route.logical_invocation_id,
                model_attempt_id=result.accepted_model_attempt_id,
                payload={
                    "route_plan_id": route.route_plan_id,
                    "execution_status": result.status.value,
                    "tool_call_ids": [call.tool_call_id for call in response.tool_calls],
                },
            )
            self._bind_response(route, response)

    def record_model_response(
        self,
        workflow_run_id: str,
        route,
        response: ModelResponse,
        *,
        accepted_attempt_id: str | None,
    ) -> None:
        """记录 M3 fallback 响应，并更新其工具调用的 episode 绑定。"""

        if workflow_run_id != self.workflow_run_id:
            return
        with self._lock:
            self._response_attempts[route.episode_id] = accepted_attempt_id
            self._emit(
                "model_response",
                episode_id=route.episode_id,
                logical_invocation_id=route.logical_invocation_id,
                model_attempt_id=accepted_attempt_id,
                payload=response.model_dump(mode="json"),
            )
            self._bind_response(route, response)

    def _bind_response(self, record_or_route, response: ModelResponse) -> None:
        route = getattr(record_or_route, "route_plan", record_or_route)
        attempt_id = getattr(record_or_route, "execution_summary", None)
        if attempt_id is not None:
            attempt_id = attempt_id.accepted_model_attempt_id
        for call in response.tool_calls:
            self._tool_bindings[call.tool_call_id] = {
                "tool_name": call.tool_name,
                "episode_id": route.episode_id,
                "logical_invocation_id": route.logical_invocation_id,
                "model_attempt_id": attempt_id,
            }

    def queue_recovery_response(self, response: ModelResponse, *, cost: float | None = None) -> None:
        with self._lock:
            self._recovery_responses.append((response, cost))

    def consume_recovery_response(self) -> tuple[ModelResponse, float | None] | None:
        with self._lock:
            if not self._recovery_responses:
                return None
            return self._recovery_responses.pop(0)

    def set_terminal_error(self, error: str) -> None:
        with self._lock:
            if self._terminal_error is None:
                self._emit("workflow_error", payload={"error": error})
            self._terminal_error = error

    @property
    def terminal_error(self) -> str | None:
        with self._lock:
            return self._terminal_error

    def record_tool_started(self, action: Mapping[str, Any], *, cwd: str = "") -> str:
        """Record the action immediately before the wrapped Environment executes it."""

        with self._lock:
            tool_call_id = action.get("tool_call_id")
            call_id = tool_call_id if isinstance(tool_call_id, str) else None
            binding = self._tool_bindings.get(call_id or "", {})
            tool_name = binding.get("tool_name") or "bash"
            entry = self._registry.get(tool_name)
            scope = entry.effect_scope_for_action(action) if entry is not None else EffectScope.UNKNOWN
            event = self._emit(
                "tool_started",
                episode_id=binding.get("episode_id"),
                logical_invocation_id=binding.get("logical_invocation_id"),
                tool_call_id=call_id,
                model_attempt_id=binding.get("model_attempt_id"),
                payload={
                    "tool_name": tool_name,
                    "action": _jsonable(dict(action)),
                    "cwd": cwd,
                    "interface": entry.interface if entry is not None else "unknown",
                    "effect_scope": scope.value,
                    "effect_risk": _effect_risk(scope).value,
                    "permissions": list(entry.permissions) if entry is not None else [],
                },
            )
            self._tool_runs[event.event_id] = (real_time(), event)
            return event.event_id

    def record_tool_finished(
        self,
        tool_run_id: str,
        output: Mapping[str, Any] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Record the terminal result of one Environment action."""

        with self._lock:
            pending = self._tool_runs.pop(tool_run_id, None)
            if pending is None:
                return
            started, event = pending
            payload: dict[str, Any] = {
                "tool_started_event_id": event.event_id,
                "duration_ms": max(0, round((real_time() - started) * 1000)),
                "result": _jsonable(dict(output or {})),
            }
            if error is not None:
                payload["error_type"] = type(error).__name__
                payload["error"] = str(error)
            self._emit(
                "tool_finished",
                episode_id=event.episode_id,
                logical_invocation_id=event.logical_invocation_id,
                tool_call_id=event.tool_call_id,
                model_attempt_id=event.model_attempt_id,
                payload=payload,
            )

    def record_tool_observations(
        self,
        message: Mapping[str, Any],
        outputs: list[Mapping[str, Any]],
    ) -> None:
        """Record the observations that the Agent will feed into its next LM span."""

        extra = message.get("extra")
        actions = extra.get("actions", []) if isinstance(extra, Mapping) else []
        if not isinstance(actions, list):
            return
        with self._lock:
            for index, action in enumerate(actions):
                if not isinstance(action, Mapping):
                    continue
                output = outputs[index] if index < len(outputs) else {}
                call_id = action.get("tool_call_id")
                tool_call_id = call_id if isinstance(call_id, str) else None
                binding = self._tool_bindings.get(tool_call_id or "", {})
                self._emit(
                    "tool_observation",
                    episode_id=binding.get("episode_id"),
                    logical_invocation_id=binding.get("logical_invocation_id"),
                    tool_call_id=tool_call_id,
                    model_attempt_id=binding.get("model_attempt_id"),
                    payload={
                        "action": _jsonable(dict(action)),
                        "output": _jsonable(dict(output)),
                    },
                )

    def finish_workflow(self, result: Mapping[str, Any] | None) -> None:
        """Record the final Agent-visible workflow result exactly once."""

        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._emit("workflow_outcome", payload={"result": _jsonable(dict(result or {}))})

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": "integration-trace-1.0",
                "workflow_run_id": self.workflow_run_id,
                "agent": {
                    "id": self._spec.agent_id,
                    "version": self._spec.agent_version,
                    "role": self._spec.agent_role.value,
                    "metadata": _jsonable(self._metadata),
                },
                "tool_registry": [entry.to_dict() for entry in self._registry.values()],
                "events": [event.to_dict() for event in self._events],
            }

    def write_json(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def snapshot_prior_execution_facts(self) -> PriorExecutionFacts | None:
        """Return read-only facts from the latest completed episode.

        The session maintains incremental prior-fact indexes as events are
        emitted, so this method does not rescan the event journal.  The current
        ``lm_invocation`` has no completed span yet, so it cannot contribute
        verifier/state facts.  Tool outcomes are the last 256 prior
        ``tool_finished`` summaries and are used later to decide which recent
        message rounds count as successful tool rounds.
        """

        with self._lock:
            latest = self._latest_completed_span_event
            if latest is None or latest.episode_id is None:
                return None
            episode_id = str(latest.episode_id)
            payload = latest.payload if isinstance(latest.payload, Mapping) else {}
            raw_span_kind = payload.get("span_kind", "normal")
            span_kind = (
                raw_span_kind
                if raw_span_kind in ("normal", "fallback", "rollback_recovery")
                else "normal"
            )
            parent_episode_id = payload.get("parent_episode_id")
            return PriorExecutionFacts(
                episode_id=episode_id,
                logical_invocation_id=latest.logical_invocation_id,
                span_kind=span_kind,
                parent_episode_id=(
                    str(parent_episode_id)
                    if isinstance(parent_episode_id, str) and parent_episode_id
                    else None
                ),
                verifier_results=tuple(
                    self._prior_verification_by_episode.get(episode_id, ())
                ),
                state_observation=self._prior_state_by_episode.get(episode_id),
                tool_outcomes=tuple(
                    self._prior_tool_outcomes[tool_call_id]
                    for tool_call_id in self._prior_tool_order
                ),
            )

    def _index_prior_fact_event(self, event: RuntimeEvent) -> None:
        """Incrementally maintain M1 prior-fact indexes when an event is emitted."""

        if event.event_type == "verifier_result" and event.episode_id is not None:
            self._prior_verification_by_episode.setdefault(
                str(event.episode_id), []
            ).append(_prior_verification_record(event))
        elif event.event_type == "state_observation" and event.episode_id is not None:
            self._prior_state_by_episode[str(event.episode_id)] = _prior_state_record(event)
        elif event.event_type == "tool_finished" and event.tool_call_id:
            tool_call_id = str(event.tool_call_id)
            if tool_call_id not in self._prior_tool_outcomes:
                self._prior_tool_order.append(tool_call_id)
            self._prior_tool_outcomes[tool_call_id] = _prior_tool_outcome(event)
            while len(self._prior_tool_order) > 256:
                expired = self._prior_tool_order.pop(0)
                self._prior_tool_outcomes.pop(expired, None)
        elif event.event_type == "span_recorded" and event.episode_id is not None:
            self._latest_completed_span_event = event

    def _upstream_context_refs(self) -> tuple[UpstreamContextRef, ...]:
        sources = [
            event
            for event in self._events
            if event.event_type in {"tool_finished", "tool_observation", "verifier_result", "state_observation"}
        ][-12:]
        return tuple(
            UpstreamContextRef(
                source_type=event.event_type,
                ref=VersionedRef(kind="runtime_event", id=event.event_id, version="v1"),
                summary=_event_summary(event),
            )
            for event in sources
        )

    def _emit(
        self,
        event_type: str,
        *,
        episode_id: str | None = None,
        logical_invocation_id: str | None = None,
        tool_call_id: str | None = None,
        model_attempt_id: str | None = None,
        payload: Mapping[str, Any],
    ) -> RuntimeEvent:
        self._event_counter += 1
        event = RuntimeEvent(
            event_id=f"integration-event-{self._event_counter:04d}",
            sequence=self._event_counter,
            event_type=event_type,
            occurred_at=self._clock(),
            workflow_run_id=self.workflow_run_id,
            episode_id=episode_id,
            logical_invocation_id=logical_invocation_id,
            tool_call_id=tool_call_id,
            model_attempt_id=model_attempt_id,
            payload=_jsonable(dict(payload)),
        )
        self._index_prior_fact_event(event)
        self._events.append(event)
        self._events_by_id[event.event_id] = event
        return event


def _event_summary(event: RuntimeEvent) -> str:
    if event.event_type == "tool_finished":
        return "tool execution result"
    if event.event_type == "tool_observation":
        return "tool observation supplied to the next model span"
    if event.event_type == "verifier_result":
        return "verifier result"
    return "state observation"



def _prior_verification_record(event: RuntimeEvent) -> PriorVerificationRecord:
    guard = GuardResult.model_validate(event.payload, strict=False)
    if event.episode_id is None:
        raise ValueError("verifier_result event is missing episode_id")
    return PriorVerificationRecord(
        episode_id=str(event.episode_id),
        logical_invocation_id=event.logical_invocation_id,
        guard_ref=guard.guard_ref,
        stage=guard.stage,
        outcome=guard.outcome,
        subject_tool_call_id=guard.subject_tool_call_id,
    )


def _prior_state_record(event: RuntimeEvent) -> PriorStateRecord:
    state = StateOutcome.model_validate(event.payload, strict=False)
    if event.episode_id is None:
        raise ValueError("state_observation event is missing episode_id")
    return PriorStateRecord(
        episode_id=str(event.episode_id),
        logical_invocation_id=event.logical_invocation_id,
        effect=state.effect,
        commit_status=state.commit_status,
        recovery_required=state.recovery_required,
        rollback_status=state.rollback_status,
        resume_status=state.resume_status,
        fault_class=state.fault_class,
    )


def _prior_tool_outcome(event: RuntimeEvent) -> PriorToolOutcome:
    if event.tool_call_id is None:
        raise ValueError("tool_finished event is missing tool_call_id")
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    result = payload.get("result")
    returncode = result.get("returncode") if isinstance(result, Mapping) else None
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        returncode = None
    error_type = payload.get("error_type")
    error_type_value = (
        str(error_type) if isinstance(error_type, str) and error_type else None
    )
    success = (
        error_type_value is None
        and "error" not in payload
        and (returncode is None or returncode == 0)
    )
    #: 事前 L1 的效应类别；由 adapter 在 tool_finished 的 result 里带上。
    #: 缺失时留 None（在线统计按"未知"处理，**不猜**），不阻断旧事件。
    raw_category = result.get("effect_category") if isinstance(result, Mapping) else None
    effect_category = (
        str(raw_category) if isinstance(raw_category, str) and raw_category else None
    )
    return PriorToolOutcome(
        tool_call_id=str(event.tool_call_id),
        episode_id=str(event.episode_id) if event.episode_id is not None else None,
        logical_invocation_id=event.logical_invocation_id,
        success=success,
        returncode=returncode,
        error_type=error_type_value,
        effect_category=effect_category,
    )


class IntegrationHub:
    """Coordinator-owned registry of framework-neutral adapter sessions."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or _utcnow
        self._lock = RLock()
        self._sessions: dict[str, AdapterSession] = {}

    def open_session(self, spec: AdapterSessionSpec) -> AdapterSession:
        with self._lock:
            existing = self._sessions.get(spec.workflow_run_id)
            if existing is not None:
                return existing
            session = AdapterSession(spec, clock=self._clock)
            self._sessions[spec.workflow_run_id] = session
            return session

    def session(self, workflow_run_id: str) -> AdapterSession | None:
        with self._lock:
            return self._sessions.get(workflow_run_id)

    def record_span(self, record: SpanRecord, response: ModelResponse | None) -> None:
        session = self.session(record.route_plan.workflow_run_id)
        if session is not None:
            session.record_span(record, response)

    def record_span_opened(self, record: SpanRecord, response: ModelResponse) -> None:
        session = self.session(record.route_plan.workflow_run_id)
        if session is not None:
            session.record_span_opened(record, response)

    def record_pending_span(
        self,
        route: RoutePlan,
        result: ExecutionResult,
        response: ModelResponse,
    ) -> None:
        """Record an open tool phase without requiring a terminal SpanRecord."""

        session = self.session(route.workflow_run_id)
        if session is not None:
            session.record_pending_span(
                route,
                result,
                response,
            )

    def record_model_response(
        self,
        workflow_run_id: str,
        route,
        response: ModelResponse,
        *,
        accepted_attempt_id: str | None,
    ) -> None:
        session = self.session(workflow_run_id)
        if session is not None:
            session.record_model_response(
                workflow_run_id,
                route,
                response,
                accepted_attempt_id=accepted_attempt_id,
            )


__all__ = [
    "AdapterSession",
    "AdapterSessionSpec",
    "IntegrationHub",
    "RuntimeEvent",
    "ToolRegistryEntry",
]
