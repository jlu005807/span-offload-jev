from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

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
    ExecutionStatus,
    ExecutionSummary,
    FeedbackSnapshot,
    FinishReason,
    GuardRef,
    MessageRole,
    ModelMessage,
    ModelParameters,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelTarget,
    ModelTargetKind,
    OutcomeEvent,
    OutcomeLabel,
    OutcomeScope,
    ParseStatus,
    ProducerRef,
    RecoveryLevel,
    RequestScope,
    ResumeStatus,
    RollbackStatus,
    RoutePlan,
    SpanRecord,
    StateOutcome,
    VerifierPlan,
    VersionedRef,
)
from traceoffload.storage import (
    AppendStatus,
    ConflictError,
    EventStore,
    ExpiredRef,
    HistoryStore,
    RefNotFound,
    RequestResponseStore,
    SQLiteHistoryStore,
    SnapshotExists,
    SnapshotNotFound,
    SnapshotStore,
    UnauthorizedRef,
)

NOW = datetime.now(timezone.utc)
LLM = ModelRef(kind="model", id="llm", version="v1")
TENANT = VersionedRef(kind="tenant", id="tenant-1", version="v1")


def _lineage() -> dict[str, object]:
    return {
        "workflow_run_id": "workflow-1",
        "workflow_plan_ref": {"kind": "workflow_plan", "id": "plan-1", "version": "v1"},
        "workflow_plan_revision": 0,
        "episode_id": "episode-1",
        "logical_invocation_id": "invocation-1",
    }


def _capability(
    scope: CapabilityScope,
    audience: Component,
    *,
    ref_id: str = "request-1",
    kind: str = "request",
    target: ModelRef | None = None,
    tenant: VersionedRef | None = TENANT,
    content_hash: str | None = None,
    expires_at: datetime = NOW + timedelta(hours=1),
) -> CapabilityRef:
    if scope == CapabilityScope.EXECUTION_READ and content_hash is None:
        content_hash = f"sha256:{ref_id}"
    return CapabilityRef(
        kind=kind,
        id=ref_id,
        version="v1",
        scope=scope,
        audience=audience,
        content_hash=content_hash,
        tenant_ref=tenant,
        expires_at=expires_at,
        binding=CapabilityBinding(**_lineage(), target_model_ref=target),
    )


def _request() -> ModelRequest:
    return ModelRequest(
        request_version="1.0",
        messages=(ModelMessage(role=MessageRole.USER, content="hello"),),
        input_text="hello",
        model_parameters=ModelParameters(temperature=0.0),
        request_scope=RequestScope(
            tenant_ref=TENANT,
            data_policy_ref=VersionedRef(kind="policy", id="default", version="v1"),
        ),
    )


def _response() -> ModelResponse:
    return ModelResponse(
        response_id="response-1",
        model_attempt_id="attempt-1",
        output_text="fixed patch",
        finish_reason=FinishReason.STOP,
        parse_status=ParseStatus.NOT_NEEDED,
        raw_response_ref=VersionedRef(kind="raw", id="raw-1", version="v1"),
    )


def _route(route_plan_id: str = "route-1") -> RoutePlan:
    return RoutePlan(
        schema_version="1.0",
        message_id=f"message-route-{route_plan_id}",
        producer_ref=ProducerRef(component=Component.M2, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=1),
        **_lineage(),
        route_plan_id=route_plan_id,
        target=ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=LLM),
        execution_mode=ExecutionMode.DIRECT,
        verifier_plan=VerifierPlan(
            verifier_plan_id="verifier-none",
            version="v1",
            overall_timeout_ms=1000,
            empty_plan_reason="large_model_direct_no_guard",
        ),
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


def _span(record_message_id: str = "message-span-1", **summary_overrides: object) -> SpanRecord:
    route = _route()
    overrides = {"latency_ms": 60, **summary_overrides}
    summary = ExecutionSummary(
        route_plan_id=route.route_plan_id,
        **_lineage(),
        status=ExecutionStatus.SUCCESS,
        model_attempts=(
            {
                "model_attempt_id": "attempt-1",
                "role": AttemptRole.LARGE_MODEL_DIRECT,
                "model_ref": LLM,
                "status": AttemptStatus.SUCCESS,
                "latency_ms": 50,
                "cost": 0.01,
            },
        ),
        state_outcome=StateOutcome(
            effect=ActualEffect.NONE,
            commit_status=CommitStatus.NOT_ATTEMPTED,
            recovery_required=False,
            rollback_status=RollbackStatus.NOT_NEEDED,
            resume_status=ResumeStatus.NOT_NEEDED,
        ),
        accepted_model_attempt_id="attempt-1",
        **overrides,
    )
    return SpanRecord(
        schema_version="1.0",
        message_id=record_message_id,
        producer_ref=ProducerRef(component=Component.COORDINATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=3),
        runtime_projection={"descriptor_key": route.experience_key},
        route_plan=route,
        execution_summary=summary,
    )


def _outcome(
    event_id: str = "event-1",
    revision: int = 1,
    **overrides: object,
) -> OutcomeEvent:
    field_overrides = {
        "scope": OutcomeScope.CANDIDATE,
        "model_attempt_id": "attempt-1",
        "label": OutcomeLabel.SUCCESS,
        **overrides,
    }
    return OutcomeEvent(
        schema_version="1.0",
        message_id=f"message-{event_id}-r{revision}",
        producer_ref=ProducerRef(component=Component.EVALUATOR, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=4),
        event_id=event_id,
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        evaluator_ref=GuardRef(kind="evaluator", id="swebench", version="v1"),
        observed_at=NOW + timedelta(seconds=4),
        revision=revision,
        **field_overrides,
    )


def _snapshot(version: str = "snapshot-1") -> FeedbackSnapshot:
    return FeedbackSnapshot(
        schema_version="1.0",
        message_id=f"message-{version}",
        producer_ref=ProducerRef(component=Component.M4, version="0.1.0"),
        produced_at=NOW + timedelta(seconds=5),
        snapshot_version=version,
        safety_calibration_ref=VersionedRef(kind="calibration", id="safety", version="v1"),
    )


class TestRequestResponseStore:
    def test_request_resolves_for_m1_profile_view(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.PROFILE_READ, Component.M1)
        request = _request()

        store.save_request(capability, request)

        assert store.resolve_request(capability, Component.M1) == request

    def test_request_resolves_for_m3_execution_view(self) -> None:
        store = RequestResponseStore()
        capability = _capability(
            CapabilityScope.EXECUTION_READ,
            Component.M3,
            target=LLM,
        )
        request = _request()

        store.save_request(capability, request)

        assert store.resolve_request(capability, Component.M3, expected_target=LLM) == request

    def test_unauthorized_viewer_cannot_resolve_request(self) -> None:
        store = RequestResponseStore()
        profile_ref = _capability(CapabilityScope.PROFILE_READ, Component.M1)
        store.save_request(profile_ref, _request())

        with pytest.raises(UnauthorizedRef):
            store.resolve_request(profile_ref, Component.M3)
        with pytest.raises(UnauthorizedRef):
            store.resolve_request(profile_ref, Component.M2)

        execution_ref = _capability(
            CapabilityScope.EXECUTION_READ,
            Component.M3,
            ref_id="request-exec",
            target=LLM,
        )
        store.save_request(execution_ref, _request())
        with pytest.raises(UnauthorizedRef):
            store.resolve_request(execution_ref, Component.M1)

    def test_unknown_ref_is_rejected(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.PROFILE_READ, Component.M1)

        with pytest.raises(RefNotFound):
            store.resolve_request(capability, Component.M1)

    def test_tampered_binding_is_rejected(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.PROFILE_READ, Component.M1)
        store.save_request(capability, _request())

        tampered = capability.model_copy(
            update={
                "binding": capability.binding.model_copy(update={"episode_id": "episode-other"})
            }
        )
        with pytest.raises(UnauthorizedRef):
            store.resolve_request(tampered, Component.M1)

    def test_expired_ref_is_rejected(self) -> None:
        store = RequestResponseStore()
        capability = _capability(
            CapabilityScope.PROFILE_READ,
            Component.M1,
            expires_at=NOW + timedelta(minutes=1),
        )
        store.save_request(capability, _request())

        with pytest.raises(ExpiredRef):
            store.resolve_request(
                capability,
                Component.M1,
                now=NOW + timedelta(minutes=2),
            )

    def test_expired_capability_cannot_be_saved(self) -> None:
        store = RequestResponseStore()
        capability = _capability(
            CapabilityScope.PROFILE_READ,
            Component.M1,
            expires_at=NOW - timedelta(minutes=1),
        )

        with pytest.raises(ExpiredRef):
            store.save_request(capability, _request())

    def test_target_model_mismatch_is_rejected(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.EXECUTION_READ, Component.M3, target=LLM)
        store.save_request(capability, _request())

        other = ModelRef(kind="model", id="other-llm", version="v1")
        with pytest.raises(UnauthorizedRef):
            store.resolve_request(capability, Component.M3, expected_target=other)

    def test_execution_view_requires_bound_target(self) -> None:
        store = RequestResponseStore()
        capability = _capability(
            CapabilityScope.EXECUTION_READ,
            Component.M3,
            target=None,
            content_hash="sha256:request-1",
        )
        store.save_request(capability, _request())

        with pytest.raises(UnauthorizedRef):
            store.resolve_request(capability, Component.M3, expected_target=LLM)

    def test_tenant_mismatch_is_rejected(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.PROFILE_READ, Component.M1, tenant=None)

        with pytest.raises(UnauthorizedRef):
            store.save_request(capability, _request())

        tenant_ref = _capability(
            CapabilityScope.PROFILE_READ,
            Component.M1,
            tenant=TENANT,
        )
        store.save_request(tenant_ref, _request())
        assert store.resolve_request(tenant_ref, Component.M1) == _request()

    def test_duplicate_save_with_same_content_is_idempotent(self) -> None:
        store = RequestResponseStore()
        capability = _capability(CapabilityScope.PROFILE_READ, Component.M1)

        store.save_request(capability, _request())
        store.save_request(capability, _request())

        with pytest.raises(ConflictError):
            store.save_request(capability, _request().model_copy(update={"input_text": "other"}))

    def test_accepted_output_resolves_to_full_response(self) -> None:
        store = RequestResponseStore()
        accepted_ref = _capability(
            CapabilityScope.ACCEPTED_OUTPUT,
            Component.AGENT_ADAPTER,
            kind="model_response",
            ref_id="accepted-1",
            target=LLM,
        )
        audit_ref = GuardRef(kind="model_response", id="audit-1", version="v1")
        response = _response()

        store.save_response(accepted_ref, response, audit_refs=(audit_ref,))

        projection = store.resolve_accepted_output(accepted_ref, Component.AGENT_ADAPTER)
        assert projection.output_text == response.output_text
        assert projection.raw_response_ref == response.raw_response_ref

        full = store.resolve_audit_response(audit_ref, Component.M3)
        assert full == response
        assert full.raw_response_ref is not None

    def test_accepted_output_is_restricted_to_agent_adapter(self) -> None:
        store = RequestResponseStore()
        accepted_ref = _capability(
            CapabilityScope.ACCEPTED_OUTPUT,
            Component.AGENT_ADAPTER,
            kind="model_response",
            ref_id="accepted-1",
            target=LLM,
        )
        audit_ref = GuardRef(kind="model_response", id="audit-1", version="v1")
        store.save_response(accepted_ref, _response(), audit_refs=(audit_ref,))

        with pytest.raises(UnauthorizedRef):
            store.resolve_accepted_output(accepted_ref, Component.M3)
        with pytest.raises(UnauthorizedRef):
            store.resolve_audit_response(audit_ref, Component.AGENT_ADAPTER)

    def test_expired_accepted_output_is_rejected(self) -> None:
        store = RequestResponseStore()
        accepted_ref = _capability(
            CapabilityScope.ACCEPTED_OUTPUT,
            Component.AGENT_ADAPTER,
            kind="model_response",
            ref_id="accepted-1",
            target=LLM,
            expires_at=NOW + timedelta(minutes=1),
        )
        store.save_response(accepted_ref, _response())

        with pytest.raises(ExpiredRef):
            store.resolve_accepted_output(
                accepted_ref,
                Component.AGENT_ADAPTER,
                now=NOW + timedelta(minutes=2),
            )

    def test_save_response_rejects_non_accepted_capability(self) -> None:
        store = RequestResponseStore()
        profile_ref = _capability(CapabilityScope.PROFILE_READ, Component.M1)

        with pytest.raises(UnauthorizedRef):
            store.save_response(profile_ref, _response())

    def test_conflicting_response_registration_is_rejected(self) -> None:
        store = RequestResponseStore()
        accepted_ref = _capability(
            CapabilityScope.ACCEPTED_OUTPUT,
            Component.AGENT_ADAPTER,
            kind="model_response",
            ref_id="accepted-1",
            target=LLM,
        )
        store.save_response(accepted_ref, _response())

        with pytest.raises(ConflictError):
            store.save_response(accepted_ref, _response().model_copy(update={"output_text": "other"}))


class TestEventStore:
    def test_span_record_is_deduplicated_by_message_id(self) -> None:
        store = EventStore()
        record = _span()

        assert store.append_span_record(record) is AppendStatus.ADDED
        assert store.append_span_record(record) is AppendStatus.IGNORED
        assert store.span_records() == (record,)

    def test_span_record_conflict_raises(self) -> None:
        store = EventStore()
        store.append_span_record(_span())

        with pytest.raises(ConflictError):
            store.append_span_record(_span(latency_ms=61))

    def test_outcome_event_keeps_highest_revision(self) -> None:
        store = EventStore()
        rev1 = _outcome(revision=1)
        rev2 = _outcome(revision=2, label=OutcomeLabel.FAILURE)

        assert store.append_outcome_event(rev1) is AppendStatus.ADDED
        assert store.append_outcome_event(rev2) is AppendStatus.ADDED
        assert store.outcome_events() == (rev2,)

        assert store.append_outcome_event(rev1) is AppendStatus.IGNORED
        assert store.append_outcome_event(rev2) is AppendStatus.IGNORED
        assert store.outcome_events() == (rev2,)

    def test_out_of_order_old_revision_does_not_overwrite(self) -> None:
        store = EventStore()
        rev2 = _outcome(revision=2, label=OutcomeLabel.FAILURE)
        rev1 = _outcome(revision=1)

        store.append_outcome_event(rev2)
        assert store.append_outcome_event(rev1) is AppendStatus.IGNORED
        assert store.outcome_events() == (rev2,)

    def test_same_revision_conflict_is_isolated(self) -> None:
        store = EventStore()
        first = _outcome(revision=1, label=OutcomeLabel.SUCCESS)
        conflict = _outcome(revision=1, label=OutcomeLabel.FAILURE)

        assert store.append_outcome_event(first) is AppendStatus.ADDED
        assert store.append_outcome_event(conflict) is AppendStatus.CONFLICT
        assert store.outcome_events() == (first,)
        assert store.conflicts() == (conflict,)

    def test_multiple_event_ids_are_kept_separately(self) -> None:
        store = EventStore()
        first = _outcome(event_id="event-1", revision=2)
        second = _outcome(event_id="event-2", revision=1)

        store.append_outcome_event(first)
        store.append_outcome_event(second)

        assert store.outcome_events() == (first, second)

    def test_span_and_outcome_can_coexist(self) -> None:
        store = EventStore()
        store.append_span_record(_span())
        store.append_outcome_event(_outcome())

        assert len(store.span_records()) == 1
        assert len(store.outcome_events()) == 1


class TestSnapshotStore:
    def test_publish_read_and_latest_round_trip(self) -> None:
        store = SnapshotStore()
        snapshot = _snapshot()

        store.publish(snapshot)

        assert store.read(snapshot.snapshot_version) == snapshot
        assert store.latest() == snapshot
        assert store.versions() == (snapshot.snapshot_version,)

    def test_published_version_cannot_be_replaced(self) -> None:
        store = SnapshotStore()
        store.publish(_snapshot("snapshot-1"))

        with pytest.raises(SnapshotExists):
            store.publish(_snapshot("snapshot-1"))

    def test_unknown_version_is_rejected(self) -> None:
        store = SnapshotStore()

        with pytest.raises(SnapshotNotFound):
            store.read("missing")

    def test_latest_is_last_published_version(self) -> None:
        store = SnapshotStore()
        first = _snapshot("snapshot-1")
        second = _snapshot("snapshot-2")

        store.publish(first)
        store.publish(second)

        assert store.latest() == second
        assert store.versions() == ("snapshot-1", "snapshot-2")

    def test_latest_on_empty_store_is_rejected(self) -> None:
        store = SnapshotStore()

        with pytest.raises(SnapshotNotFound):
            store.latest()


class TestSQLiteHistoryStore:
    def test_history_survives_restart_with_revision_and_conflict_semantics(
        self,
        tmp_path,
    ) -> None:
        path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(path)
        span = _span()
        revision_1 = _outcome(revision=1)
        conflict = _outcome(revision=1, label=OutcomeLabel.FAILURE)
        revision_2 = _outcome(revision=2, label=OutcomeLabel.FAILURE)
        snapshot = _snapshot()

        assert isinstance(store, HistoryStore)
        assert store.append_span_record(span) is AppendStatus.ADDED
        assert store.append_outcome_event(revision_1) is AppendStatus.ADDED
        assert store.append_outcome_event(conflict) is AppendStatus.CONFLICT
        assert store.append_outcome_event(revision_2) is AppendStatus.ADDED
        store.publish(snapshot)

        reopened = SQLiteHistoryStore(path)

        assert reopened.span_records() == (span,)
        assert reopened.outcome_events() == (revision_2,)
        assert reopened.conflicts() == (conflict,)
        assert reopened.read(snapshot.snapshot_version) == snapshot
        assert reopened.latest() == snapshot
        assert reopened.storage_metadata() == {"storage_schema_version": "1"}
        assert reopened.append_span_record(span) is AppendStatus.IGNORED
        assert reopened.append_outcome_event(revision_1) is AppendStatus.IGNORED

    def test_concurrent_writers_do_not_lose_spans(self, tmp_path) -> None:
        path = tmp_path / "history.sqlite3"
        stores = (SQLiteHistoryStore(path), SQLiteHistoryStore(path))
        records = tuple(_span(f"message-span-{index}") for index in range(100))

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = tuple(
                pool.map(
                    lambda item: stores[item[0] % 2].append_span_record(item[1]),
                    enumerate(records),
                )
            )

        assert statuses == (AppendStatus.ADDED,) * len(records)
        persisted = SQLiteHistoryStore(path).span_records()
        assert len(persisted) == len(records)
        assert {record.message_id for record in persisted} == {
            record.message_id for record in records
        }

    def test_snapshot_version_cannot_be_replaced_after_restart(self, tmp_path) -> None:
        path = tmp_path / "history.sqlite3"
        SQLiteHistoryStore(path).publish(_snapshot("snapshot-1"))

        with pytest.raises(SnapshotExists):
            SQLiteHistoryStore(path).publish(_snapshot("snapshot-1"))
