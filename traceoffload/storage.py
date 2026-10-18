"""不可变、能力校验的 TraceOffload 记录存储。

三个 Store：

- RequestResponseStore：以能力引用为键保存完整 ModelRequest / ModelResponse。
  解析时强制校验 scope、audience、绑定身份、租户、目标模型与过期时间；
  这些检查控制访问权限，不删除响应字段。
- EventStore：以 message_id 去重追加 SpanRecord，以 event revision 语义追加
  OutcomeEvent，供 M4 聚合使用。
- SnapshotStore：按版本发布不可变 FeedbackSnapshot；已发布的版本不可覆盖。
- SQLiteHistoryStore：实现同一 ``HistoryStore`` 接口的持久化事件与快照存储，
  保留去重、revision、冲突隔离和不可变发布语义。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import AwareDatetime

from .contracts import (
    CapabilityRef,
    CapabilityScope,
    Component,
    FeedbackSnapshot,
    ModelRef,
    ModelRequest,
    ModelResponse,
    OutcomeEvent,
    SpanRecord,
    VersionedRef,
)


class StorageError(RuntimeError):
    """存储失败基类。"""


class RefNotFound(StorageError):
    """该引用下没有注册任何对象。"""


class UnauthorizedRef(StorageError):
    """该引用不能被此查看者（viewer）或为此用途解析。"""


class ExpiredRef(StorageError):
    """能力引用已过期，不能再使用。"""


class ConflictError(StorageError):
    """同一个键被注册了两次且内容不同（不可变对象冲突）。"""


class SnapshotNotFound(StorageError):
    """请求的版本不存在任何快照。"""


class SnapshotExists(StorageError):
    """该版本快照已经发布。"""


class PublishJobStatus(StrEnum):
    """Durable M4 outbox state."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class M4PublishJob:
    """One durable request to publish the latest feedback snapshot."""

    job_id: str
    workflow_run_id: str
    target_sequence: int
    status: PublishJobStatus
    attempts: int
    available_at: datetime
    lease_until: datetime | None = None
    snapshot_version: str | None = None
    error: str | None = None


# 存储键：引用三元组 (kind, id, version)
RefKey = tuple[str, str, str]


def _key(ref: VersionedRef) -> RefKey:
    return (ref.kind, ref.id, ref.version)


def _utcnow() -> AwareDatetime:
    return datetime.now(timezone.utc)


def _checkpoint_row(session: Any) -> dict[str, Any]:
    """Serialize only recovery facts; executable objects never enter the journal."""
    state = session.state.value if hasattr(session.state, "value") else str(session.state)
    return {
        "session_id": str(session.session_id),
        "workflow_run_id": str(session.workflow_run_id),
        "transaction_ids": list(session.transaction_ids),
        "scopes": dict(session.scopes),
        "state": state,
        "handle_retained": bool(session.handle_retained),
        "fault_class": session.fault_class,
        "fault_detail": session.fault_detail,
        "rollback_status": str(session.rollback_status),
        "decision": session.decision,
        "operation_seq": int(session.operation_seq),
    }


class RequestResponseStore:
    """在能力引用下保存不可变请求与响应。"""

    def __init__(self, *, clock: Callable[[], AwareDatetime] | None = None) -> None:
        self._clock = clock or _utcnow
        # 键 -> (注册时的能力引用, 请求对象)；解析时要求能力引用完全相等
        self._requests: dict[RefKey, tuple[CapabilityRef, ModelRequest]] = {}
        # 键 -> (接受能力引用或 None(仅审计), 响应对象)
        self._responses: dict[RefKey, tuple[CapabilityRef | None, ModelResponse]] = {}

    def save_request(self, capability: CapabilityRef, request: ModelRequest) -> None:
        """保存请求：拒绝已过期能力、租户不匹配；同键同内容幂等。"""
        if capability.expires_at <= self._clock():
            raise ExpiredRef("cannot save a request under an already-expired capability")
        if request.request_scope.tenant_ref != capability.tenant_ref:
            raise UnauthorizedRef("capability tenant_ref must match the request scope")
        key = _key(capability)
        stored = self._requests.get(key)
        if stored is not None:
            stored_capability, stored_request = stored
            if stored_request != request or stored_capability != capability:
                raise ConflictError(f"request already registered under {key} with different content")
            return
        self._requests[key] = (capability, request)

    def resolve_request(
        self,
        capability: CapabilityRef,
        viewer: Component,
        *,
        now: AwareDatetime | None = None,
        expected_target: ModelRef | None = None,
    ) -> ModelRequest:
        """按查看者分派视图：M1 需 profile_read，M3 需 execution_read+内容哈希+目标模型。"""
        stored = self._requests.get(_key(capability))
        if stored is None:
            raise RefNotFound("no request registered under this ref")
        stored_capability, request = stored
        if stored_capability != capability:
            raise UnauthorizedRef("capability does not match the registered request")
        if capability.audience != viewer:
            raise UnauthorizedRef(
                f"{viewer.value} cannot resolve a ref bound for {capability.audience.value}"
            )
        if viewer == Component.M1:
            if capability.scope != CapabilityScope.PROFILE_READ:
                raise UnauthorizedRef("m1 profile view requires a profile_read capability")
        elif viewer == Component.M3:
            if capability.scope != CapabilityScope.EXECUTION_READ:
                raise UnauthorizedRef("m3 execution view requires an execution_read capability")
            if capability.content_hash is None:
                raise UnauthorizedRef("m3 execution view requires a content-hashed capability")
            if capability.binding.target_model_ref is None:
                raise UnauthorizedRef("m3 execution view requires a bound target model")
        else:
            raise UnauthorizedRef(f"{viewer.value} cannot resolve model requests")
        resolved_at = now or self._clock()
        if resolved_at >= capability.expires_at:
            raise ExpiredRef(f"capability expired at {capability.expires_at}")
        if expected_target is not None and capability.binding.target_model_ref != expected_target:
            raise UnauthorizedRef("capability target model does not match expected target")
        return request

    def save_response(
        self,
        capability: CapabilityRef,
        response: ModelResponse,
        *,
        audit_refs: Iterable[VersionedRef] = (),
    ) -> None:
        """保存被接受的响应：能力必须是 ACCEPTED_OUTPUT 且 audience 为 AGENT_ADAPTER。"""
        if capability.scope != CapabilityScope.ACCEPTED_OUTPUT:
            raise UnauthorizedRef("accepted output capability requires accepted_output scope")
        if capability.audience != Component.AGENT_ADAPTER:
            raise UnauthorizedRef("accepted output capability must target agent_adapter")
        if capability.expires_at <= self._clock():
            raise ExpiredRef("cannot save a response under an already-expired capability")
        self._register_response(_key(capability), capability, response)
        for audit_ref in audit_refs:
            self._register_response(_key(audit_ref), None, response)

    def _register_response(
        self,
        key: RefKey,
        capability: CapabilityRef | None,
        response: ModelResponse,
    ) -> None:
        """内部注册：同键同内容幂等，异内容抛冲突。"""
        stored = self._responses.get(key)
        if stored is not None:
            stored_capability, stored_response = stored
            if stored_response != response or stored_capability != capability:
                raise ConflictError(f"response already registered under {key} with different content")
            return
        self._responses[key] = (capability, response)

    def register_audit_response(
        self,
        audit_ref: VersionedRef,
        response: ModelResponse,
    ) -> None:
        """注册从未被接受的响应（仅审计可访问）。

        被拒候选与失败路径的记录仍可通过 ``resolve_audit_response`` 解析，
        但不携带任何 accepted-output 能力。
        """
        self._register_response(_key(audit_ref), None, response)

    def resolve_accepted_output(
        self,
        capability: CapabilityRef,
        viewer: Component,
        *,
        now: AwareDatetime | None = None,
    ) -> ModelResponse:
        """仅 AGENT_ADAPTER 可解析；返回完整的已接受模型响应。"""
        stored = self._responses.get(_key(capability))
        if stored is None:
            raise RefNotFound("no response registered under this ref")
        stored_capability, response = stored
        if stored_capability != capability:
            raise UnauthorizedRef("capability does not match the registered response")
        if capability.scope != CapabilityScope.ACCEPTED_OUTPUT:
            raise UnauthorizedRef("accepted output requires an accepted_output capability")
        if capability.audience != Component.AGENT_ADAPTER or viewer != Component.AGENT_ADAPTER:
            raise UnauthorizedRef("accepted output is only resolvable by the agent adapter")
        resolved_at = now or self._clock()
        if resolved_at >= capability.expires_at:
            raise ExpiredRef(f"capability expired at {capability.expires_at}")
        return response

    def resolve_audit_response(
        self,
        result_ref: VersionedRef,
        viewer: Component,
    ) -> ModelResponse:
        """仅 M3 可解析审计响应。"""
        if viewer != Component.M3:
            raise UnauthorizedRef("full responses are only resolvable by m3")
        stored = self._responses.get(_key(result_ref))
        if stored is None:
            raise RefNotFound("no response registered under this ref")
        _, response = stored
        return response


class AppendStatus(StrEnum):
    """追加结果三态：新增 / 忽略（重复或旧 revision）/ 冲突隔离。"""

    ADDED = "added"
    IGNORED = "ignored"
    CONFLICT = "conflict"


@runtime_checkable
class EventHistoryStore(Protocol):
    """M4 事件历史边界；内存与持久化实现共享同一契约。"""

    def append_span_record(self, record: SpanRecord) -> AppendStatus: ...

    def append_outcome_event(self, event: OutcomeEvent) -> AppendStatus: ...

    def span_records(self) -> tuple[SpanRecord, ...]: ...

    def outcome_events(self) -> tuple[OutcomeEvent, ...]: ...

    def conflicts(self) -> tuple[OutcomeEvent, ...]: ...

    def read_evidence(
        self,
    ) -> tuple[tuple[SpanRecord, ...], tuple[OutcomeEvent, ...]]: ...

    # M4 is intentionally decoupled from workflow latency.  These methods are
    # an outbox contract, not part of evidence aggregation itself.  Keeping
    # them on the first-party protocol makes a custom store fail at startup
    # instead of silently dropping a publication request.
    def enqueue_m4_publish(self, workflow_run_id: str) -> str: ...

    def claim_m4_publish_job(
        self,
        *,
        lease_seconds: float = 60.0,
    ) -> M4PublishJob | None: ...

    def complete_m4_publish_job(self, job_id: str, snapshot_version: str) -> None: ...

    def fail_m4_publish_job(
        self,
        job_id: str,
        error: str,
        *,
        retry_delay_seconds: float = 5.0,
        max_attempts: int = 5,
    ) -> None: ...

    def m4_publish_jobs(self) -> tuple[M4PublishJob, ...]: ...


@runtime_checkable
class SnapshotHistoryStore(Protocol):
    """不可变反馈快照的发布与读取边界。"""

    def publish(self, snapshot: FeedbackSnapshot) -> None: ...

    def read(self, version: str) -> FeedbackSnapshot: ...

    def latest(self) -> FeedbackSnapshot: ...

    def versions(self) -> tuple[str, ...]: ...


@runtime_checkable
class HistoryStore(EventHistoryStore, SnapshotHistoryStore, Protocol):
    """M4 完整历史接口：事件日志与不可变反馈快照。"""


class EventStore:
    """span 记录与延迟结果标签的追加式事件日志。

    Thread-safe: All append operations are protected by a lock for concurrent access.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._span_records: list[SpanRecord] = []
        self._span_ids: dict[str, SpanRecord] = {}
        # event_id -> 有效事件（只保留最高 revision）
        self._outcome_events: dict[str, OutcomeEvent] = {}
        # event_id -> 被隔离的同 revision 冲突
        self._outcome_conflicts: dict[str, list[OutcomeEvent]] = {}
        self._history_revision = 0
        self._m4_jobs: dict[str, M4PublishJob] = {}
        self._m4_job_seq = 0
        self._checkpoint_rows: dict[str, dict[str, Any]] = {}

    def append_span_record(self, record: SpanRecord) -> AppendStatus:
        """按 message_id 去重；同 ID 异内容抛冲突（防止重复投递重复累计）。

        Thread-safe: Protected by internal lock.
        """
        with self._lock:
            return self._append_span_record(record)

    def _append_span_record(self, record: SpanRecord) -> AppendStatus:
        existing = self._span_ids.get(record.message_id)
        if existing is None:
            self._span_ids[record.message_id] = record
            self._span_records.append(record)
            self._history_revision += 1
            return AppendStatus.ADDED
        if existing == record:
            return AppendStatus.IGNORED
        raise ConflictError(
            f"span_record {record.message_id} re-registered with different content"
        )

    def append_outcome_event(self, event: OutcomeEvent) -> AppendStatus:
        """按 event_id 保留最高 revision；旧 revision 忽略（乱序不覆盖）；
        同 revision 异内容隔离为冲突，不参与聚合。

        Thread-safe: Protected by internal lock.
        """
        with self._lock:
            return self._append_outcome_event(event)

    def _append_outcome_event(self, event: OutcomeEvent) -> AppendStatus:
        existing = self._outcome_events.get(event.event_id)
        if existing is None:
            self._outcome_events[event.event_id] = event
            self._history_revision += 1
            return AppendStatus.ADDED
        if event.revision < existing.revision:
            return AppendStatus.IGNORED
        if event == existing:
            return AppendStatus.IGNORED
        if event.revision == existing.revision:
            conflicts = self._outcome_conflicts.setdefault(event.event_id, [])
            if event in conflicts:
                return AppendStatus.IGNORED
            conflicts.append(event)
            return AppendStatus.CONFLICT
        self._outcome_events[event.event_id] = event
        self._history_revision += 1
        return AppendStatus.ADDED

    def span_records(self) -> tuple[SpanRecord, ...]:
        """Thread-safe snapshot of all span records."""
        with self._lock:
            return tuple(self._span_records)

    def outcome_events(self) -> tuple[OutcomeEvent, ...]:
        """有效事件（每个 event_id 只含最高 revision）。

        Thread-safe snapshot of all outcome events.
        """
        with self._lock:
            return tuple(self._outcome_events.values())

    def conflicts(self) -> tuple[OutcomeEvent, ...]:
        """被隔离的同 revision 冲突事件。"""
        with self._lock:
            return tuple(
                event
                for events in self._outcome_conflicts.values()
                for event in events
            )

    def read_evidence(
        self,
    ) -> tuple[tuple[SpanRecord, ...], tuple[OutcomeEvent, ...]]:
        """在同一临界区读取 M4 聚合所需的有效证据。"""
        with self._lock:
            return tuple(self._span_records), tuple(self._outcome_events.values())

    def enqueue_m4_publish(self, workflow_run_id: str) -> str:
        """Enqueue or coalesce one background publication request."""
        with self._lock:
            target = self._history_revision
            for job in reversed(tuple(self._m4_jobs.values())):
                if (
                    job.workflow_run_id == workflow_run_id
                    and job.status == PublishJobStatus.PENDING
                ):
                    self._m4_jobs[job.job_id] = replace(
                        job,
                        target_sequence=max(job.target_sequence, target),
                    )
                    return job.job_id
            self._m4_job_seq += 1
            job_id = f"m4-job-{self._m4_job_seq}"
            self._m4_jobs[job_id] = M4PublishJob(
                job_id=job_id,
                workflow_run_id=workflow_run_id,
                target_sequence=target,
                status=PublishJobStatus.PENDING,
                attempts=0,
                available_at=_utcnow(),
            )
            return job_id

    def claim_m4_publish_job(self, *, lease_seconds: float = 60.0) -> M4PublishJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _utcnow()
        with self._lock:
            candidates = [
                job
                for job in self._m4_jobs.values()
                if (
                    job.status == PublishJobStatus.PENDING
                    and job.available_at <= now
                )
                or (
                    job.status == PublishJobStatus.RUNNING
                    and job.lease_until is not None
                    and job.lease_until <= now
                )
            ]
            if not candidates:
                return None
            job = min(candidates, key=lambda item: (item.available_at, item.job_id))
            claimed = replace(
                job,
                status=PublishJobStatus.RUNNING,
                attempts=job.attempts + 1,
                lease_until=now + timedelta(seconds=lease_seconds),
            )
            self._m4_jobs[job.job_id] = claimed
            return claimed

    def complete_m4_publish_job(self, job_id: str, snapshot_version: str) -> None:
        with self._lock:
            job = self._m4_jobs.get(job_id)
            if job is None:
                raise StorageError(f"unknown M4 publish job {job_id}")
            if job.status != PublishJobStatus.RUNNING:
                raise StorageError(f"M4 publish job {job_id} is not running")
            self._m4_jobs[job_id] = replace(
                job,
                status=PublishJobStatus.SUCCEEDED,
                lease_until=None,
                snapshot_version=snapshot_version,
                error=None,
            )

    def fail_m4_publish_job(
        self,
        job_id: str,
        error: str,
        *,
        retry_delay_seconds: float = 5.0,
        max_attempts: int = 5,
    ) -> None:
        if retry_delay_seconds < 0 or max_attempts <= 0:
            raise ValueError("invalid M4 publish retry parameters")
        with self._lock:
            job = self._m4_jobs.get(job_id)
            if job is None:
                raise StorageError(f"unknown M4 publish job {job_id}")
            terminal = job.attempts >= max_attempts
            self._m4_jobs[job_id] = replace(
                job,
                status=(PublishJobStatus.FAILED if terminal else PublishJobStatus.PENDING),
                available_at=_utcnow() + timedelta(seconds=retry_delay_seconds),
                lease_until=None,
                error=error[:2048],
            )

    def m4_publish_jobs(self) -> tuple[M4PublishJob, ...]:
        with self._lock:
            return tuple(self._m4_jobs.values())

    def save_checkpoint_session(self, session: Any) -> None:
        with self._lock:
            self._checkpoint_rows[session.session_id] = _checkpoint_row(session)

    def delete_checkpoint_session(self, session_id: str) -> None:
        with self._lock:
            self._checkpoint_rows.pop(session_id, None)

    def load_checkpoint_sessions(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._checkpoint_rows.values())


class SnapshotStore:
    """按版本键控的不可变反馈快照发布物。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, FeedbackSnapshot] = {}
        self._publish_order: list[str] = []

    def publish(self, snapshot: FeedbackSnapshot) -> None:
        """发布快照；同版本重复发布一律拒绝（不可覆盖）。"""
        with self._lock:
            if snapshot.snapshot_version in self._snapshots:
                raise SnapshotExists(
                    f"snapshot {snapshot.snapshot_version} is already published"
                )
            self._snapshots[snapshot.snapshot_version] = snapshot
            self._publish_order.append(snapshot.snapshot_version)

    def read(self, version: str) -> FeedbackSnapshot:
        with self._lock:
            try:
                return self._snapshots[version]
            except KeyError:
                raise SnapshotNotFound(f"no snapshot published as {version}") from None

    def latest(self) -> FeedbackSnapshot:
        """最后发布的快照（反馈闭环的读取入口）。"""
        with self._lock:
            if not self._publish_order:
                raise SnapshotNotFound("no snapshot has been published")
            return self._snapshots[self._publish_order[-1]]

    def versions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._publish_order)


_SQLITE_SCHEMA_VERSION = "1"


def _model_json(model: SpanRecord | OutcomeEvent | FeedbackSnapshot) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


class SQLiteHistoryStore:
    """SQLite-backed ``HistoryStore`` suitable for process restarts.

    Each operation uses its own connection. Writers acquire an SQLite
    ``BEGIN IMMEDIATE`` transaction, so duplicate checks and appends remain
    atomic across both threads and processes. Full validated contract JSON is
    retained for replay; selected lineage/version columns are indexed for
    operational audit.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_dir():
            raise StorageError(f"history database path is a directory: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout_seconds = timeout_seconds
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}"
        )
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(self):
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS history_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS span_records (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id TEXT NOT NULL UNIQUE,
                        schema_version TEXT NOT NULL,
                        workflow_run_id TEXT NOT NULL,
                        policy_version TEXT NOT NULL,
                        feedback_snapshot_version TEXT,
                        payload_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_span_workflow
                    ON span_records(workflow_run_id, sequence);

                    CREATE TABLE IF NOT EXISTS outcome_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        message_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        workflow_run_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(event_id, revision)
                    );

                    CREATE INDEX IF NOT EXISTS idx_outcome_workflow
                    ON outcome_events(workflow_run_id, sequence);

                    CREATE TABLE IF NOT EXISTS outcome_conflicts (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(event_id, revision, payload_json)
                    );

                    CREATE TABLE IF NOT EXISTS feedback_snapshots (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_version TEXT NOT NULL UNIQUE,
                        schema_version TEXT NOT NULL,
                        calibration_version TEXT NOT NULL,
                        produced_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS m4_publish_jobs (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL UNIQUE,
                        workflow_run_id TEXT NOT NULL,
                        target_sequence INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        available_at TEXT NOT NULL,
                        lease_until TEXT,
                        snapshot_version TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_m4_publish_claim
                    ON m4_publish_jobs(status, available_at, lease_until, sequence);

                    CREATE INDEX IF NOT EXISTS idx_m4_publish_workflow
                    ON m4_publish_jobs(workflow_run_id, sequence);

                    CREATE TABLE IF NOT EXISTS checkpoint_sessions (
                        session_id TEXT PRIMARY KEY,
                        workflow_run_id TEXT NOT NULL,
                        transaction_ids_json TEXT NOT NULL,
                        scopes_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        handle_retained INTEGER NOT NULL,
                        fault_class TEXT,
                        fault_detail TEXT,
                        rollback_status TEXT NOT NULL,
                        decision TEXT,
                        operation_seq INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_checkpoint_workflow
                    ON checkpoint_sessions(workflow_run_id);
                    """
                )
                connection.execute(
                    "INSERT OR IGNORE INTO history_metadata(key, value) VALUES (?, ?)",
                    ("storage_schema_version", _SQLITE_SCHEMA_VERSION),
                )
                row = connection.execute(
                    "SELECT value FROM history_metadata WHERE key = ?",
                    ("storage_schema_version",),
                ).fetchone()
                if row is None or row["value"] != _SQLITE_SCHEMA_VERSION:
                    found = None if row is None else row["value"]
                    raise StorageError(
                        "unsupported history storage schema: "
                        f"expected {_SQLITE_SCHEMA_VERSION}, found {found}"
                    )
                connection.commit()
        except sqlite3.Error as error:
            raise StorageError(f"cannot initialize history database {self.path}: {error}") from error

    @staticmethod
    def _span_from_json(payload: str) -> SpanRecord:
        try:
            return SpanRecord.model_validate_json(payload)
        except Exception as error:
            raise StorageError("stored span record does not match the current schema") from error

    @staticmethod
    def _outcome_from_json(payload: str) -> OutcomeEvent:
        try:
            return OutcomeEvent.model_validate_json(payload)
        except Exception as error:
            raise StorageError("stored outcome event does not match the current schema") from error

    @staticmethod
    def _snapshot_from_json(payload: str) -> FeedbackSnapshot:
        try:
            return FeedbackSnapshot.model_validate_json(payload)
        except Exception as error:
            raise StorageError("stored feedback snapshot does not match the current schema") from error

    def append_span_record(self, record: SpanRecord) -> AppendStatus:
        payload = _model_json(record)
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM span_records WHERE message_id = ?",
                    (record.message_id,),
                ).fetchone()
                if row is not None:
                    if self._span_from_json(row["payload_json"]) == record:
                        return AppendStatus.IGNORED
                    raise ConflictError(
                        f"span_record {record.message_id} re-registered with different content"
                    )
                meta = record.route_plan.decision_meta
                connection.execute(
                    """
                    INSERT INTO span_records(
                        message_id, schema_version, workflow_run_id,
                        policy_version, feedback_snapshot_version, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.message_id,
                        record.schema_version,
                        record.route_plan.workflow_run_id,
                        meta.policy_version,
                        meta.feedback_snapshot_version,
                        payload,
                    ),
                )
                return AppendStatus.ADDED
        except sqlite3.Error as error:
            raise StorageError(f"cannot append span record: {error}") from error

    def append_outcome_event(self, event: OutcomeEvent) -> AppendStatus:
        payload = _model_json(event)
        try:
            with self._write_transaction() as connection:
                current = connection.execute(
                    """
                    SELECT revision, payload_json
                    FROM outcome_events
                    WHERE event_id = ?
                    ORDER BY revision DESC
                    LIMIT 1
                    """,
                    (event.event_id,),
                ).fetchone()
                if current is None:
                    self._insert_outcome(connection, event, payload)
                    return AppendStatus.ADDED
                current_event = self._outcome_from_json(current["payload_json"])
                if event.revision < current["revision"] or event == current_event:
                    return AppendStatus.IGNORED
                if event.revision == current["revision"]:
                    duplicate = connection.execute(
                        """
                        SELECT 1 FROM outcome_conflicts
                        WHERE event_id = ? AND revision = ? AND payload_json = ?
                        """,
                        (event.event_id, event.revision, payload),
                    ).fetchone()
                    if duplicate is not None:
                        return AppendStatus.IGNORED
                    connection.execute(
                        """
                        INSERT INTO outcome_conflicts(event_id, revision, payload_json)
                        VALUES (?, ?, ?)
                        """,
                        (event.event_id, event.revision, payload),
                    )
                    return AppendStatus.CONFLICT
                self._insert_outcome(connection, event, payload)
                return AppendStatus.ADDED
        except sqlite3.Error as error:
            raise StorageError(f"cannot append outcome event: {error}") from error

    @staticmethod
    def _insert_outcome(
        connection: sqlite3.Connection,
        event: OutcomeEvent,
        payload: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO outcome_events(
                event_id, revision, message_id, schema_version,
                workflow_run_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.revision,
                event.message_id,
                event.schema_version,
                event.workflow_run_id,
                payload,
            ),
        )

    @staticmethod
    def _read_span_rows(connection: sqlite3.Connection) -> tuple[sqlite3.Row, ...]:
        return tuple(
            connection.execute(
                "SELECT payload_json FROM span_records ORDER BY sequence"
            ).fetchall()
        )

    @staticmethod
    def _read_outcome_rows(connection: sqlite3.Connection) -> tuple[sqlite3.Row, ...]:
        return tuple(
            connection.execute(
                """
                WITH selected AS (
                    SELECT event_id, MAX(revision) AS revision, MIN(sequence) AS first_sequence
                    FROM outcome_events
                    GROUP BY event_id
                )
                SELECT event.payload_json
                FROM selected
                JOIN outcome_events AS event
                  ON event.event_id = selected.event_id
                 AND event.revision = selected.revision
                ORDER BY selected.first_sequence
                """
            ).fetchall()
        )

    def span_records(self) -> tuple[SpanRecord, ...]:
        try:
            with self._connection() as connection:
                rows = self._read_span_rows(connection)
        except sqlite3.Error as error:
            raise StorageError(f"cannot read span records: {error}") from error
        return tuple(self._span_from_json(row["payload_json"]) for row in rows)

    def outcome_events(self) -> tuple[OutcomeEvent, ...]:
        try:
            with self._connection() as connection:
                rows = self._read_outcome_rows(connection)
        except sqlite3.Error as error:
            raise StorageError(f"cannot read outcome events: {error}") from error
        return tuple(self._outcome_from_json(row["payload_json"]) for row in rows)

    def conflicts(self) -> tuple[OutcomeEvent, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT payload_json FROM outcome_conflicts ORDER BY sequence"
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read outcome conflicts: {error}") from error
        return tuple(self._outcome_from_json(row["payload_json"]) for row in rows)

    def read_evidence(
        self,
    ) -> tuple[tuple[SpanRecord, ...], tuple[OutcomeEvent, ...]]:
        """Read spans and effective outcome revisions from one DB snapshot."""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN")
                span_rows = self._read_span_rows(connection)
                outcome_rows = self._read_outcome_rows(connection)
                connection.commit()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read history evidence: {error}") from error
        return (
            tuple(self._span_from_json(row["payload_json"]) for row in span_rows),
            tuple(self._outcome_from_json(row["payload_json"]) for row in outcome_rows),
        )

    def publish(self, snapshot: FeedbackSnapshot) -> None:
        payload = _model_json(snapshot)
        try:
            with self._write_transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM feedback_snapshots WHERE snapshot_version = ?",
                    (snapshot.snapshot_version,),
                ).fetchone()
                if exists is not None:
                    raise SnapshotExists(
                        f"snapshot {snapshot.snapshot_version} is already published"
                    )
                connection.execute(
                    """
                    INSERT INTO feedback_snapshots(
                        snapshot_version, schema_version, calibration_version,
                        produced_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_version,
                        snapshot.schema_version,
                        snapshot.safety_calibration_ref.version,
                        snapshot.produced_at.isoformat(),
                        payload,
                    ),
                )
        except sqlite3.Error as error:
            raise StorageError(f"cannot publish feedback snapshot: {error}") from error

    def read(self, version: str) -> FeedbackSnapshot:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM feedback_snapshots WHERE snapshot_version = ?",
                    (version,),
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read feedback snapshot: {error}") from error
        if row is None:
            raise SnapshotNotFound(f"no snapshot published as {version}")
        return self._snapshot_from_json(row["payload_json"])

    def latest(self) -> FeedbackSnapshot:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT payload_json FROM feedback_snapshots
                    ORDER BY sequence DESC LIMIT 1
                    """
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read latest feedback snapshot: {error}") from error
        if row is None:
            raise SnapshotNotFound("no snapshot has been published")
        return self._snapshot_from_json(row["payload_json"])

    def versions(self) -> tuple[str, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT snapshot_version FROM feedback_snapshots
                    ORDER BY sequence
                    """
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read feedback snapshot versions: {error}") from error
        return tuple(row["snapshot_version"] for row in rows)

    @staticmethod
    def _m4_job_from_row(row: sqlite3.Row) -> M4PublishJob:
        try:
            available_at = datetime.fromisoformat(row["available_at"])
            lease_until = (
                datetime.fromisoformat(row["lease_until"])
                if row["lease_until"] is not None
                else None
            )
            return M4PublishJob(
                job_id=row["job_id"],
                workflow_run_id=row["workflow_run_id"],
                target_sequence=int(row["target_sequence"]),
                status=PublishJobStatus(row["status"]),
                attempts=int(row["attempts"]),
                available_at=available_at,
                lease_until=lease_until,
                snapshot_version=row["snapshot_version"],
                error=row["error"],
            )
        except (TypeError, ValueError, KeyError) as error:
            raise StorageError("stored M4 publish job is invalid") from error

    @staticmethod
    def _history_sequence(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            """
            SELECT
                COALESCE((SELECT MAX(sequence) FROM span_records), 0) +
                COALESCE((SELECT MAX(sequence) FROM outcome_events), 0)
                AS target_sequence
            """
        ).fetchone()
        return int(row["target_sequence"])

    def enqueue_m4_publish(self, workflow_run_id: str) -> str:
        now = _utcnow().isoformat()
        job_id = f"m4-job-{uuid4().hex}"
        try:
            with self._write_transaction() as connection:
                target = self._history_sequence(connection)
                pending = connection.execute(
                    """
                    SELECT job_id, target_sequence
                    FROM m4_publish_jobs
                    WHERE workflow_run_id = ? AND status = ?
                    ORDER BY sequence DESC LIMIT 1
                    """,
                    (workflow_run_id, PublishJobStatus.PENDING.value),
                ).fetchone()
                if pending is not None:
                    connection.execute(
                        """
                        UPDATE m4_publish_jobs
                        SET target_sequence = ?, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (max(int(pending["target_sequence"]), target), now, pending["job_id"]),
                    )
                    return str(pending["job_id"])
                connection.execute(
                    """
                    INSERT INTO m4_publish_jobs(
                        job_id, workflow_run_id, target_sequence, status,
                        attempts, available_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        job_id,
                        workflow_run_id,
                        target,
                        PublishJobStatus.PENDING.value,
                        now,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as error:
            raise StorageError(f"cannot enqueue M4 publish job: {error}") from error
        return job_id

    def claim_m4_publish_job(self, *, lease_seconds: float = 60.0) -> M4PublishJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _utcnow()
        lease_until = now + timedelta(seconds=lease_seconds)
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM m4_publish_jobs
                    WHERE (status = ? AND available_at <= ?)
                       OR (status = ? AND lease_until IS NOT NULL AND lease_until <= ?)
                    ORDER BY sequence LIMIT 1
                    """,
                    (
                        PublishJobStatus.PENDING.value,
                        now.isoformat(),
                        PublishJobStatus.RUNNING.value,
                        now.isoformat(),
                    ),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    """
                    UPDATE m4_publish_jobs
                    SET status = ?, attempts = attempts + 1,
                        lease_until = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        PublishJobStatus.RUNNING.value,
                        lease_until.isoformat(),
                        now.isoformat(),
                        row["job_id"],
                    ),
                )
                claimed = connection.execute(
                    "SELECT * FROM m4_publish_jobs WHERE job_id = ?",
                    (row["job_id"],),
                ).fetchone()
                return self._m4_job_from_row(claimed)
        except sqlite3.Error as error:
            raise StorageError(f"cannot claim M4 publish job: {error}") from error

    def complete_m4_publish_job(self, job_id: str, snapshot_version: str) -> None:
        now = _utcnow().isoformat()
        try:
            with self._write_transaction() as connection:
                cursor = connection.execute(
                    """
                    UPDATE m4_publish_jobs
                    SET status = ?, snapshot_version = ?, error = NULL,
                        lease_until = NULL, updated_at = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (
                        PublishJobStatus.SUCCEEDED.value,
                        snapshot_version,
                        now,
                        job_id,
                        PublishJobStatus.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageError(f"M4 publish job {job_id} is not running")
        except sqlite3.Error as error:
            raise StorageError(f"cannot complete M4 publish job: {error}") from error

    def fail_m4_publish_job(
        self,
        job_id: str,
        error: str,
        *,
        retry_delay_seconds: float = 5.0,
        max_attempts: int = 5,
    ) -> None:
        if retry_delay_seconds < 0 or max_attempts <= 0:
            raise ValueError("invalid M4 publish retry parameters")
        now = _utcnow()
        try:
            with self._write_transaction() as connection:
                row = connection.execute(
                    "SELECT attempts FROM m4_publish_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None:
                    raise StorageError(f"unknown M4 publish job {job_id}")
                terminal = int(row["attempts"]) >= max_attempts
                connection.execute(
                    """
                    UPDATE m4_publish_jobs
                    SET status = ?, available_at = ?, lease_until = NULL,
                        error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        (
                            PublishJobStatus.FAILED.value
                            if terminal
                            else PublishJobStatus.PENDING.value
                        ),
                        (now + timedelta(seconds=retry_delay_seconds)).isoformat(),
                        error[:2048],
                        now.isoformat(),
                        job_id,
                    ),
                )
        except sqlite3.Error as sqlite_error:
            raise StorageError(f"cannot fail M4 publish job: {sqlite_error}") from sqlite_error

    def m4_publish_jobs(self) -> tuple[M4PublishJob, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT * FROM m4_publish_jobs ORDER BY sequence"
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read M4 publish jobs: {error}") from error
        return tuple(self._m4_job_from_row(row) for row in rows)

    def save_checkpoint_session(self, session: Any) -> None:
        row = _checkpoint_row(session)
        now = _utcnow().isoformat()
        try:
            with self._write_transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO checkpoint_sessions(
                        session_id, workflow_run_id, transaction_ids_json,
                        scopes_json, state, handle_retained, fault_class,
                        fault_detail, rollback_status, decision, operation_seq,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        workflow_run_id = excluded.workflow_run_id,
                        transaction_ids_json = excluded.transaction_ids_json,
                        scopes_json = excluded.scopes_json,
                        state = excluded.state,
                        handle_retained = excluded.handle_retained,
                        fault_class = excluded.fault_class,
                        fault_detail = excluded.fault_detail,
                        rollback_status = excluded.rollback_status,
                        decision = excluded.decision,
                        operation_seq = excluded.operation_seq,
                        updated_at = excluded.updated_at
                    WHERE excluded.operation_seq >= checkpoint_sessions.operation_seq
                    """,
                    (
                        row["session_id"],
                        row["workflow_run_id"],
                        json.dumps(row["transaction_ids"], separators=(",", ":")),
                        json.dumps(row["scopes"], sort_keys=True, separators=(",", ":")),
                        row["state"],
                        int(row["handle_retained"]),
                        row["fault_class"],
                        row["fault_detail"],
                        row["rollback_status"],
                        row["decision"],
                        row["operation_seq"],
                        now,
                    ),
                )
        except sqlite3.Error as error:
            raise StorageError(f"cannot save checkpoint session: {error}") from error

    def delete_checkpoint_session(self, session_id: str) -> None:
        try:
            with self._write_transaction() as connection:
                connection.execute(
                    "DELETE FROM checkpoint_sessions WHERE session_id = ?",
                    (session_id,),
                )
        except sqlite3.Error as error:
            raise StorageError(f"cannot delete checkpoint session: {error}") from error

    def load_checkpoint_sessions(self) -> tuple[dict[str, Any], ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT * FROM checkpoint_sessions ORDER BY updated_at, session_id"
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"cannot load checkpoint sessions: {error}") from error
        result = []
        for row in rows:
            try:
                result.append(
                    {
                        "session_id": row["session_id"],
                        "workflow_run_id": row["workflow_run_id"],
                        "transaction_ids": json.loads(row["transaction_ids_json"]),
                        "scopes": json.loads(row["scopes_json"]),
                        "state": row["state"],
                        "handle_retained": bool(row["handle_retained"]),
                        "fault_class": row["fault_class"],
                        "fault_detail": row["fault_detail"],
                        "rollback_status": row["rollback_status"],
                        "decision": row["decision"],
                        "operation_seq": int(row["operation_seq"]),
                    }
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise StorageError("stored checkpoint session is invalid") from error
        return tuple(result)

    def storage_metadata(self) -> dict[str, str]:
        """Return immutable store metadata for audit and migration checks."""
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT key, value FROM history_metadata ORDER BY key"
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read history metadata: {error}") from error
        return {row["key"]: row["value"] for row in rows}


class MemoryProbeLedger:
    """内存探针台账：进程内按实验作用域累计（无持久化，供测试/手动入口）。

    用锁保护 try_consume 的“检查-插入”，同一进程内并发也绝不超额；
    decision_id 全局唯一，但幂等仅在同 experiment+workflow 下成立。
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_workflow: dict[tuple[str, str], int] = {}
        self._by_experiment: dict[str, int] = {}
        self._seen: dict[str, tuple[str, str]] = {}

    def workflow_used(self, experiment_id: str, workflow_run_id: str) -> int:
        with self._lock:
            return self._by_workflow.get((experiment_id, workflow_run_id), 0)

    def total_used(self, experiment_id: str) -> int:
        with self._lock:
            return self._by_experiment.get(experiment_id, 0)

    def try_consume(
        self,
        experiment_id: str,
        workflow_run_id: str,
        decision_id: str,
        reason: str,
        *,
        workflow_limit: int | None,
        total_limit: int | None,
    ) -> bool:
        del reason  # 内存台账不落原因；sqlite 台账保留
        with self._lock:
            seen = self._seen.get(decision_id)
            if seen is not None:
                # 幂等仅在 experiment+workflow 完全一致时成立；
                # decision_id 被别的 experiment/workflow 占用 → 失败。
                return seen == (experiment_id, workflow_run_id)
            if workflow_limit is not None and self._by_workflow.get(
                (experiment_id, workflow_run_id), 0
            ) >= workflow_limit:
                return False
            if total_limit is not None and self._by_experiment.get(experiment_id, 0) >= total_limit:
                return False
            self._seen[decision_id] = (experiment_id, workflow_run_id)
            self._by_workflow[(experiment_id, workflow_run_id)] = (
                self._by_workflow.get((experiment_id, workflow_run_id), 0) + 1
            )
            self._by_experiment[experiment_id] = self._by_experiment.get(experiment_id, 0) + 1
            return True


class SQLiteProbeLedger:
    """SQLite 探针台账：跨进程持久化累计 + 消耗原因审计（文档 A7）。

    与 SQLiteHistoryStore 共享同一数据库文件（独立连接，WAL 并发安全）；
    每条消费以 route_plan_id 为主键幂等，进程重启/重复记账不重复扣减。
    """

    def __init__(self, path: Path | str, *, timeout_seconds: float = 30.0) -> None:
        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_dir():
            raise StorageError(f"probe ledger path is a directory: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout_seconds = timeout_seconds
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}"
        )
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS probe_budget_usage (
                    decision_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    workflow_run_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    consumed_at TEXT NOT NULL
                );
                """
            )
            # 迁移：旧库无 experiment_id 列时补列，存量行 scope 到空实验
            # （不参与任何真实实验的统计，避免污染新实验预算）。
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(probe_budget_usage)").fetchall()
            }
            if "experiment_id" not in columns:
                connection.execute(
                    "ALTER TABLE probe_budget_usage"
                    " ADD COLUMN experiment_id TEXT NOT NULL DEFAULT ''"
                )
            indexes = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA index_list(probe_budget_usage)"
                ).fetchall()
            }
            if "idx_probe_budget_experiment_workflow" not in indexes:
                connection.execute("DROP INDEX IF EXISTS idx_probe_budget_workflow")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_probe_budget_experiment_workflow"
                    " ON probe_budget_usage (experiment_id, workflow_run_id)"
                )

    def workflow_used(self, experiment_id: str, workflow_run_id: str) -> int:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM probe_budget_usage"
                    " WHERE experiment_id = ? AND workflow_run_id = ?",
                    (experiment_id, workflow_run_id),
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read probe ledger: {error}") from error
        return int(row["n"]) if row is not None else 0

    def total_used(self, experiment_id: str) -> int:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM probe_budget_usage WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"cannot read probe ledger: {error}") from error
        return int(row["n"]) if row is not None else 0

    def try_consume(
        self,
        experiment_id: str,
        workflow_run_id: str,
        decision_id: str,
        reason: str,
        *,
        workflow_limit: int | None,
        total_limit: int | None,
    ) -> bool:
        """原子扣减：BEGIN IMMEDIATE 串行化并发，检查与插入在同一事务内。

        decision_id 全局唯一：已存在且 experiment+workflow 一致 → 幂等成功；
        已被别的 experiment/workflow 占用 → False。任一上限耗尽回滚并返回 False。
        """
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    exists = connection.execute(
                        "SELECT experiment_id, workflow_run_id FROM probe_budget_usage"
                        " WHERE decision_id = ?",
                        (decision_id,),
                    ).fetchone()
                    if exists is not None:
                        connection.commit()
                        return (exists["experiment_id"], exists["workflow_run_id"]) == (
                            experiment_id,
                            workflow_run_id,
                        )
                    if workflow_limit is not None:
                        row = connection.execute(
                            "SELECT COUNT(*) AS n FROM probe_budget_usage"
                            " WHERE experiment_id = ? AND workflow_run_id = ?",
                            (experiment_id, workflow_run_id),
                        ).fetchone()
                        if int(row["n"]) >= workflow_limit:
                            connection.rollback()
                            return False
                    if total_limit is not None:
                        row = connection.execute(
                            "SELECT COUNT(*) AS n FROM probe_budget_usage"
                            " WHERE experiment_id = ?",
                            (experiment_id,),
                        ).fetchone()
                        if int(row["n"]) >= total_limit:
                            connection.rollback()
                            return False
                    connection.execute(
                        "INSERT INTO probe_budget_usage"
                        " (decision_id, experiment_id, workflow_run_id, reason, consumed_at)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            decision_id,
                            experiment_id,
                            workflow_run_id,
                            reason,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    connection.commit()
                    return True
                except BaseException:
                    connection.rollback()
                    raise
        except sqlite3.Error as error:
            raise StorageError(f"cannot consume probe budget: {error}") from error


__all__ = [
    "AppendStatus",
    "ConflictError",
    "EventHistoryStore",
    "EventStore",
    "ExpiredRef",
    "HistoryStore",
    "M4PublishJob",
    "MemoryProbeLedger",
    "PublishJobStatus",
    "RefNotFound",
    "RequestResponseStore",
    "SQLiteHistoryStore",
    "SnapshotExists",
    "SnapshotHistoryStore",
    "SnapshotNotFound",
    "SnapshotStore",
    "SQLiteProbeLedger",
    "StorageError",
    "UnauthorizedRef",
]
