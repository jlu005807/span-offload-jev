"""Single-owner checkpoint lifecycle for one guarded span.

The executor remains responsible for orchestration, but this object owns the
span's transaction set and its monotonic lifecycle.  Backend calls are still
made by ToolProxy; they cannot clear this state directly.  A journal may be
provided by a persistent history store so a process restart can discover
non-terminal sessions instead of silently losing their handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Any, Mapping, Protocol


class CheckpointState(StrEnum):
    NEW = "new"
    PREPARING = "preparing"
    PREPARED = "prepared"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMMITTING = "committing"
    COMMITTED_RELEASE_PENDING = "committed_release_pending"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK_RELEASE_PENDING = "rolled_back_release_pending"
    CLOSED = "closed"
    RECOVERY_REQUIRED = "recovery_required"


_TERMINAL = {CheckpointState.CLOSED}
_RECOVERY = {
    CheckpointState.RECOVERY_REQUIRED,
    CheckpointState.COMMITTED_RELEASE_PENDING,
    CheckpointState.ROLLED_BACK_RELEASE_PENDING,
}

_ALLOWED_TRANSITIONS: dict[CheckpointState, frozenset[CheckpointState]] = {
    CheckpointState.NEW: frozenset({CheckpointState.PREPARING, CheckpointState.PREPARED}),
    CheckpointState.PREPARING: frozenset(
        {
            CheckpointState.PREPARED,
            CheckpointState.ROLLING_BACK,
            CheckpointState.RECOVERY_REQUIRED,
            CheckpointState.CLOSED,
        }
    ),
    CheckpointState.PREPARED: frozenset(
        {
            CheckpointState.EXECUTING,
            CheckpointState.ROLLING_BACK,
            CheckpointState.RECOVERY_REQUIRED,
            CheckpointState.CLOSED,
        }
    ),
    CheckpointState.EXECUTING: frozenset(
        {
            CheckpointState.VERIFYING,
            CheckpointState.ROLLING_BACK,
            CheckpointState.RECOVERY_REQUIRED,
            CheckpointState.CLOSED,
        }
    ),
    CheckpointState.VERIFYING: frozenset(
        {
            CheckpointState.COMMITTING,
            CheckpointState.ROLLING_BACK,
            CheckpointState.RECOVERY_REQUIRED,
            CheckpointState.CLOSED,
        }
    ),
    CheckpointState.COMMITTING: frozenset(
        {
            CheckpointState.COMMITTED_RELEASE_PENDING,
            CheckpointState.ROLLING_BACK,
            CheckpointState.RECOVERY_REQUIRED,
        }
    ),
    CheckpointState.COMMITTED_RELEASE_PENDING: frozenset(
        {CheckpointState.ROLLING_BACK, CheckpointState.RECOVERY_REQUIRED, CheckpointState.CLOSED}
    ),
    CheckpointState.ROLLING_BACK: frozenset(
        {CheckpointState.ROLLED_BACK_RELEASE_PENDING, CheckpointState.RECOVERY_REQUIRED}
    ),
    CheckpointState.ROLLED_BACK_RELEASE_PENDING: frozenset(
        {CheckpointState.RECOVERY_REQUIRED, CheckpointState.CLOSED}
    ),
    CheckpointState.RECOVERY_REQUIRED: frozenset(
        {CheckpointState.ROLLING_BACK, CheckpointState.CLOSED}
    ),
    CheckpointState.CLOSED: frozenset(),
}


@dataclass(frozen=True)
class LifecycleResult:
    """Normalized result of a backend lifecycle operation."""

    ok: bool
    operation: str = ""
    fault_class: str | None = None
    handle_retained: bool = False
    retryable: bool = False
    detail: str | None = None


@dataclass
class CheckpointSession:
    session_id: str
    workflow_run_id: str
    transaction_ids: tuple[str, ...] = ()
    scopes: dict[str, str] = field(default_factory=dict)
    actions: dict[str, Any] = field(default_factory=dict)
    verdicts: dict[str, Any] = field(default_factory=dict)
    l2_before: Any = None
    state: CheckpointState = CheckpointState.NEW
    handle_retained: bool = False
    fault_class: str | None = None
    fault_detail: str | None = None
    rollback_status: str = "not_needed"
    decision: str | None = None
    operation_seq: int = 0

    @property
    def active(self) -> bool:
        return self.state not in _TERMINAL

    @property
    def recovery_required(self) -> bool:
        # A live PREPARED/EXECUTING session necessarily owns a handle; that is
        # normal resource ownership, not a recovery lock.  Recovery begins only
        # after commit/rollback is waiting for release or an operation failed.
        return self.state in _RECOVERY

    def transition(
        self,
        state: CheckpointState,
        *,
        operation: str = "",
        handle_retained: bool | None = None,
        fault_class: str | None = None,
        fault_detail: str | None = None,
        rollback_status: str | None = None,
        decision: str | None = None,
    ) -> None:
        """Apply a monotonic state transition and retain operation facts."""

        if self.state == CheckpointState.CLOSED:
            raise RuntimeError(f"checkpoint session {self.session_id} is already closed")
        if state == CheckpointState.NEW:
            raise ValueError("checkpoint sessions cannot transition back to NEW")
        if state != self.state and state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"invalid checkpoint transition for {self.session_id}: "
                f"{self.state.value} -> {state.value}"
            )
        if state == CheckpointState.RECOVERY_REQUIRED:
            self.handle_retained = True if handle_retained is None else handle_retained
        elif handle_retained is not None:
            self.handle_retained = handle_retained
        self.state = state
        self.operation_seq += 1
        if operation:
            self.fault_detail = operation if fault_detail is None else fault_detail
        if fault_class is not None:
            self.fault_class = fault_class
        if fault_detail is not None:
            self.fault_detail = fault_detail
        if rollback_status is not None:
            self.rollback_status = rollback_status
        if decision is not None:
            self.decision = decision


class CheckpointJournal(Protocol):
    """Durable metadata boundary; backend handles remain opaque strings."""

    def save_checkpoint_session(self, session: CheckpointSession) -> None: ...

    def delete_checkpoint_session(self, session_id: str) -> None: ...

    def load_checkpoint_sessions(self) -> tuple[Mapping[str, Any], ...]: ...


class MemoryCheckpointJournal:
    """Small journal used by unit tests and non-persistent adapters."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._rows: dict[str, dict[str, Any]] = {}

    def save_checkpoint_session(self, session: CheckpointSession) -> None:
        with self._lock:
            self._rows[session.session_id] = _session_row(session)

    def delete_checkpoint_session(self, session_id: str) -> None:
        with self._lock:
            self._rows.pop(session_id, None)

    def load_checkpoint_sessions(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._rows.values())


def _session_row(session: CheckpointSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "workflow_run_id": session.workflow_run_id,
        "transaction_ids": list(session.transaction_ids),
        "scopes": dict(session.scopes),
        "state": session.state.value,
        "handle_retained": session.handle_retained,
        "fault_class": session.fault_class,
        "fault_detail": session.fault_detail,
        "rollback_status": session.rollback_status,
        "decision": session.decision,
        "operation_seq": session.operation_seq,
    }


class CheckpointSessionManager:
    """Own active sessions and make terminal cleanup explicit."""

    def __init__(self, *, journal: CheckpointJournal | None = None) -> None:
        self._journal = journal or MemoryCheckpointJournal()
        self._sessions: dict[str, CheckpointSession] = {}
        self._load_active_sessions()

    def create(
        self,
        session_id: str,
        *,
        workflow_run_id: str,
        transaction_ids: tuple[str, ...],
        scopes: Mapping[str, Any],
        actions: Mapping[str, Any],
        verdicts: Mapping[str, Any],
        l2_before: Any = None,
    ) -> CheckpointSession:
        session = self.reserve(session_id, workflow_run_id=workflow_run_id)
        for transaction_id in transaction_ids:
            self.plan_transaction(
                session_id,
                transaction_id=transaction_id,
                scope=scopes.get(transaction_id, "unknown"),
                action=actions.get(transaction_id),
                verdict=verdicts.get(transaction_id),
            )
        session.l2_before = l2_before
        return self.transition(
            session_id,
            CheckpointState.PREPARED,
            operation="checkpoint_prepared",
        )

    def reserve(
        self,
        session_id: str,
        *,
        workflow_run_id: str,
    ) -> CheckpointSession:
        """Persist PREPARING before any backend checkpoint operation."""
        existing = self._sessions.get(session_id)
        if existing is not None and existing.active:
            raise RuntimeError(f"checkpoint session {session_id} already exists")
        session = CheckpointSession(
            session_id=session_id,
            workflow_run_id=workflow_run_id,
            state=CheckpointState.PREPARING,
        )
        self._sessions[session_id] = session
        self._persist(session)
        return session

    def plan_transaction(
        self,
        session_id: str,
        *,
        transaction_id: str,
        scope: Any,
        action: Any,
        verdict: Any,
    ) -> CheckpointSession:
        """Journal a possible handle before asking ToolProxy to create it."""
        session = self._sessions[session_id]
        if session.state != CheckpointState.PREPARING:
            raise RuntimeError("transactions can only be planned while PREPARING")
        scope_value = str(scope.value if hasattr(scope, "value") else scope)
        if transaction_id not in session.transaction_ids:
            session.transaction_ids += (transaction_id,)
        session.scopes[transaction_id] = scope_value
        if action is not None:
            session.actions[transaction_id] = action
        if verdict is not None:
            session.verdicts[transaction_id] = verdict
        if scope_value not in {"read_only", "EffectScope.READ_ONLY"}:
            session.handle_retained = True
        session.transition(
            CheckpointState.PREPARING,
            operation="transaction_planned",
        )
        self._persist(session)
        return session

    def get(self, session_id: str) -> CheckpointSession | None:
        return self._sessions.get(session_id)

    def active_sessions(self) -> tuple[CheckpointSession, ...]:
        return tuple(session for session in self._sessions.values() if session.active)

    def transition(self, session_id: str, state: CheckpointState, **kwargs: Any) -> CheckpointSession:
        session = self._sessions[session_id]
        session.transition(state, **kwargs)
        self._persist(session)
        return session

    def close(self, session_id: str, *, decision: str) -> None:
        session = self._sessions[session_id]
        session.transition(
            CheckpointState.CLOSED,
            decision=decision,
            handle_retained=False,
            operation="release_succeeded",
        )
        self._persist(session)
        self._journal.delete_checkpoint_session(session_id)
        self._sessions.pop(session_id, None)

    def abandon_unstarted(self, session_id: str) -> None:
        """Drop a reservation only when no backend handle could have existed."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        if session.handle_retained or session.transaction_ids:
            raise RuntimeError("cannot abandon a checkpoint session with planned handles")
        try:
            self._journal.delete_checkpoint_session(session_id)
        finally:
            self._sessions.pop(session_id, None)

    def retain_for_recovery(
        self,
        session_id: str,
        *,
        detail: str,
        decision: str | None = None,
        rollback_status: str | None = None,
    ) -> CheckpointSession:
        return self.transition(
            session_id,
            CheckpointState.RECOVERY_REQUIRED,
            operation="recovery_required",
            handle_retained=True,
            fault_class="env_fault",
            fault_detail=detail,
            decision=decision,
            rollback_status=rollback_status,
        )

    def begin_rollback(self, transaction_ids: tuple[str, ...]) -> None:
        """Move the single owning session to ROLLING_BACK before backend I/O."""
        transaction_set = set(transaction_ids)
        for session in self.active_sessions():
            if not transaction_set.intersection(session.transaction_ids):
                continue
            if session.state != CheckpointState.ROLLING_BACK:
                self.transition(
                    session.session_id,
                    CheckpointState.ROLLING_BACK,
                    operation="rollback_started",
                )

    def finish_rollback(
        self,
        transaction_ids: tuple[str, ...],
        *,
        rollback_status: str,
    ) -> None:
        """Record rollback success while retaining the handle until release."""
        transaction_set = set(transaction_ids)
        for session in self.active_sessions():
            if not transaction_set.intersection(session.transaction_ids):
                continue
            if session.state == CheckpointState.ROLLING_BACK:
                self.transition(
                    session.session_id,
                    CheckpointState.ROLLED_BACK_RELEASE_PENDING,
                    operation="rollback_succeeded",
                    rollback_status=rollback_status,
                )

    def journal_rows(self) -> tuple[Mapping[str, Any], ...]:
        return self._journal.load_checkpoint_sessions()

    def install_legacy_sessions(self, sessions: Mapping[str, tuple[str, ...]]) -> None:
        """Import the old executor map for adapters upgrading in place."""
        for session_id, transaction_ids in sessions.items():
            if session_id in self._sessions:
                continue
            session = CheckpointSession(
                session_id=session_id,
                workflow_run_id=session_id,
                transaction_ids=tuple(transaction_ids),
                state=CheckpointState.RECOVERY_REQUIRED,
                handle_retained=True,
                fault_class="env_fault",
                fault_detail="legacy_session_import",
            )
            self._sessions[session_id] = session
            self._persist(session)

    def _persist(self, session: CheckpointSession) -> None:
        self._journal.save_checkpoint_session(session)

    def _load_active_sessions(self) -> None:
        """Rehydrate durable non-terminal sessions after process restart."""
        for row in self._journal.load_checkpoint_sessions():
            try:
                state = CheckpointState(str(row["state"]))
                if state == CheckpointState.CLOSED:
                    self._journal.delete_checkpoint_session(str(row["session_id"]))
                    continue
                session = CheckpointSession(
                    session_id=str(row["session_id"]),
                    workflow_run_id=str(row["workflow_run_id"]),
                    transaction_ids=tuple(str(value) for value in row.get("transaction_ids", ())),
                    scopes={str(key): str(value) for key, value in dict(row.get("scopes", {})).items()},
                    state=state,
                    handle_retained=bool(row.get("handle_retained", False)),
                    fault_class=(str(row["fault_class"]) if row.get("fault_class") else None),
                    fault_detail=(str(row["fault_detail"]) if row.get("fault_detail") else None),
                    rollback_status=str(row.get("rollback_status", "not_needed")),
                    decision=(str(row["decision"]) if row.get("decision") else None),
                    operation_seq=int(row.get("operation_seq", 0)),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("invalid durable checkpoint session") from error
            if session.session_id in self._sessions:
                raise RuntimeError(f"duplicate durable checkpoint session {session.session_id}")
            if session.state != CheckpointState.RECOVERY_REQUIRED:
                prior_state = session.state
                session.state = CheckpointState.RECOVERY_REQUIRED
                session.handle_retained = bool(session.transaction_ids) or session.handle_retained
                session.fault_class = "env_fault"
                session.fault_detail = f"process_restart_from:{prior_state.value}"
                session.operation_seq += 1
            self._sessions[session.session_id] = session
            self._persist(session)


__all__ = [
    "CheckpointJournal",
    "CheckpointSession",
    "CheckpointSessionManager",
    "CheckpointState",
    "LifecycleResult",
    "MemoryCheckpointJournal",
]
