import pytest

from traceoffload.m3.checkpoint import (
    CheckpointSession,
    CheckpointSessionManager,
    CheckpointState,
    MemoryCheckpointJournal,
)
from traceoffload.storage import SQLiteHistoryStore


def _create(manager: CheckpointSessionManager) -> None:
    manager.create(
        "execution-1",
        workflow_run_id="workflow-1",
        transaction_ids=("transaction-1",),
        scopes={"transaction-1": "reversible_write"},
        actions={},
        verdicts={},
    )


def test_nonterminal_session_survives_process_restart(tmp_path) -> None:
    store = SQLiteHistoryStore(tmp_path / "history.sqlite3")
    manager = CheckpointSessionManager(journal=store)
    _create(manager)
    manager.transition("execution-1", CheckpointState.EXECUTING)

    reopened = CheckpointSessionManager(
        journal=SQLiteHistoryStore(tmp_path / "history.sqlite3")
    )
    session = reopened.get("execution-1")

    assert session is not None
    assert session.state == CheckpointState.RECOVERY_REQUIRED
    assert session.transaction_ids == ("transaction-1",)
    assert session.handle_retained is True
    assert session.fault_detail == "process_restart_from:executing"


def test_terminal_release_removes_the_durable_session(tmp_path) -> None:
    path = tmp_path / "history.sqlite3"
    manager = CheckpointSessionManager(journal=SQLiteHistoryStore(path))
    _create(manager)

    manager.close("execution-1", decision="aborted")

    reopened = CheckpointSessionManager(journal=SQLiteHistoryStore(path))
    assert reopened.get("execution-1") is None
    assert SQLiteHistoryStore(path).load_checkpoint_sessions() == ()


def test_rollback_has_explicit_release_pending_state() -> None:
    manager = CheckpointSessionManager()
    _create(manager)
    manager.transition("execution-1", CheckpointState.EXECUTING)

    manager.begin_rollback(("transaction-1",))
    assert manager.get("execution-1").state == CheckpointState.ROLLING_BACK

    manager.finish_rollback(
        ("transaction-1",),
        rollback_status="succeeded",
    )
    session = manager.get("execution-1")
    assert session is not None
    assert session.state == CheckpointState.ROLLED_BACK_RELEASE_PENDING
    assert session.handle_retained is True


def test_invalid_checkpoint_state_jump_is_rejected() -> None:
    session = CheckpointSession(
        session_id="execution-1",
        workflow_run_id="workflow-1",
        state=CheckpointState.PREPARED,
    )

    with pytest.raises(RuntimeError, match="invalid checkpoint transition"):
        session.transition(CheckpointState.COMMITTED_RELEASE_PENDING)


def test_transaction_is_journaled_before_backend_begin() -> None:
    journal = MemoryCheckpointJournal()
    manager = CheckpointSessionManager(journal=journal)
    manager.reserve("execution-1", workflow_run_id="workflow-1")

    manager.plan_transaction(
        "execution-1",
        transaction_id="transaction-1",
        scope="reversible_write",
        action=object(),
        verdict=object(),
    )

    row = journal.load_checkpoint_sessions()[0]
    assert row["state"] == "preparing"
    assert row["transaction_ids"] == ["transaction-1"]
    assert row["handle_retained"] is True


def test_restart_converts_any_nonterminal_state_to_recovery_required(tmp_path) -> None:
    path = tmp_path / "history.sqlite3"
    first = CheckpointSessionManager(journal=SQLiteHistoryStore(path))
    _create(first)
    first.transition("execution-1", CheckpointState.EXECUTING)
    first.transition("execution-1", CheckpointState.VERIFYING)

    recovered = CheckpointSessionManager(journal=SQLiteHistoryStore(path))
    session = recovered.get("execution-1")

    assert session is not None
    assert session.recovery_required is True
    assert session.state == CheckpointState.RECOVERY_REQUIRED
    assert session.fault_class == "env_fault"
