from __future__ import annotations

from threading import Thread

from traceoffload.storage import MemoryProbeLedger, SQLiteProbeLedger


def test_experiment_scoping_isolates_total_budget() -> None:
    ledger = MemoryProbeLedger()
    assert ledger.try_consume('exp-old', 'wf', 'd1', 'r', workflow_limit=None, total_limit=1) is True
    assert ledger.total_used('exp-old') == 1
    assert ledger.total_used('exp-new') == 0
    assert ledger.try_consume('exp-new', 'wf2', 'd2', 'r', workflow_limit=None, total_limit=1) is True
    assert ledger.total_used('exp-new') == 1


def test_decision_id_mismatch_fails_closed_memory() -> None:
    ledger = MemoryProbeLedger()
    assert ledger.try_consume('exp-a', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is True
    assert ledger.try_consume('exp-b', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is False
    assert ledger.try_consume('exp-a', 'wf-b', 'd1', 'r', workflow_limit=None, total_limit=None) is False
    assert ledger.try_consume('exp-a', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is True
    assert ledger.total_used('exp-a') == 1


def test_decision_id_mismatch_fails_closed_sqlite(tmp_path) -> None:
    path = tmp_path / 'history.sqlite3'
    first = SQLiteProbeLedger(path)
    assert first.try_consume('exp-a', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is True
    assert first.try_consume('exp-b', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is False
    assert first.try_consume('exp-a', 'wf-a', 'd1', 'r', workflow_limit=None, total_limit=None) is True
    assert first.total_used('exp-a') == 1
    assert first.total_used('exp-b') == 0


def test_sqlite_experiment_scoping_across_instances(tmp_path) -> None:
    path = tmp_path / 'history.sqlite3'
    first = SQLiteProbeLedger(path)
    first.try_consume('exp-1', 'wf-1', 'd1', 'r', workflow_limit=None, total_limit=None)
    second = SQLiteProbeLedger(path)
    assert second.total_used('exp-1') == 1
    assert second.total_used('exp-2') == 0
    assert second.workflow_used('exp-1', 'wf-1') == 1
    assert second.workflow_used('exp-2', 'wf-1') == 0


def test_sqlite_concurrent_threads_single_winner(tmp_path) -> None:
    path = tmp_path / 'history.sqlite3'
    SQLiteProbeLedger(path)  # prewarm schema to avoid concurrent DDL contention
    results: list[bool] = []

    def consume(wf: str, decision_id: str) -> None:
        ledger = SQLiteProbeLedger(path)
        results.append(
            ledger.try_consume('exp-c', wf, decision_id, 'r', workflow_limit=None, total_limit=1)
        )

    threads = [Thread(target=consume, args=(f'wf-{i}', f'd{i}')) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert SQLiteProbeLedger(path).total_used('exp-c') == 1
