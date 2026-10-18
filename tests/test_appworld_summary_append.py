"""一轮的 summary 改成**追加**写 —— 因为驱动要改成「一个进程一个任务」。

为什么非要换进程结构
------------------
AppWorld 的 `Requester.close()` 无条件 `freezer.stop()`，而且**不检查停过没有** ——
`local_remote.unset_local_date_and_time` 里那句 `freezer._freezer = None`（本该是
「已停过」的护栏）写在 `stop()` **之后**，永远拦不住第二次。于是：

    同一个进程里跑第 N+1 个任务
      → AppWorld.initialize() → close_all()
      → 对第 N 个任务留下的冻结器**停第二次**
      → freezegun 全局栈弹空 → IndexError: pop from empty list

实测：45 次任务执行崩 1 次 ⇒ 90 任务一轮存活率 ≈ **(44/45)^90 ≈ 13%**。

native 那条路（`run_native_5rounds_5workers.sh:130-132`）**一个进程只跑一个任务**，
冻结器列表本来就是空的，**从来不产生这个条件** —— 900 次采集 0 崩溃。

⇒ 我们照做。驱动每个任务起一次 python；脚本这边就得把 summary 从**覆盖**改成
**追加**，否则后一个任务会把前一个的抹掉，一轮跑完只剩最后一条。

⚠️ 追加必须按 `instance_id` **去重**：重跑某一格时应当**替换**，不是多出一条。
否则「90 条」这个验收判据会在重跑后变成 91 条，而看起来像是任务数写错了。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments/run_appworld.py"


def _load():
    """按路径 import —— `bench-resources` 带连字符，不是合法的包名。"""
    spec = importlib.util.spec_from_file_location("_ra_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ra():
    return _load()


def _row(task: str, **over: object) -> dict:
    return {"record_type": "workflow", "instance_id": task, "task_success": True, **over}


# --------------------------------------------------------------------------
# 合并语义
# --------------------------------------------------------------------------

def test_empty_existing_yields_just_the_new_rows(ra) -> None:
    merged = ra._merge_summary_rows([], [_row("a"), _row("b")])

    assert [r["instance_id"] for r in merged] == ["a", "b"]


def test_existing_rows_are_kept_in_order_and_new_ones_appended(ra) -> None:
    merged = ra._merge_summary_rows([_row("a")], [_row("b")])

    assert [r["instance_id"] for r in merged] == ["a", "b"]


def test_a_rerun_replaces_its_own_row_instead_of_duplicating(ra) -> None:
    """**这条是判据**：重跑某一格后总数仍是 2，不是 3。

    驱动会重跑崩掉的那几个任务。多出一条的话，验收的「每轮 =90 条」会变成 91，
    而那个症状看起来像是实例列表写重了 —— 两件完全不同的事。
    """
    merged = ra._merge_summary_rows(
        [_row("a", task_success=True), _row("b")],
        [_row("a", task_success=False)],
    )

    assert [r["instance_id"] for r in merged] == ["a", "b"]
    assert merged[0]["task_success"] is False, "重跑的结果没盖过旧的"
    assert merged[1] == _row("b"), "替换某一个不该动到别的行"


def test_replaced_row_keeps_its_original_position(ra) -> None:
    """替换是**原地**盖 —— 不把重跑的那条挪到末尾。

    挪到末尾会让 summary 的顺序取决于「哪个任务崩过」，而下游按行号对齐
    （`aggregate_sweep` 那类）就会错位。
    """
    merged = ra._merge_summary_rows(
        [_row("a"), _row("b"), _row("c")],
        [_row("b", note="rerun")],
    )

    assert [r["instance_id"] for r in merged] == ["a", "b", "c"]
    assert merged[1].get("note") == "rerun"


# --------------------------------------------------------------------------
# 读盘：坏了不能拦着跑
# --------------------------------------------------------------------------

def test_reads_back_an_empty_file_as_no_rows(ra, tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text("", encoding="utf-8")

    assert ra._read_summary_rows(f) == []


def test_reads_back_a_missing_file_as_no_rows(ra, tmp_path: Path) -> None:
    assert ra._read_summary_rows(tmp_path / "nope.json") == []


def test_a_corrupt_file_does_not_take_the_run_down(ra, tmp_path: Path) -> None:
    """**半截文件是最可能的那种坏**：上一个进程被杀在写盘中途。

    这里抛的话，这一格的数据全丢；当空处理的话，最多丢前面那些格，
    而后面的还会照常追加 —— 后者更好。
    """
    f = tmp_path / "s.json"
    f.write_text('[{"record_type": "workflow"', encoding="utf-8")

    assert ra._read_summary_rows(f) == []


def test_a_non_list_payload_is_ignored(ra, tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"workflows": []}), encoding="utf-8")

    assert ra._read_summary_rows(f) == []


# --------------------------------------------------------------------------
# 端到端：三次「一个进程一个任务」跑完，盘上是三条
# --------------------------------------------------------------------------

def test_three_separate_invocations_accumulate(ra, tmp_path: Path) -> None:
    """模拟驱动连起三次进程 —— 每次只产一条，盘上要攒到三条。"""
    path = tmp_path / "appworld-classifier_route-summary.json"

    for task in ("a", "b", "c"):
        rows = ra._merge_summary_rows(ra._read_summary_rows(path), [_row(task)])
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert [r["instance_id"] for r in on_disk] == ["a", "b", "c"]


def test_a_crash_then_a_rerun_leaves_exactly_the_tasks(ra, tmp_path: Path) -> None:
    """崩掉的那格没落盘 ⇒ 重跑补上 ⇒ 盘上是 2 条，不是 3 条。"""
    path = tmp_path / "s.json"
    path.write_text(json.dumps([_row("a")]), encoding="utf-8")

    rows = ra._merge_summary_rows(ra._read_summary_rows(path), [_row("b")])
    path.write_text(json.dumps(rows), encoding="utf-8")

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(r["instance_id"] for r in on_disk) == ["a", "b"]


# --------------------------------------------------------------------------
# 任务级崩溃：要落成一条**带 `failure` 的行**，不是「没有这条」
# --------------------------------------------------------------------------

def test_a_crash_produces_an_explicit_failure_row(ra) -> None:
    """为什么必须单独标出来（`原生agent采集…1010.md` §4.3 的教训）：

    `task_success: false` 有**两个来源** —— 真实评估失败，和进程崩了。
    只统计 `task_success` 会把崩溃当成模型失败，而这两件事在这个实验里的
    含义**完全相反**：一个是「小模型答错了」，一个是「框架自己死了」。
    """
    row = ra._failure_row(
        instance_id="t1",
        mode="classifier_route",
        agent_type="react_code",
        workflow_run_id="wf-1",
        e2e_wall_clock_ms=1234,
        error=IndexError("pop from empty list"),
    )

    assert row["record_type"] == "workflow"
    assert row["instance_id"] == "t1"
    assert row["workflow_run_id"] == "wf-1"
    assert row["task_success"] is False
    assert row["appworld_success"] is False
    assert "IndexError" in row["failure"]
    assert "pop from empty list" in row["failure"]


def test_a_failure_row_does_not_invent_predictor_stats(ra) -> None:
    """**崩在哪儿，就缺哪儿。**

    崩在 `AppWorld.initialize()`（我们实测的那种）时，模型一次都没调过 ——
    这时补一个 `l2_add4: {hits: 0, ...}` 上去，汇总里会和「跑了但一次没出分」
    长得一模一样。缺键才是诚实的：下游一眼看出这条不可用。
    """
    row = ra._failure_row(
        instance_id="t1", mode="classifier_route", agent_type="react_code",
        workflow_run_id="wf-1", e2e_wall_clock_ms=1, error=RuntimeError("boom"),
    )

    for absent in ("mechanism", "l2_add4", "l2_v4", "m4_memory"):
        assert absent not in row, f"崩溃行不该编出 {absent}"


def test_a_crash_is_still_a_countable_data_point(ra) -> None:
    """崩溃行要能进 summary **并**被单独筛出来。

    这两条一起才成立：进得去（验收的「每轮 =N 条」不会被崩溃搞成 N-1），
    筛得出（分析时能把框架崩溃和模型失败分开）。
    """
    crashed = ra._failure_row(
        instance_id="t1", mode="classifier_route", agent_type="react_code",
        workflow_run_id="wf-1", e2e_wall_clock_ms=1, error=RuntimeError("boom"),
    )
    ok = _row("t2")

    merged = ra._merge_summary_rows([], [crashed, ok])

    assert len(merged) == 2
    assert sorted(r["instance_id"] for r in merged) == ["t1", "t2"]
    assert [r["instance_id"] for r in merged if "failure" in r] == ["t1"]
