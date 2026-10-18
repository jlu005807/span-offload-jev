"""`run_task` 的终点时刻**不能**在 `solve_tasks()` 之后读时钟。

为什么单独一条
--------------
AppWorld 的 world 用 freezegun 冻结时间，而 world 是在 `agent.solve_tasks()`
**内部**创建并关闭的。返回时冻结已解、freezegun 的时区栈被 pop，此后任何
`datetime.now()` 都会走进被 patch 的 `FakeDatetime.now`：

    freezegun/api.py:437  _tz_offset()  →  tz_offsets[-1]  →  IndexError

而 `traceoffload/runner.py:195  _utcnow()` 正是 `datetime.now(timezone.utc)`。

实测（验证实例 `634f342_2`，两次独立复现、且对照组 `09b0ee6_1` 从不触发）
崩在 `run_task` 里那句 `observed_at = self._clock()`，而它**只在 `m4_records`
非空时才执行** —— 所以只有落了 M4 记录的任务会踩到，代价是**整轮 run 在写
summary 之前退出**（任务本身跑完了，数据一条不落）。5-worker sweep 里 w5 的
三个轮次就是这样全丢的。

为什么是源码层面的断言
----------------------
这个状态在单测里造不出来：我试过手动 `start() → stop() → stop()`（复现了
`pop from empty list`），但之后 `datetime.now()` 反而正常返回真实时间 ——
说明触发它还需要 AppWorld 内部的某种顺序，不是「栈空」一件事。

所以这里退而求其次，钉住**调用顺序**这条不变量。它拦不住所有情况，
但正好拦住「有人把时钟调用挪回 `solve_tasks` 之后」这个回归 ——
而那就是这个 bug 的全部成因。
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "experiments" / "run_appworld.py"


def _run_task_source() -> str:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_task":
            return ast.get_source_segment(SCRIPT.read_text(encoding="utf-8"), node) or ""
    raise AssertionError("run_task 不在了 —— 这条测试要跟着改")


def test_run_task_reads_the_clock_before_solving_tasks() -> None:
    """起点时刻必须在 `solve_tasks()` **之前**取。

    之后取会撞上被解冻的 freezegun（见模块 docstring）。
    """

    source = _run_task_source()
    calls = source.index("self._clock()")
    solve = source.index("agent.solve_tasks(")

    assert calls < solve, (
        "`self._clock()` 出现在 `solve_tasks()` 之后 —— 那时 AppWorld 已经关掉 "
        "world、freezegun 的时区栈被 pop，`datetime.now()` 会抛 IndexError，"
        "整轮 run 在写 summary 之前就退出。见本文件 docstring。"
    )


def test_observed_at_is_derived_from_the_start_not_read_late() -> None:
    """终点时刻要用「起点 + 实测时长」还原，不能直接再读一次时钟。"""

    source = _run_task_source()

    assert "started_wall" in source, "起点时刻没留下变量名 —— 修法被改掉了"
    assert "observed_at = started_wall" in source or "observed_at = (\n" in source, (
        "`observed_at` 不是从起点推出来的 —— 又回去读被 patch 的时钟了"
    )
