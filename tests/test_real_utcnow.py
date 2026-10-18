"""`real_utcnow()`：穿透 freezegun 的真实 UTC 时刻。

为什么需要
----------
AppWorld 的 `world` 用 freezegun 冻结时间，而 world 在 `agent.solve_tasks()`
**内部**创建并销毁。销毁之后 freezegun 的 `tz_offsets` 栈被 pop —— 此后任何
`datetime.now()` 都会走进被 patch 的 `FakeDatetime.now`：

    freezegun/api.py:437  _tz_offset()  →  tz_offsets[-1]  →  IndexError

**实测代价（0925，T-hi r3）**：整个轮次丢失 —— 崩在 `run_task` 里
`open_adapter_session` 那条链上，**summary 一个字都没写出来**。

     run_appworld_traceoffload.py:692  run_task
       → coordinator.open_adapter_session
          → integration.AdapterSession.__init__
             → _emit(occurred_at=self._clock())
                → runner._utcnow() → datetime.now(timezone.utc)  → 崩

⚠️ **它和「起点时刻提前取」那个修法是两回事。** 那个只堵住了 `run_task`
**末尾**（写 summary 时）的一次读时钟；而 `open_adapter_session` 在**开头**，
跑的是**上一个**任务留下的坏状态 —— 所以是**第 2 个及以后**的任务才中招，
而且只在特定顺序下（实测 12 个任务里崩在第 12 个）。

为什么不能用 `datetime.now()`
----------------------------
freezegun 会**遍历 `sys.modules`，把值等于 `datetime.now` 的模块级名字就地
替换** —— 所以「导入时存一份原函数」也没用（同 `clock.py` 的 D10 记录）。
只有直接进 libc 的 `clock_gettime(CLOCK_REALTIME)` 不受影响。
"""

from __future__ import annotations

import datetime as _dt

from freezegun import freeze_time

from traceoffload.clock import real_time, real_utcnow


def test_returns_timezone_aware_utc() -> None:
    """runner 的 `clock` 参数要的是 `AwareDatetime` —— 朴素 datetime 会在别处炸。"""
    v = real_utcnow()

    assert v.tzinfo is not None, "返回了朴素 datetime"
    assert v.utcoffset() == _dt.timedelta(0), f"不是 UTC: {v.utcoffset()}"


def test_bypasses_freezegun() -> None:
    """**核心**：冻结期间必须返回真实时间，不是冻结时间。"""
    with freeze_time("2020-01-01 00:00:00"):
        assert _dt.datetime.now(_dt.timezone.utc).year == 2020, "前提：冻结生效了"
        assert real_utcnow().year > 2020, (
            f"real_utcnow() 返回了冻结时间 {real_utcnow()} —— 没穿透 freezegun"
        )


def test_agrees_with_the_real_wall_clock() -> None:
    """冻结之外，它必须和普通的 `datetime.now(utc)` 一致（差 < 1s）。

    ⚠️ **不能拿 `real_time()` 比** —— 那是 `CLOCK_MONOTONIC`（从开机算起），
    和 epoch 秒是两个不同的时钟。
    """
    before = _dt.datetime.now(_dt.timezone.utc)
    v = real_utcnow()
    after = _dt.datetime.now(_dt.timezone.utc)

    assert before <= v <= after, f"real_utcnow={v} 不在 [{before}, {after}] 内"


def test_real_time_is_monotonic_but_a_different_clock() -> None:
    """留一条记录：`real_time` 与 `real_utcnow` 不是同一个时钟，别混用。"""
    assert real_time() < 1e9, "CLOCK_MONOTONIC 不该是 epoch 量级"
    assert real_utcnow().timestamp() > 1e9, "CLOCK_REALTIME 该是 epoch 量级"


def test_survives_a_torn_down_freeze_context() -> None:
    """**复现那个崩溃的状态**：freeze 起过又停掉之后，仍要能读。

    真实场景比这更糟（AppWorld 的 world 嵌套销毁），但这一条能拦住
    「又回去用 `datetime.now()`」这个回归。
    """
    freezer = freeze_time("2020-01-01 00:00:00")
    freezer.start()
    freezer.stop()

    v = real_utcnow()

    assert v.year > 2020, "freeze 起停之后读到了假时间"


def test_is_monotonic_in_practice() -> None:
    """两次调用不倒退。"""
    a, b = real_utcnow(), real_utcnow()

    assert b >= a
