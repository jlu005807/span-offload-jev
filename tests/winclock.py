"""pytest 插件：在 Windows 上补出 ``time.clock_gettime``。

为什么需要
----------
``traceoffload/clock.py`` 用 ``time.clock_gettime(CLOCK_MONOTONIC)`` /
``time.clock_gettime(CLOCK_REALTIME)`` 拿不受 freezegun 影响的时间
（见该文件的模块 docstring）。这两个 API 是 **Unix 专有**，
Windows 的 ``time`` 模块没有 —— 于是本机跑测试会集体失败：

    AttributeError: module 'time' has no attribute 'clock_gettime'

本插件只在 ``time.clock_gettime`` 缺失时补一个等价实现，
因此在 Linux 上加载它也**不会改变任何行为**。

用法
----
    # Unix（直接跑，不需要本插件）
    python -m pytest tests -q

    # Windows
    python -m pytest tests -q -p winclock
    #   其中 winclock.py 放在 PYTHONPATH 上，或用 --rootdir 下的 conftest 引入

    # 或者：把它拷进 tests/ 目录作为 conftest.py 的补充
    #   copy winclock.py tests/winclock.py && python -m pytest tests -q

已知偏差
--------
补出来的实现用 ``time.perf_counter()`` / ``time.time()``。
``time.time()`` 会被 freezegun 冻结，所以
``tests/test_real_utcnow.py::test_bypasses_freezegun``
在 Windows + 本插件下仍会失败 —— 那是 shim 的固有限制，
不是被测代码的问题（被测代码在 Linux 上走的是真的 libc ``clock_gettime``）。
"""

from __future__ import annotations

import time as _time

if not hasattr(_time, "clock_gettime"):
    _time.CLOCK_MONOTONIC = _time.perf_counter
    _time.CLOCK_REALTIME = _time.time

    def _clock_gettime(clk_id: int) -> float:
        if clk_id == _time.CLOCK_MONOTONIC:
            return _time.perf_counter()
        return _time.time()

    _time.clock_gettime = _clock_gettime
