"""不受 AppWorld / freezegun 时间冻结影响的单调时钟（D10）。

## 为什么需要它

AppWorld 在 ``world`` 存续期间装着 freezegun。freezegun 不只是换掉
``time`` 模块上的属性 —— 它会**遍历 ``sys.modules``，把任何"值等于
``time.perf_counter``"的模块级名字就地替换成 ``fake_perf_counter``**
（``__module__`` 会变成 ``freezegun.api``）。

⇒ **``from time import perf_counter`` 这种写法一样中招**，不只是
``time.perf_counter()``。实测（受控实验，world 内真睡 0.2s）：

```
world 外: perf_counter Δ=200.116 ms   __module__='time'           qualname='perf_counter'
world 内: perf_counter Δ=  0.000 ms   __module__='freezegun.api'  qualname='fake_perf_counter'
          clock_gettime Δ=200.437 ms  ← 只它不受影响
```

⚠️ **不要用 `is` / `id()` 判断有没有被换掉** —— 原函数被释放后，fake 对象
可能拿到**同一个内存地址**，``perf_counter is _ORIG`` 会**假阳性为 True**。
上面的 ``__module__`` 才是可靠判据。

## 后果

**任何"两次读数都落在 world 存续期内"的 ``perf_counter`` 对，差值恒为 0。**
这已经在本仓咬过三次（D10），线上实测受害的有：

- ``runner.py`` 的 ``embedding_latency_ms``（4/4 恒 0，已修）
- ``components/tool_proxy.py`` 的 ``ToolExecutionResult.latency_ms``
  （线上 4/4 恒 0；且它被 ``executor`` 累加进 phase latency，**会让整条 span
  的时延系统性低估**）

用法::

    from traceoffload.clock import real_time
    started = real_time()
    ...
    latency_ms = (real_time() - started) * 1000.0

对照实现（历史遗留，未合并）：
``integrations/appworld_simplified.py:_real_time``、
``bench-resources/run_appworld_native.py:82``。
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone as _timezone

__all__ = ["real_time", "real_utcnow"]


def real_time() -> float:
    """单调时钟秒数；``CLOCK_MONOTONIC`` 直接进 libc，不受 freezegun 影响。"""

    return _time.clock_gettime(_time.CLOCK_MONOTONIC)


def real_utcnow() -> datetime:
    """**真实** UTC 时刻，不受 freezegun 影响。

    和 `real_time` 同一个理由，但返回 `datetime` —— 给需要「时刻」而不是
    「间隔」的地方用（`runner` 的 `clock` 参数就是）。

    ⚠️ **不要写成 `datetime.now(timezone.utc)`**，也不要在导入时存一份原函数
    —— freezegun 会遍历 `sys.modules` 把值相等的模块级名字**就地替换**，
    存引用一样中招。只有直接进 libc 的 `clock_gettime(CLOCK_REALTIME)` 可靠。

    实测代价（0925，T-hi r3）：AppWorld 拆掉 world 后 freezegun 的
    `tz_offsets` 栈被 pop 空，`datetime.now()` 抛 `IndexError`
    —— **整个轮次丢失，summary 一个字都没写出来**。
    见 `tests/test_real_utcnow.py`。
    """

    return datetime.fromtimestamp(
        _time.clock_gettime(_time.CLOCK_REALTIME), tz=_timezone.utc
    )
