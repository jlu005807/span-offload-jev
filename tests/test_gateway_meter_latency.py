"""`_GatewayMeter` 必须能**分模型**报时延，不只是总和。

为什么单独一条：两个模型的价格差 3.5×（输入）/ 17×（输出），时延差也在同一量级
—— 只给总和的话，「卸载率上去了但延迟也上去了」这种结论根本判不出来。
一开始只有 `framework_model_latency_ms`（总和），所以补一个按 target 分的键。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "experiments" / "run_appworld.py"
SPEC = importlib.util.spec_from_file_location("_appworld_bench_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _Inner:
    """假 gateway：按预设脚本返回每个调用的耗时/token。"""

    def __init__(self, script):
        self._script = list(script)

    def complete(self, target, request, *, model_attempt_id, deadline_at):
        del request, model_attempt_id, deadline_at
        return self._script.pop(0)

    def __getattr__(self, name):  # pragma: no cover - 只为通过代理
        raise AttributeError(name)


def _target(kind: str):
    return SimpleNamespace(kind=SimpleNamespace(value=kind))


def _result(*, latency_ms, input_tokens, output_tokens):
    return SimpleNamespace(
        latency_ms=latency_ms, input_tokens=input_tokens, output_tokens=output_tokens, cost=0.0
    )


def test_summary_reports_latency_per_target() -> None:
    meter = runner._GatewayMeter(
        _Inner([
            _result(latency_ms=1000.0, input_tokens=100, output_tokens=10),   # small
            _result(latency_ms=4000.0, input_tokens=500, output_tokens=50),   # large
            _result(latency_ms=2000.0, input_tokens=200, output_tokens=20),   # small
        ])
    )
    meter.complete(_target("small_model"), None, model_attempt_id="a", deadline_at=None)
    meter.complete(_target("large_model"), None, model_attempt_id="b", deadline_at=None)
    meter.complete(_target("small_model"), None, model_attempt_id="c", deadline_at=None)

    summary = meter.summary()

    assert summary["framework_latency_ms_by_target"] == {
        "small_model": 3000.0,
        "large_model": 4000.0,
    }
    assert summary["framework_latency_ms_mean_by_target"] == {
        "small_model": 1500.0,
        "large_model": 4000.0,
    }
    # 旧键不动（下游有消费者）
    assert summary["framework_model_latency_ms"] == 7000
    assert summary["framework_model_calls_by_target"] == {"small_model": 2, "large_model": 1}


def test_summary_is_empty_safe() -> None:
    summary = runner._GatewayMeter(_Inner([])).summary()

    assert summary["framework_latency_ms_by_target"] == {}
    assert summary["framework_latency_ms_mean_by_target"] == {}
    assert summary["framework_model_calls"] == 0
