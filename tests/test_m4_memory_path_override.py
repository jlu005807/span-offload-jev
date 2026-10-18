"""M4 记忆库的路径必须能钉成**绝对路径** —— 否则跑批里 M4 永远不累积。

为什么需要这个开关
------------------
`runner._start_m4_memory` 里是这么解析的：

    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = self._output_dir / path        # ← 相对路径拼到 --output-dir 下

而跑批的 `--output-dir` 是 **per (task, round)** 的（协议 §6 的
`<task>/<GROUP>/round-NN/result.json` 布局要求如此）。两者一叠加：

    每个 (task, 轮) 一份**全新的空库** ⇒ M4 从来不累积

实测证据（我自己的 5-worker sweep，w1）：round-1 的库只含 `aa2d21b8` 那次调用，
round-2 的库只含 `c33f2cb3` —— **零交集**。而协议 §4 明文要求
「worker 内跨实例、**跨轮次**累积」。⇒ 那条要求从未实现过。

绝对路径跳过那次拼接，于是：
  * worker 内跨实例、跨轮次累积（同一个库）
  * worker 之间物理隔离（各写各的库）
  * `--output-dir` 的 per-(task, round) 布局不受影响

⚠️ 相对路径必须**报错**而不是静默接受 —— 静默接受就是重新制造上面那个 bug，
而且不报错（跑得通、有数据、只是记忆是空的）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "experiments" / "run_appworld.py"
SPEC = importlib.util.spec_from_file_location("_appworld_runner_m4path", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
# 先注册再 exec：脚本里有 `@dataclass`，dataclasses 解析字段类型要查 sys.modules。
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config(path: Any = "m4_memory.sqlite3") -> dict[str, Any]:
    return {"runner": {"m4_memory": {"enabled": True, "path": path}}}


def test_absolute_override_replaces_the_config_path(tmp_path) -> None:
    config = _config()
    target = tmp_path / "w1" / "m4_memory.sqlite3"

    runner._apply_m4_memory_path_override(config, str(target))

    assert config["runner"]["m4_memory"]["path"] == str(target)
    assert Path(config["runner"]["m4_memory"]["path"]).is_absolute()


def test_no_override_leaves_the_config_untouched() -> None:
    """不传就得原样不动 —— 单任务/调试跑不该被这个开关影响。"""

    config = _config("m4_memory.sqlite3")

    runner._apply_m4_memory_path_override(config, None)

    assert config["runner"]["m4_memory"]["path"] == "m4_memory.sqlite3"


def test_relative_override_is_rejected(tmp_path, monkeypatch) -> None:
    """相对路径必须**报错**。

    报错而不是静默接受，是因为静默接受等于重新制造「每格一份空库」那个 bug ——
    跑得通、有数据、只是记忆永远是空的，翻遍日志也看不出来。
    """

    config = _config()

    with pytest.raises(ValueError, match="绝对路径"):
        runner._apply_m4_memory_path_override(config, "runs/relative/m4.sqlite3")


def test_missing_m4_memory_section_is_tolerated(tmp_path) -> None:
    """config 里没有 `runner.m4_memory` 时也是 `path` 覆盖不了，但要明确报错。

    这一条钉的是「不许悄悄建出一个没人读的键」—— 那是我们在别处踩过的形状。
    """

    with pytest.raises(ValueError, match="m4_memory"):
        runner._apply_m4_memory_path_override({"runner": {}}, str(tmp_path / "m.sqlite3"))


def test_cli_flag_is_registered() -> None:
    """`--m4-memory-path` 必须真的挂在 CLI 上（不是只写了个函数没人调）。"""

    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--m4-memory-path"' in source
    assert "_apply_m4_memory_path_override(" in source
    # 至少两处：定义 + 调用。只定义不调用是最容易漏的形状。
    assert source.count("_apply_m4_memory_path_override") >= 2
