"""M3 的副作用判定类型。

真正的判定器按环境分两种：
  * AppWorld  -> integrations/appworld.py 里的 AppWorldCodeEffectAnalyzer
    （看 apis.* 调用在 OpenAPI 里的 HTTP method）
  * 其它环境  -> 由调用方注入；不支持时 executor 一律判 UNKNOWN 并 fail-closed

这个模块只放**共享的数据类型**，不放任何分析逻辑 —— 这样 executor 和
AppWorld 适配器都能 import 它，而不用把一个 1300 行的 bash 语法分析器带进依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import EffectScope

#: 效应类别。执行前 fail-closed 的是 external / irreversible / unknown。
EFFECT_CATEGORIES: tuple[str, ...] = (
    "read_only",
    "declared_write",
    "opaque_exec",
    "verify_exec",
    "external",
    "irreversible",
)


@dataclass(frozen=True)
class LeafEffect:
    """一个可执行节点或重定向的效应（审计用）。"""

    node_type: str = ""
    executable: str = ""
    category: str = "unknown"
    scope: EffectScope = EffectScope.UNKNOWN
    write_targets: tuple[str, ...] = ()
    requires_checkpoint: bool = False
    confidence: str = "certain"
    reason: str = ""
    start_byte: int = 0
    end_byte: int = 0


@dataclass(frozen=True)
class EffectVerdict:
    """合并后的判定 + 叶子级审计轨迹。"""

    category: str
    scope: EffectScope
    write_targets: tuple[str, ...] = ()
    risk_flags: frozenset = frozenset()
    requires_checkpoint: bool = False
    confidence: str = "certain"
    reason: str = ""
    leaf_effects: tuple[LeafEffect, ...] = ()
    parse_complete: bool = True
    grammar_version: str = "appworld-api"
    policy_version: str = ""


__all__ = ["EFFECT_CATEGORIES", "EffectVerdict", "LeafEffect"]
