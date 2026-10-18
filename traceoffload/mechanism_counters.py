"""T2 机制计数（spec `实验_四策略可比性协议.md` §3.2 / 消融设计 §7.3）。

为什么需要
----------
不加这个，**摘掉守卫的臂和满血臂在报告里长得一模一样** —— 同样的成功率、
同样的 token。分不清「没效果」和「没触发」。

实测（0925，A5-full 60 次跑）三条观测路径全断：

* `summary.json` 里没有 guard/memory 字段
* AppWorld 的 `logger.jsonl` 是空的
* `M4FailureMemoryVerifier` 拒绝时**复用内层 verifier 的 `guard_ref`**
  （`guards.py`：`GuardResult(guard_ref=result.guard_ref, ...)`）
  ⇒ 记忆那一层在 `verifier_refs` 里也不可见

设计取舍
--------
* **键必须先注册**：写错键名报 `KeyError`。静默建新键 = 报告里多一列恒为 0 的
  假数据，而假数据比缺数据坏 —— 缺数据会被发现，假数据不会。
* **未 bump 的键出现在快照里（值为 0）**：缺键会让汇总脚本以为「字段没实现」
  而跳过断言；0 才是「实现了但没触发」。§3.2 的自洽性断言全靠这个区别。
* **`null()` 空接收器**：组件默认不接线（`counters=None`），免得每个调用点都写
  `if counters is not None` —— 漏一处就是 AttributeError 打断主链路。
* **线程安全**：写入在 M4 worker 线程、读取在主线程。丢计数会让断言**假通过**。
* **放顶层而不是 `components/`**：`guards.py` 要用它，而 `components/__init__`
  会 import coordinator → router → 回头 import `guards`。放在 `components/` 下
  就构成循环导入（实测 `ImportError: cannot import name 'ToolPolicy' from
  partially initialized module 'traceoffload.guards'`）。本模块是叶子（零依赖），
  放顶层最安全。
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["MechanismCounters", "T2_KEYS", "T2_MAPS", "ABSOLUTE_KEYS", "delta_of"]

#: 标量计数（spec §3.2 的字段清单）
T2_KEYS: tuple[str, ...] = (
    # 准入（executor）
    "admission_accepted",
    # 守卫（verifier）
    "guard_checks",
    "guard_shape_rejects",
    "guard_memory_rejects",
    "guard_post_check_rejects",
    # 事务（executor / tool_proxy）
    "checkpoints_created",
    "transactions_committed",
    "transactions_rolled_back",
    "rollback_failures",
    "recovery_entries",
    # M4
    "m4_records_at_start",
    "m4_records_at_end",
    "m4_records_stored",
    "m4_skipped_success",
    "m4_guard_lookups",
    "m4_guard_hits",
)

#: 按原因分开数的计数
T2_MAPS: tuple[str, ...] = (
    "admission_rejected_by_rule",
    "fallback_by_reason",
)

#: **不是事件计数**，做差得到的是垃圾 —— `delta_of` 原样透传 `after` 的值。
#:   · `m4_records_at_start/_at_end` 是**库存快照**（差 = 「跑了几个任务时库长了几条」）
#:   · `*_avg_ms` 是**派生值**（avg 的差没有意义；调用方应按增量重算）
#:   · `*_max_ms` 是**进程级极值**（每任务 max 不可从快照恢复）
#:   · `queue_size` / `pending_*` 是**瞬时值**（差可能是负数）
#:   · `thread_alive` 是 bool
#:   · `input_dim` / `threshold` 是**常量** —— before/after 里一样，做差必然得 0。
#:     这一条尤其阴：`input_dim` 会从 52 变成 0，读起来像「维度是 0」。
#:
#: ⚠️ 这是**逐键白名单**：新增一个 int 字段就得在这里做一次决定。
#:    `tests/components/test_mechanism_counters.py` 里那两个「真实形状」测试
#:    是这道防线 —— 结构变了它们会挂，逼你回来决定。
ABSOLUTE_KEYS: frozenset[str] = frozenset(
    {
        "m4_records_at_start",
        "m4_records_at_end",
        "m1_online_embedding_avg_ms",
        "m1_online_embedding_max_ms",
        "queue_size",
        "pending_embeddings",
        "pending_texts",
        "thread_alive",
        "input_dim",
        "threshold",
        # add4 两阶段的**两个工作点**。与 `threshold` 同一形状：是选出来的常量，
        # before/after 里一样 ⇒ 做差得 0。实测踩过一次：summary 里
        # `tau_gate: 0.0 / tau_net: 0.0`，而跑的时候是 0.5381 / 0.5523 ——
        # 读起来像「阈值是 0，所以什么都卸载」。
        "tau_gate",
        "tau_net",
    }
)


def _is_counter(value: Any) -> bool:
    """能安全做差的数。**bool 要排除** —— `isinstance(True, int)` 是 `True`，
    不特判的话 `True - True = 0`，`thread_alive` 会变成整数 0（假），
    看起来像「线程死了」。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def delta_of(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """`after` 相对 `before` 的**增量**，用来写「该任务自己的」计数。

    为什么需要
    ----------
    计数器活在 runner 上，而一个进程顺序跑多个任务 ⇒**第 N 行的快照含前 N-1 个
    任务的量**。汇总时求和会重复计算，**而且不报错**。

    实测（0925）踩到三次，都是同一个形状：

    =========================  ======（求和 vs 末行）
    `p_values`（守卫/预测器）       —
    `mechanism.guard_checks`       1245 vs **74**
    `m4_memory.worker.received`    6867 vs **1065**

    改用增量之后所有行**口径统一，求和就对**，下游不用再记「这个字段要取末行」。

    行为
    ----
    * 计数器（int/float，非 bool）→ 做差
    * 嵌套 Mapping → 递归
    * `ABSOLUTE_KEYS` 里的键、bool、非数值 → 原样透传
    """
    out: dict[str, Any] = {}
    for key, value in after.items():
        # ⚠️ 顺序要紧：Mapping 先判，否则会被 `not _is_counter` 截住、递归不可达。
        if key in ABSOLUTE_KEYS:
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = delta_of(before.get(key) or {}, value)
        elif isinstance(value, list):
            # 追加型日志（`l2_v4.p_values`）：每行含前面所有任务的量。
            # **只在 `before` 确实是前缀时才切** —— 猜错方向会静默丢掉前半截，
            # 而且切出来的长度看着正常。
            prior = before.get(key)
            if isinstance(prior, list) and list(value[: len(prior)]) == prior:
                out[key] = list(value[len(prior) :])
            else:
                out[key] = list(value)
        elif _is_counter(value):
            prior = before.get(key, 0)
            out[key] = value - (prior if _is_counter(prior) else 0)
        else:
            out[key] = value  # bool / None / 字符串等，原样透传
    return out


class MechanismCounters:
    """分层的计数袋。见模块 docstring。"""

    __slots__ = ("_counts", "_maps", "_keys", "_map_names", "_lock", "_null")

    def __init__(
        self,
        *,
        keys: Iterable[str] = T2_KEYS,
        maps: Iterable[str] = T2_MAPS,
        _null: bool = False,
    ) -> None:
        self._keys = frozenset(keys)
        self._map_names = frozenset(maps)
        self._counts: dict[str, int] = {k: 0 for k in self._keys}
        self._maps: dict[str, dict[str, int]] = {m: {} for m in self._map_names}
        self._lock = threading.Lock()
        self._null = _null

    # ------------------------------------------------------------------ 构造
    @classmethod
    def null(cls) -> MechanismCounters:
        """空接收器：接受任何键、什么都记不住、任何情况下都不抛。

        给「没接线」的默认值用。它**不校验键名** —— 校验是给真的计数器用的，
        空接收器本来就没有要保护的状态。
        """
        return cls(keys=(), maps=(), _null=True)

    # ------------------------------------------------------------------ 写
    def bump(self, key: str, n: int = 1) -> None:
        """标量 +n。键没注册就报错。"""
        if self._null:
            return
        with self._lock:
            if key not in self._counts:
                raise KeyError(
                    f"未注册的 T2 计数键 {key!r}。"
                    f"要新增就加到 mechanism_counters.T2_KEYS —— "
                    f"写错键名会静默多出一列恒为 0 的假数据。"
                )
            self._counts[key] += n

    def bump_map(self, name: str, key: str, n: int = 1) -> None:
        """`name[key]` +n。`name` 没注册就报错。"""
        if self._null:
            return
        with self._lock:
            if name not in self._maps:
                raise KeyError(
                    f"未注册的 T2 分组计数 {name!r}。"
                    f"要新增就加到 mechanism_counters.T2_MAPS。"
                )
            self._maps[name][key] = self._maps[name].get(key, 0) + n

    def set_value(self, key: str, value: int) -> None:
        """覆盖（不是累加）。给 `m4_records_at_start/_at_end` 这类**快照值**用 ——
        它们不是事件计数，是「此刻库里有几条」。"""
        if self._null:
            return
        with self._lock:
            if key not in self._counts:
                raise KeyError(f"未注册的 T2 计数键 {key!r}")
            self._counts[key] = int(value)

    # ------------------------------------------------------------------ 读
    def snapshot(self) -> dict[str, Any]:
        """深拷贝一份。调用方改它**不能**影响内部状态。"""
        if self._null:
            return {}
        with self._lock:
            return {
                **self._counts,
                **{name: dict(entries) for name, entries in self._maps.items()},
            }
