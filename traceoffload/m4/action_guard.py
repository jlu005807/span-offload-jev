"""M4 动作守卫的**查询侧**：把「候选动作 + span 身份」变成「M4 里的命中」。

它在链路里的位置
----------------
    M3 出候选响应
      → `M4FailureMemoryVerifier.check`      负责**顺序**（形状 → 记忆 → fail-open）
      → **本模块**                            负责**取值**（身份→52 维→逐个 tool call）
      → `M4Memory.query_failed_action`        负责**判定**（相似度 AND 签名）

三段分开是为了各自可测：顺序那段不碰记忆库、判定那段不碰响应对象、
取值这段（本模块）两边都要碰但只有十几行。

两条 fail-open
--------------
**拿不到向量就放行，不抛异常。** 身份是 None（非 guarded 路径）、或它不在
向量缓存里（M1 还没算完、换了进程），都算「没有证据」而不是「有罪」。
守卫是增强，不是依赖 —— 它坏了不能把路由卡住。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .memory import DEFAULT_MEMORY_GUARD_THRESHOLD, M4Memory, MemoryNeighbor
from .span_features import action_signature
from ..mechanism_counters import MechanismCounters

__all__ = ["lookup_failed_actions"]


def lookup_failed_actions(
    candidate: Any,
    identity: Any,
    *,
    guard_vectors: Mapping[str, Sequence[float]],
    memory: M4Memory,
    vector_key: Callable[[Any], str],
    threshold: float = DEFAULT_MEMORY_GUARD_THRESHOLD,
    counters: MechanismCounters | None = None,
) -> tuple[MemoryNeighbor, ...]:
    """对候选响应里的**每一个** tool call 查一次 M4，返回命中的邻居并集。

    参数
    ----
    ``guard_vectors``
        ``{vector_key(identity): 52 维}`` —— 由 M1 在线派生后缓存。
    ``vector_key``
        ``identity → 缓存键``。**注入而不是内联**，这样键的格式只有一处定义
        （`runner._m4_record_key`），本模块不用复刻它 —— 复刻就会漂移，
        而漂移的坏法是不报错。
    """

    counters = counters if counters is not None else MechanismCounters.null()

    if identity is None:
        return ()
    vector = guard_vectors.get(vector_key(identity))
    if vector is None:
        return ()

    hits: list[MemoryNeighbor] = []
    for call in getattr(candidate, "tool_calls", ()) or ():
        signature = action_signature(
            str(getattr(call, "tool_name", "")),
            getattr(call, "arguments", None) or {},
        )
        # T2（spec §3.2）：一次 tool call = 一次查询。**只数真正查了的** ——
        # 身份缺失/向量缺失在上面就 return 了，那些不该记成 lookup，
        # 否则「守卫没接线」和「守卫查了但没命中」看起来一样。
        counters.bump("m4_guard_lookups")
        found = memory.query_failed_action(
            vector,
            signature,
            similarity_threshold=threshold,
        )
        if found:
            counters.bump("m4_guard_hits", len(found))
        hits.extend(found)
    return tuple(hits)
