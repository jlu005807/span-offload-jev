"""T2 机制计数（spec §3.2 / §7.3）：让「机制有没有跑」变得可观测。

为什么需要
----------
不加这个，**`B1-no-guard` 和 `A5-full` 在报告里长得一模一样** —— 同样的成功率、
同样的 token。你分不清「没效果」和「没触发」。

实测（0925，A5-full 60 次跑）：summary 里没有任何 guard/memory 字段，
`logger.jsonl` 是空的，而 `M4FailureMemoryVerifier` 拒绝时**复用内层 verifier 的
`guard_ref`** ⇒ 记忆那一层在 `verifier_refs` 里也不可见。**三条路全断。**

⇒ 守卫触发过几次，从这次跑批的产物里**问不出来**。这是 spec 标为「阻塞全部臂」
的那个窟窿。
"""
from __future__ import annotations

import threading

import pytest

from traceoffload.mechanism_counters import MechanismCounters, delta_of


def test_counts_start_at_zero_not_missing() -> None:
    """没 bump 过的键也要出现（值为 0）。

    缺键和 0 在报告里是两回事：**缺键会让汇总脚本以为「这个字段没实现」**，
    于是断言被跳过；0 才是「实现了，但没触发」。§3.2 的自洽性断言全靠这个区别。
    """
    c = MechanismCounters(keys=("guard_checks", "guard_memory_rejects"), maps=())

    snap = c.snapshot()

    assert snap == {"guard_checks": 0, "guard_memory_rejects": 0}


def test_unbumped_maps_appear_as_empty_dicts() -> None:
    """分组计数同理 —— 空 dict 表示「实现了但没触发」，缺键表示「没实现」。"""
    c = MechanismCounters(keys=(), maps=("fallback_by_reason",))

    assert c.snapshot() == {"fallback_by_reason": {}}


def test_unregistered_key_is_rejected() -> None:
    """写错键名必须报错 —— 静默建一个新键 = 报告里多一列恒为 0 的假数据。"""
    c = MechanismCounters(keys=("guard_checks",))

    with pytest.raises(KeyError, match="guard_check"):
        c.bump("guard_check")  # 少个 s


def test_bump_accumulates() -> None:
    c = MechanismCounters(keys=("guard_checks",))

    c.bump("guard_checks")
    c.bump("guard_checks", 3)

    assert c.snapshot()["guard_checks"] == 4


def test_bump_map_records_reasons_separately() -> None:
    """`admission_rejected_by_rule` 这类要按原因分开数。"""
    c = MechanismCounters(keys=(), maps=("admission_rejected_by_rule",))

    c.bump_map("admission_rejected_by_rule", "scope_unknown")
    c.bump_map("admission_rejected_by_rule", "scope_unknown")
    c.bump_map("admission_rejected_by_rule", "opaque_exec_not_allowed")

    assert c.snapshot()["admission_rejected_by_rule"] == {
        "scope_unknown": 2,
        "opaque_exec_not_allowed": 1,
    }


def test_unknown_map_key_is_rejected() -> None:
    c = MechanismCounters(keys=(), maps=("fallback_by_reason",))

    with pytest.raises(KeyError, match="fallback_by_reason"):
        c.bump_map("fallback_by_reasons", "guard_reject")


def test_snapshot_is_a_copy() -> None:
    """调用方改快照不能改到计数器的内部状态（汇总会遍历它）。"""
    c = MechanismCounters(keys=("guard_checks",), maps=("m",))
    c.bump_map("m", "a")
    snap = c.snapshot()

    snap["guard_checks"] = 999
    snap["m"]["a"] = 999

    assert c.snapshot()["guard_checks"] == 0
    assert c.snapshot()["m"]["a"] == 1


def test_concurrent_bumps_do_not_lose_counts() -> None:
    """写入在 worker 线程、读取在主线程 —— 丢计数就会让断言**假通过**。"""
    c = MechanismCounters(keys=("guard_checks",))

    def worker() -> None:
        for _ in range(200):
            c.bump("guard_checks")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert c.snapshot()["guard_checks"] == 1600


def test_none_counters_is_a_noop_sink() -> None:
    """`None` 要能安全地当计数器用 —— 组件默认就是 None（未接线）。

    不这么做的话，每个调用点都要写 `if counters is not None`，
    漏一处就是 AttributeError 打断主链路。
    """
    c = MechanismCounters.null()

    c.bump("anything")
    c.bump_map("anything", "x")

    assert c.snapshot() == {}


def test_null_sink_accepts_unknown_keys() -> None:
    """空接收器不校验键名 —— 它本来就是「没接线」的意思。"""
    MechanismCounters.null().bump("whatever_key")  # 不抛


# ─────────────────── 每任务增量（不要进程内累积值）───────────────────
#
# 实测踩过两次：计数器活在 runner 上、一个进程顺序跑 12 个任务，所以第 N 行的
# 快照含前 N-1 个任务的量。汇总时求和 ⇒ 重复计算。
# 实测数字：求和得 1245 次 guard_checks，取最后一行才是 74。


def test_delta_of_subtracts_what_the_previous_task_already_counted() -> None:
    """写进 summary 的必须是**该任务自己的**增量。"""
    c = MechanismCounters(keys=("guard_checks",), maps=("fallback_by_reason",))

    before = c.snapshot()
    c.bump("guard_checks", 3)
    c.bump_map("fallback_by_reason", "guard_reject")
    after = c.snapshot()

    d = delta_of(before, after)

    assert d["guard_checks"] == 3
    assert d["fallback_by_reason"] == {"guard_reject": 1}


def test_delta_of_is_zero_for_a_task_that_did_nothing() -> None:
    c = MechanismCounters(keys=("guard_checks",), maps=())
    snap = c.snapshot()

    assert delta_of(snap, c.snapshot()) == {"guard_checks": 0}


def test_deltas_sum_to_the_total_across_tasks() -> None:
    """**这条是整件事的判据**：把每行的增量加起来，必须等于真实总数。

    累积口径下这个和会变成 1+2+3=6（真实是 3）。
    """
    c = MechanismCounters(keys=("guard_checks",), maps=())
    per_task = []
    for _ in range(3):
        before = c.snapshot()
        c.bump("guard_checks", 1)
        per_task.append(delta_of(before, c.snapshot())["guard_checks"])

    assert per_task == [1, 1, 1]
    assert sum(per_task) == c.snapshot()["guard_checks"] == 3


def test_delta_of_keeps_snapshot_keys_absolute() -> None:
    """`m4_records_at_start` / `_at_end` 是**库存快照**，不是事件计数。

    对它们做差得到的是「跑了几个任务时库长了几条」—— 没有意义的数。
    快照值必须原样透传（取 `after` 的值）。
    """
    c = MechanismCounters(keys=("guard_checks", "m4_records_at_start", "m4_records_at_end"), maps=())

    before = c.snapshot()
    c.bump("guard_checks", 2)
    c.set_value("m4_records_at_start", 11)
    c.set_value("m4_records_at_end", 16)
    after = c.snapshot()

    d = delta_of(before, after)

    assert d["guard_checks"] == 2
    assert d["m4_records_at_start"] == 11
    assert d["m4_records_at_end"] == 16


def test_delta_of_tolerates_a_key_missing_from_before() -> None:
    """`before` 里没有的键按 0 起算 —— 中途接线不该炸。"""
    d = delta_of({}, {"guard_checks": 5})

    assert d["guard_checks"] == 5


# ─────────────── 嵌套与非计数（`m4_memory` 那种结构）───────────────
#
# `m4_memory_stats()` 的形状是嵌套的 + 混了三种不同性质的字段：
#   计数器（received / stored / m1_online_embedding_count）→ 该做差
#   派生值（avg_ms = total/count）                        → 做差没意义
#   瞬时值（queue_size / pending_* / thread_alive）        → 做「差」得到垃圾


def test_delta_of_recurses_into_nested_dicts() -> None:
    before = {"worker": {"received": 10, "stored": 1}}
    after = {"worker": {"received": 15, "stored": 3}}

    d = delta_of(before, after)

    assert d["worker"] == {"received": 5, "stored": 2}


def test_delta_of_recurses_twice_for_source_counts() -> None:
    """`worker.embedding_source_counts` 是第二层嵌套的计数器。"""
    before = {"worker": {"embedding_source_counts": {"m1_online": 4}}}
    after = {"worker": {"embedding_source_counts": {"m1_online": 9, "cache": 1}}}

    d = delta_of(before, after)

    assert d["worker"]["embedding_source_counts"] == {"m1_online": 5, "cache": 1}


def test_delta_of_passes_gauges_through_absolute() -> None:
    """`queue_size` / `pending_*` 是**瞬时值** —— 做差得到的是垃圾（可能是负数）。"""
    before = {"worker": {"queue_size": 7, "pending_embeddings": 3}}
    after = {"worker": {"queue_size": 0, "pending_embeddings": 0}}

    d = delta_of(before, after)

    assert d["worker"]["queue_size"] == 0
    assert d["worker"]["pending_embeddings"] == 0


def test_delta_of_passes_bools_through_absolute() -> None:
    """`thread_alive` 是 bool。

    ⚠️ Python 里 `isinstance(True, int)` 是 **True** —— 不特判的话
    `True - True = 0`，`thread_alive` 会变成整数 0（假），看起来像「线程死了」。
    """
    d = delta_of({"worker": {"thread_alive": True}}, {"worker": {"thread_alive": True}})

    assert d["worker"]["thread_alive"] is True


def test_delta_of_handles_the_real_m4_memory_shape() -> None:
    """拿 `m4_memory_stats()` 的真实形状跑一遍 —— 保证接线时不会踩到没考虑到的键。"""
    before = {
        "m1_online_embedding_count": 11,
        "m1_online_embedding_errors": 0,
        "m1_online_embedding_total_ms": 1000.0,
        "m1_online_embedding_avg_ms": 90.9,
        "m1_online_embedding_max_ms": 210.0,
        "worker": {
            "received": 13, "stored": 0, "skipped_success": 11, "errors": 0,
            "embedded": 0, "m1_embedding_registered": 11,
            "embedding_source_counts": {}, "pending_embeddings": 2,
            "pending_texts": 1, "queue_size": 3, "thread_alive": True,
        },
    }
    after = {
        "m1_online_embedding_count": 27,
        "m1_online_embedding_errors": 0,
        "m1_online_embedding_total_ms": 2400.0,
        "m1_online_embedding_avg_ms": 88.9,
        "m1_online_embedding_max_ms": 210.0,
        "worker": {
            "received": 31, "stored": 0, "skipped_success": 27, "errors": 0,
            "embedded": 0, "m1_embedding_registered": 27,
            "embedding_source_counts": {"m1_online": 16}, "pending_embeddings": 0,
            "pending_texts": 0, "queue_size": 0, "thread_alive": True,
        },
    }

    d = delta_of(before, after)

    # 计数器：做差
    assert d["m1_online_embedding_count"] == 16
    assert d["m1_online_embedding_total_ms"] == 1400.0
    assert d["worker"]["received"] == 18
    assert d["worker"]["m1_embedding_registered"] == 16
    # 瞬时值：透传
    assert d["worker"]["queue_size"] == 0
    assert d["worker"]["pending_embeddings"] == 0
    assert d["worker"]["thread_alive"] is True
    # 派生值：透传（avg 由调用方按增量重算，max 是进程级）
    assert d["m1_online_embedding_max_ms"] == 210.0


# ─────────────── 追加型列表（`l2_v4.p_values`）───────────────
#
# `p_values` 是「按调用序」的日志，每行含前面所有任务的 p。求和算卸载率会把
# 每个 span 重复算多遍 —— 和计数器是同一个坑，只是形状是 list。


def test_delta_of_slices_append_only_lists() -> None:
    """前缀扩展的列表 ⇒ 取尾部那一截。"""
    before = {"p_values": [0.1, 0.2]}
    after = {"p_values": [0.1, 0.2, 0.7, 0.9]}

    d = delta_of(before, after)

    assert d["p_values"] == [0.7, 0.9]


def test_delta_of_returns_whole_list_when_not_an_extension() -> None:
    """不是前缀扩展 ⇒ **不是追加日志**，原样给出去。

    猜错方向很危险：硬切 `value[len(before):]` 会静默丢掉前半截，
    而且切出来的长度看着正常。
    """
    before = {"p_values": [0.1, 0.2]}
    after = {"p_values": [0.5, 0.6]}

    d = delta_of(before, after)

    assert d["p_values"] == [0.5, 0.6]


def test_delta_of_handles_empty_before_list() -> None:
    d = delta_of({"p_values": []}, {"p_values": [0.3, 0.4]})

    assert d["p_values"] == [0.3, 0.4]


def test_p_values_deltas_concatenate_to_the_total() -> None:
    """**判据**：把所有行的切片拼起来 == 真实总数，且长度对。"""
    c: dict = {"p_values": []}
    per_task = []
    for i in range(4):
        before = {"p_values": list(c["p_values"])}
        c["p_values"] = c["p_values"] + [float(i)]
        per_task.append(delta_of(before, c)["p_values"])

    assert per_task == [[0.0], [1.0], [2.0], [3.0]]
    assert [x for part in per_task for x in part] == c["p_values"]


def test_delta_of_handles_the_real_l2_v4_shape() -> None:
    before = {
        "hits": 11, "miss_no_vector": 0, "input_dim": 52,
        "model_type": "mlp", "embedding_segment": "V2_goal_recent",
        "threshold": 0.52, "p_values": [0.1, 0.9],
    }
    after = {
        "hits": 19, "miss_no_vector": 1, "input_dim": 52,
        "model_type": "mlp", "embedding_segment": "V2_goal_recent",
        "threshold": 0.52, "p_values": [0.1, 0.9, 0.4, 0.6, 0.8],
    }

    d = delta_of(before, after)

    assert d["hits"] == 8
    assert d["miss_no_vector"] == 1
    assert d["p_values"] == [0.4, 0.6, 0.8]
    # 常量原样透传
    assert d["input_dim"] == 52 and d["model_type"] == "mlp" and d["threshold"] == 0.52


def test_delta_of_handles_the_real_l2_add4_shape() -> None:
    """add4 的 `l2_add4_stats()` 真实形状。

    ⚠️ **两个 τ 也在白名单里。** 它们是选出来的常量，不是计数器 —— 做差会在
    summary 里写成 `tau_gate: 0.0`，读起来像「阈值是 0，所以什么都卸载」。
    实测踩过一次（本文件加这条之前）。
    """
    before = {
        "hits": 3, "miss_no_vector": 0, "input_dim": 116,
        "model_type": "logistic", "gate_segment": "V2_goal_recent",
        "net_segment": "C_bare_w10",
        "tau_gate": 0.53807248, "tau_net": 0.55231072,
        "p_gate_values": [0.1, 0.9], "p_net_values": [0.2, 0.8],
    }
    after = {
        "hits": 7, "miss_no_vector": 2, "input_dim": 116,
        "model_type": "logistic", "gate_segment": "V2_goal_recent",
        "net_segment": "C_bare_w10",
        "tau_gate": 0.53807248, "tau_net": 0.55231072,
        "p_gate_values": [0.1, 0.9, 0.3, 0.7, 0.5, 0.2],
        "p_net_values": [0.2, 0.8, 0.4, 0.6, 0.1, 0.9],
    }

    d = delta_of(before, after)

    assert d["hits"] == 4
    assert d["miss_no_vector"] == 2
    assert d["p_gate_values"] == [0.3, 0.7, 0.5, 0.2]
    assert d["p_net_values"] == [0.4, 0.6, 0.1, 0.9]
    # 常量原样透传 —— **不能变成 0.0**
    assert d["tau_gate"] == 0.53807248
    assert d["tau_net"] == 0.55231072
    assert d["input_dim"] == 116 and d["model_type"] == "logistic"


def test_delta_of_does_not_mutate_its_inputs() -> None:
    before = {"worker": {"received": 1}}
    after = {"worker": {"received": 2}}

    delta_of(before, after)

    assert before == {"worker": {"received": 1}}
    assert after == {"worker": {"received": 2}}
