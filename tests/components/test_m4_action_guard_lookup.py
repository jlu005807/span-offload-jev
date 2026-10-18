"""守卫的查询侧：把「候选动作 + span 身份」变成「M4 里的命中」。

缺失的那一环
------------
`M4FailureMemoryVerifier` 只负责**顺序**（形状 → 记忆 → fail-open），
`M4Memory.query_failed_action` 只负责**判定**（相似度 AND 签名）。
中间这一段没人做：拿 span 身份去 `_m4_gate_vectors` 换 52 维、
再对候选响应里的**每一个** tool call 查一遍。

`ModelResponse` 里没有 span 标识，所以身份由 M3 传下来 —— 这一段就是它的消费者。

两条 fail-open，都要有测试
-------------------------
* 身份不在向量缓存里（M1 还没算完 / 换进程了）⇒ 空，放行
* 身份是 None（非 guarded 路径传下来的）⇒ 空，放行
"""

from types import SimpleNamespace

from traceoffload.m4.action_guard import lookup_failed_actions
from traceoffload.m4.memory import M4ClassSpec, M4Memory
from traceoffload.m4.span_features import action_signature
from traceoffload.interfaces import SpanIdentity

VEC = (1.0, 0.0)
IDENTITY = SpanIdentity(
    workflow_run_id="workflow-1",
    episode_id="episode-1",
    logical_invocation_id="invocation-1",
)


def _span() -> SimpleNamespace:
    return SimpleNamespace(
        route_plan=SimpleNamespace(
            workflow_run_id="workflow-1",
            episode_id="episode-1",
            logical_invocation_id="invocation-1",
            target=SimpleNamespace(kind="small_model"),
            verifier_plan=SimpleNamespace(steps=()),
        ),
        execution_summary=SimpleNamespace(status="success"),
    )


def _candidate(*calls: tuple[str, str]) -> SimpleNamespace:
    """``calls`` 是 ``(tool_call_id, code)``。"""

    return SimpleNamespace(
        tool_calls=tuple(
            SimpleNamespace(
                tool_call_id=call_id,
                tool_name="execute_code_batch",
                arguments={"code": code},
            )
            for call_id, code in calls
        )
    )


def _memory_with_failure(code: str) -> M4Memory:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),), top_k=5)
    memory.observe(
        "r1",
        VEC,
        _span(),
        failed_actions=(action_signature("execute_code_batch", {"code": code}),),
    )
    return memory


def test_returns_the_match_when_identity_and_action_both_hit() -> None:
    hits = lookup_failed_actions(
        _candidate(("c1", "boom()")),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with_failure("boom()"),
        vector_key=lambda identity: "workflow-1|invocation-1",
    )

    assert hits
    assert hits[0].record.record_id == "r1"


def test_returns_empty_when_the_identity_has_no_cached_vector() -> None:
    """M1 还没算完 / 换进程了 ⇒ 放行，不报错。"""

    hits = lookup_failed_actions(
        _candidate(("c1", "boom()")),
        IDENTITY,
        guard_vectors={},
        memory=_memory_with_failure("boom()"),
        vector_key=lambda identity: "workflow-1|invocation-1",
    )

    assert hits == ()


def test_returns_empty_when_identity_is_missing() -> None:
    """非 guarded 路径不会传身份进来 —— 必须放行。"""

    hits = lookup_failed_actions(
        _candidate(("c1", "boom()")),
        None,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with_failure("boom()"),
        vector_key=lambda identity: "workflow-1|invocation-1",
    )

    assert hits == ()


def test_finds_a_hit_among_several_tool_calls() -> None:
    """一个响应可能带多个 tool call —— 逐个查，命中任意一个即可。"""

    hits = lookup_failed_actions(
        _candidate(("c1", "fine()"), ("c2", "boom()")),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with_failure("boom()"),
        vector_key=lambda identity: "workflow-1|invocation-1",
    )

    assert hits


def test_returns_empty_when_no_tool_call_matches() -> None:
    hits = lookup_failed_actions(
        _candidate(("c1", "fine()")),
        IDENTITY,
        guard_vectors={"workflow-1|invocation-1": VEC},
        memory=_memory_with_failure("boom()"),
        vector_key=lambda identity: "workflow-1|invocation-1",
    )

    assert hits == ()


# ------------------------------------------------------------------ 端到端
def test_real_52_dim_vector_round_trips_through_store_and_query(tmp_path) -> None:
    """**真实 52 维**：写库 → 读回 → 守卫查询命中。

    这条补的是一个只有端到端才会暴露的洞：守卫拿 52 维、库里存 1024 维。
    那个 `_cosine_distance` 的 `ValueError` **只在库里有 failed_actions 时
    才会触发** —— 第一次 AppWorld 冒烟里 `failed_actions` 全是空的，
    循环提前 `continue` 了，所以「跑得通、日志干净、其实错了」。

    这里用真 `derive_span_features` 产出 52 维，不手工塞小向量 ——
    维度对不对正是要验的东西。
    """

    import random

    from traceoffload.m4.memory import M4Memory, M4ClassSpec
    from traceoffload.m4.store import SQLiteM4MemoryStore
    from traceoffload.m4.span_features import (
        derive_span_features,
        guard_vector_dimension,
    )

    rng = random.Random(20260924)
    raw_embedding = [rng.uniform(-1.0, 1.0) for _ in range(1024)]
    features = {"prefix_n_A": 3.0, "log_est_prompt_tokens": 9.5}
    vector = derive_span_features(
        raw_embedding, (features["prefix_n_A"], features["log_est_prompt_tokens"])
    )
    assert len(vector) == guard_vector_dimension()

    signature = action_signature("execute_code_batch", {"code": "boom()"})
    dimension = guard_vector_dimension()
    store = SQLiteM4MemoryStore(tmp_path / "m4.sqlite3", embedding_dimension=dimension)
    memory = M4Memory(
        classes=(M4ClassSpec(class_key=0),), persistence=store
    )
    memory.observe("r1", vector, _span(), failed_actions=(signature,))

    # 换个进程读回来 —— 走的是 SQLite 的 52 维 BLOB
    reopened_store = SQLiteM4MemoryStore(
        tmp_path / "m4.sqlite3", embedding_dimension=dimension
    )
    reopened = M4Memory(classes=(M4ClassSpec(class_key=0),), persistence=reopened_store)

    hits = reopened.query_failed_action(vector, signature, similarity_threshold=0.8)

    assert hits, "真实 52 维写进去、读回来、查不到 —— 维度或落盘有问题"
    assert hits[0].record.record_id == "r1"
    store.close()
    reopened_store.close()
