"""换 52 维之后，老库里的 1024 维向量必须被**跳过**，不能被误读。

为什么这是一道必须有的闸
------------------------
M4 的嵌入维度从 1024 换成 52（= 2 前缀标量 + PCA50），这不是「缩小」——
**是新旧两个完全不同的向量空间**，坐标之间没有任何换算关系。

而 `runs/` 下已经有 60 个 `to-classifier` 跑出来的库，里面全是 1024 维的 BLOB。
新代码读到它们会 `_decode_embedding` 出一个 1024 元的元组，然后拿去和 52 维的
查询向量算 `_cosine_distance` —— 要么抛异常，要么**算出毫无意义的相似度**。
后者更糟：它不报错，只会让守卫基于垃圾证据拒绝动作。

⇒ 读的时候按维度过滤，**对不上的一律当作不存在**（fail-open 的一侧）。
"""

from types import SimpleNamespace

from traceoffload.m4.memory import M4Memory, placeholder_class_specs
from traceoffload.m4.store import SQLiteM4MemoryStore

FOREIGN_DIM = 1024
GUARD_DIM = 52


def _span() -> SimpleNamespace:
    return SimpleNamespace(
        route_plan=SimpleNamespace(
            workflow_run_id="workflow-1",
            episode_id="episode-1",
            logical_invocation_id="invocation-1",
            target=SimpleNamespace(kind="large_model"),
            verifier_plan=SimpleNamespace(steps=()),
        ),
        execution_summary=SimpleNamespace(status="success"),
    )


def _write_legacy_record(path, record_id: str, dimension: int) -> None:
    store = SQLiteM4MemoryStore(path)
    memory = M4Memory(
        classes=placeholder_class_specs(dimension=dimension),
        persistence=store,
    )
    memory.register_pending(record_id, tuple([0.1] * dimension))
    memory.complete(record_id, _span())
    store.close()


def test_load_records_skips_embeddings_of_a_foreign_dimension(tmp_path) -> None:
    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "legacy", FOREIGN_DIM)

    store = SQLiteM4MemoryStore(path)

    assert store.load_records(embedding_dimension=GUARD_DIM) == []
    store.close()


def test_load_records_without_dimension_keeps_old_behaviour(tmp_path) -> None:
    """不传维度 = 不过滤 —— 这是既有调用方的行为，不能被这次改动改掉。"""

    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "legacy", FOREIGN_DIM)

    store = SQLiteM4MemoryStore(path)

    loaded = store.load_records()
    assert len(loaded) == 1
    assert len(loaded[0]["embedding"]) == FOREIGN_DIM
    store.close()


def test_load_records_keeps_records_of_the_matching_dimension(tmp_path) -> None:
    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "current", GUARD_DIM)

    store = SQLiteM4MemoryStore(path)

    loaded = store.load_records(embedding_dimension=GUARD_DIM)
    assert [row["record_id"] for row in loaded] == ["current"]
    store.close()


def test_constructor_dimension_filters_loads_without_a_per_call_argument(tmp_path) -> None:
    """维度闸应当由**构造时**定，而不是每次调用都记得传。

    M4Memory 启动时会 `load_records()` 恢复历史 —— 那一次调用由库内部发起，
    没有「调用方记得传维度」这回事。构造时定死才不会漏。
    """

    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "legacy", FOREIGN_DIM)
    _write_legacy_record(path, "current", GUARD_DIM)

    store = SQLiteM4MemoryStore(path, embedding_dimension=GUARD_DIM)

    assert [row["record_id"] for row in store.load_records()] == ["current"]
    store.close()


def test_constructor_dimension_can_be_overridden_per_call(tmp_path) -> None:
    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "legacy", FOREIGN_DIM)

    store = SQLiteM4MemoryStore(path, embedding_dimension=GUARD_DIM)

    assert len(store.load_records(embedding_dimension=FOREIGN_DIM)) == 1
    store.close()


def test_memory_does_not_load_records_of_a_foreign_dimension(tmp_path) -> None:
    """`M4Memory` 必须**自己**挡住换空间之前的记录，不管 store 怎么构造的。

    读进来的下场是在 `_cosine_distance` 上撞「维度不符」。异常是响的，
    但**崩在启动路径上**同样是事故 —— 而且这一类「调用方忘了传参数」的
    漏法靠约定防不住，只能靠库自己知道自己要多大。
    """

    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "legacy", FOREIGN_DIM)

    # 故意**不传** embedding_dimension —— 模拟调用方漏传
    store = SQLiteM4MemoryStore(path)
    memory = M4Memory(
        classes=placeholder_class_specs(dimension=GUARD_DIM), persistence=store
    )

    assert memory.size == 0
    store.close()


def test_memory_still_loads_records_of_its_own_dimension(tmp_path) -> None:
    path = tmp_path / "m4_memory.sqlite3"
    _write_legacy_record(path, "current", GUARD_DIM)

    store = SQLiteM4MemoryStore(path)
    memory = M4Memory(
        classes=placeholder_class_specs(dimension=GUARD_DIM), persistence=store
    )

    assert memory.size == 1
    store.close()
