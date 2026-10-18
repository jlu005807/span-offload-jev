from array import array
import sqlite3
from types import SimpleNamespace

from traceoffload.m4.memory import (
    DangerTier,
    M4ClassSpec,
    M4Memory,
    M4MemoryJob,
    M4MemoryWorker,
    compute_danger_tier,
    compute_local_success,
    placeholder_class_specs,
)
from traceoffload.m4.store import SQLiteM4MemoryStore
from traceoffload.m4.span_features import action_signature
from traceoffload.contracts import (
    ExecutionMode,
    ExecutionStatus,
    GuardOutcome,
    ModelTargetKind,
    OutcomeLabel,
    OutcomeScope,
    RollbackStatus,
    ToolExecutionStatus,
)


def _tool(*, returncode=0, status=ToolExecutionStatus.SUCCESS, error_type=None):
    return SimpleNamespace(
        status=status,
        returncode=returncode,
        error_type=error_type,
    )


def _worker(memory, **kwargs):
    """测试用的 worker：守卫投影**直通**。

    投影本身（读 artifact 做 PCA、维度必须 1024）由
    `test_m4_span_features.py` 和本文件末尾那两条 worker 测试单独覆盖。
    其余用例关心的是别的东西，不该被迫造 1024 维假嵌入。
    """

    kwargs.setdefault("guard_projector", lambda job, embedding: tuple(embedding))
    return M4MemoryWorker(memory, **kwargs)


def _span(
    *,
    online_features: bool = False,
    kind=ModelTargetKind.LARGE_MODEL,
    status=ExecutionStatus.SUCCESS,
    guard_outcomes=(),
    tools=(),
    rollback=RollbackStatus.NOT_NEEDED,
    recovery_required=False,
    recovery_check=None,
):
    profile = (
        SimpleNamespace(
            online_features={
                "prefix_n_A": 3.0,
                "log_est_prompt_tokens": 9.5,
            }
        )
        if online_features
        else None
    )
    return SimpleNamespace(
        runtime_profile=profile,
        route_plan=SimpleNamespace(
            workflow_run_id="workflow-1",
            episode_id="episode-1",
            logical_invocation_id="invocation-1",
            target=SimpleNamespace(kind=kind),
            verifier_plan=SimpleNamespace(steps=()),
        ),
        execution_summary=SimpleNamespace(
            status=status,
            guard_results=tuple(
                SimpleNamespace(outcome=outcome) for outcome in guard_outcomes
            ),
            tool_executions=tuple(tools),
            state_outcome=SimpleNamespace(
                rollback_status=rollback,
                recovery_required=recovery_required,
            ),
        ),
        recovery_check=recovery_check,
    )


def _failed_span(**kwargs):
    """一个**本地执行失败**的 span —— M4 只收这一类（成功的 span 一条都不进）。"""

    kwargs.setdefault("kind", ModelTargetKind.SMALL_MODEL)
    return _span(tools=(_tool(status=ToolExecutionStatus.FAILED, returncode=1),), **kwargs)


def test_verifier_filter_overrides_embedding_distance() -> None:
    memory = M4Memory(
        classes=(
            M4ClassSpec(
                class_key=0,
                verifier_set=frozenset({"schema:v1"}),
                center_embedding=(1.0, 0.0),
            ),
            M4ClassSpec(
                class_key=1,
                verifier_set=frozenset({"exec:v1"}),
                center_embedding=(0.0, 1.0),
            ),
        )
    )

    record = memory.observe(
        "r1",
        (1.0, 0.0),
        _span(),
        verifier_refs=("exec:v1",),
    )

    assert record.class_key == 1


def test_top_five_average_distance_breaks_class_tie() -> None:
    memory = M4Memory(
        classes=(
            M4ClassSpec(class_key=0, center_embedding=(0.0, 1.0)),
            M4ClassSpec(class_key=1, center_embedding=(1.0, 0.0)),
        )
    )
    memory.observe("seed-0", (0.0, 1.0), _span())
    memory.observe("seed-1", (1.0, 0.0), _span())

    record = memory.observe("r2", (1.0, 0.0), _span())

    assert record.class_key == 1
    assert record.assignment_status == "empty_verifier"


def test_tie_falls_back_to_first_class_in_order() -> None:
    memory = M4Memory(
        classes=(
            M4ClassSpec(class_key=0, center_embedding=(1.0, 0.0)),
            M4ClassSpec(class_key=1, center_embedding=(1.0, 0.0)),
        )
    )

    record = memory.observe("r1", (1.0, 0.0), _span())

    assert record.class_key == 0
    assert record.forced_class is True
    assert record.assignment_status == "tie"


def test_no_compatible_verifier_class_falls_back_to_first_class() -> None:
    memory = M4Memory(
        classes=(
            M4ClassSpec(class_key=0, verifier_set=frozenset({"a:v1"})),
            M4ClassSpec(class_key=1, verifier_set=frozenset({"b:v1"})),
        )
    )

    record = memory.observe(
        "r1",
        (1.0, 0.0),
        _span(),
        verifier_refs=("unknown:v1",),
    )

    assert record.class_key == 0
    assert record.forced_class is True
    assert record.assignment_status == "no_candidate"


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        self._now += 1.0
        return self._now


def test_lru_evicts_least_recently_accessed_record_in_class() -> None:
    clock = _FakeClock()
    memory = M4Memory(
        classes=(M4ClassSpec(class_key=0),),
        max_records_per_class=2,
        max_records_total=10,
        clock=clock,
    )

    memory.observe("r1", (1.0, 0.0), _span())
    memory.observe("r2", (0.0, 1.0), _span())
    assert memory.touch("r1") is True
    memory.observe("r3", (1.0, 1.0), _span())

    assert memory.get("r1") is not None
    assert memory.get("r2") is None
    assert memory.get("r3") is not None


def test_total_lru_cap_across_classes() -> None:
    clock = _FakeClock()
    memory = M4Memory(
        classes=(M4ClassSpec(class_key=0),),
        max_records_per_class=10,
        max_records_total=2,
        clock=clock,
    )

    memory.observe("r1", (1.0, 0.0), _span())
    memory.observe("r2", (0.0, 1.0), _span())
    assert memory.touch("r1") is True
    memory.observe("r3", (1.0, 1.0), _span())

    assert memory.get("r1") is not None
    assert memory.get("r2") is None
    assert memory.get("r3") is not None


def test_pending_seed_is_not_evicted_and_can_complete_later() -> None:
    memory = M4Memory(
        classes=(M4ClassSpec(class_key=0),),
        max_records_per_class=1,
        max_records_total=1,
    )
    memory.register_pending("pending-1", (1.0, 0.0))
    memory.observe("r1", (1.0, 0.0), _span())

    assert memory.pending_size == 1
    record = memory.complete("pending-1", _span())

    assert record.record_id == "pending-1"
    assert memory.size == 1


def test_local_success_and_danger_tiers() -> None:
    success = _span(tools=(_tool(returncode=0),))
    assert compute_local_success(success) is True
    assert compute_danger_tier(success) == DangerTier.SUCCESS

    tool_failure = _span(tools=(_tool(returncode=1),))
    assert compute_local_success(tool_failure) is False
    assert compute_danger_tier(tool_failure) == DangerTier.LOW

    rollback_failure = _span(
        status=ExecutionStatus.ROLLBACK_FAILED,
        tools=(_tool(returncode=0),),
        rollback=RollbackStatus.FAILED,
        recovery_required=True,
    )
    assert compute_local_success(rollback_failure) is False
    assert compute_danger_tier(rollback_failure) == DangerTier.HIGH


def test_memory_worker_ignores_same_model_kind() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    worker.enqueue(M4MemoryJob("r1", _span(kind=ModelTargetKind.LARGE_MODEL), embedding=(1.0, 0.0)))
    worker.enqueue(M4MemoryJob("r2", _span(kind=ModelTargetKind.LARGE_MODEL), embedding=(0.0, 1.0)))

    assert worker.process_one() is True
    assert worker.process_one() is True

    assert memory.size == 0


def test_memory_worker_stores_local_failure_only() -> None:
    """入库判据是**本地执行失败**，与路由是否切换**无关**。

    旧的「切换配对」口径（`previous_kind == current_kind` 就两条都丢）是 M4 当
    决策器时留下的 —— 那时它要学的是「切换那一刻发生了什么」。决策角色已经被
    证据否掉，但配对逻辑没跟着删，于是：

        local 失败的 span 不在切换点上 ⇒ **一条都不记** ⇒ 守卫永远学不到它
        成功的 span 正好在切换点上   ⇒ **记了** ⇒ 记忆里混进没失败的

    两个方向都错。新口径：`compute_local_success(span) is False` 就记。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    # 同一个路由目标，中途没有切换 —— 旧口径下这三条一条都不会入库
    ok = _span(kind=ModelTargetKind.SMALL_MODEL)
    bad = _span(
        kind=ModelTargetKind.SMALL_MODEL,
        tools=(_tool(status=ToolExecutionStatus.FAILED, returncode=1),),
    )
    ok2 = _span(kind=ModelTargetKind.SMALL_MODEL)
    worker.enqueue(M4MemoryJob("ok-1", ok, embedding=(1.0, 0.0)))
    worker.enqueue(M4MemoryJob("bad", bad, embedding=(0.0, 1.0)))
    worker.enqueue(M4MemoryJob("ok-2", ok2, embedding=(1.0, 1.0)))
    while worker.process_one():
        pass

    assert memory.get("bad") is not None, "本地失败的 span 必须入库"
    assert memory.get("ok-1") is None, "成功的 span 不入库"
    assert memory.get("ok-2") is None


def test_memory_worker_skips_a_span_that_never_executed_anything() -> None:
    """**有工具真的执行过并且失败** —— 这才是唯一判据。

    不是 `not compute_local_success(...)`：那个把两种「没失败」也算了进来。

        准入被拒的 span      `status=verification_failed`、`tool_executions` **空**
                             ⇒ 它证明的是「分析器认不出这段代码」，不是「这个动作会失败」
        工具成功但 post-check 拒  `status=verification_failed`、工具 `SUCCESS`
                             ⇒ 动作本身执行成功了，被拒的是最终产物

    把这两种收进去，守卫就会基于**无关证据**拒动作 —— 而且查得到、拦得下，
    因为它俩都会被记成「失败样本」。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)

    rejected_before_running = _span(status=ExecutionStatus.VERIFICATION_FAILED)
    assert rejected_before_running.execution_summary.tool_executions == ()

    ran_and_succeeded_but_post_check_rejected = _span(
        status=ExecutionStatus.VERIFICATION_FAILED,
        tools=(_tool(status=ToolExecutionStatus.SUCCESS, returncode=0),),
    )
    ran_and_failed = _span(
        tools=(_tool(status=ToolExecutionStatus.FAILED, returncode=1),),
    )

    for record_id, span in (
        ("never-ran", rejected_before_running),
        ("ran-ok-rejected", ran_and_succeeded_but_post_check_rejected),
        ("ran-and-failed", ran_and_failed),
    ):
        worker.enqueue(M4MemoryJob(record_id, span, embedding=(1.0, 0.0)))
    while worker.process_one():
        pass

    assert memory.get("never-ran") is None, "根本没执行 ⇒ 不是失败样本"
    assert memory.get("ran-ok-rejected") is None, "工具执行成功了 ⇒ 不是失败样本"
    assert memory.get("ran-and-failed") is not None


def test_memory_worker_stores_a_route_switch_that_did_not_fail() -> None:
    """反向的同一件事：**切换本身不再是入库理由**。

    旧口径会把这一对记下来（large→small 是个切换），但两条 span 本地都成功了
    ——「切换」不是「会失败」的证据，留着只会让守卫基于无关证据拒动作。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    worker.enqueue(
        M4MemoryJob("large", _span(kind=ModelTargetKind.LARGE_MODEL), embedding=(1.0, 0.0))
    )
    worker.enqueue(
        M4MemoryJob("small", _span(kind=ModelTargetKind.SMALL_MODEL), embedding=(0.0, 1.0))
    )
    while worker.process_one():
        pass

    assert memory.size == 0
    assert worker.stats()["skipped_success"] == 2


def test_memory_worker_does_not_store_a_span_twice() -> None:
    """同一条失败的 span 被 enqueue 两次，只落一条 —— 幂等。

    旧版本叫 `..._seen_in_two_pairs`：同一条 span 会同时出现在「进入切换」和
    「离开切换」两个配对里，靠 `_store_job` 开头的 `memory.get(record_id)` 挡住重复。
    配对没了，但**幂等这件事还得成立**（重连、重试都可能重放同一条 span）。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    bad = _span(
        kind=ModelTargetKind.SMALL_MODEL,
        tools=(_tool(status=ToolExecutionStatus.FAILED, returncode=1),),
    )
    worker.enqueue(M4MemoryJob("bad", bad, embedding=(1.0, 0.0)))
    worker.enqueue(M4MemoryJob("bad", bad, embedding=(1.0, 0.0)))

    while worker.process_one():
        pass

    assert memory.size == 1
    assert memory.get("bad") is not None


def test_memory_worker_runs_async_and_drains() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    worker.enqueue(M4MemoryJob("before", _failed_span(), embedding=(1.0, 0.0)))
    worker.enqueue(M4MemoryJob("after", _failed_span(), embedding=(0.0, 1.0)))

    worker.start()
    try:
        assert worker.drain(timeout_seconds=2.0) is True
        assert memory.size == 2
        assert worker.stats()["thread_alive"] is True
    finally:
        assert worker.stop(timeout_seconds=2.0) is True


def test_query_returns_thresholded_neighbors_and_touches() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    memory.observe("r1", (1.0, 0.0), _span())
    memory.observe("r2", (0.0, 1.0), _span())

    neighbors = memory.query((1.0, 0.0), top_k=1, similarity_threshold=0.5)

    assert len(neighbors) == 1
    assert neighbors[0].record.record_id == "r1"
    assert neighbors[0].similarity > 0.99
    assert memory.get("r1").last_access_at >= memory.get("r2").last_access_at


def test_apply_outcome_updates_local_success_and_danger() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    memory.observe("r1", (1.0, 0.0), _span())
    candidate_failure = SimpleNamespace(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        scope=OutcomeScope.CANDIDATE,
        label=OutcomeLabel.FAILURE,
    )
    unsafe = SimpleNamespace(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        scope=OutcomeScope.SAFETY,
        label=OutcomeLabel.UNSAFE,
    )

    assert memory.apply_outcome(candidate_failure) is True
    assert memory.get("r1").local_success is False
    assert memory.get("r1").danger_tier == DangerTier.LOW
    assert memory.apply_outcome(unsafe) is True
    assert memory.get("r1").danger_tier == DangerTier.HIGH


def test_apply_outcome_before_record_is_stored_then_backfilled() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    event = SimpleNamespace(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        scope=OutcomeScope.CANDIDATE,
        label=OutcomeLabel.FAILURE,
    )
    assert memory.apply_outcome(event) is False

    memory.observe("r1", (1.0, 0.0), _span())

    assert memory.get("r1").local_success is False
    assert memory.get("r1").danger_tier == DangerTier.LOW


def test_sqlite_persistence_round_trips_pending_and_records(tmp_path) -> None:
    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    memory.register_pending("r1", (1.0, 0.0, 0.0, 0.0, 0.0))
    memory.complete("r1", _span())

    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    assert reloaded.size == 1
    record = reloaded.get("r1")
    assert record is not None
    assert record.workflow_run_id == "workflow-1"
    assert record.class_key in range(5)
    store.close()


def test_worker_lazily_embeds_only_failed_spans() -> None:
    class _Embedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed_one(self, text):
            self.calls += 1
            return SimpleNamespace(values=(1.0, 0.0, 0.0, 0.0, 0.0))

    embedder = _Embedder()
    memory = M4Memory(classes=placeholder_class_specs(dimension=5))
    worker = _worker(
        memory,
        embedder=embedder,
        text_builder=lambda record: "embedding text",
    )
    worker.enqueue(
        M4MemoryJob(
            "same-1",
            _span(kind=ModelTargetKind.LARGE_MODEL),   # 成功 ⇒ 不入库、不嵌入
            embedding_text="ignored",
        )
    )
    worker.enqueue(
        M4MemoryJob(
            "same-2",
            _span(kind=ModelTargetKind.LARGE_MODEL),   # 成功 ⇒ 不入库、不嵌入
            embedding_text="ignored",
        )
    )
    while worker.process_one():
        pass
    assert embedder.calls == 0
    assert memory.size == 0

    worker.enqueue(
        M4MemoryJob(
            "before",
            _failed_span(),
            embedding_text="before text",
        )
    )
    worker.enqueue(
        M4MemoryJob(
            "after",
            _failed_span(),
            embedding_text="after text",
        )
    )
    while worker.process_one():
        pass
    assert embedder.calls == 2
    assert memory.size == 2


def test_worker_applies_delayed_outcome_async() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    memory.observe("r1", (1.0, 0.0), _span())
    event = SimpleNamespace(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        scope=OutcomeScope.SAFETY,
        label=OutcomeLabel.UNSAFE,
    )
    worker = _worker(memory)

    worker.enqueue_outcome(event)
    assert worker.process_one() is True

    assert memory.get("r1").danger_tier == DangerTier.HIGH


def test_worker_prefers_m1_online_embedding_and_records_latency() -> None:
    class _Embedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed_one(self, text):
            self.calls += 1
            return SimpleNamespace(values=(9.0, 9.0), latency_ms=99.0)

    embedder = _Embedder()
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory, embedder=embedder)
    worker.register_pending_embedding("before", (1.0, 0.0), latency_ms=12.5)
    worker.register_pending_embedding("after", (0.0, 1.0), latency_ms=13.5)
    worker.enqueue(M4MemoryJob("before", _failed_span()))
    worker.enqueue(M4MemoryJob("after", _failed_span()))

    while worker.process_one():
        pass

    assert embedder.calls == 0
    assert memory.get("before").embedding_source == "m1_online"
    assert memory.get("before").embedding_latency_ms == 12.5
    assert memory.get("after").embedding_latency_ms == 13.5


def test_sqlite_persistence_round_trips_embedding_source_and_latency(tmp_path) -> None:
    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    memory.observe(
        "r1",
        (1.0, 0.0, 0.0, 0.0, 0.0),
        _span(),
        embedding_latency_ms=8.25,
        embedding_source="m1_online",
    )
    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    record = reloaded.get("r1")
    assert record.embedding_latency_ms == 8.25
    assert record.embedding_source == "m1_online"
    store.close()


# ---------------------------------------------------------------------------
# 决策 -> 结局 账本字段（L2 x M4 联动设计 §4 第 0 步）
# ---------------------------------------------------------------------------


def _decided_span(
    *,
    kind=ModelTargetKind.SMALL_MODEL,
    execution_mode=ExecutionMode.GUARDED_SLM,
    reason_code="classifier_slm",
    predictor_p=0.83,
    threshold=0.5,
    online_features=None,
    attempts=(),
    latency_ms=1200,
    workflow_run_id="workflow-1",
    episode_id="episode-1",
    logical_invocation_id="invocation-1",
):
    """带完整决策元数据的 Span；用于校验 M4 账本字段的投影。"""

    return SimpleNamespace(
        route_plan=SimpleNamespace(
            workflow_run_id=workflow_run_id,
            episode_id=episode_id,
            logical_invocation_id=logical_invocation_id,
            target=SimpleNamespace(kind=kind),
            execution_mode=execution_mode,
            verifier_plan=SimpleNamespace(steps=()),
            decision_meta=SimpleNamespace(
                policy_version="policy-test",
                reason_code=reason_code,
                classifier_p=predictor_p,
                classifier_threshold=threshold,
                classifier_version="l2-demo",
                classifier_feature_version="fv-1",
            ),
        ),
        runtime_profile=SimpleNamespace(online_features=online_features or {}),
        execution_summary=SimpleNamespace(
            status=ExecutionStatus.SUCCESS,
            guard_results=(),
            tool_executions=(),
            state_outcome=SimpleNamespace(
                rollback_status=RollbackStatus.NOT_NEEDED,
                recovery_required=False,
            ),
            model_attempts=tuple(attempts),
            latency_ms=latency_ms,
        ),
        recovery_check=None,
    )


def _attempt(*, input_tokens=None, output_tokens=None, cost=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )


def test_record_projects_decision_fields_from_route_plan() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    record = memory.observe(
        "r1",
        (1.0, 0.0),
        _decided_span(
            online_features={"prefix_n_A": 2.0, "log_est_prompt_tokens": 8.5},
        ),
    )

    assert record.action == "small_model:guarded_slm"
    assert record.decision_reason == "classifier_slm"
    assert record.policy_version == "policy-test"
    assert record.predictor_p == 0.83
    assert record.predictor_threshold == 0.5
    assert record.predictor_version == "l2-demo"
    assert record.predictor_feature_version == "fv-1"
    assert record.router_features_json == (
        '{"log_est_prompt_tokens": 8.5, "prefix_n_A": 2.0}'
    )


def test_record_decision_fields_default_to_unknown_on_legacy_span() -> None:
    """旧 Span 没有 decision_meta / execution_mode，不得伪造出有效决策。"""

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    record = memory.observe("r1", (1.0, 0.0), _span())

    assert record.action == "unknown"
    assert record.decision_reason == "unknown"
    assert record.policy_version == "unknown"
    assert record.predictor_p is None
    assert record.router_features_json is None


def test_record_sums_span_cost_across_attempts() -> None:
    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    record = memory.observe(
        "r1",
        (1.0, 0.0),
        _decided_span(
            attempts=(
                _attempt(input_tokens=100, output_tokens=20, cost=0.01),
                _attempt(input_tokens=150, output_tokens=30, cost=0.02),
            ),
            latency_ms=1500,
        ),
    )

    assert record.span_input_tokens == 250
    assert record.span_output_tokens == 50
    assert record.span_cost == 0.03
    assert record.span_latency_ms == 1500


def test_span_cost_missing_attempt_usage_stays_none() -> None:
    """没有 token 统计时保持 None，不写 0 冒充「零开销」。"""

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    record = memory.observe(
        "r1",
        (1.0, 0.0),
        _decided_span(attempts=(_attempt(),)),
    )

    assert record.span_input_tokens is None
    assert record.span_output_tokens is None
    assert record.span_cost is None


def _outcome(scope, label):
    return SimpleNamespace(
        workflow_run_id="workflow-1",
        episode_id="episode-1",
        logical_invocation_id="invocation-1",
        scope=scope,
        label=label,
    )


def test_task_success_only_written_from_task_level_scopes() -> None:
    """只有 episode_final / workflow_final 携带任务级结局语义。"""

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    memory.observe("r1", (1.0, 0.0), _span())
    record = memory.get("r1")

    # downstream / candidate / safety 都不是任务级结局。
    memory.apply_outcome(_outcome(OutcomeScope.DOWNSTREAM, OutcomeLabel.FAILURE))
    memory.apply_outcome(_outcome(OutcomeScope.CANDIDATE, OutcomeLabel.FAILURE))
    memory.apply_outcome(_outcome(OutcomeScope.SAFETY, OutcomeLabel.SAFE))
    assert record.task_success is None
    assert record.label_scope is None

    memory.apply_outcome(_outcome(OutcomeScope.EPISODE_FINAL, OutcomeLabel.FAILURE))
    assert record.task_success is False
    assert record.label_scope == "episode_final"


def test_task_success_unknown_label_stays_censored() -> None:
    """unknown 标签 = censored：既不算成功也不算失败。"""

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    memory.observe("r1", (1.0, 0.0), _span())
    record = memory.get("r1")

    memory.apply_outcome(_outcome(OutcomeScope.WORKFLOW_FINAL, OutcomeLabel.SUCCESS))
    assert record.task_success is True
    memory.apply_outcome(_outcome(OutcomeScope.WORKFLOW_FINAL, OutcomeLabel.UNKNOWN))
    assert record.task_success is None


def test_sqlite_round_trips_decision_and_outcome_fields(tmp_path) -> None:
    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    memory.observe(
        "r1",
        (1.0, 0.0, 0.0, 0.0, 0.0),
        _decided_span(
            attempts=(_attempt(input_tokens=10, output_tokens=4, cost=0.5),),
            online_features={"prefix_n_A": 1.0},
        ),
        embedding_source="m1_online",
    )
    memory.apply_outcome(_outcome(OutcomeScope.EPISODE_FINAL, OutcomeLabel.FAILURE))

    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    record = reloaded.get("r1")
    assert record.action == "small_model:guarded_slm"
    assert record.decision_reason == "classifier_slm"
    assert record.predictor_p == 0.83
    assert record.predictor_version == "l2-demo"
    assert record.router_features_json == '{"prefix_n_A": 1.0}'
    assert record.span_input_tokens == 10
    assert record.span_output_tokens == 4
    assert record.span_cost == 0.5
    assert record.task_success is False
    assert record.label_scope == "episode_final"
    store.close()


def test_sqlite_round_trips_unknown_task_success_as_null(tmp_path) -> None:
    """未知结局必须往返成 None，不能被读成 False。"""

    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    memory.observe("r1", (1.0, 0.0, 0.0, 0.0, 0.0), _span())

    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    assert reloaded.get("r1").task_success is None
    store.close()


def test_legacy_sqlite_database_gains_new_columns(tmp_path) -> None:
    """旧库（无决策/结局/成本列）必须能平滑迁移，历史行读成保守默认值。"""

    path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE m4_records(
            record_id TEXT PRIMARY KEY,
            workflow_run_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            logical_invocation_id TEXT NOT NULL,
            class_key INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            full_span_json TEXT NOT NULL,
            local_success INTEGER NOT NULL,
            danger_tier INTEGER NOT NULL,
            verifier_refs TEXT NOT NULL,
            entered_at REAL NOT NULL,
            last_access_at REAL NOT NULL,
            pair_id TEXT,
            switch_direction TEXT,
            forced_class INTEGER NOT NULL,
            assignment_status TEXT NOT NULL
        );
        """
    )
    legacy.execute(
        """
        INSERT INTO m4_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-1",
            "workflow-1",
            "episode-1",
            "invocation-1",
            0,
            array("f", (1.0, 0.0, 0.0, 0.0, 0.0)).tobytes(),
            "{}",
            1,
            0,
            "[]",
            0.0,
            0.0,
            None,
            None,
            0,
            "normal",
        ),
    )
    legacy.commit()
    legacy.close()

    store = SQLiteM4MemoryStore(path)
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    record = memory.get("legacy-1")

    assert record is not None
    assert record.action == "unknown"
    assert record.policy_version == "unknown"
    assert record.predictor_p is None
    assert record.task_success is None
    assert record.span_latency_ms == 0
    store.close()


def test_sqlite_persistence_round_trips_failed_action_signatures(tmp_path) -> None:
    """失败动作签名必须落盘。

    没有它，进程一重启 M4 的记忆就是空的 —— 守卫会退化成**永远放行**，
    而且日志里看不出任何异常（fail-open 会把它伪装成「查不到证据」）。
    M4 的全部价值都在「跨 span / 跨运行的历史」，只活在内存里等于没有。
    """

    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    signature = action_signature("query", {"sql": "boom"})
    memory.register_pending("r1", (1.0, 0.0, 0.0, 0.0, 0.0))
    memory.complete("r1", _span(), failed_actions=(signature,))

    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    record = reloaded.get("r1")
    assert record is not None
    assert record.failed_actions == frozenset({signature})
    store.close()


def test_sqlite_persistence_round_trips_an_empty_failed_action_set(tmp_path) -> None:
    """没有失败动作的 span 也要能读回来 —— 区分「空集合」与「字段丢了」。"""

    store = SQLiteM4MemoryStore(tmp_path / "m4_memory.sqlite3")
    memory = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    memory.register_pending("r1", (1.0, 0.0, 0.0, 0.0, 0.0))
    memory.complete("r1", _span())

    reloaded = M4Memory(classes=placeholder_class_specs(dimension=5), persistence=store)
    record = reloaded.get("r1")
    assert record is not None
    assert record.failed_actions == frozenset()
    store.close()


def test_memory_worker_carries_failed_action_signatures_into_the_record() -> None:
    """失败签名必须**穿过 job 到达记录**。

    丢在这一段的话守卫永远查不到东西 —— 而 fail-open 会把它伪装成
    「没有证据」，日志里看不出任何异常。这是最容易漏、也最难发现的一段：
    每一段单看都对，只有端到端连起来才知道签名有没有到。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    signature = action_signature("execute_code_batch", {"code": "boom()"})
    worker.enqueue(
        M4MemoryJob(
            "before",
            _failed_span(),
            embedding=(1.0, 0.0),
        )
    )
    worker.enqueue(
        M4MemoryJob(
            "after",
            _failed_span(),
            embedding=(0.0, 1.0),
            failed_actions=(signature,),
        )
    )

    while worker.process_one():
        pass

    record = memory.get("after")
    assert record is not None
    assert record.failed_actions == frozenset({signature})


def test_enqueue_completed_span_carries_failed_action_signatures() -> None:
    """走**主链路真正的入口** `enqueue_completed_span`，而不是直接构造 job。

    前一条测试直接 `M4MemoryJob(..., failed_actions=...)`，盖不到
    `enqueue_completed_span` 里那次转发 —— 而 runner 调的是后者。
    变异测试在这里抓到过漏：把转发那一行删掉，前一条测试照样绿。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = _worker(memory)
    signature = action_signature("execute_code_batch", {"code": "boom()"})
    worker.enqueue_completed_span("before", _failed_span(), embedding=(1.0, 0.0))
    worker.enqueue_completed_span(
        "after",
        _failed_span(),
        embedding=(0.0, 1.0),
        failed_actions=(signature,),
    )

    while worker.process_one():
        pass

    record = memory.get("after")
    assert record is not None
    assert record.failed_actions == frozenset({signature})


def test_worker_stores_the_52_dim_guard_vector_not_the_raw_embedding() -> None:
    """M4 落库的是 **52 维守卫向量**，不是 1024 维原始嵌入。

    为什么这条必须钉住：存错维度**不会当场报错**。守卫拿 52 维去比库里的
    1024 维，那个 `_cosine_distance` 的 `ValueError` 只在「库里正好有一条
    带失败签名的记录」时才触发 —— 平时完全看不见（实测第一次冒烟就是这样，
    `failed_actions` 是空的，循环提前 `continue` 了）。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    # **不注入投影器** —— 这条测试的全部意义就是验真实维度
    worker = M4MemoryWorker(memory)
    # 必须是**本地失败**的 span —— 成功的会被 `_handle_span_job` 直接跳过
    worker.enqueue_completed_span(
        "large-1",
        _failed_span(kind=ModelTargetKind.LARGE_MODEL, online_features=True),
        embedding=tuple([0.1] * 1024),
    )
    worker.enqueue_completed_span(
        "small-1",
        _failed_span(kind=ModelTargetKind.SMALL_MODEL, online_features=True),
        embedding=tuple([0.2] * 1024),
    )

    while worker.process_one():
        pass

    record = memory.get("small-1")
    assert record is not None
    assert len(record.embedding) == 52


def test_worker_skips_a_span_that_cannot_be_projected() -> None:
    """拿不到在线特征 ⇒ **不存**这条记录。

    宁可漏记，不可存一个维度不对的向量 —— 后者会让守卫在某个深夜拿 52 维
    去比 1024 维，然后炸在一个跟根因隔了三层的地方。
    """

    memory = M4Memory(classes=(M4ClassSpec(class_key=0),))
    worker = M4MemoryWorker(memory)
    # online_features=False ⇒ span 上没有 runtime_profile，两个都派生不出来
    worker.enqueue_completed_span(
        "large-1",
        _span(kind=ModelTargetKind.LARGE_MODEL, online_features=False),
        embedding=tuple([0.1] * 1024),
    )
    worker.enqueue_completed_span(
        "small-1",
        _span(kind=ModelTargetKind.SMALL_MODEL, online_features=False),
        embedding=tuple([0.2] * 1024),
    )

    while worker.process_one():
        pass

    assert memory.size == 0
