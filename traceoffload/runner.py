"""Pilot runner: drives the frozen pilot end to end for one ablation mode.

The runner reads every model, dataset and threshold from the unique pilot
configuration and wires the real M1-M4 components around an injected
gateway, model and environment factory.  It supports seven ablation modes and
writes raw JSONL records per workflow/episode plus an aggregate summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import AwareDatetime

from .clock import real_time
from .config import default_results_dir
from .m3.integration import ToolRegistryEntry
from .contracts import (
    AllowedTargets,
    AttemptRole,
    Attribution,
    Component,
    EffectScope,
    ExecutionMode,
    FailureAction,
    FallbackPlan,
    FeedbackSnapshot,
    GuardRef,
    ModelParameters,
    ModelRef,
    ModelTarget,
    ModelTargetKind,
    OutcomeEvent,
    OutcomeLabel,
    OutcomeScope,
    PassAction,
    PassCondition,
    PassConditionKind,
    ProducerRef,
    RecoveryLevel,
    RoutePlan,
    RoutePolicy,
    RuntimeProfile,
    StateAccessMode,
    VerifierPlan,
    VerifierStage,
    VerifierStep,
    VerifierSubject,
    WorkflowGoal,
    ToolDefinition,
    VersionedRef,
)
from .coordinator import Coordinator
from .fakes import FakeEvaluator, FakeVerifier
from .interfaces import Evaluator, ModelGateway, Router, Verifier
from .m1.profiler import RuntimeProfiler
from .m1.span_embedding import (
    SpanEmbeddingTextBuilder,
    SpanEmbeddingTextBuilderV1,
    SpanEmbeddingTextBuilderV2,
    TEMPLATE_VERSION_V1,
    TEMPLATE_VERSION_V2,
    SpanActionsTextBuilder,
    build_embedding_text_from_span_record,
    build_embedding_text_from_span_record_v1,
    build_embedding_text_from_span_record_v2,
)
from .m2.predictor import load_l2_add4_predictor
from .m2.routers import (
    _AlwaysLargeRouter,
    _AlwaysSmallRouter,
    _ClassifierRouter,
    _RandomRouter,
    _ShadowRouter,
    _guarded_slm_plan,
)
from .mechanism_counters import MechanismCounters
from .m1.profiler import ClassifierFeatureProfiler
from .m1.appworld_features import AppWorldOnlineFeatureBuilder
from .m3.executor import FULL_STATE_COVERAGE, ControlledExecutor
from .m1.embedding_client import OllamaEmbeddingClient
from .m4.action_guard import lookup_failed_actions
from .m4.memory import (
    DEFAULT_MEMORY_GUARD_THRESHOLD,
    M4Memory,
    M4MemoryWorker,
    placeholder_class_specs,
)
from .m4.span_features import (
    failed_action_signatures,
    guard_vector,
    guard_vector_dimension,
)
from .m4.store import SQLiteM4MemoryStore
from .storage import (
    EventHistoryStore,
    EventStore,
    HistoryStore,
    RequestResponseStore,
    SQLiteHistoryStore,
    MemoryProbeLedger,
    PublishJobStatus,
    SnapshotHistoryStore,
    SnapshotStore,
    SQLiteProbeLedger,
)
from .m3.guards import ToolPolicy

#: 只保留实验真正用到的四条路由。
#:   llm_only / slm_only  —— 两个固定模型基线
#:   random_route         —— 随机替换对照（论文的 random substitution）
#:   classifier_route     —— 116 维两阶段路由器（+ shadow 装饰器）
PILOT_MODES = (
    "llm_only",
    "slm_only",
    "random_route",
    "classifier_route",
)

_PRE_COMMIT_STAGE = VerifierStage.PRE_COMMIT


def _candidate_response_for_m4(record: Any, accepted_response: Any) -> Any:
    """给 M4 取「这一步 SLM 到底产出了什么」的响应。

    ⚠️ Coordinator 的回调传的是 ``accepted_response``，
    **只有被接受时它才非 None** —— 失败路径（`verification_failed`）下必然是
    None。而失败恰恰是最该记的时候（那正是「这个动作不该做」的证据）。

    只看顶层参数的话，每一次失败都会记下空集 ⇒ 守卫**永远学不到东西**，
    而且不报错。候选响应一直躺在
    ``record.execution_summary.model_attempts`` 里（`small_model_candidate`
    那条的 `response`），所以取不到顶层参数时回头从那里取。
    """

    if accepted_response is not None:
        return accepted_response
    summary = getattr(record, "execution_summary", None)
    for attempt in getattr(summary, "model_attempts", ()) or ():
        if str(getattr(attempt, "role", "")) == "small_model_candidate":
            return getattr(attempt, "response", None)
    return None


def _utcnow() -> AwareDatetime:
    return datetime.now(timezone.utc)




def _model_refs(config: dict[str, Any]) -> tuple[ModelRef, ModelRef]:
    pilot_id = config["pilot"]["id"]
    models = config["models"]

    def ref(kind: str) -> ModelRef:
        model_id = models[kind]["model_id"].replace("/", "-")
        return ModelRef(kind="model", id=model_id, version=pilot_id)

    return ref("large"), ref("small")


def _decode_parameters(config: dict[str, Any]) -> ModelParameters:
    decoding = config["models"]["decoding"]

    def optional_float(name: str) -> float | None:
        value = decoding.get(name)
        return None if value is None else float(value)

    def optional_int(name: str) -> int | None:
        value = decoding.get(name)
        return None if value is None else int(value)

    return ModelParameters(
        temperature=optional_float("temperature"),
        top_p=optional_float("top_p"),
        top_k=optional_int("top_k"),
        max_output_tokens=optional_int("max_output_tokens"),
        seed=optional_int("seed"),
    )



class PilotRunner:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        mode: str,
        output_dir: Path | str | None = None,
        gateway: ModelGateway,
        model: Any,
        env_factory: Callable[[], Any],
        verifier: Verifier | None = None,
        evaluator: Evaluator | None = None,
        oracle: Callable[[RuntimeProfile], bool] | None = None,
        step_limit: int | None = None,
        cost_limit: float | None = None,
        agent_output_path: Path | str | None = None,
        clock: Callable[[], AwareDatetime] | None = None,
        system_template: str = "{{ task }}",
        instance_template: str = "{{ task }}",
        experiment_id: str | None = None,
    ) -> None:
        if mode not in PILOT_MODES:
            raise ValueError(f"unknown pilot mode {mode!r}; expected one of {PILOT_MODES}")
        # T2 机制计数（spec §3.2）。**子类可能在 `super().__init__` 之前就建好了**
        # —— 守卫要在那时构造，而守卫要收这个计数器。所以见到已有的就不重造，
        # 否则守卫手里的那个和 runner 暴露的是两份，计数全丢。
        if getattr(self, "_counters", None) is None:
            self._counters = MechanismCounters()
        self._config = config
        self._mode = mode
        self._output_dir = Path(output_dir) if output_dir is not None else default_results_dir()
        self._gateway = gateway
        self._model = model
        self._env_factory = env_factory
        self._verifier = verifier or FakeVerifier()
        self._evaluator = evaluator or FakeEvaluator()
        self._oracle = oracle
        self._step_limit = step_limit if step_limit is not None else int(config["agent"]["step_limit"])
        configured_cost_limit = config.get("agent", {}).get("cost_limit")
        self._cost_limit = (
            cost_limit
            if cost_limit is not None
            else (
                float(configured_cost_limit)
                if configured_cost_limit is not None
                else None
            )
        )
        self._agent_output_path = (
            Path(agent_output_path) if agent_output_path is not None else None
        )
        self._clock = clock or _utcnow
        self._system_template = system_template
        self._instance_template = instance_template
        self._policy = self._route_policy(config)
        self._large, self._small = _model_refs(config)
        self._history: HistoryStore | None = None
        self._events, self._snapshots = self._make_history_stores(config)
        # 探针台账（A7）：与 history 同后端——sqlite 跨进程持久化，内存仅测试。
        if isinstance(self._history, SQLiteHistoryStore):
            self._probe_ledger = SQLiteProbeLedger(self._history.path)
        else:
            self._probe_ledger = MemoryProbeLedger()
        # 实验作用域（A7 探针预算）：同一实验的所有 workflow/进程共享同一
        # experiment_id；新实验必须换新 id，避免冷启动重启时复用旧台账预算。
        self._experiment_id = experiment_id
        #: 每次**进程**一个的运行标识 —— `workflow_run_id` 的唯一性来源。
        #: 不能复用 `experiment_id`：它在 AppWorld 路径上恒等于 `pilot.id`。
        self._run_nonce = uuid4().hex[:8]
        self._calibration_version = config["runner"]["calibration_version"]
        self._workflow_rows: dict[str, dict[str, Any]] = {}
        self._m4_memory: M4Memory | None = None
        self._m4_memory_store: SQLiteM4MemoryStore | None = None
        self._m4_memory_worker: M4MemoryWorker | None = None
        # 嵌入链路的属性也先给兜底值：`_m1_embedding_needed()` 会在构造期被调用，
        # 而 `_l2_v4_predictor` 要到下面才赋值。没有兜底就会 AttributeError。
        self._m4_embedder: OllamaEmbeddingClient | None = None
        self._m4_text_builder: Any = None
        self._m4_record_text_builder: Any = None
        self._m4_template_version: str = TEMPLATE_VERSION_V1
        self._m4_online_embedding = False
        self._span_feature_vectors: dict[str, tuple[float, ...]] = {}
        self._m4_guard_threshold = DEFAULT_MEMORY_GUARD_THRESHOLD
        self._m1_online_embedding_count = 0
        self._m1_online_embedding_errors = 0
        self._m1_online_embedding_total_ms = 0.0
        self._m1_online_embedding_max_ms = 0.0
        # ⚠️ 顺序：嵌入链路先建（`_start_m4_memory` 用它建的文本构建器，
        # 下面的 l2_v4 校验块读 `_m4_template_version`）。
        self._start_m1_embedding(config)
        self._start_m4_memory(config)
        # shadow_route 的强制目标。**先给兜底值** —— `_make_router` 在构造尾部
        # 才被调用，但任何早于校验块的读取都应当看到「没开」而不是 AttributeError。
        self._shadow_force: str | None = None
        #: 第 ④ 段（`C_bare_w10`）**算不算**。见配置块里的说明。
        self._m1_force_actions_text = False
        #: 52 维 v4 预测器已移除；本仓库只保留 116 维 add4。
        # ---- add4 两阶段预测器（`runner.l2_add4`）----------------------------
        #: 116 维向量。**与 `_span_feature_vectors`（52 维）分开存** —— 两者是
        #: 不同的向量空间，共用一个槽会让 M4 守卫和别人读到的不是一回事，
        #: 而且不报错。见 `l2_add4_predictor` 的模块说明。
        self._add4_span_vectors: dict[str, tuple[float, ...]] = {}
        self._l2_add4_predictor: Any | None = None
        #: 出分次数 / 因为「116 维那组没算出来」而**直接走大模型**的次数。
        self._l2_add4_hits = 0
        self._l2_add4_miss_no_vector = 0
        #: 每次判定的两个 p（按调用序）。审计「全没卸载」时先看这两个分布。
        self._l2_add4_p_gate: list[float] = []
        self._l2_add4_p_net: list[float] = []
        #: 两个 τ。默认取 artifact 里的，配置可覆盖（阈值敏感性实验用）。
        self._l2_add4_tau_gate: float | None = None
        self._l2_add4_tau_net: float | None = None
        #: 调试：设了 `TRACEOFFLOAD_DUMP_EMBED_TEXT` 就把每个 span 的 embedding
        #: 输入文本追加到这个文件。默认关。
        _dump = os.environ.get("TRACEOFFLOAD_DUMP_EMBED_TEXT")
        self._dump_embed_text_path: str | None = _dump.strip() if _dump else None
        l2_v4_raw = config.get("runner", {}).get("l2_v4", {})
        # 52 维 v4 已移除。配了就明确报错，不要让旧配置静默跑成「没装预测器」。
        if isinstance(l2_v4_raw, dict) and bool(l2_v4_raw.get("enabled", False)):
            raise ValueError(
                "runner.l2_v4 已移除：本仓库只保留 116 维 add4 预测器。"
                "请改用 runner.l2_add4，或把 l2_v4.enabled 设为 false。"
            )

        # ---- add4 两阶段预测器：`runner.l2_add4` -----------------------------
        # 与 v4 **同一套路数**（一个 `runner:` 下的 key，不是新 mode），
        # 校验也放在同一个 `classifier_route` 门里，理由同上。
        l2_add4_raw = config.get("runner", {}).get("l2_add4", {})
        if (
            self._mode == "classifier_route"
            and isinstance(l2_add4_raw, dict)
            and bool(l2_add4_raw.get("enabled", False))
        ):
            add4_path = l2_add4_raw.get("model_path")
            if not isinstance(add4_path, str) or not add4_path.strip():
                raise ValueError("runner.l2_add4.enabled 为真但 model_path 为空")
            self._l2_add4_predictor = load_l2_add4_predictor(add4_path)
            if self._l2_add4_predictor is None:
                raise FileNotFoundError(
                    f"runner.l2_add4.model_path 指向的文件不存在: {add4_path}"
                )
            # 第 ③ 段（V2）的向量空间由 `template_version` 决定；第 ④ 段由
            # `SpanActionsTextBuilder` 决定，与 template_version **无关**。
            if self._m4_template_version != TEMPLATE_VERSION_V2:
                raise ValueError(
                    "runner.l2_add4 的第 ③ 段是 V2_goal_recent 的 PCA，"
                    "所以要求 `m4_memory.template_version: span-embed-v2`，"
                    f"现在是 {self._m4_template_version!r}。"
                )
            if not self._m1_embedding_online():
                raise ValueError(
                    "runner.l2_add4 的两段向量都来自 M1（`_observe_m4_profile`），"
                    "所以必须 `m4_memory.embedding_mode: m1_online`。"
                    "否则每次都取不到向量、一次都不会卸载。"
                )
            # τ 默认取 artifact 里的；配置给了就覆盖（必须都在 [0,1]）。
            self._l2_add4_tau_gate = self._l2_add4_predictor.tau_gate
            self._l2_add4_tau_net = self._l2_add4_predictor.tau_net
            for key, attr in (("tau_gate", "_l2_add4_tau_gate"),
                              ("tau_net", "_l2_add4_tau_net")):
                override = l2_add4_raw.get(key)
                if override is None:
                    continue
                if not isinstance(override, int | float) or isinstance(override, bool):
                    raise ValueError(f"runner.l2_add4.{key} 必须是数字")
                if not 0.0 <= float(override) <= 1.0:
                    raise ValueError(f"runner.l2_add4.{key}={override} 不在 [0,1]")
                setattr(self, attr, float(override))
        # ---- M1 第 ④ 段：**不装 add4 也照算** -------------------------------
        #
        # 第 ④ 段（`C_bare_w10`，最近 10 步的动作签名）本来是**为 add4 建的** ——
        # 只在 `_l2_add4_predictor` 在场时才嵌。要比较「原版 L2_new（52 维）」
        # 和「add4（116 维）」时，这会让两组**差一份嵌入开销**，于是分不清
        # 差别是预测器带来的还是嵌入成本带来的。
        #
        # 打开这个开关 = 那一段照算、算完**不交给任何人**，纯粹为了把两组的
        # 嵌入对齐。代价是多一次文本构建（`embed_many` 仍是一次 HTTP，
        # 两个文本一起发）。
        #
        # ⚠️ 默认 `false` —— 既有配置没写这个键，行为**逐字不变**。
        raw_m1 = config.get("runner", {}).get("m1_embedding", {})
        raw_m1 = raw_m1 if isinstance(raw_m1, dict) else {}
        self._m1_force_actions_text = bool(raw_m1.get("force_actions_text", False))
        # ---- shadow_route：预测器照常算，但决定不生效 ------------------------
        #
        # 用在「接入 TraceOffload 但**不生效**」这个对照组：预测器真的跑
        # （`classifier_p` / `classifier_p_gate` 照常有值），只是路由被强制到
        # 一个固定目标。两批的差别就只剩「决定生不生效」这一件事。
        #
        # ⚠️ **键在 `runner.` 下，不在任何一个预测器块里** —— 这是它可拔插的
        #    关键：换预测器（v4 ↔ add4）、换预测器的**数量**，这段都不用改。
        #
        # ⚠️ 校验放在**构造期**。留到 `decide()` 才炸的话，8 小时的跑批会死在
        #    半路；而且那时报的是个不相干的错。
        raw_shadow = config.get("runner", {}).get("shadow_route", {})
        raw_shadow = raw_shadow if isinstance(raw_shadow, dict) else {}
        if bool(raw_shadow.get("enabled", False)):
            force = raw_shadow.get("force_target")
            if force not in ("large", "small"):
                raise ValueError(
                    "runner.shadow_route.enabled 为真时必须给 force_target，"
                    f"且只能是 'large' 或 'small'，收到 {force!r}。"
                    "（写错会在执行期以一个不相干的错误冒出来。）"
                )
            self._shadow_force = str(force)
        self._pilot_id = config["pilot"]["id"]
        self._config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
        self._guard = GuardRef(**config["runner"]["guarded_fallback"]["verifier_guard"])
        raw_tool_policy = config.get("runner", {}).get("tool_policy", {})
        self._tool_policy = ToolPolicy.from_mapping(
            raw_tool_policy if isinstance(raw_tool_policy, dict) else {},
            default_allowed_tools=("bash",),
            default_require_tool_calls=False,
        )
        # 副作用分析器由接入层注入：AppWorld 用 integrations/appworld.py 里的
        # AppWorldCodeEffectAnalyzer（读 OpenAPI 的 HTTP method）。Runner 自己
        # 不再内置 bash 语法分析器 —— 那是 SWE-bench 线的东西，本仓库不包含。
        self._analyzer = None
        # 阶段 4 预测权重表：缺失/损坏/版本不符时 weights=None，
        # BucketRouter 自身 fail closed 到 direct（见 _make_router）。
        # 探针预算（A7）：每 workflow 受控探针上限。默认 1（文档 §2.2
        # “prototype 每 workflow ≤1 次”）；A/B 预热阶段可通过配置调高以加速
        # 桶画像积累。
        raw_probe_budget = config.get("runner", {}).get("probe_budget_per_workflow", 1)
        self._probe_budget_per_workflow = (
            int(raw_probe_budget)
            if isinstance(raw_probe_budget, int) and raw_probe_budget >= 0
            else 1
        )
        # 全实验探针上限（文档 §2.1：全实验 ≤10% 请求）。None = 只按
        # 每 workflow 上限控制；实验员按批次规模显式设置。
        raw_probe_total = config.get("runner", {}).get("probe_budget_total")
        self._probe_budget_total = (
            int(raw_probe_total)
            if isinstance(raw_probe_total, int) and raw_probe_total >= 0
            else None
        )
        self._route_weights = None
        weights_path = Path(
            str(config.get("runner", {}).get("route_weights", "configs/route_weights.yaml"))
        )
        try:
            
            self._route_weights = load_route_weights(str(weights_path))
        except Exception:
            pass
        # 权重表版本匹配（阶段 4 修复）：配置声明期望版本时，加载版本不符
        # 视为缺失 -> BucketRouter fail closed 到 direct。
        expected_weights_version = config.get("runner", {}).get("route_weights_version")
        if (
            self._route_weights is not None
            and isinstance(expected_weights_version, str)
            and self._route_weights.version != expected_weights_version
        ):
            self._route_weights = None

    def _make_history_stores(
        self,
        config: dict[str, Any],
    ) -> tuple[EventHistoryStore, SnapshotHistoryStore]:
        raw = config.get("runner", {}).get("history_store", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("runner.history_store must be a mapping")
        configured_modes = raw.get("modes", ("traceoffload",))
        if (
            not isinstance(configured_modes, (list, tuple))
            or not configured_modes
            or not all(isinstance(item, str) for item in configured_modes)
        ):
            raise ValueError("runner.history_store.modes must be a non-empty string list")
        # 未知模式**忽略**而不是报错：老配置里还写着已移除的 mode 名，
        # 直接拒绝会让一份能跑 llm_only 的配置因为无关字段而失败。
        configured_modes = tuple(m for m in configured_modes if m in PILOT_MODES)
        backend = raw.get("backend", "sqlite")
        if not isinstance(backend, str):
            raise ValueError("runner.history_store.backend must be a string")
        backend = backend.strip().lower()
        if backend not in {"memory", "sqlite"}:
            raise ValueError(
                "runner.history_store.backend must be either 'memory' or 'sqlite'"
            )
        if backend == "memory" or self._mode not in configured_modes:
            return EventStore(), SnapshotStore()

        configured_path = raw.get("path", "history.sqlite3")
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError("runner.history_store.path must be a non-empty path string")
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self._output_dir / self._mode / path
        self._history = SQLiteHistoryStore(path)
        return self._history, self._history

    def _start_m1_embedding(self, config: dict[str, Any]) -> None:
        """构造 **M1 的在线嵌入链路**：文本配方 + embedder + 52 维缓存 + 计数器。

        **与 M4 记忆库无关。** 这条链路有两个消费者：

        * M4 记忆 / 动作守卫（`m4_memory.enabled: true` 时）
        * v4 预测器（`l2_v4.enabled: true`，读 `_span_feature_vectors`）

        只要有一个在场，向量就得算 ⇒ **无条件构造**。

        ⚠️ **不能再用 `m4_memory.enabled` 挡住这里。** 那会让「关 M4」连带把预测器
        一起关掉 —— 而预测器会**静默退回旧 artifact**，跑得通、日志干净、数字变了。
        `tests/test_memory_guard_switch.py` 的 docstring 记的就是这个耦合。

        ⚠️ **顺序**：必须在 `_start_m4_memory`（用 `_m4_record_text_builder`）与
        l2_v4 校验块（读 `_m4_template_version`）**之前**调用。
        """

        raw = config.get("runner", {}).get("m4_memory", {})
        raw = raw if isinstance(raw, dict) else {}

        # 检索时排掉**本轮**自己的 Span（用 `self._experiment_id`）。
        # 为什么不能用 `workflow_run_id` 或 `record_id`：两者都是
        # `workflow-{task_id}` 派生的 —— 同一 task 重跑会撞在一起，
        # `INSERT OR REPLACE` 直接覆盖，库里永远只有那几条，而它们恰好全在
        # 「本轮已见」里 ⇒ 实测 n_neighbors 恒为 0，历史经验根本用不上。
        #
        # --- M2 决策路径上的检索（第二道门）----------------------------------
        # M1 算完向量后按 record_key 暂存，M2 决策时取用。
        # ⚠️ 决策路径**不给 M4 建索引**：`query()` 是同步的、持 `M4Memory._lock`，
        # 而写入在 worker 线程 —— 决策时只读已经落库的历史，不等当前这批 Span。
        #: span 的 **52 维表示**（= 2 前缀标量 + PCA50，已标准化），
        #: 键是 `_m4_record_key`。**两个消费者共用这一份**：
        #:   · `_l2_v4_lookup`（v4 预测器的输入）
        #:   · `_m4_action_guard_lookup`（M4 检索的查询向量）
        #:
        #: **1024 维原始嵌入不缓存** —— 它只在 `_observe_m4_profile` 里活一瞬间，
        #: 派生成这 52 维之后就丢了。所以这里存的和 M4 库里存的是同一个东西，
        #: 不存在「守卫拿 52 维比 1024 维」那种错配。
        self._span_feature_vectors: dict[str, tuple[float, ...]] = {}
        #: 动作守卫的相似度下限。查不到邻居 / 没开 M4 时一律放行（fail-open）。
        self._m4_guard_threshold = float(
            raw.get("memory_guard_threshold", DEFAULT_MEMORY_GUARD_THRESHOLD)
        )
        self._m4_embedder = OllamaEmbeddingClient(
            base_url=raw.get("ollama_base_url") or None,
            model=str(raw.get("embedding_model", "qwen3-embedding:0.6b")),
            timeout_seconds=float(raw.get("embedding_timeout_seconds", 30.0)),
        )
        mode = str(raw.get("embedding_mode", "worker")).strip().lower()
        self._m4_online_embedding = mode in {"m1_online", "online", "sync", "m1"}
        self._m1_online_embedding_count = 0
        self._m1_online_embedding_errors = 0
        self._m1_online_embedding_total_ms = 0.0
        self._m1_online_embedding_max_ms = 0.0
        # 文本配方：默认 v1（`[GOAL]` + (思考+输出+工具名/参数)×K）。
        # `template_version: span-embed-v0` 可退回旧配方（10 行 header + 窗口内
        # 全部 messages）。**v0 保留不改** —— 它是审计字段的历史值。
        template_version = str(
            raw.get("template_version", TEMPLATE_VERSION_V1)
        ).strip().lower()
        if template_version == "span-embed-v0":
            self._m4_text_builder: Any = SpanEmbeddingTextBuilder()
            self._m4_record_text_builder: Any = build_embedding_text_from_span_record
        elif template_version == TEMPLATE_VERSION_V2:
            # v2 = 离线 `aw3_build.span_text_v2` 的在线复刻（`[TOOLS]/[REASONING]/
            # [ASSISTANT]` + 窗口 10）。**v4 预测器（`runner.l2_v4`）要求这一段** ——
            # 它的 PCA/coef 是在 v2 文本的 embedding 上拟合的，换成 v1 向量空间
            # 就完全不同（实测两个配方 761 个 key **零字符相同**），而且**不报错**。
            self._m4_text_builder = SpanEmbeddingTextBuilderV2()
            self._m4_record_text_builder = build_embedding_text_from_span_record_v2
        else:
            self._m4_text_builder = SpanEmbeddingTextBuilderV1()
            self._m4_record_text_builder = build_embedding_text_from_span_record_v1
        self._m4_template_version = template_version
        # ---- 第 ④ 段（`C_bare_w10`，add4 预测器的第二次嵌入）------------------
        # **无条件构造**，理由与整条嵌入链路一样：add4 没配时它没人消费，
        # 但配了就必须在。它与上面的 `_m4_text_builder` **互不相干** ——
        # 不同的文本、不同的向量空间，只是碰巧都由 M1 的 profile 回调触发。
        self._add4_actions_text_builder = SpanActionsTextBuilder()
        #: 所有需要「跳过 AppWorld prompt 模板」的构建器。注入时统一遍历，
        #: **避免只记得接一个** —— 漏掉某个的话它的窗口会被 few-shot 示例填满，
        #: 而这是静默的（见 `set_skip_leading_messages` 的警告）。
        self._m1_text_builders: tuple[Any, ...] = (
            self._m4_text_builder,
            self._add4_actions_text_builder,
        )

    def _set_skip_leading_messages(self, count: int) -> None:
        """把「模板消息数」注入**所有**窗口型构建器。

        由 AppWorld 入口在 agent 就绪后调用（那个数只有 `initialize()` 之后才有）。
        集中成一处是为了让「新增一个构建器」时**没得漏**：只加进
        `_m1_text_builders` 就会自动被注入。
        """
        for builder in self._m1_text_builders:
            setter = getattr(builder, "set_skip_leading_messages", None)
            if callable(setter):
                setter(count)

    def _m1_embedding_online(self) -> bool:
        """嵌入链路**可不可用** —— 只看「建好了没 + 配成同步在线」，与有无消费者无关。

        用在两处：l2_v4 的配置校验、`_observe_m4_profile` 的内层守卫。
        这两处问的是「向量**算得出来**吗」，不是「有没有人要」。
        """
        return bool(self._m4_online_embedding) and self._m4_embedder is not None

    def _m1_embedding_needed(self) -> bool:
        """有没有**活消费者**要这份向量 —— 决定要不要为它付嵌入成本。

        只用在 `profile_observer` 的注入判断上。M4 记忆与任一预测器在场就要。
        """
        return (self._m4_memory_worker is not None
                or self._l2_add4_predictor is not None)

    def _start_m4_memory(self, config: dict[str, Any]) -> None:
        """按配置启动 M4 Memory worker；默认关闭，避免影响既有实验。

        ⚠️ **关掉这里不等于不算嵌入。** 嵌入链路由 `_start_m1_embedding` 无条件
        构造 —— 预测器（v4）也读那份向量。这里只建记忆库 / store / worker。
        """

        raw = config.get("runner", {}).get("m4_memory", {})
        raw = raw if isinstance(raw, dict) else {}
        if not bool(raw.get("enabled", False)):
            return
        class_count = int(raw.get("class_count", 5))
        dimension = int(raw.get("embedding_dimension", guard_vector_dimension()))
        if class_count < 1 or dimension < class_count:
            raise ValueError("runner.m4_memory class_count/dimension is invalid")
        configured_path = raw.get("path", "m4_memory.sqlite3")
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError("runner.m4_memory.path must be a non-empty string")
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self._output_dir / path
        self._m4_memory_store = SQLiteM4MemoryStore(path)
        # T2（spec §3.2）：**起跑时库里几条**。子类可能在 `super().__init__` 之前
        # 就已经建了计数器（守卫要在那时构造），所以用现成的那个。
        # 没有这个数，「seed 没加载上」和「守卫就是不触发」看起来一样。
        self._counters.set_value(
            "m4_records_at_start", self._m4_memory_store.count_records()
        )
        self._m4_memory = M4Memory(
            classes=placeholder_class_specs(
                class_count=class_count,
                dimension=dimension,
            ),
            max_records_per_class=int(raw.get("max_records_per_class", 200)),
            max_records_total=int(raw.get("max_records_total", 2_000)),
            top_k=int(raw.get("top_k", 5)),
            persistence=self._m4_memory_store,
        )
        self._m4_memory_worker = M4MemoryWorker(
            self._m4_memory,
            embedder=self._m4_embedder,
            text_builder=self._m4_record_text_builder,
            thread_name=f"m4-memory-{self._mode}",
        )
        self._m4_memory_worker.start()

    @staticmethod
    def _m4_record_key(workflow_run_id: str, logical_invocation_id: str) -> str:
        return f"{workflow_run_id}|{logical_invocation_id}"

    def _m4_action_guard_lookup(self, candidate: Any, identity: Any) -> Any:
        """动作守卫的查询入口 —— 注入给 `M4FailureMemoryVerifier`。

        **没开 M4 就返回空**（fail-open 放行），而不是抛异常：守卫是增强，
        不是依赖。M4 关着时这条路径的行为与纯形状检查完全一致。
        """

        memory = getattr(self, "_m4_memory", None)
        if memory is None:
            return ()
        return lookup_failed_actions(
            candidate,
            identity,
            guard_vectors=self._span_feature_vectors,
            memory=memory,
            # 键的格式只有 `_m4_record_key` 一处定义 —— 不在这里复刻，
            # 复刻就会漂移，而漂移的坏法是不报错（永远是「查不到」）。
            vector_key=lambda ident: self._m4_record_key(
                ident.workflow_run_id, ident.logical_invocation_id
            ),
            threshold=self._m4_guard_threshold,
            counters=self._counters,
        )

    def _observe_m4_profile(self, context, request, profile) -> None:
        """Coordinator profile_observer 回调：M1 后在线计算 embedding。

        **两个消费者共用算出来的那个向量**：M4 记忆/守卫（在场时入 pending），
        v4 预测器（填 `_span_feature_vectors`）。M4 关掉时**只剩预测器** ——
        所以这里的守卫是「链路可用吗」，不是「M4 开着吗」。
        """

        if not self._m1_embedding_online():
            return
        embedding_input = None
        try:
            embedding_input = self._m4_text_builder.build(context, request, profile)
            # 调试用：`TRACEOFFLOAD_DUMP_EMBED_TEXT=<路径>` 时把每个 span 的
            # **embedding 输入文本**落盘。用途是核对"在线产出的文本 == 离线训练用的
            # 文本" —— 只对模板不报错，所以必须能拿到原文比对。
            if self._dump_embed_text_path is not None:
                try:
                    with open(self._dump_embed_text_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "key": self._m4_record_key(
                                context.workflow_run_id, context.logical_invocation_id
                            ),
                            "template_version": self._m4_template_version,
                            "sha256": embedding_input.text_sha256,
                            "chars": len(embedding_input.text),
                            "text": embedding_input.text,
                        }, ensure_ascii=False) + "\n")
                except Exception:  # noqa: BLE001 - 调试钩子不能影响主链路
                    pass
            # add4 在场时要**第二段文本**（`C_bare_w10`）。两段一次发出去 ——
            # `OllamaEmbeddingClient.embed_many` 就是一次 HTTP 请求，不会付两次往返。
            #
            # ⚠️ `_m1_force_actions_text` 让它**不装 add4 也算**：这是给
            #「比较两个预测器」用的 —— 两组的嵌入开销必须一模一样，否则差的
            # 那一份嵌入会和预测器本身的差别混在一起。算完**不交给任何人**。
            actions_input = (
                self._add4_actions_text_builder.build(context, request, profile)
                if (self._l2_add4_predictor is not None or self._m1_force_actions_text)
                else None
            )
            # ⚠️ 必须用 `real_time()`，见 `traceoffload/clock.py`：这条回调在
            # world 存续期内跑，用 `perf_counter` 的话 `latency_ms` 恒为 0.0
            # （线上实测过，见 D10）。
            started = real_time()
            actions_result = None
            if actions_input is None or not actions_input.text.strip():
                # 第 ④ 段为空（`k=1` 时前缀里一个动作都没有）⇒ **零向量，不发请求**。
                # 离线就是这么做的（`sf_actions_variants` 的 embed 阶段对该情形
                # `np.zeros((n,1024))` 且跳过 HTTP）。**发空串给 Ollama 不是一回事**
                # —— 那是"空文本的嵌入"，与零向量差得很远，而且不报错。
                result = self._m4_embedder.embed_one(embedding_input.text)
            else:
                both = self._m4_embedder.embed_many(
                    (embedding_input.text, actions_input.text)
                )
                result, actions_result = both[0], both[1]
            latency_ms = (real_time() - started) * 1000.0
            key = self._m4_record_key(
                context.workflow_run_id, context.logical_invocation_id
            )
            # M4 关掉时没有 worker，这段跳过 —— 但下面的派生与缓存照做，
            # 因为那正是预测器要的。
            if self._m4_memory_worker is not None:
                self._m4_memory_worker.register_pending_embedding(
                    key,
                    result.values,
                    latency_ms=latency_ms,
                )
            # ⚠️ 缓存现在只服务 v4 预测器（经济门已删除）。
            # 仍按「任一需要就填」写：将来再加消费者时不必回头改这里。
            # **1024 维在这里变成 52 维，然后丢掉** —— 之后全链路（L2_v4、
            # 守卫、M4 库）看到的都是这同一个 52 维。
            #
            # 无条件做：两个消费者（v4 预测器、动作守卫）任缺一个都会让行为
            # 静默退化，而成本只有 50×1024 次乘加。
            span_features = guard_vector(
                result.values, getattr(profile, "online_features", None) or {}
            )
            if span_features is not None:
                # **有界**：超了淘汰最早的（正常一条轨迹也就几十步，
                # 这个上限只是防长跑进程泄漏）。
                if len(self._span_feature_vectors) >= 4096:
                    self._span_feature_vectors.pop(
                        next(iter(self._span_feature_vectors))
                    )
                self._span_feature_vectors[key] = span_features
            # add4 的 116 维**另存一个槽**。两段 1024 维在这里用完即弃。
            self._cache_add4_vector(key, result.values, actions_result, profile)
            self._m1_online_embedding_count += 1
            self._m1_online_embedding_total_ms += latency_ms
            self._m1_online_embedding_max_ms = max(
                self._m1_online_embedding_max_ms, latency_ms
            )
        except Exception:
            self._m1_online_embedding_errors += 1
            # ⚠️ 这里**必须**判 worker 是否为 None。M4 关掉时 worker 是 None，
            # 而本回调的异常会穿过 `coordinator.py` 的回调点直接进请求路径 ——
            # 少了这个判断，一次嵌入失败就会二次抛 AttributeError，把
            # 「fail-soft 记录一次错误」变成「整个 workflow 炸掉」。
            if embedding_input is not None and self._m4_memory_worker is not None:
                key = self._m4_record_key(
                    context.workflow_run_id, context.logical_invocation_id
                )
                self._m4_memory_worker.register_pending_text(key, embedding_input.text)

    def _cache_add4_vector(self, key, v2_values, actions_result, profile) -> None:
        """把 add4 的 **116 维**算出来存进 `_add4_span_vectors`。

        `actions_result` 为 None ⇒ 第 ④ 段文本是空的 ⇒ 用**零向量**
        （与离线 `sf_actions_variants` 的 embed 阶段一致）。
        """
        predictor = self._l2_add4_predictor
        if predictor is None:
            return
        features = getattr(profile, "online_features", None) or {}
        missing = [n for n in predictor.prefix_features if n not in features]
        if missing:
            # 这里**不抛** —— 与 `guard_vector` 的 fail-soft 一致：缺列时只是
            # 「这一刻没有向量」，由 `_l2_add4_lookup` 抛（那才是配置错该响的地方）。
            # 反过来写的话，一次 profiler 抖动会让整条轨迹崩。
            return
        net_values = (
            actions_result.values
            if actions_result is not None
            else tuple([0.0] * predictor.net_embedding_dim)
        )
        vector = predictor.features_from(
            gate_embedding=v2_values,
            net_embedding=net_values,
            prefix_values=[float(features[name]) for name in predictor.prefix_features],
        )
        # **有界**，与 `_span_feature_vectors` 同策略。
        if len(self._add4_span_vectors) >= 4096:
            self._add4_span_vectors.pop(next(iter(self._add4_span_vectors)))
        self._add4_span_vectors[key] = vector

    def _l2_add4_lookup(self, profile: RuntimeProfile) -> Any | None:
        """add4 预测器：116 维 → `(p_gate, p_net)` → 两阶段判定。

        **不重算任何嵌入** —— 读 `_cache_add4_vector` 存的 116 维。

        三种结局，与 `_l2_v4_lookup` 同构：
        - `online_features` **缺列** → **抛**（配置错）
        - **没有向量** → 计数 + 返回 None ⇒ router 直接走大模型
        - 有向量 → 返回 `Add4Decision`
        """
        if self._l2_add4_predictor is None:
            return None
        # ⚠️ **缺列检查必须排在「没有向量」之前**，理由同 v4：否则配置错会被
        # 静默降级成「暂时没向量」。
        features = profile.online_features or {}
        missing = [
            name for name in self._l2_add4_predictor.prefix_features
            if name not in features
        ]
        if missing:
            raise ValueError(
                f"add4 预测器需要 online_features 里有 {missing}，"
                f"实际只有 {sorted(features)}。请检查 profiler 的特征构建是否按 "
                f"`prefix_features` 走。"
            )
        key = self._m4_record_key(profile.workflow_run_id, profile.logical_invocation_id)
        vector = self._add4_span_vectors.get(key)
        if vector is None:
            self._l2_add4_miss_no_vector += 1
            return None
        self._l2_add4_hits += 1
        decision = self._l2_add4_predictor.decide_from_span_features(
            vector,
            tau_gate=self._l2_add4_tau_gate,
            tau_net=self._l2_add4_tau_net,
        )
        self._l2_add4_p_gate.append(round(float(decision.p_gate), 6))
        self._l2_add4_p_net.append(round(float(decision.p_net), 6))
        return decision

    def l2_add4_stats(self) -> dict[str, object] | None:
        """add4 预测器的使用统计。没配就 None。

        **两个 p 都要留** —— 「全没卸载」可能是门控挡的，也可能是经济挡的，
        只看一个 `p` 分不出来。`miss_no_vector > 0` 说明有 span 因为
        116 维没算出来而直接走了大模型。
        """
        if self._l2_add4_predictor is None:
            return None
        return {
            "hits": self._l2_add4_hits,
            "miss_no_vector": self._l2_add4_miss_no_vector,
            "input_dim": self._l2_add4_predictor.input_dim,
            "model_type": self._l2_add4_predictor.gate_model_type,
            "gate_segment": self._l2_add4_predictor.gate_segment,
            "net_segment": self._l2_add4_predictor.net_segment,
            "tau_gate": self._l2_add4_tau_gate,
            "tau_net": self._l2_add4_tau_net,
            "p_gate_values": list(self._l2_add4_p_gate),
            "p_net_values": list(self._l2_add4_p_net),
        }

    def mechanism_stats(self) -> dict[str, object]:
        """T2 机制计数（spec §3.2）。**不是可选的** —— 没有它，
        摘掉机制的臂和满血臂在报告里长得一模一样。

        收尾时把 `m4_records_at_end` 也填上（它要的是「跑完之后几条」，
        只能在这时候读）。`m4_records_at_start` 在 `_start_m4_memory` 里填。
        """
        if self._m4_memory_store is not None:
            self._counters.set_value(
                "m4_records_at_end", self._m4_memory_store.count_records()
            )
        # M4 worker 自己已经在数 `stored` / `skipped_success` —— 提到扁平字段里，
        # 免得汇总脚本要去嵌套 dict 里刨。`m4_records_stored == 0` 正是
        # 「关掉 M4 写入」那条消融臂的自洽性断言（spec §3.2）。
        if self._m4_memory_worker is not None:
            worker = self._m4_memory_worker.stats()
            self._counters.set_value("m4_records_stored", int(worker.get("stored", 0)))
            self._counters.set_value(
                "m4_skipped_success", int(worker.get("skipped_success", 0))
            )
        return self._counters.snapshot()

    def m4_memory_stats(self) -> dict[str, object] | None:
        """Return online M1 embedding stats + M4 worker stats for run summaries.

        ⚠️ **M4 关掉时也必须报嵌入那几栏。** 计数器是**无条件**累加的（在
        `_observe_m4_profile` 里，不看 worker 在不在），而预测器照样要付这份
        嵌入成本。这里曾经在 worker 缺席时直接 `return None`，把它们一起吞掉
        —— 实测：summary 里 `m4_memory` 是 `{}`，而那次嵌入跑了 16 次、
        花了 6.9 秒。量「框架自身开销」的实验会因此得出「框架不花时间」的
        假结论，而那个结论看起来完全正常。

        嵌入链路**本身没开**（没预测器、M4 也关）时才返回 None —— 那种情况下
        返回一份恒为 0 的块会让「没实现」和「没触发」在汇总里分不清
        （见 `mechanism_counters` 的 docstring）。
        """
        if not self._m1_embedding_online():
            return None
        count = self._m1_online_embedding_count
        stats: dict[str, object] = {
            "m1_online_embedding_count": count,
            "m1_online_embedding_errors": self._m1_online_embedding_errors,
            "m1_online_embedding_total_ms": round(self._m1_online_embedding_total_ms, 3),
            "m1_online_embedding_avg_ms": (
                round(self._m1_online_embedding_total_ms / count, 3)
                if count
                else None
            ),
            "m1_online_embedding_max_ms": round(self._m1_online_embedding_max_ms, 3),
        }
        # **没有 worker 时不许编一个出来** —— 汇总脚本靠这个键判断
        # 「M4 在不在」，塞个空 dict 会让两批看起来一样。
        if self._m4_memory_worker is not None:
            stats["worker"] = self._m4_memory_worker.stats()
        return stats

    def _observe_m4_span(self, record, response) -> None:
        """Coordinator span_observer 回调：O(1) enqueue，不在主链路做 embedding。

        这里顺手把**失败动作签名**连出来 —— 参数在 ``response`` 上、成败在
        ``record.execution_summary.tool_executions`` 上，靠 ``tool_call_id`` 对齐。
        两处都不常在同一个对象里，只有这个回调同时握着它们。

        ``failed_action_signatures`` 自己吞掉「对不上号」的情形（返回空集），
        所以这里不用再防一遍。
        """

        if self._m4_memory_worker is None:
            return
        key = self._m4_record_key(
            record.route_plan.workflow_run_id,
            record.route_plan.logical_invocation_id,
        )
        self._m4_memory_worker.enqueue_completed_span(
            key,
            record,
            failed_actions=failed_action_signatures(
                _candidate_response_for_m4(record, response),
                getattr(
                    getattr(record, "execution_summary", None),
                    "tool_executions",
                    (),
                )
                or (),
            ),
            template_version=getattr(self, "_m4_template_version", "unknown"),
            experiment_id=str(self._experiment_id or "unknown"),
        )

    def _route_policy(self, config: dict[str, Any]) -> RoutePolicy:
        policy = config["runner"]["policy"]
        routing = config.get("runner", {}).get("routing", {})
        routing = routing if isinstance(routing, dict) else {}
        utility_raw = dict(policy["utility_weights"])
        configured_weights = routing.get("utility_weights", {})
        if isinstance(configured_weights, dict):
            utility_raw.update(
                {
                    key: value
                    for key, value in configured_weights.items()
                    if key in {"success", "cost", "latency", "risk", "fallback", "overhead"}
                }
            )
        admission = routing.get("admission", {})
        evidence = routing.get("evidence", {})
        exploration = routing.get("exploration_constraints", {})
        return RoutePolicy(
            policy_version=str(
                routing.get("policy_version", f"policy-{config['pilot']['id']}")
            ),
            correctness_lower_bound=float(policy["correctness_lower_bound"]),
            risk_upper_bound=float(policy["risk_upper_bound"]),
            budget={
                "max_cost": float(policy["budget"]["max_cost"]),
                "max_latency_ms": float(policy["budget"]["max_latency_ms"]),
            },
            utility_weights={
                key: float(value) for key, value in utility_raw.items()
            },
            exploration_constraints=dict(exploration) if isinstance(exploration, dict) else {},
            conservative_prior=dict(policy["conservative_prior"]),
            min_utility_gain=float(routing.get("min_utility_gain", 0.0)),
            admission_constraints=dict(admission) if isinstance(admission, dict) else {},
            evidence_constraints=dict(evidence) if isinstance(evidence, dict) else {},
        )

    def _snapshot_provider(self) -> FeedbackSnapshot | None:
        if not self._snapshots.versions():
            return None
        snapshot = self._snapshots.latest()
        # A snapshot produced under another calibration or contract schema is
        # historical audit data, not admissible evidence for this run.  Fail
        # closed so a calibration change cannot silently reuse old routing
        # confidence after a process restart.
        if snapshot.safety_calibration_ref.version != self._calibration_version:
            return None
        if snapshot.schema_version != "1.0":
            return None
        return snapshot

    def _make_profiler(self, allowed_targets: AllowedTargets):
        """M1：基础 profiler，classifier_route 下再包一层在线特征构建。

        要建哪两列由 artifact 的 prefix_features 决定（顺序也要对，因为是按名字
        取值再按该顺序拼进 116 维）。
        """
        base = RuntimeProfiler(allowed_targets=allowed_targets)
        if self._mode != "classifier_route":
            return base
        if self._l2_add4_predictor is not None:
            return ClassifierFeatureProfiler(
                base,
                AppWorldOnlineFeatureBuilder(
                    self._l2_add4_predictor.prefix_features,
                    feature_version="l2-add4-two-stage",
                ),
            )
        return base


    def _make_router(self) -> Router:
        """按 mode 造 router；`shadow_route` 开着就**包一层**。

        装饰放在这里（而不是某个 mode 的分支里）是为了**与 mode 无关**：
        `classifier_route` / `random_route` / 任何 router 都能被同一个开关包住。
        换预测器、换预测器的数量，这个函数一行都不用改。
        """
        router = self._build_router()
        if self._shadow_force is not None:
            router = _ShadowRouter(
                router, self._large, self._small, force=self._shadow_force
            )
        return router

    def _build_router(self) -> Router:
        mode = self._mode
        if mode == "llm_only":
            return _AlwaysLargeRouter(self._large, self._small)
        if mode == "slm_only":
            return _AlwaysSmallRouter(self._large, self._small)
        if mode == "random_route":
            raw = self._config.get("runner", {}).get("random_route", {})
            if not isinstance(raw, Mapping):
                raise ValueError("runner.random_route must be a mapping")
            return _RandomRouter(
                self._large,
                self._small,
                probability_large=float(raw.get("probability_large", 0.5)),
                seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            )
        if mode == "classifier_route":
            if self._l2_add4_predictor is None:
                raise ValueError(
                    "classifier_route 需要 runner.l2_add4.enabled: true —— "
                    "本仓库只保留 116 维两阶段预测器。"
                )
            return _ClassifierRouter(
                self._large,
                self._small,
                lookup=self._l2_add4_lookup,
                guard=self._guard,
            )
        raise ValueError(f"unsupported mode: {mode}")


    def _workflow_run_id(self, key: str) -> str:
        """`workflow-{experiment_id}-{key}` —— **运行唯一**的 workflow 标识。

        为什么不沿用 `workflow-{task_id}`：同一 task 重跑会得到**完全相同**的
        id，而 M4 的**所有** join 第一分量都是它 ——
          · 结局扇出（`M4Memory.apply_outcome` 的 WORKFLOW_FINAL 分支）
          · 成本回填（`apply_observed_cost`）
          · 待定结局（`_find_record_by_lineage` / `_apply_stored_outcomes`）
        ⇒ 第 2 轮的结局会**扇到第 1 轮的记录上**。静默，只在两轮结局不同时显形。

        并进运行标识后 `workflow_run_id` 本身唯一 ⇒ 三处 join 同时修好；
        `record_id = f"{workflow_run_id}|{invocation}"` 是它的派生，自动变 run-scoped。

        ⚠️ **`experiment_id` 不能当运行标识用** —— 实测它在 AppWorld 路径上恒等于
        配置里的 `pilot.id`（`AppWorldPilotRunner.__init__` 把它预置了，
        于是 `_resolve_experiment_id` 里那个「生成 uuid」的分支**从来没触发过**）。
        三个独立进程跑同一个 task，`experiment_id` 完全相同。
        所以运行唯一性必须来自**每进程一个的 nonce**。

        实测：全仓 **零处反解**（`removeprefix` / `split` / `startswith("workflow-")`
        均无命中），所以改格式不会踩到解析。
        长度 `workflow-{pilot_id}-{8hex}-{key}` ≈ 49，`Identifier` 上限 128 ✓。
        """
        return f"workflow-{self._experiment_id or 'unknown'}-{self._run_nonce}-{key}"

    def _resolve_experiment_id(self, *, instance_keys: Sequence[str] = ()) -> str:
        """解析并缓存 experiment_id（首次调用生成，进程内稳定共享）。

        有实例键时用“实例集哈希 + 随机 nonce”标识本次实验（重跑同实例集也换新 id，
        从而冷启动重启不污染旧预算）；无实例键（直接 run_workflow）退化为随机 id。
        """
        if self._experiment_id is None:
            if instance_keys:
                digest = hashlib.sha256(
                    json.dumps(sorted(instance_keys), default=str).encode()
                ).hexdigest()[:12]
                self._experiment_id = f"exp-{digest}-{uuid4().hex[:8]}"
            else:
                self._experiment_id = f"exp-{uuid4().hex[:16]}"
        return self._experiment_id

    def wait_for_m4(self, *, timeout_seconds: float = 30.0) -> bool:
        """Explicit synchronization point for shutdown/tests, never routing."""
        if self._m4_worker is None:
            if self._m4_memory_worker is not None:
                return self._m4_memory_worker.drain(
                    timeout_seconds=timeout_seconds
                )
            return True
        job_ids = tuple(
            job_id
            for workflow in self._workflow_rows.values()
            if isinstance((job_id := workflow.get("m4_job_id")), str)
        )
        completed = self._m4_worker.drain(
            timeout_seconds=timeout_seconds,
            job_ids=job_ids,
        )
        if self._m4_memory_worker is not None:
            completed = (
                self._m4_memory_worker.drain(timeout_seconds=timeout_seconds)
                and completed
            )
        self._refresh_m4_rows(tuple(self._workflow_rows.values()))
        return completed

    def close(self, *, drain: bool = False, timeout_seconds: float = 5.0) -> bool:
        """Stop local workers; unfinished durable jobs remain in their stores."""

        completed = True
        if drain and self._m4_worker is not None:
            completed = self.wait_for_m4(timeout_seconds=timeout_seconds)
        if self._m4_memory_worker is not None:
            if drain:
                completed = (
                    self._m4_memory_worker.drain(timeout_seconds=timeout_seconds)
                    and completed
                )
            self._m4_memory_worker.stop(timeout_seconds=timeout_seconds)
        if self._m4_worker is not None:
            self._m4_worker.stop(timeout_seconds=timeout_seconds)
        if self._m4_memory_store is not None:
            self._m4_memory_store.close()
        return completed
