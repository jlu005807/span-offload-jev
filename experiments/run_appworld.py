#!/usr/bin/env python3
"""Run AppWorld Simplified agents through TraceOffload.

This entry point never edits ``appworld/experiments``.  It constructs the
native Simplified Agent, wraps its ``language_model.generate`` and
``world.batch_execute`` boundaries, and runs one TraceOffload workflow per
AppWorld ``task_id``.

Model configuration is intentionally read from the same YAML as
``configs/spanoffload-qwen-local-full.yaml``: Qwen3.8-27B (large) and
Qwen3.5-9B (small), with fixed seed/temperature/top_p (top_k intentionally omitted).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import traceback
from functools import wraps
from datetime import timedelta
from pathlib import Path
from time import perf_counter as _perf_counter
from types import MethodType
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from traceoffload.clock import real_time, real_utcnow  # noqa: E402
from traceoffload.config import load_env_file  # noqa: E402
from traceoffload.contracts import (  # noqa: E402
    AllowedTargets,
    Attribution,
    Component,
    DownstreamDependency,
    EffectRisk,
    EffectScope,
    GuardRef,
    ModelTarget,
    ModelTargetKind,
    OutcomeEvent,
    OutcomeLabel,
    OutcomeScope,
    ProducerRef,
    RecoveryLevel,
    StateAccessMode,
    WorkflowGoal,
)
from traceoffload.fakes import FakeEvaluator  # noqa: E402
from traceoffload.m3.guards import M4FailureMemoryVerifier, ToolPolicy  # noqa: E402
from traceoffload.mechanism_counters import (  # noqa: E402
    MechanismCounters,
    delta_of,
)
from traceoffload.integrations.appworld import (  # noqa: E402
    BATCH_TOOL_NAME,
    AppWorldCodeEffectAnalyzer,
    AppWorldLanguageModelGateway,
    AppWorldLanguageModelProxy,
    AppWorldToolProxy,
    AppWorldTraceOffloadRuntime,
)
from traceoffload.runner import PILOT_MODES, PilotRunner, _decode_parameters  # noqa: E402

_AGENT_TYPES = ("react_code", "function_calling", "full_code")
def _timedelta_milliseconds(milliseconds: int) -> Any:
    """`timedelta(milliseconds=...)` —— 单独包一层，避免在 freezegun 的
    补丁作用域里构造（那正是我们要绕开的东西）。"""

    return _TIMEDELTA(milliseconds=milliseconds)


#: 在模块导入时绑好，早于 AppWorld 的任何 time-freeze。
_TIMEDELTA = timedelta

_FIXED_SAMPLING_KEYS = ("temperature", "top_p", "seed")


# ---------------------------------------------------------------------------
# Config preparation
# ---------------------------------------------------------------------------


def _runner_defaults() -> dict[str, Any]:
    return {
        "calibration_version": "appworld-qwen-local-v1",
        "history_store": {
            "backend": "memory",
            "modes": [
                "llm_only",
                "slm_only",
                "random_route",
                "cyclic_route",
                "guarded_fallback",
                "traceoffload",
            ],
        },
        "policy": {
            "correctness_lower_bound": 0.0,
            "risk_upper_bound": 1.0,
            "budget": {"max_cost": 100.0, "max_latency_ms": 600000.0},
            "utility_weights": {
                "success": 1.0,
                "cost": 0.1,
                "latency": 0.01,
                "risk": 1.0,
                "fallback": 0.5,
                "overhead": 0.1,
            },
            "conservative_prior": {},
        },
        "routing": {
            "policy_version": "appworld-policy-v1",
            "utility_weights": {},
            "admission": {},
            "evidence": {},
            "exploration_constraints": {"enabled": False},
        },
        "guarded_fallback": {
            "verifier_guard": {
                "kind": "guard",
                "id": "appworld-basic-verifier",
                "version": "v1",
            }
        },
        "tool_policy": {
            "allowed_tools": [BATCH_TOOL_NAME],
            "require_tool_calls": False,
            "tools": {
                BATCH_TOOL_NAME: {
                    "required_argument_paths": ["codes"],
                    "allowed_argument_keys": ["codes"],
                    "max_argument_bytes": 2_000_000,
                    "result": {
                        "accepted_statuses": ["success"],
                        "accepted_returncodes": [0],
                    },
                }
            },
        },
        "static_route": {"threshold_tokens": 30000},
        "random_route": {"probability_large": 0.5, "seed": 42},
        "cyclic_route": {"pattern": ["large", "large", "small"]},
        "classifier": {
            "feature_builder": "appworld",
            # L2_new 预测器（`classfierTrain/04_预测器训练/l2_predictor.py` 训练，`classfierTrain/06_接入TraceOffload/export_l2_artifact.py` 打包）。
            # 两列：`prefix_n_A` + `log_est_prompt_tokens`；在线特征由
            # `AppWorldOnlineFeatureBuilder` 提供，`prefix_n_A` 取自
            # `context.prior_execution_facts`（结构化，不解析文本）。
            #
            # 旧的 14 维 demo artifact 仍在
            # `classfierTrain/out/appworld_classifier/model/classifier.json`，
            # 可在 yaml 的 `runner.classifier.model_path` 里覆盖回来
            # （第 150-156 行的按 key 合并）。
            "model_path": str(
                REPO_ROOT / "classfierTrain" / "out" / "router" / "l2_new_aw" / "classifier.json"
            ),
            # τ **占位**：artifact 里的 threshold 只是占位值，部署时应按目标集现算后
            # 在这里覆盖（开发 0.589 / 测试 0.875，差 0.29 —— 见待办 §4.3 坑 1）。
            "threshold": 0.5,
        },
        "confidence_z": 1.96,
        "probe_budget_per_workflow": 0,
        "command_effects_kb": "configs/command_effects.yaml",
    }


def _prepare_runner_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    config.setdefault("pilot", {})
    config.setdefault("agent", {})
    config.setdefault("models", {})
    config.setdefault("runner", {})
    for key, value in _runner_defaults().items():
        if key not in config["runner"]:
            config["runner"][key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(config["runner"][key], dict):
            merged = copy.deepcopy(value)
            merged.update(copy.deepcopy(config["runner"][key]))
            config["runner"][key] = merged
    config["agent"].setdefault("step_limit", raw.get("agent", {}).get("step_limit", 40))
    config["agent"].setdefault("cost_limit", raw.get("agent", {}).get("cost_limit", 100.0))
    config["pilot"].setdefault("id", raw.get("pilot", {}).get("id", "appworld-traceoffload"))
    if "decoding" not in config["models"]:
        large_kwargs = config["models"].get("large", {}).get("request_kwargs", {}) or {}
        config["models"]["decoding"] = {
            "temperature": large_kwargs.get("temperature", 0.0),
            "top_p": large_kwargs.get("top_p", 1.0),
            "seed": large_kwargs.get("seed", 42),
            "max_output_tokens": large_kwargs.get("max_tokens", 8192),
        }
    return config


def _fixed_sampling(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("large", "small"):
        model = raw.get("models", {}).get(kind)
        if not isinstance(model, dict):
            raise ValueError(f"models.{kind} is missing")
        kwargs = model.get("request_kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"models.{kind}.request_kwargs must be a mapping")
        missing = [key for key in _FIXED_SAMPLING_KEYS if key not in kwargs]
        if missing:
            raise ValueError(
                f"models.{kind}.request_kwargs must pin {', '.join(missing)}"
            )
        result[kind] = {key: kwargs[key] for key in _FIXED_SAMPLING_KEYS}
    if result["large"] != result["small"]:
        raise ValueError(
            "large and small models must use identical seed/temperature/top_p"
        )
    return result


def _build_language_model_config(raw_model: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    model_name = raw_model.get("served_model_name") or raw_model.get("model_id")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model config requires served_model_name or model_id")
    request_kwargs = raw_model.get("request_kwargs", {}) or {}
    config: dict[str, Any] = {
        "name": model_name,
        "client_name": "openai",
        "api_type": "chat_completions",
        "base_url": str(raw_model["api_base"]),
        "api_key": str(raw_model.get("api_key", "dummy")),
        "temperature": float(fixed["temperature"]),
        "top_p": float(fixed["top_p"]),
        "seed": int(fixed["seed"]),
        "max_tokens": int(request_kwargs.get("max_tokens", 8192)),
        "drop_reasoning_content": False,
        "use_cache": False,
        "retry_after_n_seconds": 5,
        "max_retries": 10,
        "cost_per_token": {
            "input_cache_miss": 0.0,
            "input_cache_hit": 0.0,
            "input_cache_write": 0.0,
            "output": 0.0,
        },
    }
    if isinstance(request_kwargs.get("parallel_tool_calls"), bool):
        config["parallel_tool_calls"] = request_kwargs["parallel_tool_calls"]
    return config


def _ensure_appworld_agents_importable() -> None:
    try:
        __import__("appworld_agents")
        return
    except ImportError:
        pass
    experiments_root = REPO_ROOT / "appworld" / "experiments"
    if not experiments_root.is_dir():
        raise RuntimeError(
            "Cannot import appworld_agents and source experiments directory was not found: "
            f"{experiments_root}"
        )
    shim_root = Path(tempfile.mkdtemp(prefix="traceoffload-appworld-agents-"))
    link = shim_root / "appworld_agents"
    if not link.exists():
        link.symlink_to(experiments_root, target_is_directory=True)
    if str(shim_root) not in sys.path:
        sys.path.insert(0, str(shim_root))
    __import__("appworld_agents")


def _task_instruction(task_id: str) -> str:
    """AppWorld 任务的**真实指令**（公开 API `appworld.task.Task.load`）。

    为什么必须读真的：原来两处都写 `WorkflowGoal(goal_text=f"AppWorld task {task_id}")`
    —— 那只是一个 task_id。而这个值就是 embedding 文本 `[GOAL]` 段的唯一来源
    （`span_embedding._goal_text` 读 `context.workflow_context.workflow_goal.goal_text`）。
    结果：M4 的向量里**从来没有过任务指令**，只有 "AppWorld task 024c982_1"。

    真值在 `<APPWORLD_ROOT>/data/tasks/<id>/specs.json` 的 `instruction` 字段，
    `Task.load` 是 AppWorld 自己的公开读取入口（不碰私有 API、不改它的源码）。
    读不到时返回空串 —— `[GOAL]` 会退化成 `none`，**不猜**。
    """
    try:
        src = REPO_ROOT / "appworld" / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from appworld.task import Task  # noqa: PLC0415

        return str(getattr(Task.load(task_id), "instruction", "") or "")
    except Exception:  # noqa: BLE001 - 读不到就当没有，不阻断任务
        return ""


def _workflow_goal(task_id: str) -> WorkflowGoal:
    """构造带真实指令的 `WorkflowGoal`；读不到则退化成 task_id 文案。"""
    instruction = _task_instruction(task_id)
    return WorkflowGoal(goal_text=instruction or f"AppWorld task {task_id}")


def _prompt_path(relative: str) -> str:
    path = REPO_ROOT / "appworld" / "experiments" / "prompts" / relative
    if not path.is_file():
        raise FileNotFoundError(f"AppWorld prompt file not found: {path}")
    return str(path)


def _build_agent_config(raw: dict[str, Any], agent_type: str, large_model_config: dict[str, Any]) -> dict[str, Any]:
    agent_cfg = raw.get("agent", {})
    step_limit = int(agent_cfg.get("step_limit", 40))
    cost_limit = float(agent_cfg.get("cost_limit", 100.0))
    appworld_config: dict[str, Any] = {
        "random_seed": int(_fixed_sampling(raw)["large"]["seed"]),
        "raise_on_extra_parameters": True,
    }
    logger_config = {"color": False, "verbose": False}
    usage_tracker_config = {
        "max_cost_overall": cost_limit,
        "max_cost_per_task": cost_limit,
    }
    if agent_type == "react_code":
        return {
            "type": "simplified_react_code_agent",
            "model_config": copy.deepcopy(large_model_config),
            "appworld_config": appworld_config,
            "logger_config": logger_config,
            "usage_tracker_config": usage_tracker_config,
            "prompt_file_path": _prompt_path("react_code_agent/instructions.txt"),
            "ignore_multiple_calls": True,
            "max_prompt_length": None,
            "max_output_length": None,
            "max_steps": step_limit,
            "log_lm_calls": False,
        }
    api_predictor_config = {
        "mode": "predicted",
        "model_config": copy.deepcopy(large_model_config),
        "prompt_file_path": _prompt_path("api_predictor.txt"),
        "demo_task_ids": [],
        "max_predicted_apis": 20,
    }
    if agent_type == "function_calling":
        appworld_config = {
            **appworld_config,
            "include_direct_functions": True,
            "direct_function_separator": "__",
        }
        return {
            "type": "simplified_function_calling",
            "model_config": copy.deepcopy(large_model_config),
            "api_predictor_config": api_predictor_config,
            "appworld_config": appworld_config,
            "logger_config": logger_config,
            "usage_tracker_config": usage_tracker_config,
            "prompt_file_path": _prompt_path("function_calling_agent/instructions.txt"),
            "demo_messages_file_path": _prompt_path("function_calling_agent/demos.json"),
            "max_steps": step_limit,
            "log_lm_calls": False,
        }
    if agent_type == "full_code":
        return {
            "type": "simplified_full_code_agent",
            "model_config": copy.deepcopy(large_model_config),
            "api_predictor_config": api_predictor_config,
            "appworld_config": appworld_config,
            "logger_config": logger_config,
            "usage_tracker_config": usage_tracker_config,
            "code_prompt_file_path": _prompt_path("full_code_agent/full_code_instructions.txt"),
            "retrial_prompt_file_path": _prompt_path("full_code_agent/reflexion_instructions.txt"),
            "demo_task_ids": [],
            "remove_code_demo_comments": True,
            "max_num_retrials": 5,
            "compress_api_docs": True,
            "max_steps": step_limit,
            "log_lm_calls": False,
        }
    raise ValueError(f"unsupported agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _GatewayMeter:
    """Observational proxy over ``AppWorldLanguageModelGateway``.

    Every TraceOffload model attempt -- direct, fallback, replay or recovery --
    goes through ``gateway.complete``.  Wrapping it records what the framework
    actually spent (real vLLM requests, tokens and latency) without touching
    routing, planning or the response that reaches the agent.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def complete(
        self,
        target: Any,
        request: Any,
        *,
        model_attempt_id: str,
        deadline_at: Any,
    ) -> Any:
        result = self._inner.complete(
            target,
            request,
            model_attempt_id=model_attempt_id,
            deadline_at=deadline_at,
        )
        target_kind = getattr(getattr(target, "kind", None), "value", None) or str(
            getattr(target, "kind", target)
        )
        self.calls.append(
            {
                "target": target_kind,
                "input_tokens": getattr(result, "input_tokens", None),
                "output_tokens": getattr(result, "output_tokens", None),
                "latency_ms": getattr(result, "latency_ms", None),
                "cost": getattr(result, "cost", None),
            }
        )
        return result

    def summary(self) -> dict[str, Any]:
        target_calls: dict[str, int] = {}
        target_tokens: dict[str, int] = {}
        # 分模型 **且** 分输入/输出。现有的两个键各缺一维：
        #   `framework_tokens_by_target` 分模型但不分 in/out
        #   `prompt_tokens`/`completion_tokens` 分 in/out 但不分模型
        # 而两条 token 的**价格差 3.5×（输入）/ 17×（输出）**，所以必须同时有。
        # 只加这一个键，不动上面任何一个（下游有消费者）。
        target_io: dict[str, dict[str, int]] = {}
        # 分模型的时延。总和（`framework_model_latency_ms`）判不出
        # 「卸载率上去了、延迟是不是也上去了」—— 两个模型的量级不同。
        target_latency: dict[str, float] = {}
        prompt = completion = 0
        latency = 0.0
        cost = 0.0
        for call in self.calls:
            target = str(call["target"])
            target_calls[target] = target_calls.get(target, 0) + 1
            input_tokens = int(call["input_tokens"] or 0)
            output_tokens = int(call["output_tokens"] or 0)
            prompt += input_tokens
            completion += output_tokens
            target_tokens[target] = target_tokens.get(target, 0) + input_tokens + output_tokens
            slot = target_io.setdefault(target, {"in": 0, "out": 0})
            slot["in"] += input_tokens
            slot["out"] += output_tokens
            latency += float(call["latency_ms"] or 0.0)
            target_latency[target] = target_latency.get(target, 0.0) + float(
                call["latency_ms"] or 0.0
            )
            cost += float(call["cost"] or 0.0)
        return {
            "framework_model_calls": len(self.calls),
            "framework_model_calls_by_target": target_calls,
            "framework_tokens_by_target": target_tokens,
            "framework_tokens_by_target_io": target_io,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "framework_model_latency_ms": round(latency),
            "framework_latency_ms_by_target": {
                target: round(total, 1) for target, total in target_latency.items()
            },
            "framework_latency_ms_mean_by_target": {
                target: round(target_latency[target] / target_calls[target], 1)
                for target in target_latency
            },
            "cost": round(cost, 10),
        }


def _jev_task_lifecycle(method):
    """Finalize optional Jev evidence on both returned and raised task failures."""
    @wraps(method)
    def run(self, task_id):
        if getattr(self, "_jev_config", None) is None:
            return method(self, task_id)
        self._jev_task_workflow_id = None
        row = None
        started = real_time()
        try:
            row = method(self, task_id)
        except Exception as error:
            traceback.print_exc()
            workflow = self._jev_task_workflow_id
            row = _failure_row(
                instance_id=task_id, mode=self._mode,
                agent_type=getattr(self, "_appworld_agent_type", "unknown"),
                workflow_run_id=workflow or f"initialization-{task_id}",
                e2e_wall_clock_ms=max(0, round((real_time() - started) * 1000)), error=error,
            )
            if workflow:
                self._workflow_rows[workflow] = row
        finally:
            workflow = self._jev_task_workflow_id
            if workflow:
                try:
                    if row is not None:
                        self._finalize_jev_task(workflow, row)
                finally:
                    self._clear_jev_workflow(workflow)
            self._jev_task_workflow_id = None
        return row
    return run


class AppWorldPilotRunner(PilotRunner):
    """PilotRunner with AppWorld Simplified model/environment hooks."""

    def __init__(
        self,
        *,
        raw_config: dict[str, Any],
        mode: str,
        agent_type: str,
        task_ids: list[str],
        output_dir: Path | str,
        experiment_name: str | None = None,
        jev_input_preflight: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        os.environ.setdefault("MODEL_SERVER_URL", "")
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
        if mode not in PILOT_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if agent_type not in _AGENT_TYPES:
            raise ValueError(f"unknown AppWorld agent type: {agent_type}")
        self._appworld_raw_config = copy.deepcopy(raw_config)
        self._appworld_agent_type = agent_type
        self._appworld_task_ids = [str(task_id) for task_id in task_ids]
        # AppWorld writes every artefact of a run under
        # ``<root>/experiments/outputs/<experiment_name>/``.  Without an
        # explicit name all modes of one task collapse onto the same
        # directory, so the caller (the group driver) passes a name that
        # encodes mode and attempt.  Falls back to the historical name for
        # ad-hoc single-run invocations.
        self._appworld_experiment_name = experiment_name
        fixed = _fixed_sampling(self._appworld_raw_config)
        self._appworld_large_model_config = _build_language_model_config(
            self._appworld_raw_config["models"]["large"], fixed["large"]
        )
        self._appworld_small_model_config = _build_language_model_config(
            self._appworld_raw_config["models"]["small"], fixed["small"]
        )
        config = _prepare_runner_config(self._appworld_raw_config)
        tool_policy_raw = config["runner"]["tool_policy"]
        tool_policy = ToolPolicy.from_mapping(
            tool_policy_raw, default_allowed_tools=(BATCH_TOOL_NAME,)
        )
        # 守卫 = 形状检查（`BasicToolCallVerifier`，包一层保留）**+ M4 失败记忆**。
        #
        # `lookup` 是**绑定方法**，此刻 `self._m4_guard_vectors` / `self._m4_memory`
        # 还没建（它们由下面的 `super().__init__` 填）—— 但 lookup 要到执行期才被
        # 调用，那时一定已经就绪。所以这里用方法引用而不是先取值。
        # T2 机制计数（spec §3.2）。**必须在构造守卫之前建** —— 守卫要收它，
        # 而守卫在 `super().__init__` 之前就得就绪。基类那边见到已有就不再建。
        self._counters = MechanismCounters()
        verifier = M4FailureMemoryVerifier(
            tool_policy=tool_policy,
            # `runner.m4_memory.guard_enabled: false` ⇒ None ⇒ 记忆层不参与，
            # 但形状检查与计数照旧（消融臂 B1-no-guard）。见 `_resolve_guard_lookup`。
            lookup=_resolve_guard_lookup(config, self._m4_action_guard_lookup),
            counters=self._counters,
        )
        super().__init__(
            config=config,
            mode=mode,
            output_dir=output_dir,
            gateway=object(),
            model=object(),
            env_factory=lambda: None,
            verifier=verifier,
            evaluator=FakeEvaluator(),
            step_limit=int(config["agent"]["step_limit"]),
            cost_limit=float(config["agent"]["cost_limit"]),
            system_template="",
            instance_template="",
            experiment_id=str(config.get("pilot", {}).get("id", "appworld-traceoffload")),
            # ⚠️ **必须注入穿透 freezegun 的时钟。**
            #
            # 默认的 `runner._utcnow()` 是 `datetime.now(timezone.utc)`，而 AppWorld
            # 的 world 在 `solve_tasks()` 内部创建并销毁，销毁后 freezegun 的
            # `tz_offsets` 栈被 pop —— 此后任何 `datetime.now()` 都抛
            # `IndexError: list index out of range`。
            #
            # 崩点在 `run_task` 里 `open_adapter_session` 那条链上（**开头**，
            # 跑的是**上一个**任务留下的坏状态 ⇒ 第 2 个及以后的任务才中招）。
            # 实测代价（0925，T-hi r3）：**整个轮次丢失，summary 一个字都没写出来**。
            #
            # 见 `traceoffload/clock.py:real_utcnow` 与 `tests/test_real_utcnow.py`。
            clock=real_utcnow,
            jev_input_preflight=jev_input_preflight,
        )
        self._appworld_agent_config = _build_agent_config(
            self._appworld_raw_config,
            agent_type,
            self._appworld_large_model_config,
        )

    def _build_native_agent(self) -> Any:
        _ensure_appworld_agents_importable()
        from appworld_agents.code.simplified.agent import Agent

        return Agent.from_dict(copy.deepcopy(self._appworld_agent_config))

    def _build_small_language_model(self) -> Any:
        _ensure_appworld_agents_importable()
        from appworld_agents.code.simplified.language_model import LanguageModel

        return LanguageModel(**self._appworld_small_model_config)

    def _adapter_spec(self, runtime: AppWorldTraceOffloadRuntime, task_id: str) -> Any:
        from traceoffload.m3.integration import AdapterSessionSpec, ToolRegistryEntry

        return AdapterSessionSpec(
            workflow_run_id=runtime.workflow_run_id,
            agent_id=f"appworld-{self._appworld_agent_type}",
            agent_version="0.1.0",
            workflow_goal=_workflow_goal(task_id),
            tool_definitions=runtime.tool_definitions_for(None),
            state_access_mode=StateAccessMode.TRANSACTIONAL_WRITE,
            expected_effect=EffectRisk.REVERSIBLE_WRITE,
            recovery_capability=RecoveryLevel.ROLLBACK_AND_RESUME,
            downstream_dependency=DownstreamDependency.FULL,
            node_id="appworld_agent_loop",
            node_type="react_loop",
            framework_metadata={
                "appworld_agent_type": self._appworld_agent_type,
                "appworld_task_id": task_id,
                "model_pair": ["Qwen3.8-27B", "Qwen3.5-9B"],
            },
            tool_registry=(
                ToolRegistryEntry(
                    tool_name=BATCH_TOOL_NAME,
                    interface="python",
                    default_effect_scope=EffectScope.REVERSIBLE_WRITE,
                    version="v1",
                    description="Execute generated AppWorld code as one DB transaction.",
                ),
            ),
        )

    def _build_coordinator(self, runtime: AppWorldTraceOffloadRuntime) -> Any:
        from traceoffload.coordinator import Coordinator
        from traceoffload.m3.executor import ControlledExecutor
        from traceoffload.storage import RequestResponseStore

        store = RequestResponseStore(clock=self._clock)
        allowed_targets = AllowedTargets(
            policy_version=f"appworld-{self._mode}",
            targets=(
                ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=self._large),
                ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=self._small),
            ),
        )
        executor = ControlledExecutor(
            counters=self._counters,
            gateway=runtime.gateway,
            verifier=self._verifier,
            store=store,
            tool_proxy=runtime.tool_proxy,
            clock=self._clock,
            tool_action_analyzers={BATCH_TOOL_NAME: runtime.analyzer},
        )
        return Coordinator(
            counters=self._counters,
            profiler=self._make_profiler(allowed_targets),
            router=self._make_router(),
            executor=executor,
            store=store,
            events=self._events,
            policy=self._policy,
            clock=self._clock,
            admission_gate=None,
            step_limit=self._step_limit,
            probe_budget_per_workflow=self._probe_budget_per_workflow,
            probe_budget_total=self._probe_budget_total,
            probe_ledger=self._probe_ledger,
            experiment_id=self._resolve_experiment_id(instance_keys=self._appworld_task_ids),
            sandbox_backend="appworld-db-checkpoint",
            supports_transactional_write=True,
            registered_tools=(BATCH_TOOL_NAME,),
            large_model=self._large,
            span_observer=(
                self._observe_m4_span if self._m4_memory_worker is not None else None
            ),
            # ⚠️ 这里**不能**跟着 `_m4_memory_worker` 一起关：`_observe_m4_profile`
            # 算的是 span 向量，v4 预测器读的就是它。M4 关掉时预测器还得活着。
            # 用基类的谓词，别再复刻一遍条件（复刻就会漂）。
            profile_observer=(
                self._observe_profile if self._profile_observer_needed() else None
            ),
        )

    @_jev_task_lifecycle
    def run_task(self, task_id: str) -> dict[str, Any]:
        _ensure_appworld_agents_importable()
        # T2 计数活在 runner 上、而一个进程顺序跑多个任务 ⇒ **本行快照含前面任务的量**。
        # 记下起点，写 summary 时做差，得到「该任务自己的」计数。
        # 不做差的话汇总求和会重复计算 —— 实测求和得 1245 次 guard_checks，
        # 取最后一行才是 74（差 17 倍，且不报错）。
        counters_before = self._counters.snapshot()
        # `m4_memory` 同理：它是**进程内累积**的（第 N 行含前 N-1 个任务）。
        # 实测：12 行的 `worker.received` 求和得 6867，取末行才是 1065（虚高 6.4 倍）。
        _m4_stats = getattr(self, "m4_memory_stats", None)
        m4_before = _m4_stats() if callable(_m4_stats) else None
        # `l2_v4` 同理：`hits` / `miss_no_vector` 是计数器，`p_values` 是追加日志。
        # 实测：12 行的 `p_values` 长度 10→191、`hits` 11→193，求和会重复计算。
        _v4_stats = getattr(self, "l2_v4_stats", None)
        v4_before = _v4_stats() if callable(_v4_stats) else None
        # `l2_add4` 同理：两个 `p_*_values` 都是追加日志，`hits` 是计数器。
        _a4_stats = getattr(self, "l2_add4_stats", None)
        a4_before = _a4_stats() if callable(_a4_stats) else None
        # 幂等：`run_tasks` 已调过；直接调 `run_task` 的入口（探针/测试）在这里兜底。
        # 必须在建 runtime 之前 —— `workflow_run_id` 要用它。
        self._resolve_experiment_id(instance_keys=self._appworld_task_ids)
        agent = self._build_native_agent()
        large_language_model = agent.language_model
        runtime = AppWorldTraceOffloadRuntime(
            workflow_run_id=self._workflow_run_id(task_id),
            agent=agent,
            original_language_model=large_language_model,
            model_parameters=_decode_parameters(self._config),
            workflow_goal=_workflow_goal(task_id),
            deadline_minutes=10.0,
            snapshot_provider=self._snapshot_provider,
            record_lm_step_index=(self._mode == "classifier_route"),
        )
        self._jev_task_workflow_id = runtime.workflow_run_id
        runtime.analyzer = AppWorldCodeEffectAnalyzer()
        runtime.tool_proxy = AppWorldToolProxy(runtime)
        runtime.gateway = AppWorldLanguageModelGateway(
            runtime,
            large_language_model,
            self._build_small_language_model(),
        )
        # The meter must be installed on ``runtime.gateway`` *before* the
        # coordinator is built: ``ControlledExecutor`` captures the gateway it
        # is handed at construction time, so wrapping afterwards would leave
        # the executor calling the unmetered gateway and every count at zero.
        meter = _GatewayMeter(runtime.gateway)
        runtime.gateway = meter
        runtime.coordinator = self._build_coordinator(runtime)
        runtime.adapter_session = runtime.coordinator.open_adapter_session(
            self._adapter_spec(runtime, task_id)
        )
        runtime.proxy = AppWorldLanguageModelProxy(runtime)
        agent.language_model = runtime.proxy

        original_initialize = agent.initialize

        def initialize(_self: Any, world: Any) -> None:
            original_initialize(world)
            runtime.bind_world(world)
            # M1 的 embedding 文本要**跳过 prompt 模板消息**。
            #
            # 为什么：AppWorld 的 prompt 模板自带一段**完整的 few-shot 示例轨迹**
            # （实测 23 条消息、其中 10 条 assistant，内容是另一个 task 的完整
            # rollout，以 `complete_task` 结尾）。`initialize()` 把模板渲染成
            # `self.messages`，所以**第一次 `generate()` 就已经有 10 条 assistant**。
            # 而 v2 的窗口正好是 10 —— 不跳过的话整窗全是示例，而离线同一步只有
            # `[GOAL]`。实测真机：11 次调用每次 10 个示例块，逐字比对离线 **0/11**。
            #
            # 数量由 agent 自己记（`SimplifiedReActCodeAgent.__init__` 里的
            # `num_instruction_messages`），**不靠文案匹配**。
            #
            # ⚠️ 交给 `_set_skip_leading_messages` 统一注入，**不要**只设
            # `_m4_text_builder` —— add4 的动作文本构建器（`C_bare_w10`）窗口
            # 也是 10，漏掉它的话 k=1 的窗口会被示例动作填满，而离线在 k=1
            # 是空串。实测：跳过 20~23 时 588 个验证 cell **逐字一致**，
            # 不跳过只有 37%。
            count = getattr(agent, "num_instruction_messages", None)
            if isinstance(count, int):
                self._set_skip_leading_messages(count)

        agent.initialize = MethodType(initialize, agent)

        workflow_run_id = runtime.workflow_run_id
        started = _perf_counter()
        # ⚠️ 任务的**起点**墙上时间必须在 `solve_tasks()` 之前取。
        #
        # 为什么：`solve_tasks()` 内部会创建 AppWorld 的 world，而 world 用
        # freezegun 冻结时间；返回时 world 已关、freezegun 的时区栈被 pop。
        # 此后任何 `datetime.now()` 都会走进被 patch 的 `FakeDatetime.now` →
        # `_tz_offset()` → `tz_offsets[-1]` → **IndexError: list index out of range**。
        #
        # 实测（验证实例 `634f342_2`，两次独立复现）：崩在下面那句
        # `observed_at = self._clock()`，而它**只在 `m4_records` 非空时才执行** ——
        # 所以只有落了 M4 记录的任务会踩到。代价是整轮 run 在写 summary 之前退出：
        # 任务本身跑完了，数据一条不落（w5 三个轮次就是这样全丢的）。
        #
        # 所以这里先取起点，下面用「起点 + 实测时长」还原终点时刻 —— 既不碰
        # 那口被 patch 的时钟，语义也仍然是对的（不用退回"任务开始时刻"）。
        started_wall = self._clock()
        try:
            agent.solve_tasks(
                task_ids=[task_id],
                experiment_name=self._appworld_experiment_name or f"traceoffload-{task_id}",
                num_processes=1,
                process_index=0,
            )
        except Exception as error:  # noqa: BLE001 - 崩了也仍是一条数据点
            # 任务级兜底。**驱动已经是一个进程一个任务**（AppWorld 的
            # `Requester.close()` 会无条件 `freezer.stop()`，同一个进程里跑
            # 第二个任务就会 `freezegun` 弹空栈 → IndexError），所以这里接住
            # 不会污染下一个任务 —— 下一个任务在**另一个进程**里。
            #
            # 不接的话：这一格在 summary 里**整个消失**，验收的「每轮 =N 条」
            # 变成 N-1，而那个症状看起来像实例列表写错了。
            traceback.print_exc()
            elapsed = max(0, round((_perf_counter() - started) * 1000))
            row = _failure_row(
                instance_id=task_id,
                mode=self._mode,
                agent_type=self._appworld_agent_type,
                workflow_run_id=workflow_run_id,
                e2e_wall_clock_ms=elapsed,
                error=error,
            )
            self._workflow_rows[workflow_run_id] = row
            return row
        runtime.coordinator.flush_pending(workflow_run_id)
        runtime.coordinator.close_workflow(workflow_run_id)
        e2e_wall_clock_ms = max(0, round((_perf_counter() - started) * 1000))
        evaluation = runtime.evaluation
        success = bool(getattr(evaluation, "success", False))

        # ── 结局投递给 M4 ────────────────────────────────────────────────
        # 为什么必须在这里补：`PilotRunner.run_workflow()`（runner.py:1326-1360）
        # 里有完整的 evaluate → enqueue_outcome 流程，但 **AppWorld 路径走的是
        # 本方法，绕过了它**。结果是 M4 的记录 `task_success` 恒为 None ——
        # 检索出来的邻居一个带结局标签的都没有。
        #
        # 也不能复用 `self._evaluator.evaluate_workflow_final`：那个
        # FakeEvaluator（fakes.py:194）的 success 是从 **span 的
        # `execution_summary.status`** 算的，不是 AppWorld 的任务成败，
        # 在这里会给出错的标签。所以直接用 `runtime.evaluation.success`。
        if self._m4_memory_worker is not None:
            m4_records = tuple(
                record
                for record in self._events.span_records()
                if record.route_plan.workflow_run_id == workflow_run_id
            )
            if m4_records:
                attribution = None
                if len(m4_records) > 1:
                    share = 1.0 / len(m4_records)
                    attribution = tuple(
                        Attribution(
                            episode_id=record.route_plan.episode_id,
                            weight=share,
                            method="equal_weight",
                        )
                        for record in m4_records
                    )
                # 终点时刻 = 起点 + 实测时长。**不要**在这里直接读时钟：
                # 那时 AppWorld 已经关掉 world 了（见上面 `started_wall` 的注释）。
                observed_at = started_wall + _timedelta_milliseconds(
                    e2e_wall_clock_ms
                )
                self._m4_memory_worker.enqueue_outcome(
                    OutcomeEvent(
                        schema_version="1.0",
                        message_id=f"message-workflow_final-{workflow_run_id}",
                        producer_ref=ProducerRef(
                            component=Component.EVALUATOR, version="0.1.0"
                        ),
                        produced_at=observed_at,
                        event_id=f"workflow_final-{workflow_run_id}",
                        workflow_run_id=workflow_run_id,
                        episode_id=None,
                        scope=OutcomeScope.WORKFLOW_FINAL,
                        logical_invocation_id=(
                            None
                            if attribution is not None
                            else m4_records[-1].route_plan.logical_invocation_id
                        ),
                        label=(
                            OutcomeLabel.SUCCESS if success else OutcomeLabel.FAILURE
                        ),
                        evaluator_ref=GuardRef(
                            kind="evaluator",
                            id=f"appworld-{self._appworld_agent_type}",
                            version="v1",
                        ),
                        attribution=attribution,
                        observed_at=observed_at,
                        revision=1,
                    )
                )

            # ── 观测成本：suffix_cost / trajectory_cost / position_ratio ─────
            # 线上一次只跑**一条臂**，所以算不出离线实验里的 Δ（卸载 vs 不卸载）。
            # 能实测的是「这条轨迹实际烧了多少」—— 存下来，检索时按位置归一化再比。
            #
            # `suffix_cost` 强依赖位置（越早的 span 后缀越长），所以必须同时记
            # `position_ratio = k/n`；**直接横向比 suffix_cost 会得出「早的 span
            # 都贵」这种假结论**。
            from traceoffload.m4.memory import compute_span_cost_fields

            def _step_index(record: Any) -> int:
                # `Coordinator` 用单调计数器生成 `invocation-{n}`（coordinator.py:666），
                # 所以数字后缀就是轨迹内顺序。
                tail = str(record.route_plan.logical_invocation_id).rsplit("-", 1)[-1]
                return int(tail) if tail.isdigit() else 0

            def _span_tokens(record: Any) -> int:
                fields = compute_span_cost_fields(record)
                return int(fields.get("span_input_tokens") or 0) + int(
                    fields.get("span_output_tokens") or 0
                )

            ordered = sorted(m4_records, key=_step_index)
            costs = [_span_tokens(record) for record in ordered]
            trajectory_cost = float(sum(costs))
            span_count = len(ordered)
            per_span: dict[str, dict[str, float | int | None]] = {}
            prefix = 0
            for position, (record, cost) in enumerate(zip(ordered, costs), start=1):
                prefix += cost
                per_span[
                    self._m4_record_key(
                        record.route_plan.workflow_run_id,
                        record.route_plan.logical_invocation_id,
                    )
                ] = {
                    # 含本 Span：从它起到轨迹结束
                    "suffix_cost": float(trajectory_cost - prefix + cost),
                    "trajectory_cost": trajectory_cost,
                    "position_ratio": position / span_count,
                    "trajectory_span_count": span_count,
                    # 绝对步号 —— M2 决策时的分桶坐标（见 m4_retrieval.STEP_BIN_EDGES）
                    "step_index": _step_index(record),
                }
            self._m4_memory_worker.enqueue_observed_cost(workflow_run_id, per_span)

        row = {
            "record_type": "workflow",
            "workflow_run_id": workflow_run_id,
            "instance_id": task_id,
            "appworld_agent_type": self._appworld_agent_type,
            "mode": self._mode,
            "framework": "traceoffload",
            "appworld_success": success,
            "task_success": success,
            "appworld_pass_count": int(getattr(evaluation, "pass_count", 0) or 0),
            "appworld_fail_count": int(getattr(evaluation, "fail_count", 0) or 0),
            "e2e_wall_clock_ms": e2e_wall_clock_ms,
        }
        row.update(meter.summary())
        # T2 机制计数（spec §3.2）—— **必须进 summary**，否则摘掉机制的臂和满血臂
        # 在报告里长得一模一样，分不清「没效果」和「没触发」。
        # 0925 实测：加这个之前，60 次跑里守卫触发过几次**问不出来**
        # （summary 无字段、logger.jsonl 空、verifier_refs 复用内层 guard_ref）。
        mechanism_stats = getattr(self, "mechanism_stats", None)
        if callable(mechanism_stats):
            # **增量**，不是累积快照 —— 见 `run_task` 开头。把所有行的同名字段
            # 加起来就等于真实总数；累积口径下会变成 1+2+…+12 倍。
            row["mechanism"] = delta_of(counters_before, mechanism_stats())
        # M4 worker 的统计（没配 M4 就 None）。`stored`=**该任务**往库里写了几条。
        # **增量**，不是累积快照 —— 见 `run_task` 开头。
        if callable(_m4_stats) and m4_before is not None:
            stats = _m4_stats()
            if stats is not None:
                delta = delta_of(m4_before, stats)
                # `avg_ms` 在 `delta_of` 里走透传（avg 的差没有意义）——
                # 这里按增量重算成「该任务期间的每次 embedding 平均耗时」。
                count = delta.get("m1_online_embedding_count") or 0
                total = delta.get("m1_online_embedding_total_ms") or 0.0
                delta["m1_online_embedding_avg_ms"] = (
                    round(total / count, 3) if count else None
                )
                row["m4_memory"] = delta
        # v4 预测器的使用统计（没配就 None）。`miss_no_vector > 0` = 有 span 因为
        # 那 52 个数没算出来而**直接走了大模型**（不卸载）—— 不是退回旧预测器。
        # **增量**，不是累积快照 —— `p_values` 是追加日志，`hits` 是计数器。
        if callable(_v4_stats) and v4_before is not None:
            stats = _v4_stats()
            if stats is not None:
                row["l2_v4"] = delta_of(v4_before, stats)
        # add4 预测器的使用统计（没配就 None）。**增量** —— 两个 `p_*_values`
        # 都是追加日志，求和会重复计算（与 `l2_v4` 同一个坑）。
        # `miss_no_vector > 0` = 有 span 因为 116 维没算出来而直接走了大模型。
        if callable(_a4_stats) and a4_before is not None:
            stats = _a4_stats()
            if stats is not None:
                row["l2_add4"] = delta_of(a4_before, stats)
        self._workflow_rows[workflow_run_id] = row
        return row

    def run_tasks(self) -> dict[str, Any]:
        self._resolve_experiment_id(instance_keys=self._appworld_task_ids)
        rows = [self.run_task(task_id) for task_id in self._appworld_task_ids]
        output_path = Path(self._output_dir) / f"appworld-{self._mode}-summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # ⚠️ **读旧 + 合并**，不是直接覆盖。驱动现在是「一个进程一个任务」
        # （见 `run_ablation_5workers.sh` 的 run_round 与
        # `tests/test_appworld_summary_append.py` 的说明），一张 summary 会被
        # 追加写 N 次。这里直接覆盖的话，一轮跑完盘上只剩最后一个任务。
        merged = _merge_summary_rows(_read_summary_rows(output_path), rows)
        output_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # M4 memory worker 也要排空：它收的是 SpanRecord 和 OutcomeEvent，
        # 都是异步 job。不等就退出进程 → 最后一批（含结局回填）全丢。
        if self._m4_memory_worker is not None:
            self._m4_memory_worker.drain(timeout_seconds=30.0)
        return {"workflows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _failure_row(
    *,
    instance_id: str,
    mode: str,
    agent_type: str,
    workflow_run_id: str,
    e2e_wall_clock_ms: int,
    error: BaseException,
) -> dict[str, Any]:
    """任务级崩溃落成的一条行。**它仍然是一条数据点，但被单独标出来。**

    为什么必须带 `failure`（`原生agent采集…1010.md` §4.3 的教训）：
    `task_success: false` 有**两个来源** —— 真实评估失败，和进程崩了。
    只统计 `task_success` 会把崩溃当成模型失败，而这两件事在这个实验里的含义
    **完全相反**：一个是「小模型答错了」，一个是「框架自己死了」。

    ⚠️ **不补 `mechanism` / `l2_add4` / `m4_memory` 这些键。** 崩在
    `AppWorld.initialize()`（我们实测的那种）时模型一次都没调过 —— 这时补一个
    `{"hits": 0}` 上去，汇总里会和「跑了但一次没出分」长得一模一样。
    缺键才是诚实的：下游一眼看出这条不可用。
    """
    return {
        "record_type": "workflow",
        "workflow_run_id": workflow_run_id,
        "instance_id": instance_id,
        "appworld_agent_type": agent_type,
        "mode": mode,
        "framework": "traceoffload",
        "appworld_success": False,
        "task_success": False,
        "appworld_pass_count": 0,
        "appworld_fail_count": 0,
        "e2e_wall_clock_ms": e2e_wall_clock_ms,
        "failure": f"{type(error).__name__}: {error}",
    }


def _read_summary_rows(path: Path) -> list[dict[str, Any]]:
    """读回盘上已有的行。**任何读不动的情形都当「没有」**，不抛。

    为什么当空而不是抛：驱动现在是**一个进程一个任务**，一张 summary 被追加
    写很多次。半截文件（上一个进程被杀在写盘中途）是最可能的那种坏 ——
    这里抛的话这一格的数据全丢；当空的话最多丢前面那些格，后面的照常追加。
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _merge_summary_rows(
    existing: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """把新产出的行并进已有的 —— 按 `instance_id` **原地替换**，不是追加到末尾。

    为什么必须去重：崩掉的格会被驱动重跑。多出一条的话，验收的「每轮 =N 条」
    会变成 N+1 —— 而那个症状看起来像是实例列表写重了，两件完全不同的事。

    为什么替换要**留在原位**：挪到末尾会让 summary 的顺序取决于「哪个任务崩过」，
    下游按行号对齐的地方就会错位。
    """
    out = list(existing)
    index_by_id: dict[Any, int] = {}
    for position, row in enumerate(out):
        key = row.get("instance_id") if isinstance(row, dict) else None
        if key is not None:
            index_by_id[key] = position
    for row in rows:
        key = row.get("instance_id")
        position = index_by_id.get(key)
        if position is None:
            index_by_id[key] = len(out)
            out.append(row)
        else:
            out[position] = row
    return out


def _memory_guard_enabled(config: dict[str, Any]) -> bool:
    """`runner.m4_memory.guard_enabled`，缺省 `True`。

    **默认必须是开**：反过来会让所有既有配置静默失去守卫 —— 跑得通、有数据、
    只是守卫没了，翻日志也看不出来。

    字符串按字面判（`"false"` / `"0"` / `"no"` 都算关）。按 `bool(s)` 走的话
    `bool("false") is True` ⇒ **静默保持开启**，而调用方以为自己跑的是
    `B1-no-guard`，实际是 `A5-full` —— 最坏的方向。
    """
    m4 = (config.get("runner") or {}).get("m4_memory")
    if not isinstance(m4, dict):
        return True
    raw = m4.get("guard_enabled", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"false", "0", "no", "off", ""}
    return bool(raw)


def _resolve_guard_lookup(config: dict[str, Any], lookup: Any) -> Any:
    """`guard_enabled: false` ⇒ 不注入 lookup（记忆层不参与判定）。

    ⚠️ **不要换成 `BasicToolCallVerifier`。** `M4FailureMemoryVerifier.check()`
    在 `lookup is None` 时就等于它，但形状判定仍走本类 ⇒
    `guard_checks` / `guard_shape_rejects` **照样计数**。换成基类的话那两个计数
    会变成 0，报告里看不出「形状检查还在跑」—— 而 spec §2.2 明确说 B1
    「不是『无守卫』，是『只有形状守卫』」。
    """
    return lookup if _memory_guard_enabled(config) else None


def _apply_m4_memory_path_override(config: dict[str, Any], override: str | None) -> None:
    """把 M4 记忆库的路径钉成**绝对路径** —— 让一个 worker 的多轮共享同一个库。

    为什么必须是绝对路径
    --------------------
    `runner._start_m4_memory` 是这么解析的::

        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self._output_dir / path        # ← 相对路径拼到 --output-dir 下

    而跑批的 `--output-dir` 是 **per (task, round)** 的（协议 §6 的
    `<task>/<GROUP>/round-NN/result.json` 布局要求如此）。两者一叠加，
    **每个 (task, 轮) 一份全新的空库** ⇒ M4 从来不累积。

    实测证据（5-worker sweep，w1）：round-1 的库只含 `aa2d21b8` 那次调用，
    round-2 只含 `c33f2cb3` —— 零交集。而协议 §4 明文要求
    「worker 内跨实例、**跨轮次**累积」。⇒ 那条要求从未实现过。

    绝对路径跳过那次拼接：worker 内跨实例/跨轮次累积（同一个库），
    worker 之间物理隔离（各写各的），`--output-dir` 的布局不受影响。

    ⚠️ 相对路径**报错**而不是静默接受 —— 静默接受等于重新制造上面那个 bug，
    而且跑得通、有数据、只是记忆永远是空的，日志里看不出来。
    """

    if override is None:
        return
    if not Path(override).expanduser().is_absolute():
        raise ValueError(
            f"--m4-memory-path 必须是绝对路径，收到 {override!r}。"
            "相对路径会被拼到 per-(task, round) 的 --output-dir 下，"
            "于是每个格子一份空库、M4 永远不累积（而且不报错）。"
        )
    m4_section = config.get("runner", {}).get("m4_memory")
    if not isinstance(m4_section, dict):
        raise ValueError(
            "--m4-memory-path 给了，但 config 里没有 runner.m4_memory —— "
            "覆盖一个不存在的键等于没覆盖，而且不报错。"
        )
    m4_section["path"] = str(Path(override).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/spanoffload-qwen-local-full.yaml",
        help="YAML containing the exact large/small model configuration.",
    )
    parser.add_argument(
        "--agent-type",
        choices=_AGENT_TYPES,
        default="react_code",
        help="AppWorld Simplified agent to run.",
    )
    parser.add_argument(
        "--mode",
        choices=PILOT_MODES,
        default="llm_only",
        help="TraceOffload execution mode.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=None,
        help="AppWorld task id; can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="AppWorld dataset name (dev/train/test_normal/test_challenge).",
    )
    parser.add_argument(
        "--m4-memory-path",
        default=None,
        help=(
            "M4 记忆库的**绝对路径**。给了它，同一个 worker 的多轮就共享同一个库"
            "（协议 §4「worker 内跨实例、跨轮次累积」）。不给则按 config 里的相对"
            "路径解析到 --output-dir 下，即每个 (task, round) 一份空库。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/appworld-traceoffload",
        help="Directory for TraceOffload workflow rows.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "AppWorld experiment name, i.e. the directory under "
            "<root>/experiments/outputs/ that this run writes to. Defaults to "
            "'traceoffload-<task-id>'. Pass a distinct name per mode, or runs "
            "of the same task overwrite each other's AppWorld artefacts."
        ),
    )
    args = parser.parse_args()

    load_env_file()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("config must be a YAML mapping")
    _fixed_sampling(raw)
    _apply_m4_memory_path_override(raw, args.m4_memory_path)

    if args.task_ids:
        task_ids = [str(item) for item in args.task_ids]
    elif args.dataset:
        from appworld import load_task_ids

        task_ids = [str(item) for item in load_task_ids(args.dataset)]
    else:
        raise SystemExit("pass --task-id or --dataset")

    if args.experiment_name and len(task_ids) != 1:
        raise SystemExit(
            "--experiment-name names one AppWorld experiment directory, so it "
            f"only makes sense with a single --task-id (got {len(task_ids)})."
        )

    runner = AppWorldPilotRunner(
        raw_config=raw,
        mode=args.mode,
        agent_type=args.agent_type,
        task_ids=task_ids,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    result = runner.run_tasks()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
