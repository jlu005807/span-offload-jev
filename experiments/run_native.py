#!/usr/bin/env python3
"""Run the *native* AppWorld Simplified Agent with no TraceOffload in the path.

Purpose
-------
``run_appworld_spanoffload.py`` records the original trace through a
``TracingLanguageModelProxy`` that intercepts every ``language_model.generate``
and sums per-call latencies.  That number is an *instrumented* figure: it does
not include the agent-loop/Docker/AppWorld overhead around the LM calls, and it
is produced by a wrapped agent.

This entry point measures the two quantities that the TraceOffload comparison
actually needs, with nothing inserted into the agent's call path:

1. **End-to-end latency** -- a wall clock started before ``Agent.solve_tasks``
   and stopped after it returns, i.e. the real cost of solving one AppWorld
   task.
2. **Token cost** -- ``standardized_usage`` read off the dict that
   ``LanguageModel.generate`` already returns, summed over calls.

The only modification made to the agent is observational: ``generate`` is
replaced by a wrapper that calls straight through to the original bound method
and records what it returned.  Messages, prompts, tool schemas, action parsing
and the agent loop are untouched, and ``appworld/experiments/**`` is never
edited.

Caching is disabled (``use_cache=False``) so every measured call is a real
request to the vLLM endpoint.

Usage
-----
    APPWORLD_ROOT=runs/测试实例/_appworld_root \
    python experiments/run_native.py \
        --config configs/spanoffload-qwen-local-full-8300-8400.yaml \
        --model large --task-id 3aa1a22_3 \
        --output-dir "runs/测试实例/3aa1a22_3/native-large" \
        --step-limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
import time
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from experiments.task_outcomes import observe_termination, finalize_task_outcome  # noqa: E402

from traceoffload.config import load_env_file  # noqa: E402

from run_appworld import (  # noqa: E402
    _build_agent_config,
    _build_language_model_config,
    _ensure_appworld_agents_importable,
    _fixed_sampling,
    _prepare_runner_config,
)

_MODEL_KINDS = ("large", "small")


def _real_time() -> float:
    """Monotonic wall clock that is immune to AppWorld/freezegun time freezing.

    ``perf_counter`` is *not* safe here.  freezegun 1.5.1 freezes it for the
    whole ``freeze_time`` window a world installs (freezegun issue #529, which
    AppWorld pins 1.5.0/1.5.1 around), and freezing it makes every per-call
    duration exactly 0 -- the module-level reference does not escape, because
    freezegun patches behaviour rather than the module attribute.
    ``CLOCK_MONOTONIC`` goes straight to libc and is not touched, which is the
    same idiom ``traceoffload.integrations.appworld_simplified`` uses.
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def _empty_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_cache_miss_tokens": 0,
        "input_cache_hit_tokens": 0,
        "input_cache_write_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "model_latency_ms": 0,
        "error_calls": 0,
    }


def _usage_of(message: Any) -> dict[str, Any] | None:
    """Read AppWorld's ``standardized_usage`` without depending on its class."""
    if not isinstance(message, dict):
        return None
    usage = message.get("standardized_usage")
    if usage is None:
        return None
    tokens = getattr(usage, "tokens", None)
    cost = getattr(usage, "cost", None)
    if tokens is None:
        return None
    cache_miss = int(getattr(tokens, "input_cache_miss", 0) or 0)
    cache_hit = int(getattr(tokens, "input_cache_hit", 0) or 0)
    cache_write = int(getattr(tokens, "input_cache_write", 0) or 0)
    output = int(getattr(tokens, "output", 0) or 0)
    return {
        "input_cache_miss_tokens": cache_miss,
        "input_cache_hit_tokens": cache_hit,
        "input_cache_write_tokens": cache_write,
        "output_tokens": output,
        "total_tokens": cache_miss + cache_hit + cache_write + output,
        "cost": float(getattr(cost, "total", 0.0) or 0.0),
    }


def _install_measurement(language_model: Any, totals: dict[str, Any]) -> None:
    """Wrap ``generate`` observationally; behaviour is otherwise identical."""
    original_generate = language_model.generate

    def generate(
        _self: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cache_control_at: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        started = _real_time()
        try:
            message = original_generate(
                messages=messages,
                tools=tools,
                cache_control_at=cache_control_at,
                **kwargs,
            )
        finally:
            totals["model_latency_ms"] += max(0, round((_real_time() - started) * 1000))
        totals["calls"] += 1
        parsed = _usage_of(message)
        if parsed is None:
            totals["error_calls"] += 1
            return message
        for key, value in parsed.items():
            if key == "cost":
                totals[key] = round(float(totals[key]) + value, 10)
            else:
                totals[key] += value
        return message

    language_model.generate = MethodType(generate, language_model)


def _tracker_tokens(agent: Any, task_id: str) -> dict[str, int] | None:
    """Cross-check totals from AppWorld's own usage tracker, when present."""
    tracker = getattr(agent, "usage_tracker", None)
    if tracker is None:
        return None
    usage = getattr(tracker, "task_usage", {}).get(task_id)
    tokens = getattr(usage, "tokens", None)
    if tokens is None:
        usage = getattr(tracker, "overall_usage", None)
        tokens = getattr(usage, "tokens", None)
    if tokens is None:
        return None
    cache_miss = int(getattr(tokens, "input_cache_miss", 0) or 0)
    cache_hit = int(getattr(tokens, "input_cache_hit", 0) or 0)
    output = int(getattr(tokens, "output", 0) or 0)
    return {
        "prompt_tokens": cache_miss + cache_hit,
        "completion_tokens": output,
        "total_tokens": cache_miss + cache_hit + output,
    }


def run_native_task(
    *,
    raw_config: dict[str, Any],
    model_kind: str,
    task_id: str,
    output_dir: Path,
    step_limit: int,
    agent_type: str,
    experiment_name: str,
    dataset_name: str = "train",
) -> dict[str, Any]:
    # ``LanguageModel.__init__`` in appworld_agents eagerly builds an
    # ``OpenAI()`` client for its provider map even when the configured
    # provider is the vLLM one, and ``OpenAI()`` refuses to construct without
    # a key.  The TraceOffload entry point sets this too; the native baseline
    # has to mirror it or the agent never gets built.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    _ensure_appworld_agents_importable()
    from appworld import AppWorld
    from appworld_agents.code.simplified.agent import Agent

    fixed = _fixed_sampling(raw_config)
    model_config = _build_language_model_config(
        raw_config["models"][model_kind], fixed[model_kind]
    )
    # Every measured call must be a real request: the joblib LM cache would
    # otherwise return a stored response and report ~0 ms latency.
    model_config["use_cache"] = False

    agent_config = _build_agent_config(raw_config, agent_type, model_config)
    agent_config["max_steps"] = step_limit
    # The measured task ids come from the caller, not from a dataset split, so
    # pin the dataset that AppWorld records in its experiment metadata.
    agent_config["appworld_config"]["dataset_name"] = dataset_name
    agent = Agent.from_dict(agent_config)
    terminations = observe_termination(agent)

    totals = _empty_usage()
    _install_measurement(agent.language_model, totals)

    evaluation: Any = None
    task_completed: bool | None = None
    original_initialize = agent.initialize

    def initialize(_self: Any, world: Any) -> None:
        nonlocal evaluation
        original_initialize(world)
        original_close = world.close

        def close(world_self: Any) -> None:
            nonlocal evaluation, task_completed
            try:
                evaluation = world_self.evaluate()
            except Exception as error:  # noqa: BLE001 - recorded, never fatal
                evaluation = {"error": f"{type(error).__name__}: {error}"}
            # ``task_completed`` goes through the supervisor over the world's
            # sqlite connections, so it has to be read while the world is
            # still open.  Reading it afterwards raises
            # ``Cannot operate on a closed database``.
            try:
                task_completed = bool(world_self.task_completed())
            except Exception:  # noqa: BLE001 - recorded as unknown, never fatal
                task_completed = None
            original_close()

        world.close = MethodType(close, world)

    agent.initialize = MethodType(initialize, agent)

    started = _real_time()
    failure: str | None = None
    try:
        with AppWorld.initializer(update_defaults=True, experiment_name=experiment_name):
            agent.solve_tasks(
                task_ids=[task_id],
                experiment_name=experiment_name,
                num_processes=1,
                process_index=0,
            )
    except Exception as error:  # noqa: BLE001 - a crashed task is still a data point
        failure = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    e2e_wall_clock_ms = max(0, round((_real_time() - started) * 1000))

    if isinstance(evaluation, dict):
        success = bool(evaluation.get("success", False))
        pass_count = int(evaluation.get("pass_count", 0) or 0)
        fail_count = int(evaluation.get("fail_count", 0) or 0)
    elif evaluation is None:
        success, pass_count, fail_count = False, 0, 0
    else:
        success = bool(getattr(evaluation, "success", False))
        pass_count = int(getattr(evaluation, "pass_count", 0) or 0)
        fail_count = int(getattr(evaluation, "fail_count", 0) or 0)

    row = {
        "record_type": "native_workflow",
        "instance_id": task_id,
        "group": f"native-{model_kind}",
        "framework": "native",
        "mode": f"native_{model_kind}",
        "model_kind": model_kind,
        "model_name": model_config["name"],
        "api_base": model_config["base_url"],
        "agent_type": agent_type,
        "step_limit": step_limit,
        "dataset_name": dataset_name,
        "task_success": success,
        "appworld_pass_count": pass_count,
        "appworld_fail_count": fail_count,
        "e2e_wall_clock_ms": e2e_wall_clock_ms,
        "model_calls": totals["calls"],
        "model_latency_ms": totals["model_latency_ms"],
        "input_cache_miss_tokens": totals["input_cache_miss_tokens"],
        "input_cache_hit_tokens": totals["input_cache_hit_tokens"],
        "input_cache_write_tokens": totals["input_cache_write_tokens"],
        "prompt_tokens": totals["input_cache_miss_tokens"]
        + totals["input_cache_hit_tokens"],
        "completion_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "cost": totals["cost"],
        "agent_steps": int(getattr(agent, "step_number", 0) or 0),
        "task_completed": task_completed,
        "failure": failure,
        "tool_latency_ms": max(0, e2e_wall_clock_ms - totals["model_latency_ms"]),
        "usage_tracker_tokens": _tracker_tokens(agent, task_id),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    finalize_task_outcome(row, terminations)
    (output_dir / "result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/arms/offload-active.yaml",
        help="YAML holding the exact model configuration for both sides.",
    )
    parser.add_argument("--model", choices=_MODEL_KINDS, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-type", default="react_code")
    parser.add_argument("--step-limit", type=int, default=40)
    parser.add_argument("--dataset-name", default="train")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="AppWorld experiment name (defaults to native-<model>-<task-id>).",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("MODEL_SERVER_URL", "")
    load_env_file()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("config must be a YAML mapping")
    config = _prepare_runner_config(raw)
    experiment_name = args.experiment_name or f"native-{args.model}-{args.task_id}"

    row = run_native_task(
        raw_config=config,
        model_kind=args.model,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
        step_limit=args.step_limit,
        agent_type=args.agent_type,
        experiment_name=experiment_name,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
