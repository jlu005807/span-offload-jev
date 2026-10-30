"""Isolated AppWorld router comparisons with smoke gates and durable progress."""
from collections import Counter
from copy import deepcopy
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import yaml

GROUPS = ("native", "legacy", "jev08", "jev07")
HOLDOUT_GROUPS = ("native", "legacy", "jev050", "jev065", "jev070", "jev075")
THRESHOLDS = {"jev08": .8, "jev07": .7, "jev050": .5, "jev055": .55, "jev060": .6,
              "jev065": .65, "jev070": .7, "jev075": .75,
              "jev_v1_t50": .5, "jev_v1_t60": .6, "jev_v1_t70": .7}
V1_PROFILES = {
    "v1-t50": ("jev_v1_t50", "V1-T50"),
    "v1-t60": ("jev_v1_t60", "V1-T60"),
    "v1-t70": ("jev_v1_t70", "V1-T70"),
}
V1_COST_PER_MILLION = {
    "large_model": {"in": .35, "out": 2.55},
    "small_model": {"in": .10, "out": .15},
}


def active_groups(profile=None):
    profile = profile or os.environ.get("JEV_COMPARE_PROFILE", "baseline")
    if profile == "baseline":
        return GROUPS
    if profile == "thresholds":
        return ("jev055", "jev060", "jev065", "jev075")
    if profile == "holdout":
        return HOLDOUT_GROUPS
    if profile in V1_PROFILES:
        return (V1_PROFILES[profile][0],)
    raise ValueError("Unknown JEV_COMPARE_PROFILE: " + profile)



def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_configs(raw, *, groups=None):
    configs = {}
    for group in groups or active_groups():
        config = deepcopy(raw)
        runner = config.setdefault("runner", {})
        runner.setdefault("m4_memory", {})["enabled"] = False
        runner.setdefault("l2_v4", {})["enabled"] = False
        runner["shadow_route"] = {"enabled": False}
        legacy = runner.setdefault("l2_add4", {})
        legacy["enabled"] = group == "legacy"
        legacy.pop("tau_gate", None)
        legacy.pop("tau_net", None)
        if group.startswith("jev"):
            runner["jev"].update(enabled=True, mode="active", input_mode="context",
                                 threshold=THRESHOLDS[group])
            if group in {item[0] for item in V1_PROFILES.values()}:
                runner.setdefault("m4_memory", {})["guard_enabled"] = True
        else:
            runner["jev"] = {"enabled": False}
        configs[group] = config
    return configs


def normalize_native(row):
    row = deepcopy(row)
    calls = row.get("model_calls")
    if type(calls) is int and calls >= 0:
        row["framework_model_calls_by_target"] = {"large_model": calls}
        if all(type(row.get(k)) is int for k in ["prompt_tokens", "completion_tokens"]):
            row["framework_tokens_by_target_io"] = {"large_model": {
                "in": row["prompt_tokens"], "out": row["completion_tokens"]}}
    return row


def acceptance_errors(group, row):
    errors = []
    if row.get("consumption_complete") is False:
        errors.append("interrupted_consumption")
    if type(row.get("task_success")) is not bool:
        errors.append("missing_boolean_outcome")
    counts = row.get("framework_model_calls_by_target")
    if (not isinstance(counts, dict) or not counts or
            any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) == 0):
        errors.append("missing_actual_dispatch")
    if group == "native" and isinstance(counts, dict) and counts.get("small_model", 0):
        errors.append("native_used_small_model")
    if group == "legacy":
        block = row.get("l2_add4") or {}
        if not block.get("hits") or block.get("miss_no_vector") != 0:
            errors.append("legacy_vector_or_predictor_unavailable")
    if group.startswith("jev"):
        block = row.get("jev") or {}
        if (not block.get("http_successes") or block.get("routing_audit_complete") is not True or
                block.get("input_mode") != "context" or block.get("threshold") != (THRESHOLDS[group])):
            errors.append("jev_instrumentation_unavailable")
        if group in {item[0] for item in V1_PROFILES.values()}:
            audit = row.get("jev_audit") or {}
            if audit.get("prompt_sha256") != _context_prompt_sha256():
                errors.append("jev_prompt_hash_mismatch")
    return errors


def _estimated_execution_cost_usd(row):
    by_target = row.get("framework_tokens_by_target_io") or {}
    if not by_target:
        return None
    total = 0.0
    for target, prices in V1_COST_PER_MILLION.items():
        io = by_target.get(target) or {}
        if io and any(type(io.get(kind)) is not int or io[kind] < 0 for kind in prices):
            return None
        total += sum((io.get(kind, 0) or 0) * price
                     for kind, price in prices.items()) / 1_000_000
    return round(total, 10)


def _context_prompt_sha256():
    from traceoffload.m2.jev.input_builder import (
        ROUTE_CONTEXT_INSTRUCTIONS_V3,
        ROUTE_CRITERIA_V2,
    )
    body = {"instructions": ROUTE_CONTEXT_INSTRUCTIONS_V3,
            "criteria": dict(ROUTE_CRITERIA_V2)}
    return sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")).hexdigest()


def summarize(rows, *, groups=None, profile=None):
    profile = profile or os.environ.get("JEV_COMPARE_PROFILE", "baseline")
    group_names = tuple(groups or active_groups(profile))
    group_results = {}
    native = {r["instance_id"]: r["task_success"] for r in rows if r["group"] == "native"}
    for group_name in group_names:
        items = [r for r in rows if r["group"] == group_name]
        calls, tokens, mechanisms = Counter(), {"large_model": Counter(), "small_model": Counter()}, Counter()
        fallback, embedding = Counter(), Counter()
        for r in items:
            calls.update(r.get("framework_model_calls_by_target") or {})
            for target, io in (r.get("framework_tokens_by_target_io") or {}).items():
                tokens.setdefault(target, Counter()).update(io)
            mechanisms.update({k: v for k, v in (r.get("mechanism") or {}).items() if type(v) is int})
            fallback.update((r.get("mechanism") or {}).get("fallback_by_reason", {}))
            embedding.update(r.get("embedding_usage") or {})
        paired = [r for r in items if r["instance_id"] in native]
        incomplete = any(r.get("consumption_complete") is False or r.get("task_failure_reason") in
                         {"task_timeout", "worker_error"} for r in items)
        group_results[group_name] = {"tasks": len(items), "successes": sum(r["task_success"] is True for r in items),
            "success_rate": sum(r["task_success"] is True for r in items)/len(items) if items else None,
            "failures": dict(Counter(r.get("task_failure_reason") or "evaluation_failed" for r in items if not r["task_success"])),
            "actual_calls": dict(calls), "actual_slm_rate": calls["small_model"]/sum(calls.values()) if sum(calls.values()) and not incomplete else None,
            "consumption_complete": not incomplete, "observed_totals_are_lower_bounds": incomplete,
            "tasks_with_call_counts": sum(bool(r.get("framework_model_calls_by_target")) for r in items),
            "execution_tokens_by_target_io": {k: dict(v) for k, v in tokens.items()},
            "jev_input_tokens_known": sum((r.get("jev_audit") or {}).get("input_tokens_known", 0) for r in items),
            "jev_known_usage_responses": sum((r.get("jev_audit") or {}).get("known_usage_responses", 0) for r in items),
            "jev_successes_without_usage": sum((r.get("jev_audit") or {}).get("successful_responses_without_usage", 0) for r in items),
            "jev_attempted_preflight_input_tokens": sum((r.get("jev_audit") or {}).get("attempted_preflight_input_tokens", 0) for r in items),
            "jev_failed_attempt_preflight_input_tokens": sum((r.get("jev_audit") or {}).get("failed_attempt_preflight_input_tokens", 0) for r in items),
            "mechanism": dict(mechanisms) if group_name != "native" else None,
            "fallback_by_reason": dict(fallback) if group_name != "native" else None,
            "embedding_usage": dict(embedding) if group_name == "legacy" else None,
            "execution_model_latency_ms": sum(r.get("framework_model_latency_ms", r.get("model_latency_ms")) or 0 for r in items),
            "jev_client_total_ms": sum((r.get("jev") or {}).get("client_total_ms", 0) for r in items),
            "wall_clock_ms": sum(r.get("e2e_wall_clock_ms") or 0 for r in items),
            "worker_wall_clock_ms": sum(r.get("worker_wall_clock_ms") or 0 for r in items),
            "estimated_execution_cost_usd": (
                round(sum(r["estimated_execution_cost_usd"] for r in items), 10)
                if items and all(type(r.get("estimated_execution_cost_usd")) in (int, float)
                                 for r in items) else None
            ) if group_name in {item[0] for item in V1_PROFILES.values()} else None,
            "tasks_with_known_cost": sum(type(r.get("estimated_execution_cost_usd")) in (int, float)
                                         for r in items),
            "invalid_measurement_tasks": sum(r.get("valid_for_comparison") is False for r in items),
            "paired_with_native": {"tasks": len(paired),
                "both_success": sum(r["task_success"] and native[r["instance_id"]] for r in paired),
                "native_only_success": sum(not r["task_success"] and native[r["instance_id"]] for r in paired),
                "group_only_success": sum(r["task_success"] and not native[r["instance_id"]] for r in paired)},
            "cost": None}
    return {"groups": group_results, "experiment_profile": profile,
            "development_experiment": profile != "holdout",
            "holdout_evaluation": profile == "holdout", "fee_model": None}


def check_services(config):
    import httpx
    with httpx.Client(trust_env=False, timeout=10) as client:
        endpoints = [(config["models"][kind]["api_base"].rstrip("/") + "/models",
                      config["models"][kind].get("served_model_name") or config["models"][kind]["model_id"])
                     for kind in ["large", "small"]]
        j = config["runner"]["jev"]
        endpoints.append((j["endpoint"] + "/v1/models", j["model_id"]))
        for url, model in endpoints:
            response = client.get(url)
            response.raise_for_status()
            if model not in {x.get("id") for x in response.json().get("data", [])}:
                raise RuntimeError(f"Expected model {model} unavailable at {url}")
        url = config["runner"]["m4_memory"]["ollama_base_url"].rstrip("/") + "/api/tags"
        response = client.get(url)
        response.raise_for_status()
        if "qwen3-embedding:0.6b" not in {x.get("name") for x in response.json().get("models", [])}:
            raise RuntimeError("Original embedding model unavailable")


def execute_cell(group, config, task, output, experiment, split):
    env = os.environ.copy()
    env.update(JEV_COMPARE_ACTION="worker", JEV_COMPARE_GROUP=group, JEV_COMPARE_CONFIG=str(config),
               JEV_COMPARE_TASK=task, JEV_COMPARE_CELL=str(output), JEV_COMPARE_EXPERIMENT=experiment,
               JEV_COMPARE_SPLIT=split)
    output.mkdir(parents=True)
    started = perf_counter()
    with (output / "process.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", "-m", "experiments.compare_routers"],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=int(env.get("JEV_COMPARE_TASK_TIMEOUT", "900")))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return recover_interrupted(output, task, "task_timeout", round((perf_counter()-started)*1000))
    path = output / "result.json"
    if code != 0 or not path.exists():
        return {**recover_interrupted(output, task, "worker_error", round((perf_counter()-started)*1000)),
                "worker_exit_code": code}
    row = json.loads(path.read_text(encoding="utf-8"))
    row["worker_wall_clock_ms"] = round((perf_counter()-started)*1000)
    return row


def run_comparison(raw, tasks, output, *, execute=execute_cell, health=None, task_splits=None,
                   smoke_task=None):
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Task list must be nonempty and unique")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    profile = os.environ.get("JEV_COMPARE_PROFILE", "baseline")
    groups = active_groups(profile)
    is_v1_profile = profile in V1_PROFILES
    if is_v1_profile and len(tasks) != 33:
        raise ValueError("V1 threshold runs require exactly 33 validation tasks")
    configs = build_configs(raw, groups=groups)
    health = health or check_services
    smoke_task = smoke_task or (tasks[0] if is_v1_profile else "099174d_1")
    if task_splits is None or any(t not in task_splits for t in [*tasks, smoke_task]):
        raise ValueError("Official dataset split required for every task")
    if is_v1_profile and (any(task_splits[t] != "dev" for t in tasks)
                          or task_splits[smoke_task] != "dev"):
        raise ValueError("V1 threshold runs may use only AppWorld dev tasks")
    output.mkdir(parents=True)
    config_paths, hashes = {}, {}
    for group, config in configs.items():
        f = output / (group + ".yaml")
        f.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config_paths[group] = f
        hashes[group] = sha256(f.read_bytes()).hexdigest()
    run_id = uuid4().hex
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()
    write_json(output / "manifest.json", {"task_ids": tasks, "task_splits": task_splits,
        "groups": groups, "experiment_profile": profile,
        "experiment_group_id": V1_PROFILES[profile][1] if is_v1_profile else None,
        "run_id": run_id, "git_commit_sha": commit_sha,
        "prompt_sha256": _context_prompt_sha256() if is_v1_profile else None,
        "price_table_version": "openrouter-2026-10" if is_v1_profile else None,
        "cost_basis": ("Execution-model estimate from token counts at openrouter-2026-10 list prices; "
                       "no cache discount; excludes Jev service cost") if is_v1_profile else None,
        "holdout_evaluation": profile == "holdout", "config_sha256": hashes,
        "smoke_task_id": smoke_task, "smoke_excluded_from_formal_results": True,
        "order": "single threshold group, manifest order",
        "development_experiment": profile != "holdout"})
    rows, smoke, completed = [], [], set()
    def run(stage, task, group):
        row = execute(group, config_paths[group], task, output/stage/group/task,
                      f"routercmp-{run_id}-{stage}-{group}-{task}", task_splits[task])
        if row.get("instance_id") != task:
            raise RuntimeError("Worker returned wrong task identity")
        errors = acceptance_errors(group, row)
        row.update(group=group, stage=stage, run_id=run_id, git_commit_sha=commit_sha,
                   valid_for_comparison=not errors,
                   measurement_errors=errors, config_sha256=hashes[group])
        if is_v1_profile:
            row["experiment_group_id"] = V1_PROFILES[profile][1]
            row["prompt_sha256"] = (row.get("jev_audit") or {}).get("prompt_sha256")
            row["price_table_version"] = "openrouter-2026-10"
            row["estimated_execution_cost_usd"] = _estimated_execution_cost_usd(row)
        return row
    try:
        health(configs[next(g for g in configs if g.startswith("jev"))])
        for group in groups:
            smoke.append(run("smoke", smoke_task, group))
            write_json(output / "smoke-results.json", smoke)
        errors = {r["group"]: r["measurement_errors"] for r in smoke if r["measurement_errors"]}
        write_json(output / "acceptance.json", {"ready_for_batch": not errors, "errors": errors,
                                               "experiment_profile": profile,
                                               "config_sha256": hashes})
        if errors:
            raise RuntimeError("Router comparison integration acceptance failed")
        for task_index, task in enumerate(tasks):
            health(configs[next(g for g in configs if g.startswith("jev"))])
            offset = task_index % len(groups)
            order = groups[offset:] + groups[:offset]
            for group in order:
                row = run("batch", task, group)
                rows.append(row)
                completed.add((task, group))
                write_json(output / "results.json", rows)
                write_json(output / "report.json", summarize(rows, groups=groups, profile=profile))
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"completed": len(rows), "total": len(groups)*len(tasks),
                        "task_id": task, "group": group, "success": row["task_success"]}) + "\n")
                print(f"CELL_DONE {len(rows)}/{len(groups)*len(tasks)} {group} {task} success={row['task_success']}", flush=True)
                if row["measurement_errors"] and row.get("task_failure_reason") != "task_timeout":
                    raise RuntimeError("Route instrumentation failed; preserving row and stopping")
    except Exception as error:
        write_json(output / "batch-error.json", {"type": type(error).__name__, "completed": len(rows),
            "remaining": [{"task_id": t, "group": g} for t in tasks for g in groups if (t, g) not in completed]})
        raise
    report = summarize(rows, groups=groups, profile=profile)
    report["complete"] = len(rows) == len(groups)*len(tasks)
    write_json(output / "report.json", report)
    print("COMPARISON_COMPLETED", len(rows), flush=True)
    return report


def audit_usage(output, *, tolerate_partial=False):
    count, tokens, probabilities = 0, 0, []
    prompt_hashes = set()
    attempted_tokens = failed_tokens = missing_success_usage = unknown_attempts = partial_lines = 0
    for path in output.glob("routing-decisions-*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    r = json.loads(line)
                except ValueError:
                    if not tolerate_partial:
                        raise
                    partial_lines += 1
                    continue
                if r.get("http_attempted") is True:
                    preflight = (r.get("input_metadata") or {}).get("input_tokens")
                    if type(preflight) is int and preflight > 0:
                        attempted_tokens += preflight
                        if r.get("status") != "ok":
                            failed_tokens += preflight
                    else:
                        unknown_attempts += 1
                prompt_hash = (r.get("input_metadata") or {}).get("prompt_sha256")
                if isinstance(prompt_hash, str) and prompt_hash:
                    prompt_hashes.add(prompt_hash)
                if r.get("status") == "ok":
                    response = r["response"]
                    usage = response.get("usage", {}).get("input_tokens")
                    if type(usage) is int:
                        count += 1
                        tokens += usage
                    else:
                        missing_success_usage += 1
                    probabilities.append(response["answers"]["route"]["probabilities"]["SLM"])
    return {"input_tokens_known": tokens, "known_usage_responses": count,
            "successful_responses_without_usage": missing_success_usage,
            "attempted_preflight_input_tokens": attempted_tokens,
            "failed_attempt_preflight_input_tokens": failed_tokens,
            "attempts_without_preflight_count": unknown_attempts, "partial_lines": partial_lines,
            "successful_responses": len(probabilities),
            "mean_slm_probability": sum(probabilities)/len(probabilities) if probabilities else None,
            "prompt_sha256": next(iter(prompt_hashes)) if len(prompt_hashes) == 1 else None,
            "prompt_hashes": sorted(prompt_hashes)}


def measure_embedding(embedder, *, checkpoint=None):
    """Observe one batch once; never replace or alter embedding vectors."""
    stats = {"requests": 0, "texts": 0, "input_tokens_known": 0,
             "known_usage_requests": 0, "errors": 0}
    original = embedder.embed_many
    def embed_many(texts):
        stats["requests"] += 1
        stats["texts"] += len(texts)
        try:
            vectors = original(texts)
        except Exception:
            stats["errors"] += 1
            if checkpoint is not None:
                write_json(checkpoint, stats)
            raise
        usage = vectors[0].prompt_eval_count if vectors else None
        if type(usage) is int:
            stats["input_tokens_known"] += usage
            stats["known_usage_requests"] += 1
        if checkpoint is not None:
            write_json(checkpoint, stats)
        return vectors
    embedder.embed_many = embed_many
    return stats


def recover_interrupted(output, task, reason, elapsed_ms):
    row = {}
    for name, key in [("partial-execution.json", None), ("partial-embedding.json", "embedding_usage")]:
        path = output/name
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if key:
                row[key] = saved
            else:
                row.update(saved)
    if list(output.glob("routing-decisions-*.jsonl")):
        row["jev_audit"] = audit_usage(output, tolerate_partial=True)
    row.update(instance_id=task, task_success=False, task_failure_reason=reason,
               consumption_complete=False, worker_wall_clock_ms=elapsed_ms)
    write_json(output/"result.json", row)
    return row


def checkpoint_meter_type(base, checkpoint):
    class CheckpointMeter(base):
        def complete(self, *args, **kwargs):
            try:
                return super().complete(*args, **kwargs)
            finally:
                if self.calls:
                    write_json(checkpoint, self.summary())
    return CheckpointMeter


def native_measurement_with_checkpoints(original, checkpoint):
    def install(model, totals):
        original(model, totals)
        generate = model.generate
        @wraps(generate)
        def observed(*args, **kwargs):
            try:
                return generate(*args, **kwargs)
            finally:
                write_json(checkpoint, normalize_native({"model_calls": totals["calls"],
                    "prompt_tokens": totals["input_cache_miss_tokens"] + totals["input_cache_hit_tokens"],
                    "completion_tokens": totals["output_tokens"], "total_tokens": totals["total_tokens"],
                    "model_latency_ms": totals["model_latency_ms"]}))
        model.generate = observed
    return install


def worker():
    raw = yaml.safe_load(Path(os.environ["JEV_COMPARE_CONFIG"]).read_bytes())
    group, task = os.environ["JEV_COMPARE_GROUP"], os.environ["JEV_COMPARE_TASK"]
    output = Path(os.environ["JEV_COMPARE_CELL"])
    name, split = os.environ["JEV_COMPARE_EXPERIMENT"], os.environ["JEV_COMPARE_SPLIT"]
    if group == "native":
        from experiments import run_native
        run_native._install_measurement = native_measurement_with_checkpoints(
            run_native._install_measurement, output/"partial-execution.json")
        row = normalize_native(run_native.run_native_task(raw_config=raw, model_kind="large", task_id=task,
            output_dir=output, step_limit=int(raw["agent"]["step_limit"]), agent_type="react_code",
            experiment_name=name, dataset_name=split))
    else:
        from experiments import run_appworld
        run_appworld._GatewayMeter = checkpoint_meter_type(run_appworld._GatewayMeter, output/"partial-execution.json")
        preflight = None
        if group.startswith("jev"):
            from deploy.jev.openjev_preflight import OpenJevPreflight
            preflight = OpenJevPreflight.from_directory(os.environ["JEV_MODEL_DIR"])
        runner = run_appworld.AppWorldPilotRunner(raw_config=raw, mode="classifier_route", agent_type="react_code",
            task_ids=[task], output_dir=output, experiment_name=name, jev_input_preflight=preflight)
        runner._appworld_agent_config["appworld_config"]["dataset_name"] = split
        runner._appworld_agent_config["logger_config"]["verbose"] = True
        embedding = measure_embedding(runner._m4_embedder, checkpoint=output/"partial-embedding.json") if group == "legacy" else None
        try:
            row = runner.run_task(task)
        finally:
            runner.close()
        if group.startswith("jev"):
            row["jev_audit"] = audit_usage(output)
        if embedding is not None:
            row["embedding_usage"] = embedding
    row["appworld_experiment_name"] = name
    write_json(output / "result.json", row)


def main():
    if os.environ.get("JEV_COMPARE_ACTION") == "worker":
        worker()
        return
    raw = yaml.safe_load(Path(os.environ["JEV_COMPARE_CONFIG"]).read_bytes())
    tasks = [s.strip() for s in Path(os.environ["JEV_COMPARE_TASKS"]).read_text(encoding="utf-8").splitlines() if s.strip()]
    from appworld.task import load_task_ids
    splits = {task: split for split in ["train", "dev", "test_normal", "test_challenge"] for task in load_task_ids(split)}
    run_comparison(raw, tasks, os.environ["JEV_COMPARE_OUTPUT"], task_splits=splits)


if __name__ == "__main__":
    main()
