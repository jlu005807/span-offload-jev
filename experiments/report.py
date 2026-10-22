"""把跑批结果汇总成论文主表。

输入是跑批目录，两种布局都认：

    <root>/<arm>/r<k>/appworld-<mode>-summary.json      TraceOffload 系（一轮一个文件）
    <root>/<arm>/round-<k>/<task>/result.json           native 系（一任务一个文件）

输出列（与最终实验表一致）：

    arm | n | 成功率 | 卸载率 | 总 token | 每次调用 | 模型延迟

三个口径必须先说清楚，否则数对不上
--------------------------------
* **卸载率 = 交给小模型的调用数 / 总调用数**，读的是
  framework_model_calls_by_target（实际发生的派发）。
  不能从预测器的 p 值反推：shadow 臂上预测器照算，但执行一律走大模型，
  按 p 值算会把 shadow 报成 21%，而它实际是 0%。
* **每次调用 = 模型调用数 / 任务数**（平均一个任务要调多少次模型）。
* **模型延迟 = 每个任务的模型调用总耗时**（不是单次延迟）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

SUMMARY_GLOB = "appworld-*-summary.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        return [data] if data.get("instance_id") else []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def collect(arm_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(arm_dir.glob(f"r*/{SUMMARY_GLOB}")):
        rows.extend(_load_rows(summary))
    for result in sorted(arm_dir.glob("round-*/*/result.json")):
        rows.extend(_load_rows(result))
    return rows


def calls_of(row: dict[str, Any]) -> int:
    return int(row.get("framework_model_calls") or row.get("model_calls") or 0)


def latency_of(row: dict[str, Any]) -> float | None:
    value = row.get("framework_model_latency_ms", row.get("model_latency_ms"))
    return float(value) if isinstance(value, (int, float)) else None


def routed_calls(row: dict[str, Any]) -> tuple[int, int] | None:
    """(给小模型的调用数, 总调用数)。实际派发，不是预测器的意图。"""
    by_target = row.get("framework_model_calls_by_target")
    if not isinstance(by_target, dict) or not by_target:
        # ``framework_model_calls`` is a total over all calls and is not enough
        # to identify which model actually received them.  In particular, a
        # Jev intent or a shadow decision is not a dispatch count.
        return None
    counts: dict[str, int] = {}
    for target, value in by_target.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value < 0 or int(value) != value:
            return None
        counts[str(target)] = int(value)
    small = counts.get("small_model", counts.get("small", 0))
    return small, sum(counts.values())


def predicted_offloads(row: dict[str, Any]) -> int:
    """没有实际派发计数时，用预测器分数反推（只对 classifier_route 有意义）。"""
    block = row.get("l2_add4")
    if block:
        p_gate = block.get("p_gate_values") or []
        p_net = block.get("p_net_values") or []
        tg, tn = block.get("tau_gate"), block.get("tau_net")
        if tg is None or tn is None:
            return 0
        return sum(1 for a, b in zip(p_gate, p_net) if a >= tg and b >= tn)
    return 0


_JEV_SUM_FIELDS = (
    "decisions", "http_attempts", "http_successes", "http_errors",
    "input_unavailable", "budget_unavailable", "slm_intent", "llm_intent",
    "client_total_ms", "state_chars_total", "audit_errors",
)


def _optional_sum(rows: list[dict[str, Any]], key: str) -> int | float | None:
    values = [
        value for row in rows
        if isinstance(row.get("jev"), dict)
        for value in [row["jev"].get(key)]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(values) if values else None


def _aggregate_jev(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    jev_rows = [row["jev"] for row in rows if isinstance(row.get("jev"), dict)]
    if not jev_rows:
        return None
    result: dict[str, Any] = {}
    for key in _JEV_SUM_FIELDS:
        values = [value for row in jev_rows for value in [row.get(key)]
                  if isinstance(value, (int, float)) and not isinstance(value, bool)]
        result[key] = sum(values) if values else None
    for key in ("mode", "input_mode"):
        values = {row.get(key) for row in jev_rows if row.get(key) is not None}
        result[key] = next(iter(values)) if len(values) == 1 else None
    audits = [row.get("routing_audit_complete") for row in jev_rows
              if isinstance(row.get("routing_audit_complete"), bool)]
    result["routing_audit_complete"] = all(audits) if audits else None
    attempts = result["http_attempts"]
    result["http_failure_rate"] = (
        result["http_errors"] / attempts
        if isinstance(attempts, (int, float)) and attempts > 0
        and isinstance(result["http_errors"], (int, float)) else None
    )
    return result


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    if not n:
        return {"n": 0}
    success = sum(1 for r in rows if r.get("task_success")) / n
    tokens = sum(float(r.get("total_tokens") or 0) for r in rows)
    calls = sum(calls_of(r) for r in rows)
    latency = [x for r in rows if (x := latency_of(r)) is not None]
    off = total_routed = 0
    offload_missing_rows = 0
    for r in rows:
        routed = routed_calls(r)
        if routed is None:
            offload_missing_rows += 1
        else:
            off += routed[0]
            total_routed += routed[1]
    jev = _aggregate_jev(rows)
    jev_latency = jev.get("client_total_ms") if jev else None
    jev_failure_rate = jev.get("http_failure_rate") if jev else None
    return {
        "n": n,
        "success": success,
        "tokens": tokens,
        "calls_per_run": calls / n,
        "model_latency_s": (sum(latency) / len(latency) / 1000.0) if latency else float("nan"),
        "offload": (off / total_routed) if total_routed else None,
        "offload_detail": f"{off}/{total_routed}" if total_routed else None,
        "offload_missing_rows": offload_missing_rows,
        "jev": jev,
        "jev_routing_latency_ms": jev_latency,
        "jev_http_failure_rate": jev_failure_rate,
        # No cost model is part of the Jev audit contract.  Keep this explicit
        # so consumers do not mistake a missing price for zero cost.
        "cost": None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="跑批根目录（其下每个子目录是一个臂）；可多次给")
    ap.add_argument("--arm", action="append", default=None, help="只报这些臂（默认全部）")
    ap.add_argument("--json", default=None, help="可选：把汇总写成 json")
    args = ap.parse_args(argv)

    arms: list[tuple[str, Path]] = []
    for root in args.root:
        rp = Path(root)
        if (rp / "r1").is_dir() or any(rp.glob("round-*")):
            arms.append((rp.name, rp))
            continue
        arms.extend((c.name, c) for c in sorted(p for p in rp.iterdir() if p.is_dir()))
    if args.arm:
        wanted = set(args.arm)
        arms = [(n, p) for n, p in arms if n in wanted]

    header = f"{'arm':<24}{'n':>6}{'success':>9}{'offload':>9}{'tokens':>10}{'calls/run':>11}{'latency':>10}{'jev ms':>10}{'jev err':>10}"
    print(header)
    print("-" * len(header))
    out: dict[str, Any] = {}
    for name, path in arms:
        s = summarize(collect(path))
        out[name] = s
        if not s["n"]:
            continue
        offload_text = f"{100*s['offload']:.1f}%" if s["offload"] is not None else "missing"
        jev_ms = s["jev_routing_latency_ms"]
        jev_err = s["jev_http_failure_rate"]
        jev_ms_text = f"{jev_ms:.1f}" if jev_ms is not None else "missing"
        jev_err_text = f"{100*jev_err:.1f}%" if jev_err is not None else "missing"
        print(
            f"{name:<24}{s['n']:>6}{100*s['success']:>8.1f}%{offload_text:>9}"
            f"{s['tokens']/1e6:>9.1f}M{s['calls_per_run']:>11.1f}{s['model_latency_s']:>9.1f}s"
            f"{jev_ms_text:>10}{jev_err_text:>10}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
