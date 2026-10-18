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
    by_target = row.get("framework_model_calls_by_target") or {}
    if by_target:
        small = int(by_target.get("small_model") or 0)
        large = int(by_target.get("large_model") or 0)
        return small, small + large
    kind = row.get("model_kind")
    if kind:
        total = calls_of(row)
        return (total if str(kind).lower().startswith("small") else 0), total
    return None


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


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    if not n:
        return {"n": 0}
    success = sum(1 for r in rows if r.get("task_success")) / n
    tokens = sum(float(r.get("total_tokens") or 0) for r in rows)
    calls = sum(calls_of(r) for r in rows)
    latency = [x for r in rows if (x := latency_of(r)) is not None]
    off = 0
    for r in rows:
        routed = routed_calls(r)
        if routed is not None:
            off += routed[0]
        else:
            off += predicted_offloads(r)
    return {
        "n": n,
        "success": success,
        "tokens": tokens,
        "calls_per_run": calls / n,
        "model_latency_s": (sum(latency) / len(latency) / 1000.0) if latency else float("nan"),
        "offload": (off / calls) if calls else 0.0,
        "offload_detail": f"{off}/{calls}",
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

    header = f"{'arm':<24}{'n':>6}{'success':>9}{'offload':>9}{'tokens':>10}{'calls/run':>11}{'latency':>10}"
    print(header)
    print("-" * len(header))
    out: dict[str, Any] = {}
    for name, path in arms:
        s = summarize(collect(path))
        out[name] = s
        if not s["n"]:
            continue
        print(
            f"{name:<24}{s['n']:>6}{100*s['success']:>8.1f}%{100*s['offload']:>8.1f}%"
            f"{s['tokens']/1e6:>9.1f}M{s['calls_per_run']:>11.1f}{s['model_latency_s']:>9.1f}s"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
