"""核验 M3 的两个真实现象：fallback 与 rollback。

数据源是每次跑批落盘的 span 记录（`<run>/classifier_route/history.sqlite3`），
不是自报计数器 —— summary 里那几个统计位（fallback_by_reason / checkpoints_created）
**从来没接线**，永远是 0，不能当证据。

判据
----
* fallback：`span_kind == "fallback"` 且 `span_relation == "fallback_of"`；
  它的父 span 必须是 `verification_failed` 且有 `post_check: reject`
* rollback：工具执行记录里的 `rollback_status`：
    `succeeded`  真的回滚了（该动作是 declared_write，建过 checkpoint）
    `failed`     回滚失败（会进 recovery）
    `not_needed` 没有可回滚的东西（只读 / 未知效应）

用法
----
    python3 experiments/verify_mechanisms.py runs/one_round/offload-active/r1
    python3 experiments/verify_mechanisms.py runs/one_round/*/r1
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def load_spans(db: Path) -> list[dict]:
    conn = sqlite3.connect(db)
    try:
        return [json.loads(row[0]) for row in conn.execute("select payload_json from span_records")]
    finally:
        conn.close()


def find_db(path: Path) -> Path | None:
    if path.is_file():
        return path
    for cand in sorted(path.glob("*/history.sqlite3")):
        return cand
    return None


def summarize(db: Path) -> dict:
    spans = load_spans(db)
    es = lambda d: d.get("execution_summary") or {}
    by_ep = {(es(d).get("workflow_run_id"), es(d).get("episode_id")): d for d in spans}

    kinds = Counter(d.get("span_kind") for d in spans)
    rollback = Counter()
    rejects = Counter()
    fallbacks = []
    for d in spans:
        summary = es(d)
        for g in summary.get("guard_results") or []:
            if g.get("outcome") == "reject":
                rejects[g.get("stage")] += 1
        for t in summary.get("tool_executions") or []:
            rollback[t.get("rollback_status")] += 1
        if d.get("span_kind") == "fallback":
            parent = by_ep.get((summary.get("workflow_run_id"), d.get("parent_episode_id")))
            fallbacks.append({
                "workflow": summary.get("workflow_run_id"),
                "parent_episode": d.get("parent_episode_id"),
                "parent_status": es(parent).get("status") if parent else None,
                "parent_guards": [(g.get("stage"), g.get("outcome"))
                                  for g in (es(parent).get("guard_results") or [])] if parent else [],
                "reason": ((d.get("route_plan") or {}).get("decision_meta") or {}).get("reason_code"),
                "parent_tools": [(t.get("effect_category"), t.get("status"), t.get("rollback_status"))
                                 for t in (es(parent).get("tool_executions") or [])] if parent else [],
            })
    return {"db": str(db), "spans": len(spans), "kinds": dict(kinds),
            "rejects": dict(rejects), "rollback": dict(rollback), "fallbacks": fallbacks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="跑批目录（含 */history.sqlite3）或直接给 sqlite 文件")
    ap.add_argument("--examples", type=int, default=1, help="每类现象打印几个例子")
    args = ap.parse_args(argv)

    for raw in args.paths:
        db = find_db(Path(raw))
        if db is None:
            print(f"[skip] {raw}: 没有 history.sqlite3")
            continue
        s = summarize(db)
        print(f"== {raw}")
        print(f"   spans {s['spans']}  kinds {s['kinds']}")
        print(f"   guard rejects {s['rejects'] or '{}'}   rollback {s['rollback'] or '{}'}")
        ok = [f for f in s["fallbacks"] if f["parent_status"] == "verification_failed"]
        print(f"   fallback {len(s['fallbacks'])}（父 span 为 verification_failed 的 {len(ok)}）")
        for f in s["fallbacks"][: args.examples]:
            print(f"     - {f['workflow']} {f['parent_episode']} -> {f['reason']}")
            print(f"       父 status={f['parent_status']} guards={f['parent_guards']} tools={f['parent_tools']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
