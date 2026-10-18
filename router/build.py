"""一步构建一个 split 的离线特征（不调模型，只读 runs/）。

产物（同一批 cell，行序一致）
--------------------------
    features.jsonl        cell_id / task_id / k / prefix_n_A / est_prompt_tokens / log_est_prompt_tokens
    texts.jsonl           cell_id / text                  -> router.embed texts
    actions_texts.jsonl   cell_id / text                  -> router.embed actions

用法
----
    python -m router.build --runs ../TraceOffload/runs/训练集 --out data/训练集 --openapi <appworld>/data/api_docs/openapi
    python -m router.build --runs ../TraceOffload/runs/验证集 --out data/验证集 --openapi <...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import actions, features


def build(runs_root: Path, out_dir: Path, openapi_dir: Path, *, tasks: list[str] | None = None) -> int:
    cells = actions.discover_cells(Path(runs_root), tasks=tasks)
    if not cells:
        raise SystemExit(f"{runs_root} 下没有找到任何 cell（需要 original/snapshots/k####/prefix_trace.json）")
    cells.sort(key=lambda c: c.cell_id)

    api_methods = features.load_api_methods(Path(openapi_dir))
    if len(api_methods) < features.MIN_API_METHODS:
        raise SystemExit(
            f"OpenAPI 只加载到 {len(api_methods)} 个 operationId（< {features.MIN_API_METHODS}）："
            f"检查 --openapi-dir={openapi_dir}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_feat = (out / "features.jsonl").open("w", encoding="utf-8")
    f_text = (out / "texts.jsonl").open("w", encoding="utf-8")
    f_act = (out / "actions_texts.jsonl").open("w", encoding="utf-8")
    try:
        for c in cells:
            prefix = json.loads(c.prefix_path.read_text(encoding="utf-8"))
            row = features.build_cell(c.cell_id, prefix, api_methods)
            f_feat.write(json.dumps({
                "cell_id": row["cell_id"],
                "task_id": c.task_id,
                "k": c.k,
                "prefix_n_A": row["prefix_n_A"],
                "est_prompt_tokens": row["est_prompt_tokens"],
                "log_est_prompt_tokens": row["log_est_prompt_tokens"],
            }, ensure_ascii=False) + "\n")
            f_text.write(json.dumps({"cell_id": row["cell_id"], "text": row["text"]}, ensure_ascii=False) + "\n")
            f_act.write(json.dumps({
                "cell_id": c.cell_id,
                "text": actions.render(actions.action_signatures(prefix)),
            }, ensure_ascii=False) + "\n")
    finally:
        for fh in (f_feat, f_text, f_act):
            fh.close()
    print(f"[build] {len(cells)} cell -> {out}")
    return len(cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="runs 下的一个 split（含 <task>/original/snapshots/...）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--openapi-dir", required=True,
                    help="AppWorld 的 data/api_docs/openapi（写动作判定要用它的 HTTP method）")
    ap.add_argument("--task", action="append", default=None, help="只处理指定任务（可多次）")
    args = ap.parse_args(argv)
    build(Path(args.runs), Path(args.out), Path(args.openapi_dir), tasks=args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
