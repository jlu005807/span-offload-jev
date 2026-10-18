"""动作文本（C_bare_w10）：最近 10 个动作的规范化签名，用于第 4 段嵌入。

渲染规则（与线上 SpanActionsTextBuilder 同一口径）：
    每次工具执行 -> signature_inputs(代码).signature；一次执行里有多个调用则用 "|" 连接
    整条前缀的全部动作按时间顺序拼接，只取最后 window 个
    风格 bare：不带 [ACT] 标记、不带 category/effect（因此不依赖 AppWorld 的 OpenAPI 表）

一个 cell = 一个 (task_id, k)，来自 runs/<split>/<task>/original/snapshots/k####/prefix_trace.json。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from traceoffload.m1.constants import FeatureConstants
from traceoffload.m1.signature import signature_inputs

#: C_bare_w10 的窗口。
WINDOW = 10


@dataclass(frozen=True)
class CellRef:
    task_id: str
    k: int
    root: Path

    @property
    def cell_id(self) -> str:
        return f"{self.task_id}@k{self.k}"

    @property
    def prefix_path(self) -> Path:
        return self.root / self.task_id / "original" / "snapshots" / f"k{self.k:04d}" / "prefix_trace.json"


def discover_cells(root: Path, *, tasks: Sequence[str] | None = None) -> list[CellRef]:
    """扫出一个数据根下的全部 cell；只认带 prefix_trace.json 的快照目录。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    wanted = set(tasks) if tasks else None
    cells: list[CellRef] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_dir.name.startswith("_"):
            continue
        if wanted is not None and task_dir.name not in wanted:
            continue
        snap_root = task_dir / "original" / "snapshots"
        if not snap_root.is_dir():
            continue
        for snap in sorted(snap_root.glob("k*")):
            if not (snap / "prefix_trace.json").exists():
                continue
            try:
                k = int(snap.name[1:])
            except ValueError:
                continue
            cells.append(CellRef(task_id=task_dir.name, k=k, root=root))
    return cells


def _code_of(tool_log: dict[str, Any]) -> str:
    args = tool_log.get("input_arguments")
    if isinstance(args, dict):
        return str(args.get("code") or "")
    if isinstance(args, str):
        return args
    return ""


def action_signatures(prefix_trace: dict[str, Any], constants: FeatureConstants = FeatureConstants()) -> list[str]:
    """按时间顺序返回前缀里全部动作的签名（空签名渲染成 <unparsed>）。"""
    out: list[str] = []
    for step in prefix_trace.get("prefix_steps") or []:
        logs = sorted(
            (step.get("tool_logs") or []),
            key=lambda item: int(item.get("execution_order") or 0),
        )
        for log in logs:
            sig = signature_inputs(_code_of(log), constants).signature
            out.append(sig if sig else "<unparsed>")
    return out


def render(signatures: Iterable[str], window: int = WINDOW) -> str:
    seq = list(signatures)
    return "\n".join(seq[-window:])


def render_cell(cell: CellRef, constants: FeatureConstants = FeatureConstants(), window: int = WINDOW) -> str:
    with cell.prefix_path.open(encoding="utf-8") as fh:
        prefix = json.load(fh)
    return render(action_signatures(prefix, constants), window)


def build(roots: Sequence[Path], out_dir: Path, *, window: int = WINDOW) -> list[dict[str, Any]]:
    """遍历全部 roots，写出 actions_texts.jsonl（按 cell_id 排序，与历史 cell_ids.npz 同序）。"""
    cells: list[CellRef] = []
    for root in roots:
        cells.extend(discover_cells(Path(root)))
    cells.sort(key=lambda c: c.cell_id)
    rows = [
        {"cell_id": c.cell_id, "text": render_cell(c, window=window)}
        for c in cells
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "actions_texts.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
