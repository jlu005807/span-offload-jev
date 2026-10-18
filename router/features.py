"""第 (1)(2) 段特征 + V2_goal_recent 文本。

两个标量
--------
    prefix_n_A           前缀里分类为「写」的工具执行个数（AppWorld OpenAPI 的 HTTP method 判定）
    log_est_prompt_tokens = log1p(前缀消息总字符数 / 4)

V2_goal_recent（第 (3) 段嵌入的输入）
-----------------------------------
    [GOAL] <任务指令>

    [TOOLS] <上一步调用的 api>
    [REASONING] <思考尾部 800 字>
    [ASSISTANT] <输出尾部 1500 字>
    ...（最近 10 个 span）

去掉了工具输出（v1 里占 99% 的体量），这是 v2 配方的核心。目标文本只依赖
prefix_trace.json —— 不打开 trace.json、不打开 branches/，因此不可能泄漏第 k 步。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .actions import CellRef, discover_cells
from .messages import prefix_next_messages

# ---- v2 文本配方 ----
REASONING_TAIL_CHARS = 800
ASSISTANT_TAIL_CHARS = 1500
RECENT_WINDOW_K_V2 = 10

_OBS_PREFIX = "Output:"
_CALL_RE = re.compile(r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")

#: 写动作的动词前缀（OpenAPI 查不到时的回退；与历史实现逐字一致）。
_WRITE_VERBS = (
    "create", "update", "delete", "remove", "add", "send", "insert", "complete",
    "mark", "post", "put", "pay", "transfer", "cancel", "edit", "modify", "write",
    "upload", "like", "follow", "subscribe", "share", "comment", "rate", "set",
    "login", "logout", "signup", "react", "reply", "request", "accept", "reject",
    "revoke", "clear", "move", "rename", "make", "apply", "join", "leave",
    "enable", "disable", "reset", "archive", "pin", "save", "schedule", "assign",
    "register", "verify", "confirm", "submit", "publish", "approve", "deny",
    "block", "mute", "bookmark", "favorite", "star", "flag", "report", "resolve",
    "close", "reopen", "merge", "copy", "duplicate", "import", "export",
    "backup", "restore", "refresh", "reload", "sync", "withdraw", "deposit",
    "terminate", "unsubscribe", "unfollow", "unlike", "download",
)

#: OpenAPI 规格里的 operationId 数量（实测 473）。低于它就抛，不静默退化成启发式。
MIN_API_METHODS = 470


def _tail(s: str, n: int) -> str:
    return s if len(s) <= n else s[-n:]


def _msg_text(m: Mapping[str, Any]) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else ""


def load_api_methods(openapi_dir: Path) -> dict[str, str]:
    """operationId -> HTTP method（小写）。来源：AppWorld 的 OpenAPI 规格。"""
    out: dict[str, str] = {}
    path = Path(openapi_dir)
    if not path.is_dir():
        return out
    for f in sorted(path.glob("*.json")):
        try:
            spec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for _p, item in (spec.get("paths") or {}).items():
            if not isinstance(item, dict):
                continue
            for method, body in item.items():
                if isinstance(body, dict) and body.get("operationId"):
                    out[str(body["operationId"])] = method.lower()
    return out


def api_calls_of_code(code: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _CALL_RE.finditer(code or "")]


def _looks_write(app: str, api: str) -> bool:
    core = api.lower()
    return any(core.startswith(v) for v in _WRITE_VERBS)


def tool_event_kind(tool_log: Mapping[str, Any], api_methods: Mapping[str, str]) -> str:
    """一次工具执行的 PDF 事件符号：O（只读成功）/ A（写成功）/ F（失败）。"""
    status = str(tool_log.get("execution_status") or "").lower()
    rc = tool_log.get("returncode")
    ok = (status == "success") and (rc in (0, None))
    args = tool_log.get("input_arguments")
    code = str(args.get("code") or "") if isinstance(args, dict) else (args if isinstance(args, str) else "")
    calls = api_calls_of_code(code)
    if not ok:
        return "F"
    methods = []
    for app, api in calls:
        m = api_methods.get(f"{app}__{api}")
        methods.append(m if m is not None else ("post" if _looks_write(app, api) else "get"))
    return "A" if any(m != "get" for m in methods) else "O"


def _span_text_v2(step: Mapping[str, Any], api_methods: Mapping[str, str]) -> str:
    oc = step.get("output_content") or {}
    am = oc.get("assistant_message") or {}
    logs = sorted((step.get("tool_logs") or []), key=lambda x: int(x.get("execution_order", 0)))
    names: list[str] = []
    for lg in logs:
        args = lg.get("input_arguments")
        code = args.get("code") if isinstance(args, dict) else str(args)
        for a, b in api_calls_of_code(code or ""):
            names.append(f"{a}.{b}")
    tool_line = ", ".join(dict.fromkeys(names)) or "none"
    return (
        f"[TOOLS] {tool_line}\n"
        f"[REASONING] {_tail(oc.get('reasoning_content') or '', REASONING_TAIL_CHARS)}\n"
        f"[ASSISTANT] {_tail(am.get('content') or '', ASSISTANT_TAIL_CHARS)}"
    )


def build_cell(
    cell_id: str,
    prefix: Mapping[str, Any],
    api_methods: Mapping[str, str],
) -> dict[str, Any]:
    steps = prefix.get("prefix_steps") or []
    meta = prefix.get("trace_metadata") or {}
    msgs = prefix_next_messages(prefix)

    total_chars = sum(len(_msg_text(m)) for m in msgs)
    est_prompt_tokens = total_chars / 4.0
    n_write_ops = sum(
        1
        for s in steps
        for lg in (s.get("tool_logs") or [])
        if tool_event_kind(lg, api_methods) == "A"
    )
    text = (
        f"[GOAL] {str(meta.get('task_instruction') or '')}\n\n"
        + "\n\n".join(_span_text_v2(s, api_methods) for s in steps[-RECENT_WINDOW_K_V2:])
    )
    return {
        "cell_id": cell_id,
        "prefix_n_A": n_write_ops,
        "est_prompt_tokens": round(est_prompt_tokens, 2),
        "log_est_prompt_tokens": round(math.log1p(est_prompt_tokens), 6),
        "text": text,
    }


def build(roots: Sequence[Path], out_dir: Path, *, openapi_dir: Path, min_api_methods: int = MIN_API_METHODS) -> list[dict[str, Any]]:
    api_methods = load_api_methods(openapi_dir)
    if len(api_methods) < min_api_methods:
        raise RuntimeError(
            f"OpenAPI 只加载到 {len(api_methods)} 个 operationId（< {min_api_methods}）："
            f"检查 --openapi-dir={openapi_dir}。缺它会让写动作静默退化成启发式判定。"
        )
    cells: list[CellRef] = []
    for root in roots:
        cells.extend(discover_cells(Path(root)))
    cells.sort(key=lambda c: c.cell_id)
    rows = []
    for c in cells:
        with c.prefix_path.open(encoding="utf-8") as fh:
            prefix = json.load(fh)
        rows.append(build_cell(c.cell_id, prefix, api_methods))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "features.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n")
    with (out_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"cell_id": r["cell_id"], "text": r["text"]}, ensure_ascii=False) + "\n")
    return rows
