"""从 prefix_trace.json 重建模型当时看到的完整 messages。

trace 里的 messages 有两种形状：
  * 全量形状：input_context.messages 直接就是完整列表
  * 增量形状：input_context.messages_delta 只存新增的尾部，并带 input_messages_sha256
增量形状必须重建并校验哈希 —— 否则拿到的是一条被截断的消息表，而且不会报错。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

TRACE_SCHEMA_VERSION = "spanoffload-trace-1.0"
MSG_KEY = "messages"
DELTA_KEY = "messages_delta"
HASH_KEY = "input_messages_sha256"


class NonAppendMessages(ValueError):
    """某一步的 messages 不是前一步的纯追加，不能做增量编码。"""


class MessagesHashMismatch(ValueError):
    """重建出来的 messages 与记录着的 input_messages_sha256 不符。"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _context(step: Mapping[str, Any]) -> Mapping[str, Any]:
    ctx = step.get("input_context")
    return ctx if isinstance(ctx, Mapping) else {}


def _initial_messages(trace: Mapping[str, Any]) -> list[Any]:
    meta = trace.get("trace_metadata")
    if not isinstance(meta, Mapping):
        return []
    initial = meta.get("initial_messages")
    return list(initial) if isinstance(initial, Sequence) and not isinstance(initial, str) else []


def _is_prefix(short: Sequence[Any], long: Sequence[Any]) -> bool:
    return len(short) <= len(long) and list(long[: len(short)]) == list(short)


def messages_delta(prev: Sequence[Any], cur: Sequence[Any]) -> list[Any]:
    prev_list, cur_list = list(prev or []), list(cur or [])
    if not _is_prefix(prev_list, cur_list):
        raise NonAppendMessages(
            f"messages 不是纯追加：前一步 {len(prev_list)} 条，这一步 {len(cur_list)} 条"
        )
    return cur_list[len(prev_list):]


def _decode(trace: Mapping[str, Any]) -> list[list[Any]]:
    steps = trace.get("steps") or []
    anchor = _initial_messages(trace)
    out: list[list[Any]] = []
    for index, step in enumerate(steps):
        ctx = _context(step)
        raw = ctx.get(MSG_KEY)
        delta = ctx.get(DELTA_KEY)
        if raw is not None:
            cur = list(raw)
        elif delta is not None:
            base = out[-1] if out else anchor
            cur = list(base) + list(delta)
            recorded = ctx.get(HASH_KEY)
            if recorded and sha256_json(cur) != recorded:
                raise MessagesHashMismatch(
                    f"step {step.get('step_id', index + 1)} 重建出的 messages 与 {HASH_KEY} 不符"
                )
        else:
            raise NonAppendMessages(
                f"step {step.get('step_id', index + 1)} 既没有 {MSG_KEY} 也没有 {DELTA_KEY}"
            )
        out.append(cur)
    return out


def prefix_next_messages(prefix: Mapping[str, Any]) -> list[Any]:
    """prefix_trace.json 的 next_input_context 对应的完整 messages。"""
    steps = prefix.get("prefix_steps") or []
    synthetic = {
        "schema_version": prefix.get("schema_version", TRACE_SCHEMA_VERSION),
        "trace_metadata": prefix.get("trace_metadata") or {},
        "steps": steps,
    }
    decoded = _decode(synthetic)
    ctx = prefix.get("next_input_context")
    ctx = ctx if isinstance(ctx, Mapping) else {}
    raw = ctx.get(MSG_KEY)
    if raw is not None:
        return list(raw)
    delta = ctx.get(DELTA_KEY)
    base = decoded[-1] if decoded else _initial_messages(synthetic)
    return list(base) + list(delta or [])
