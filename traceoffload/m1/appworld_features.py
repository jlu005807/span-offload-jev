"""AppWorld 在线分类器特征 (第一版最小闭环).

这版 builder 只使用 **当前 ModelRequest 的原生 messages + 当前 step_index**，
不读 spanOffload prefix_trace / branches / 任何未来信息。14 个特征与
``classfierTrain/out/appworld_classifier/cells.jsonl`` 的 ``num`` 中纯消息统计
一一对齐，因此离线导出的 JSON artifact 可以直接在 M1 阶段复用。

特征语义很弱是已知事实：这版目标是先把 AppWorld 分类器 artifact 加载、
``RuntimeProfile.online_features`` 填充、M2 路由决策这条链路打通；模型质量
提升应等标签和三臂数据扩充后替换 builder/artifact。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

APPWORLD_MESSAGE_FEATURES: tuple[str, ...] = (
    "k",
    "n_messages",
    "n_system",
    "n_user",
    "n_assistant",
    "n_tool",
    "total_chars",
    "mean_msg_chars",
    "max_msg_chars",
    "last_role_is_tool",
    "last_role_is_assistant",
    "last_role_is_user",
    "last_msg_chars",
    "last5_chars",
)

#: L2_new 预测器（``classfierTrain/04_预测器训练/l2_predictor.py``）的两列。离线口径见
#: ``classfierTrain/aw3_build.py`` 的 ``chars`` / ``total_chars`` / ``log_est_prompt_tokens``（:300-302, 482）：
#:   ``total_chars = sum(len(m["content"]) for m in messages if isinstance(content, str))``
#:   ``est_prompt_tokens = total_chars / 4.0``
#:   ``log_est_prompt_tokens = round(log1p(est_prompt_tokens), 6)``
#: **``total_chars`` 的口径必须与离线逐位一致**（差一点 p 就偏）。
APPWORLD_L2_FEATURES: tuple[str, ...] = (
    "est_prompt_tokens",
    "log_est_prompt_tokens",
)

#: ``prefix_n_A``：前缀里「**成功且是写**」的工具执行数。离线同义定义见
#: ``classfierTrain/aw3_build.py`` 的 ``tool_event_kind``（:121-148）——
#: ``ok = (status == "success") and (rc in (0, None))``；
#: ``is_write = any(method != "get")``；返回 ``"A"`` 当且仅当 ``ok and is_write``。
APPWORLD_PREFIX_FEATURES: tuple[str, ...] = ("prefix_n_A",)

#: 在线**能算**的全部列。artifact 要了之外的列会在构造时就报错（见
#: ``AppWorldOnlineFeatureBuilder.__init__``）—— 不静默补 0。
APPWORLD_SUPPORTED_FEATURES: tuple[str, ...] = (
    *APPWORLD_MESSAGE_FEATURES,
    *APPWORLD_L2_FEATURES,
    *APPWORLD_PREFIX_FEATURES,
)

#: ``build_message_features()`` 的**完整输出集合** —— 只依赖当前 messages 的那些列。
#: ``prefix_n_A`` 不在其中：它要的是 ``PriorExecutionFacts``，与消息无关。
APPWORLD_MESSAGE_DERIVED: tuple[str, ...] = (
    *APPWORLD_MESSAGE_FEATURES,
    *APPWORLD_L2_FEATURES,
)

#: 事前 L1 六类里哪些算"写"。与 ``executor`` 的 ``_WRITE_CATEGORIES`` 同源语义。
_WRITE_CATEGORIES = frozenset({"declared_write", "opaque_exec", "verify_exec"})


def _message_text(message: Mapping[str, Any]) -> str:
    """消息文本 —— **必须与离线 ``aw3_build.py`` 的 ``_msg_text``（:278-280）逐字同义**。

    离线：``return c if isinstance(c, str) else ""``。即：``content`` 不是 ``str``
    就记 **0 字符**（``None`` 和结构化块都一样）。

    这里曾经多一条 ``return str(content)``，对 list/dict 会记下 ``repr`` 的长度
    （``str(["blob"])`` = 8 字符），使 ``total_chars`` **系统性高估**。
    当前真实数据里 ``content`` 只出现 ``str``(340103) 与 ``None``(207) 两种，
    所以当时不触发 —— 但只要 AppWorld 改成结构化 content，p 就会静默偏掉。
    """

    content = message.get("content")
    return content if isinstance(content, str) else ""


def build_message_features(
    messages: Sequence[Mapping[str, Any]],
    *,
    step_index: int,
) -> dict[str, float]:
    """从原始 AppWorld messages 计算 14 维在线特征.

    ``step_index`` 使用当前 workflow 的 0-based LM span 序号，与离线
    ``cells.jsonl`` 的 ``k`` (=``invocation_index``) 对齐。
    """

    roles = [str(message.get("role", "")) for message in messages]
    texts = [_message_text(message) for message in messages]
    lengths = [len(text) for text in texts]
    total_chars = float(sum(lengths))
    n_messages = len(messages)
    last_role = roles[-1] if roles else ""
    est_prompt_tokens = total_chars / 4.0
    return {
        "k": float(step_index),
        "n_messages": float(n_messages),
        "n_system": float(sum(1 for role in roles if role == "system")),
        "n_user": float(sum(1 for role in roles if role == "user")),
        "n_assistant": float(sum(1 for role in roles if role == "assistant")),
        "n_tool": float(sum(1 for role in roles if role == "tool")),
        "total_chars": total_chars,
        "mean_msg_chars": total_chars / n_messages if n_messages else 0.0,
        "max_msg_chars": float(max(lengths)) if lengths else 0.0,
        "last_role_is_tool": 1.0 if last_role == "tool" else 0.0,
        "last_role_is_assistant": 1.0 if last_role == "assistant" else 0.0,
        "last_role_is_user": 1.0 if last_role == "user" else 0.0,
        "last_msg_chars": float(lengths[-1]) if lengths else 0.0,
        "last5_chars": float(sum(lengths[-5:])),
        # ---- L2_new 的两列（离线口径见 APPWORLD_L2_FEATURES 的注释）----
        "est_prompt_tokens": est_prompt_tokens,
        "log_est_prompt_tokens": math.log1p(est_prompt_tokens),
    }


def count_prefix_write_ops(prior_facts: Any) -> float:
    """前缀里「**成功且是写**」的工具执行数 —— 离线 ``prefix_n_A`` 的同义实现。

    数据源是 ``PriorExecutionFacts.tool_outcomes``（最近 256 条 ``tool_finished``），
    它覆盖**当前 invocation 之前**的所有可观察执行 —— 正好是"前缀"。

    **两个条件都要满足**（缺一就与离线不一致）：

    - ``outcome.success`` —— 离线 ``ok = (status == "success") and (rc in (0, None))``，
      失败的归 ``"F"``，**不计入 A**；
    - ``outcome.effect_category`` ∈ 写类别 —— 离线 ``is_write = any(method != "get")``。

    ``effect_category`` 为 None（旧事件 / 未分类）时**不计入**：宁可少算，不猜。
    """

    outcomes = getattr(prior_facts, "tool_outcomes", None) or ()
    return float(
        sum(
            1
            for outcome in outcomes
            if getattr(outcome, "success", False)
            and getattr(outcome, "effect_category", None) in _WRITE_CATEGORIES
        )
    )


def _raw_messages_from_request(request: Any) -> list[Mapping[str, Any]]:
    """优先读 AppWorld 原生消息快照，缺失时退回规范化 messages。"""

    payload = getattr(request, "native_payload", None)
    if isinstance(payload, Mapping):
        native_messages = payload.get("messages")
        if isinstance(native_messages, list):
            return [message for message in native_messages if isinstance(message, Mapping)]

    converted: list[Mapping[str, Any]] = []
    for message in getattr(request, "messages", ()):
        role = getattr(message, "role", "")
        role_value = getattr(role, "value", role)
        converted.append(
            {
                "role": str(role_value or ""),
                "content": getattr(message, "content", None),
            }
        )
    return converted


class AppWorldOnlineFeatureBuilder:
    """把 AppWorld 消息统计 + L2 特征塞进 ``RuntimeProfile.online_features``。

    ``feature_names`` 通常来自 artifact。**构造时就会校验每一列在线是否能算**，
    算不出来直接抛 —— 这是为了堵住"artifact 声明了某列、在线静默补 0"这个坑：
    补 0 得到的 p 不只是噪声，而是**自信地错**（实测：缺 ``log_est_prompt_tokens``
    时 p 从 0.32 跳到 0.9995）。
    """

    def __init__(
        self,
        feature_names: Sequence[str] = APPWORLD_MESSAGE_FEATURES,
        *,
        feature_version: str = "appworld-online-v1",
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        unsupported = [
            name for name in self.feature_names if name not in APPWORLD_SUPPORTED_FEATURES
        ]
        if unsupported:
            raise ValueError(
                "artifact 要求了在线无法计算的列 "
                f"{unsupported}；在线可算的是 {list(APPWORLD_SUPPORTED_FEATURES)}。"
                "拒绝静默补 0 —— 那会产出看似合理但实际错误的概率。"
            )
        self.feature_version = feature_version

    def build(self, context: Any, request: Any, profile: Any) -> dict[str, float]:
        del profile
        messages = _raw_messages_from_request(request)
        node_context = getattr(
            getattr(context, "workflow_context", None), "node_context", None
        )
        step_index_raw = getattr(node_context, "step_index", 0)
        try:
            step_index = max(0, int(step_index_raw))
        except (TypeError, ValueError):
            step_index = 0
        values = build_message_features(messages, step_index=step_index)
        values["prefix_n_A"] = count_prefix_write_ops(
            getattr(context, "prior_execution_facts", None)
        )
        return {name: float(values[name]) for name in self.feature_names}


__all__ = [
    "APPWORLD_L2_FEATURES",
    "APPWORLD_MESSAGE_DERIVED",
    "APPWORLD_MESSAGE_FEATURES",
    "APPWORLD_PREFIX_FEATURES",
    "APPWORLD_SUPPORTED_FEATURES",
    "AppWorldOnlineFeatureBuilder",
    "build_message_features",
    "count_prefix_write_ops",
]
