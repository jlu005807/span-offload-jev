from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from traceoffload.m1.appworld_features import (
    APPWORLD_MESSAGE_DERIVED,
    APPWORLD_MESSAGE_FEATURES,
    AppWorldOnlineFeatureBuilder,
    build_message_features,
    count_prefix_write_ops,
)


def test_build_message_features_matches_offline_message_stats() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "response"},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "latest"},
    ]
    features = build_message_features(messages, step_index=3)

    assert set(features) == set(APPWORLD_MESSAGE_DERIVED)
    assert features["k"] == 3.0
    assert features["n_messages"] == 5.0
    assert features["n_system"] == 1.0
    assert features["n_user"] == 2.0
    assert features["n_assistant"] == 1.0
    assert features["n_tool"] == 1.0
    assert features["total_chars"] == float(
        len("system prompt") + len("hello") + len("response") + len("tool output") + len("latest")
    )
    assert features["mean_msg_chars"] == features["total_chars"] / 5.0
    assert features["max_msg_chars"] == float(max(len(text) for text in (
        "system prompt", "hello", "response", "tool output", "latest"
    )))
    assert features["last_role_is_user"] == 1.0
    assert features["last_role_is_tool"] == 0.0
    assert features["last_msg_chars"] == float(len("latest"))
    assert features["last5_chars"] == features["total_chars"]


def test_online_builder_reads_native_payload_and_step_index() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    context = SimpleNamespace(
        workflow_context=SimpleNamespace(
            node_context=SimpleNamespace(step_index=4)
        )
    )
    request = SimpleNamespace(native_payload={"messages": messages})
    builder = AppWorldOnlineFeatureBuilder(
        APPWORLD_MESSAGE_FEATURES,
        feature_version="appworld-test-v1",
    )

    values = builder.build(context, request, SimpleNamespace())

    assert tuple(values) == APPWORLD_MESSAGE_FEATURES
    assert values["k"] == 4.0
    assert values["n_messages"] == 2.0
    assert values["last_role_is_user"] == 1.0


def test_l2_features_match_offline_definition_bit_for_bit() -> None:
    """L2 两列必须与离线 `aw3_build.py` 的 `chars` / `total_chars` / `log_est_prompt_tokens`（:300-302, 482）逐位一致。

    离线：``total_chars = sum(len(m["content"]) if isinstance(content, str) else "")``
    → ``est_prompt_tokens = total_chars / 4.0``
    → ``log_est_prompt_tokens = round(log1p(est_prompt_tokens), 6)``

    **口径差一点 p 就偏**，所以这里把 `content` 的三种形态都覆盖：
    str / None / 非 str 非 None（后者离线记 0，在线也必须记 0）。
    """
    messages = [
        {"role": "system", "content": "abcd"},        # 4
        {"role": "user", "content": None},            # 0
        {"role": "assistant", "content": ["blob"]},   # 0 ← 离线 isinstance(str) 为假
        {"role": "tool", "content": "efghij"},        # 6
    ]
    features = build_message_features(messages, step_index=1)

    total_chars = float(4 + 0 + 0 + 6)
    est = total_chars / 4.0
    assert features["total_chars"] == total_chars
    assert features["est_prompt_tokens"] == est
    assert features["log_est_prompt_tokens"] == pytest.approx(math.log1p(est))


def test_online_builder_requires_supported_feature_columns() -> None:
    """坑 2：artifact 要了在线算不出的列 -> **构造时抛**，不静默补 0。

    静默补 0 得到的 p 不是"噪声"，是**自信地错**：实测缺
    ``log_est_prompt_tokens`` 时 p 会从 0.32 跳到 0.9995。
    """
    with pytest.raises(ValueError, match="无法计算的列"):
        AppWorldOnlineFeatureBuilder(("prefix_n_A", "no_such_feature_v99"))

    # 支持的三列可以正常构造
    builder = AppWorldOnlineFeatureBuilder(
        ("prefix_n_A", "log_est_prompt_tokens", "total_chars")
    )
    assert builder.feature_names == ("prefix_n_A", "log_est_prompt_tokens", "total_chars")


def _outcome(*, success, category):
    return SimpleNamespace(success=success, effect_category=category)


def test_prefix_n_A_counts_only_successful_writes() -> None:
    """`prefix_n_A` 只数「**成功且是写**」，与离线 ``tool_event_kind`` 的 ``"A"`` 对齐。

    四条反例各代表一种"不能计入"：失败的写、成功的读、成功的未知类、类别缺失。
    """
    outcomes = (
        _outcome(success=True, category="declared_write"),   # ✅ A
        _outcome(success=True, category="opaque_exec"),      # ✅ A（写类）
        _outcome(success=False, category="declared_write"),  # ❌ 失败 -> 离线归 "F"
        _outcome(success=True, category="read_only"),        # ❌ 只读 -> "O"
        _outcome(success=True, category="unknown"),          # ❌ 未知不算写
        _outcome(success=True, category=None),               # ❌ 类别缺失 -> 不猜
    )
    assert count_prefix_write_ops(SimpleNamespace(tool_outcomes=outcomes)) == 2.0
    # 没有历史时是 0，不是 None
    assert count_prefix_write_ops(SimpleNamespace(tool_outcomes=())) == 0.0
    assert count_prefix_write_ops(None) == 0.0


def test_builder_reads_prefix_n_A_from_prior_execution_facts() -> None:
    """通路：`context.prior_execution_facts.tool_outcomes` -> `prefix_n_A`。

    这条同时钉住"数据源是**前缀**"——profiling 发生在工具阶段之前，
    所以此刻的 tool_outcomes 恰好覆盖当前 invocation 之前的所有执行。
    """
    context = SimpleNamespace(
        workflow_context=SimpleNamespace(node_context=SimpleNamespace(step_index=2)),
        prior_execution_facts=SimpleNamespace(
            tool_outcomes=(
                _outcome(success=True, category="declared_write"),
                _outcome(success=False, category="declared_write"),
                _outcome(success=True, category="read_only"),
            )
        ),
    )
    request = SimpleNamespace(native_payload={"messages": [{"role": "user", "content": "x"}]})
    builder = AppWorldOnlineFeatureBuilder(("prefix_n_A", "log_est_prompt_tokens"))

    values = builder.build(context, request, SimpleNamespace())

    assert values["prefix_n_A"] == 1.0
    assert values["log_est_prompt_tokens"] == pytest.approx(math.log1p(len("x") / 4.0))

