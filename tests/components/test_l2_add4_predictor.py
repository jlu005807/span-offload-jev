"""`add4` 两阶段预测器：116 维怎么拼、两个 logit 怎么判。

这个类出错的方式**全是静默的** —— 两段嵌入接反、漏掉标准化、门控那一段
没参与判定，`p` 照样落在 `[0, 1]`，只是路由行为一路错下去。所以这里每条
测试都在钉一个「错了也不报错」的点。

与 `test_l2_v4_predictor.py` 的关系：那个管单段 52 维，这个管双段 116 维。
两个类**不共用**代码路径（刻意如此，改新的不会动到在跑的旧的）。
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from traceoffload.m2.predictor import (
    ARTIFACT_KIND,
    L2Add4Predictor,
    load_l2_add4_predictor,
)

REPO = Path(__file__).resolve().parents[2]

#: 与 `span_embedding.C_BARE_WINDOW` 必须一致 —— 它决定第 ④ 段的向量空间。
C_BARE_WINDOW = 10


def _branch(*, dim: int, nc: int, segment: str, dtype: str = "float64", seed: float = 0.0):
    """一个分支的 `preprocessing` 块。数值是确定性的，便于钉住前向结果。"""
    return {
        "embedding_segment": segment,
        "embedding_dtype": dtype,
        "pca_nc": nc,
        "emb_mu": [seed + i * 0.01 for i in range(dim)],
        "emb_sd": [1.0 + i * 0.01 for i in range(dim)],
        "pca_mean": [seed + i * 0.001 for i in range(dim)],
        "pca_components": [
            [0.1 * (j + 1) / (i + 1) for i in range(dim)] for j in range(nc)
        ],
    }


def _artifact(*, gate_nc: int = 2, net_nc: int = 3, dim: int = 4, **overrides):
    """一份最小可用的 add4 artifact：`prefix 2 + gate 2 + net 3 = 7` 维。"""
    input_dim = 2 + gate_nc + net_nc
    spec = {
        "kind": ARTIFACT_KIND,
        "prefix_features": ["prefix_n_A", "log_est_prompt_tokens"],
        "input_dim": input_dim,
        "joint_normalization": False,
        "preprocessing": {
            "legacy": {"mu": [0.5, 6.0], "sd": [2.0, 1.5]},
            "gate": _branch(dim=dim, nc=gate_nc, segment="V2_goal_recent"),
            "net": _branch(dim=dim, nc=net_nc, segment="C_bare_w10",
                           dtype="float32", seed=0.5),
        },
        "models": {
            "gate": {"model_type": "logistic", "target": "l2_new",
                     "coef": [0.1] * input_dim, "intercept": 0.0, "tau": 0.6},
            "net": {"model_type": "logistic", "target": "net_saving>0",
                    "coef": [-0.05] * input_dim, "intercept": 0.0, "tau": 0.4},
        },
    }
    spec.update(overrides)
    return spec


# --------------------------------------------------------------------------
# 形状与拼装顺序
# --------------------------------------------------------------------------

def test_input_dim_is_prefix_plus_both_pcas() -> None:
    p = L2Add4Predictor(_artifact(gate_nc=50, net_nc=64, dim=8))

    assert p.input_dim == 116, "2 标量 + PCA50 + PCA64"
    assert p.gate_pca_nc == 50 and p.net_pca_nc == 64
    assert p.gate_segment == "V2_goal_recent" and p.net_segment == "C_bare_w10"


def test_declared_input_dim_mismatch_raises() -> None:
    """artifact 自报的维度与两个 PCA 加不出来的一致 ⇒ 抛，不静默。"""
    with pytest.raises(ValueError, match="input_dim"):
        L2Add4Predictor(_artifact(input_dim=999))


def test_features_are_legacy_then_gate_then_net() -> None:
    """三段顺序**是契约**：`[0:2]` 前缀、`[2:2+gate_nc]` 门控、其后是动作。

    钉法：分别只改**一个**分支的 `pca_components`，看改到的是哪一段。
    单靠「把两个嵌入换一下」测不出来 —— 那样两段一起动，接反了也照样不等。
    """
    p = L2Add4Predictor(_artifact(gate_nc=2, net_nc=3, dim=4))
    args = {
        "gate_embedding": [1.0, 2.0, 3.0, 4.0],
        "net_embedding": [5.0, 6.0, 7.0, 8.0],
        "prefix_values": [1.0, 7.0],
    }
    feats = p.features_from(**args)
    assert len(feats) == 7

    # 前两维 = 前缀标量标准化，与两段嵌入都无关
    assert feats[0] == pytest.approx((1.0 - 0.5) / 2.0)
    assert feats[1] == pytest.approx((7.0 - 6.0) / 1.5)

    # 改**门控**那一段的投影矩阵 ⇒ 只有 `[2:4]` 变，`[4:7]` 不许变
    gate_tweaked = _artifact(gate_nc=2, net_nc=3, dim=4)
    gate_tweaked["preprocessing"]["gate"]["pca_components"][0][0] += 0.5
    g = L2Add4Predictor(gate_tweaked).features_from(**args)
    assert list(g[2:4]) != pytest.approx(list(feats[2:4])), "门控那一段没接在 [2:4]"
    assert list(g[4:7]) == pytest.approx(list(feats[4:7])), "动门控改动到了动作那一段"

    # 改**动作**那一段 ⇒ 只有 `[4:7]` 变
    net_tweaked = _artifact(gate_nc=2, net_nc=3, dim=4)
    net_tweaked["preprocessing"]["net"]["pca_components"][0][0] += 0.5
    n = L2Add4Predictor(net_tweaked).features_from(**args)
    assert list(n[4:7]) != pytest.approx(list(feats[4:7])), "动作那一段没接在 [4:7]"
    assert list(n[2:4]) == pytest.approx(list(feats[2:4])), "动动作改动到了门控那一段"


def test_net_branch_is_rounded_to_float32() -> None:
    """第 ④ 段的训练向量来自 float32 的 npz，在线必须按同样精度还原。

    否则「在线 == 离线」不逐位成立（差一个舍入）。差别小到不影响决策，
    但 `test_l2_add4_golden_features.py` 的对拍正是靠它才成立的。
    """
    raw = [0.1234567890123, 1.0 / 3.0, 2.0 / 7.0, 1e-8]

    def feats_with(dtype: str) -> list[float]:
        spec = _artifact(dim=4)
        spec["preprocessing"]["net"]["embedding_dtype"] = dtype
        return list(
            L2Add4Predictor(spec).features_from(
                gate_embedding=[0.0] * 4, net_embedding=raw, prefix_values=[0.0, 0.0]
            )
        )

    assert feats_with("float32") != feats_with("float64"), (
        "net 那一段没有按 artifact 声明的精度还原"
    )


def test_unknown_embedding_dtype_is_rejected() -> None:
    spec = _artifact()
    spec["preprocessing"]["net"]["embedding_dtype"] = "float16"
    with pytest.raises(ValueError, match="embedding_dtype"):
        L2Add4Predictor(spec)


# --------------------------------------------------------------------------
# 判定：**与**，不是或
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "p_gate, p_net, tau_gate, tau_net, offload",
    [
        (0.9, 0.9, 0.6, 0.4, True),    # 两个都过
        (0.9, 0.1, 0.6, 0.4, False),   # 门控过、经济不过
        (0.1, 0.9, 0.6, 0.4, False),   # 门控不过、经济过
        (0.1, 0.1, 0.6, 0.4, False),   # 都不过
        (0.6, 0.4, 0.6, 0.4, True),    # 正好等于 τ ⇒ 卸载（`>=` 不是 `>`）
    ],
)
def test_decision_is_the_and_of_both_thresholds(
    p_gate: float, p_net: float, tau_gate: float, tau_net: float, offload: bool
) -> None:
    """**本轮最核心的语义。** 改成 `or` 会让上面第 2、3 行变红。"""
    p = L2Add4Predictor(_artifact())
    # 前向换成固定值：本测试只关心「怎么把两个 p 合起来」
    p.predict_from_span_features = lambda features: (p_gate, p_net)  # type: ignore[method-assign]

    decision = p.decide_from_span_features(
        [0.0] * p.input_dim, tau_gate=tau_gate, tau_net=tau_net
    )

    assert decision.offload is offload
    assert (decision.p_gate, decision.p_net) == (p_gate, p_net)
    assert (decision.tau_gate, decision.tau_net) == (tau_gate, tau_net)


def test_thresholds_come_from_the_artifact_unless_overridden() -> None:
    p = L2Add4Predictor(_artifact())
    feats = p.features_from(
        gate_embedding=[0.0] * 4, net_embedding=[0.0] * 4, prefix_values=[0.0, 0.0]
    )

    default = p.decide_from_span_features(feats)
    overridden = p.decide_from_span_features(feats, tau_gate=0.99, tau_net=0.01)

    assert (default.tau_gate, default.tau_net) == (0.6, 0.4)
    assert (overridden.tau_gate, overridden.tau_net) == (0.99, 0.01)


def test_predict_from_span_features_checks_the_dimension() -> None:
    p = L2Add4Predictor(_artifact())
    with pytest.raises(ValueError, match="expected 7 span features"):
        p.predict_from_span_features([0.0] * 6)


# --------------------------------------------------------------------------
# 构造期校验
# --------------------------------------------------------------------------

def test_joint_normalization_must_be_explicitly_false() -> None:
    """**缺字段也不能放过。**

    一份「抄了 v4 格式、带 norm_mu/norm_sd」的 artifact 如果被当成
    `joint_normalization=False` 收下，就等于**静默地不做**归一化 ——
    而那正是 v4 与 add4 配方唯一的结构差别。所以要显式。
    """
    spec = _artifact()
    del spec["joint_normalization"]
    with pytest.raises(ValueError, match="joint_normalization"):
        L2Add4Predictor(spec)

    with pytest.raises(ValueError, match="joint_normalization"):
        L2Add4Predictor(_artifact(joint_normalization=True))


def test_wrong_kind_is_rejected() -> None:
    """拿 v4 的 artifact 喂进来 ⇒ 必须报「kind 不对」，不是「少 preprocessing」。"""
    with pytest.raises(ValueError, match="l2_add4_two_stage"):
        L2Add4Predictor(_artifact(kind="something_else"))


def test_missing_net_branch_is_rejected() -> None:
    spec = _artifact()
    del spec["preprocessing"]["net"]
    with pytest.raises(ValueError, match="net"):
        L2Add4Predictor(spec)


def test_zero_scale_is_rejected() -> None:
    spec = _artifact()
    spec["preprocessing"]["gate"]["emb_sd"][0] = 0.0
    with pytest.raises(ValueError, match="not invertible"):
        L2Add4Predictor(spec)


def test_missing_tau_is_rejected() -> None:
    spec = _artifact()
    del spec["models"]["gate"]["tau"]
    with pytest.raises(ValueError, match="tau"):
        L2Add4Predictor(spec)


def test_coef_length_must_match_input_dim() -> None:
    spec = _artifact()
    spec["models"]["net"]["coef"] = [0.1] * 6
    with pytest.raises(ValueError, match="coef"):
        L2Add4Predictor(spec)


# --------------------------------------------------------------------------
# 加载器约定
# --------------------------------------------------------------------------

def test_loader_returns_none_for_missing_file(tmp_path: Path) -> None:
    """文件不存在 → None（调用方决定怎么办）；**格式坏 → 抛**。"""
    assert load_l2_add4_predictor(tmp_path / "nope.json") is None


def test_loader_raises_on_malformed_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": ARTIFACT_KIND}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_l2_add4_predictor(bad)


def test_loader_reads_a_written_artifact(tmp_path: Path) -> None:
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(_artifact()), encoding="utf-8")
    loaded = load_l2_add4_predictor(good)
    assert loaded is not None and loaded.input_dim == 7


# --------------------------------------------------------------------------
# 与离线常量的漂移守卫
# --------------------------------------------------------------------------

def test_logistic_clamp_keeps_probabilities_in_range() -> None:
    """极端输入不许溢出成 nan/inf —— 钳到 ±60 再 sigmoid。"""
    spec = _artifact()
    spec["models"]["gate"]["coef"] = [1e6] * 7
    p = L2Add4Predictor(spec)
    feats = tuple([1e6] * 7)

    p_gate, p_net = p.predict_from_span_features(feats)
    assert math.isfinite(p_gate) and 0.0 <= p_gate <= 1.0
    assert math.isfinite(p_net) and 0.0 <= p_net <= 1.0
