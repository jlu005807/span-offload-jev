"""在线 116 维 == 离线 `fit_transform_prep` 的 `Xa`。

**为什么必须有这条**：第 ④ 段（动作签名嵌入）是整条链上**唯一**的新
离线/在线边界。它错了不会报错 —— 两个 `p` 照样落在 `[0, 1]`，只是路由
行为一路偏。别的检查（看 summary、看卸载率）都只能在**事后**发现，
而且分不清是模型不行还是特征不对。

对拍方式：`sf_export_add4.py` 导出 artifact 时顺手落一份
`out/router/l2_add4_selfcheck.json`，里面是若干固定 cell 的
**两段原始嵌入 + 前缀标量 + 离线算出来的 116 维**。本测试用**在线的那个类**
把 116 维重算一遍，逐位比对。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceoffload.m2.predictor import (
    L2Add4Predictor,
    load_l2_add4_predictor,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "router/artifact/l2_add4_twostage_logit.json"
SELFCHECK = REPO / "router/artifact/l2_add4_selfcheck.json"

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACT.is_file() and SELFCHECK.is_file()),
    reason="add4 artifact 不在（先跑 `04_预测器训练/sf_export_add4.py`）",
)

#: `sf_export_add4.py` 把数写成 8 位小数，所以离线那一侧本身有 ~1e-8 的量化。
#: 在线用的是全精度 —— 容差要盖住导出时的量化，但不能大到放过「少了一步
#: 标准化」这种错（那会让特征整体差好几个量级）。
TOL = 1e-6


@needs_artifacts
def test_online_features_match_the_offline_row() -> None:
    """**逐位对拍。** 每个 cell 的三段都从原始输入重建。"""
    predictor = L2Add4Predictor(json.loads(ARTIFACT.read_text(encoding="utf-8")))
    rows = json.loads(SELFCHECK.read_text(encoding="utf-8"))["rows"]
    assert rows, "selfcheck 里没有对拍行"

    for row in rows:
        online = predictor.features_from(
            gate_embedding=row["gate_embedding"],
            net_embedding=row["net_embedding"],
            prefix_values=row["prefix_values"],
        )
        offline = [float(x) for x in row["span_features"]]

        assert len(online) == len(offline) == 116, row["cell_id"]
        worst = max(abs(a - b) for a, b in zip(online, offline))
        assert worst < TOL, (
            f"{row['cell_id']}: 在线与离线最多差 {worst:.3e}（容差 {TOL:.0e}）。"
            "多半是少了某一步标准化、或两段接反了。"
        )


def _arm_l2_add4() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_ablation_configs", REPO / "classfierTrain/gen_ablation_configs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ARMS["A6-add4"]["delta"]["runner"]["l2_add4"]


@needs_artifacts
def test_shipment_artifact_holds_the_validated_thresholds() -> None:
    """artifact 里的 τ 必须是验证集上选出来的那一组（四舍五入后在 1e-4 内）。

    这两个数是「在 588 个验证点 / 24 个家族上最大化 Σ净节省」选出来的，
    跟着 artifact 走。这里钉住它们，是为了让**换数据重训**时落在别的工作点上
    这件事不会悄无声息。
    """
    predictor = load_l2_add4_predictor(ARTIFACT)
    assert predictor is not None

    assert predictor.tau_gate == pytest.approx(0.5381, abs=1e-4)
    assert predictor.tau_net == pytest.approx(0.5523, abs=1e-4)

