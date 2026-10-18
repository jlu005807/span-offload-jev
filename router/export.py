"""把训练结果打包成运行时 artifact，并在**artifact 自己的分数**上定阈值。

为什么要用 artifact 自己再打一遍分
--------------------------------
导出时 β / PCA 系数被 round 到 12 位，分数会整体挪一点点，边界上的点可能掉出去
（实测 119 -> 118，Σ净节省少 2.4%）。所以 τ 必须在**落盘参数算出来的分数**上重选，
而不是在训练时的浮点分数上选。

产物
----
    <out>           l2_add4_twostage_logit.json —— 运行时唯一读的东西
    <out>.metrics   （内嵌）验证集卸载数 / oracle 捕获 / Σ净节省

在线侧 traceoffload/components/l2_add4_predictor.py 用同一份参数重建 116 维，
因此这里算出来的 p 与线上逐位一致（test_l2_add4_golden_features 钉住这件事）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traceoffload.components.l2_add4_predictor import L2Add4Predictor

from .data import load_actions, load_labels, load_net, load_split
from .select_threshold import sweep_a

ROUND_ND = 12


def _rn(v) -> float:
    return float(np.round(float(v), ROUND_ND))


def _round_tree(node):
    if isinstance(node, dict):
        return {k: _round_tree(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_round_tree(v) for v in node]
    if isinstance(node, float):
        return _rn(node)
    if isinstance(node, (np.floating, np.integer)):
        return _rn(node) if isinstance(node, np.floating) else int(node)
    return node


def build_artifact(model: dict) -> dict:
    return {
        "kind": "l2_add4_two_stage",
        "prefix_features": list(model["prefix_features"]),
        "input_dim": int(model["input_dim"]),
        "joint_normalization": False,
        "preprocessing": _round_tree(model["preprocessing"]),
        "models": {
            "gate": {**{k: v for k, v in model["models"]["gate"].items() if k != "tau"},
                     "coef": _round_tree(model["models"]["gate"]["coef"]),
                     "intercept": _rn(model["models"]["gate"]["intercept"]), "tau": 0.0},
            "net": {**{k: v for k, v in model["models"]["net"].items() if k != "tau"},
                    "coef": _round_tree(model["models"]["net"]["coef"]),
                    "intercept": _rn(model["models"]["net"]["intercept"]), "tau": 0.0},
        },
        "trained_on": model.get("trained_on"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="train.py 产出的 model.json")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--val-labels", required=True, nargs="+")
    ap.add_argument("--net", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--action-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-name", default="features.jsonl")
    ap.add_argument("--emb-name", default="aw3_emb_v2goal.jsonl")
    ap.add_argument("--expect-tau", nargs=2, type=float, default=None,
                    help="可选：核对 tau_gate tau_net（复现历史 artifact 时用）")
    args = ap.parse_args(argv)

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    artifact = build_artifact(model)
    predictor = L2Add4Predictor(artifact)  # 构造期就会校验维度/形状

    amap, act_matrix = load_actions(Path(args.action_ids), Path(args.actions))
    net = load_net(Path(args.net))
    labels = load_labels([Path(p) for p in args.val_labels])
    val = load_split(Path(args.val_dir) / args.features_name, Path(args.val_dir) / args.emb_name,
                     labels, amap, act_matrix, net)
    ids = sorted(val)
    if not ids:
        raise SystemExit("验证集为空 —— 检查路径")

    feats = [
        predictor.features_from(
            gate_embedding=val[c]["v2"],
            net_embedding=val[c]["act"],
            prefix_values=val[c]["prefix"],
        )
        for c in ids
    ]
    scores = [predictor.predict_from_span_features(f) for f in feats]
    p_gate = np.array([s[0] for s in scores])
    p_net = np.array([s[1] for s in scores])
    net_true = np.array([val[c]["net_raw"] for c in ids])

    u, tau_g, tau_t = sweep_a(p_gate, p_net, net_true)
    off = (p_gate >= tau_g) & (p_net >= tau_t)
    oracle = float(np.maximum(net_true, 0).sum())
    capture = float(net_true[off].sum() / oracle) if oracle > 0 else float("nan")

    artifact["models"]["gate"]["tau"] = tau_g
    artifact["models"]["net"]["tau"] = tau_t
    artifact["validated_on"] = (
        f"val split ({len(ids)} points / {len({val[c]['family'] for c in ids})} families)"
    )
    artifact["metrics"] = {
        "val_points": len(ids),
        "val_offload": f"{int(off.sum())}/{len(ids)}",
        "val_net_capture": round(capture, 6),
        "val_objective": round(u, 3),
    }

    print(f"[add4] {len(ids)} 点 / {len(set(val[c]['family'] for c in ids))} 家族")
    print(f"[tau ] tau_gate = {tau_g:.15f}   tau_net = {tau_t:.15f}")
    print(f"[val ] 卸载 {artifact['metrics']['val_offload']}   oracle 捕获 {100*capture:.2f}%")

    if args.expect_tau:
        dg = abs(tau_g - args.expect_tau[0])
        dt = abs(tau_t - args.expect_tau[1])
        print(f"[cmp ] vs 参考阈值：|Δtau_g| = {dg:.3e}   |Δtau_t| = {dt:.3e}")
        if dg > 1e-6 or dt > 1e-6:
            raise SystemExit("✗ 阈值与参考不一致 —— 数据或配方变了，停下来查")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact), encoding="utf-8")
    print(f"[out ] {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
