"""训练两阶段路由器：116 维 = [2 标量] + PCA50(V2_goal_recent) + PCA64(C_bare_w10)。

两个 logit
----------
    gate  <- 标签 l2_new            （这一步本地做对不对）
    net   <- 标签 net_saving > 0    （这一步省不省钱）

为什么是两个模型：结论来自实测 —— 单模型 AUC 最高 0.8236 但 oracle 捕获只有 11.3%，
两阶段 AUC 0.7634 却有 16.0%。AUC 只衡量单分数排序，看不见两个阈值的自由度。

纪律
----
* 标准化与两个 PCA **只在训练集上拟合**；验证集只做 transform。
* 阈值**不在这里选**（见 select_threshold.py）—— 训练只产出模型参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .data import load_actions, load_labels, load_net, load_split, stack
from .stats import family_equal_auc, pooled_auc

PCA_GATE = 50
PCA_NET = 64
LOGIT_C = 1.0
LOGIT_MAX_ITER = 5000


def fit_prep(prefix: np.ndarray, v2: np.ndarray, act: np.ndarray) -> dict:
    """拟合标准化 + 两个 PCA。返回导出所需的全部中间量。"""
    mu, sd = prefix.mean(0), prefix.std(0)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    mu2, sd2 = v2.mean(0), v2.std(0)
    sd2 = np.where(sd2 <= 1e-12, 1.0, sd2)
    gate = PCA(n_components=min(PCA_GATE, v2.shape[0] - 1), random_state=0).fit((v2 - mu2) / sd2)

    mu3, sd3 = act.mean(0), act.std(0)
    sd3 = np.where(sd3 <= 1e-12, 1.0, sd3)
    net = PCA(n_components=min(PCA_NET, act.shape[0] - 1), random_state=0).fit((act - mu3) / sd3)
    return {"legacy": (mu, sd), "gate": (mu2, sd2, gate), "net": (mu3, sd3, net)}


def transform(prep: dict, prefix: np.ndarray, v2: np.ndarray, act: np.ndarray) -> np.ndarray:
    mu, sd = prep["legacy"]
    lg = (prefix - mu) / sd
    mu2, sd2, gate = prep["gate"]
    v = gate.transform((v2 - mu2) / sd2)
    mu3, sd3, net = prep["net"]
    a = net.transform((act - mu3) / sd3)
    return np.hstack([lg, v, a])


def _dump_prep(prep: dict) -> dict:
    mu, sd = prep["legacy"]
    mu2, sd2, gate = prep["gate"]
    mu3, sd3, net = prep["net"]
    return {
        "legacy": {"mu": mu.tolist(), "sd": sd.tolist()},
        "gate": {
            "embedding_segment": "V2_goal_recent",
            "embedding_dtype": "float64",
            "pca_nc": int(gate.n_components_),
            "emb_mu": mu2.tolist(), "emb_sd": sd2.tolist(),
            "pca_mean": gate.mean_.tolist(),
            "pca_components": gate.components_.tolist(),
        },
        "net": {
            "embedding_segment": "C_bare_w10",
            "embedding_dtype": "float32",
            "pca_nc": int(net.n_components_),
            "emb_mu": mu3.tolist(), "emb_sd": sd3.tolist(),
            "pca_mean": net.mean_.tolist(),
            "pca_components": net.components_.tolist(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, help="含 features.jsonl 与 aw3_emb_v2goal.jsonl 的训练目录")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--train-labels", required=True, nargs="+", help="l2_new 标签 jsonl（可多份，合并）")
    ap.add_argument("--val-labels", required=True, nargs="+")
    ap.add_argument("--net", required=True, help="net_saving 标签 jsonl")
    ap.add_argument("--actions", required=True, help="emb_C_bare_w10.npz")
    ap.add_argument("--action-ids", required=True, help="cell_ids.npz（与上面同序）")
    ap.add_argument("--out", required=True, help="输出 model.json 路径")
    ap.add_argument("--val-scores", required=True, help="输出验证集打分 val_scores.jsonl")
    ap.add_argument("--features-name", default="features.jsonl")
    ap.add_argument("--emb-name", default="aw3_emb_v2goal.jsonl")
    args = ap.parse_args(argv)

    amap, act = load_actions(Path(args.action_ids), Path(args.actions))
    net = load_net(Path(args.net))
    train_labels = load_labels([Path(p) for p in args.train_labels])
    val_labels = load_labels([Path(p) for p in args.val_labels])

    train = load_split(Path(args.train_dir) / args.features_name, Path(args.train_dir) / args.emb_name,
                       train_labels, amap, act, net)
    val = load_split(Path(args.val_dir) / args.features_name, Path(args.val_dir) / args.emb_name,
                     val_labels, amap, act, net)
    tr_ids, va_ids = sorted(train), sorted(val)
    print(f"[data] 训练 {len(tr_ids)} 点 / {len({train[c]['family'] for c in tr_ids})} 家族"
          f"   验证 {len(va_ids)} 点 / {len({val[c]['family'] for c in va_ids})} 家族")
    if not tr_ids or not va_ids:
        raise SystemExit("训练集或验证集为空 —— 检查路径与 --features-name/--emb-name")

    Ptr, Vtr, Atr, yg_tr, yn_tr, _ = stack(train, tr_ids)
    Pva, Vva, Ava, yg_va, yn_va, net_va = stack(val, va_ids)

    prep = fit_prep(Ptr, Vtr, Atr)
    Xtr, Xva = transform(prep, Ptr, Vtr, Atr), transform(prep, Pva, Vva, Ava)
    print(f"[prep] X 训练 {Xtr.shape}  验证 {Xva.shape}")

    gate_clf = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER).fit(Xtr, yg_tr)
    net_clf = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER).fit(Xtr, yn_tr)

    p_gate = gate_clf.predict_proba(Xva)[:, 1]
    p_net = net_clf.predict_proba(Xva)[:, 1]
    fam = [val[c]["family"] for c in va_ids]
    print(f"[gate] 池化 AUC {pooled_auc(yg_va, p_gate):.4f}  家族均衡 AUC {family_equal_auc(yg_va, p_gate, fam):.4f}")
    print(f"[net ] 池化 AUC {pooled_auc(yn_va, p_net):.4f}  家族均衡 AUC {family_equal_auc(yn_va, p_net, fam):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "kind": "l2_add4_model",
        "prefix_features": ["prefix_n_A", "log_est_prompt_tokens"],
        "input_dim": int(Xtr.shape[1]),
        "preprocessing": _dump_prep(prep),
        "models": {
            "gate": {"model_type": "logistic", "target": "l2_new",
                     "coef": gate_clf.coef_[0].tolist(), "intercept": float(gate_clf.intercept_[0])},
            "net": {"model_type": "logistic", "target": "net_saving>0",
                    "coef": net_clf.coef_[0].tolist(), "intercept": float(net_clf.intercept_[0])},
        },
        "trained_on": {"points": len(tr_ids), "families": len({train[c]["family"] for c in tr_ids})},
    }
    out.write_text(json.dumps(model), encoding="utf-8")
    print(f"[out ] {out}  ({out.stat().st_size/1e6:.1f} MB)")

    scores = Path(args.val_scores)
    scores.parent.mkdir(parents=True, exist_ok=True)
    with scores.open("w", encoding="utf-8") as fh:
        for cid, pg, pn, nr, f in zip(va_ids, p_gate, p_net, net_va, fam):
            fh.write(json.dumps({"cell_id": cid, "p_gate": float(pg), "p_net": float(pn),
                                 "net_saving": float(nr), "family": f}) + "\n")
    print(f"[out ] {scores}  ({len(va_ids)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
