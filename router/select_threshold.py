"""验证集上挑阈值：最大化 Σ净节省（不是 F1、不是准确率）。

判据
----
    U(tau_g, tau_t) = Σ_{p_gate >= tau_g 且 p_net >= tau_t} net_saving(点)
    tau* = argmax U

算法
----
    tau_g 取 p_gate 的 60 个分位点；对每个 tau_g，在通过门控的子集里对 p_net 做
    一次一维前缀扫描（60 次 O(n)，不是 3600 次）。

稳健化
------
    家族级 bootstrap 400 轮：重采样家族（不是点 —— 同家族的点高度相关，按点会把
    区间算窄），每次重跑一遍搜索，报中位数与 90% 区间。

⚠️ 挑出来的成绩是乐观的：同一批数据既选阈值又报成绩。测试集实跑一次才是无偏的。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_scores(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sweep_a(p_gate: np.ndarray, p_net: np.ndarray, net: np.ndarray) -> tuple[float, float, float]:
    """两阈值网格搜索。返回 (U, tau_gate, tau_net)。"""
    grid = np.quantile(p_gate, np.linspace(0, 1, 60))
    best = (-np.inf, float("nan"), float("nan"))
    for a in grid:
        m = p_gate >= a
        if not m.any():
            continue
        sub_net, sub_score = net[m], p_net[m]
        order = np.argsort(-sub_score)
        cum = np.cumsum(sub_net[order])
        i = int(np.argmax(cum))
        if cum[i] > best[0]:
            best = (float(cum[i]), float(a), float(sub_score[order][i]))
    return best


def bootstrap(rows: list[dict], rounds: int = 400, seed: int = 5) -> tuple[float, float, float, float]:
    """家族级 bootstrap。返回 (tau_g 中位, tau_t 中位, q05, q95)。"""
    fams = sorted({r["family"] for r in rows})
    by_fam = {f: [r for r in rows if r["family"] == f] for f in fams}
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(rounds):
        pick = rng.choice(len(fams), size=len(fams), replace=True)
        sample = [r for i in pick for r in by_fam[fams[i]]]
        pg = np.array([r["p_gate"] for r in sample])
        pn = np.array([r["p_net"] for r in sample])
        nt = np.array([r["net_saving"] for r in sample])
        _u, tg, tt = sweep_a(pg, pn, nt)
        if np.isfinite(tg) and np.isfinite(tt):
            taus.append((tg, tt))
    if not taus:
        return (float("nan"),) * 4
    arr = np.array(taus)
    return (
        float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
        float(np.quantile(arr[:, 0], 0.05)), float(np.quantile(arr[:, 0], 0.95)),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="train.py 产出的 val_scores.jsonl")
    ap.add_argument("--out", default=None, help="可选：把阈值写成 tau.json")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args(argv)

    rows = read_scores(Path(args.scores))
    p_gate = np.array([r["p_gate"] for r in rows])
    p_net = np.array([r["p_net"] for r in rows])
    net = np.array([r["net_saving"] for r in rows])
    u, tau_g, tau_t = sweep_a(p_gate, p_net, net)

    off = (p_gate >= tau_g) & (p_net >= tau_t)
    oracle = float(np.maximum(net, 0).sum())
    capture = float(net[off].sum() / oracle) if oracle > 0 else float("nan")
    print(f"[sweep] U = {u:,.1f}   tau_gate = {tau_g:.10f}   tau_net = {tau_t:.10f}")
    print(f"[val  ] 卸载 {int(off.sum())}/{len(rows)} = {100*off.mean():.1f}%"
          f"   oracle 捕获 {100*capture:.2f}%   Σ净节省 {net[off].sum():,.1f}")

    if args.rounds > 0:
        mg, mt, q05, q95 = bootstrap(rows, args.rounds)
        print(f"[boot ] tau_gate 中位 {mg:.6f}  90% 区间 [{q05:.6f}, {q95:.6f}]  宽度 {q95-q05:.4f}")
        print(f"[boot ] tau_net  中位 {mt:.6f}")
        if q95 - q05 > 0.15:
            print("⚠️ 区间宽度 > 0.15：家族数撑不起精细选择，阈值应取粗一点")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "tau_gate": tau_g, "tau_net": tau_t, "objective": u,
            "val_offload": f"{int(off.sum())}/{len(rows)}",
            "val_net_capture": capture,
        }, indent=2), encoding="utf-8")
        print(f"[out ] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
