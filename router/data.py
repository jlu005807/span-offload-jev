"""离线数据装载：把特征 / 两段嵌入 / 标签 / 净节省拼成训练需要的点集。

所有入口都是文件路径，不假设目录结构 —— 换一份数据只要换参数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: 两个前缀标量的名字（顺序即 116 维里的前两列）。
PREFIX_FEATURES = ("prefix_n_A", "log_est_prompt_tokens")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def cell_id_of(row: Mapping[str, Any]) -> str:
    if "cell_id" in row:
        return str(row["cell_id"])
    return f"{row['task_id']}@k{row['k']}"


def load_labels(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    lab: dict[str, dict[str, Any]] = {}
    for p in paths:
        for r in read_jsonl(p):
            lab[cell_id_of(r)] = r
    return lab


def load_net(path: Path) -> dict[str, float]:
    return {str(r["cell_id"]): float(r["net_saving"]) for r in read_jsonl(path)}


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """{"key": "042a9fc_2@k3|V2_goal_recent", "vec": [...]} -> {cell_id: vector}"""
    out: dict[str, np.ndarray] = {}
    for r in read_jsonl(path):
        out[str(r["key"]).split("|", 1)[0]] = np.asarray(r["vec"], np.float64)
    return out


def load_actions(ids_path: Path, vectors_path: Path) -> tuple[dict[str, int], np.ndarray]:
    """cell_ids.npz + emb_*.npz -> ({cell_id: 行号}, 矩阵)。

    embedding 落盘是 float32，这里按 float64 还原 —— 在线侧
    （l2_add4_predictor._as_stored）做同样的还原，两边才对得上。
    """
    ids = np.load(Path(ids_path), allow_pickle=True)["ids"]
    vecs = np.load(Path(vectors_path))["v"].astype(np.float64)
    return ({str(k): i for i, k in enumerate(ids.tolist())}, vecs)


def load_split(
    features_path: Path,
    emb_path: Path,
    labels: Mapping[str, dict[str, Any]],
    amap: Mapping[str, int],
    act: np.ndarray,
    net: Mapping[str, float],
    *,
    require_net: bool = True,
) -> dict[str, dict[str, Any]]:
    """一个 split 的点集。只保留 mask==1、有 l2_new、有两段嵌入的点。"""
    vecs = load_embeddings(emb_path)
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(features_path):
        cid = cell_id_of(r)
        m = labels.get(cid)
        if not m or m.get("mask") != 1 or m.get("l2_new") is None or cid not in vecs:
            continue
        n = net.get(cid)
        if require_net and (n is None or cid not in amap):
            continue
        out[cid] = {
            "prefix": np.array([float(r.get(s) or 0.0) for s in PREFIX_FEATURES]),
            "v2": vecs[cid],
            # 没有动作点（k=1）时用零向量，与线上 runner._cache_add4_vector 同义
            "act": act[amap[cid]] if cid in amap else np.zeros(act.shape[1]),
            "y_gate": int(m["l2_new"]),
            "y_net": int(float(n) > 0.0) if n is not None else None,
            "net_raw": float(n) if n is not None else None,
            "family": str(m.get("task_id", cid.rsplit("_", 1)[0])).rsplit("_", 1)[0],
        }
    return out


def stack(rows: Mapping[str, dict[str, Any]], ids: Sequence[str]) -> tuple[np.ndarray, ...]:
    return (
        np.stack([rows[c]["prefix"] for c in ids]),
        np.stack([rows[c]["v2"] for c in ids]),
        np.stack([rows[c]["act"] for c in ids]),
        np.array([rows[c]["y_gate"] for c in ids], float),
        np.array([rows[c]["y_net"] for c in ids], float),
        np.array([rows[c]["net_raw"] for c in ids], float),
    )
