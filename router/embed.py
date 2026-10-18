"""调用本地 Ollama 生成两段嵌入（1024 维）。

两段
----
    V2_goal_recent   span 文本（第 (3) 段）
    C_bare_w10       动作签名文本（第 (4) 段）

⚠️ 三个必须与训练时一致的口径
------------------------------
1. **不归一化**：离线写入的就是 Ollama 原样返回的向量。在线客户端默认会做 L2 归一化，
   但 Ollama 对 embedding 模型返回的本来就是单位向量（实测 ||v||=1.0），所以两边等价。
2. **num_ctx=8192**：实测最长的 V2 文本 14,542 字符 = 3,787 token，默认上下文也不截断，
   但显式给大值更稳妥（历史数据就是这么产生的）。
3. **不排序、不截断**：输入行序就是输出行序。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
NUM_CTX = 8192


def embed_batch(texts: Sequence[str], *, model: str = MODEL, url: str = OLLAMA_URL,
                num_ctx: int = NUM_CTX, retries: int = 3) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": list(texts),
                          "options": {"num_ctx": num_ctx}}).encode()
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as resp:
                got = json.loads(resp.read().decode())["embeddings"]
            if len(got) != len(texts):
                raise RuntimeError(f"返回 {len(got)} 条, 请求 {len(texts)} 条")
            return got
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"embed failed: {last}")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def embed_texts(rows: list[dict], *, segment: str, batch: int = 8) -> list[dict]:
    out: list[dict] = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        vecs = embed_batch([str(r["text"]) for r in chunk])
        for r, v in zip(chunk, vecs):
            out.append({"key": f"{r['cell_id']}|{segment}", "vec": v})
        if (start // batch) % 20 == 0:
            print(f"  [{segment}] {min(start+batch, len(rows))}/{len(rows)}", flush=True)
    return out


def build_text_embeddings(texts_path: Path, out_path: Path, segment: str, batch: int = 8) -> Path:
    rows = read_jsonl(texts_path)
    if not rows:
        raise SystemExit(f"{texts_path} 是空的")
    emb = embed_texts(rows, segment=segment, batch=batch)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in emb:
            fh.write(json.dumps(r) + "\n")
    print(f"[out] {out_path}  {len(emb)} 行 x {len(emb[0]['vec'])} 维")
    return out_path


def build_action_embeddings(texts_path: Path, vectors_path: Path, ids_path: Path, batch: int = 8) -> None:
    rows = read_jsonl(texts_path)
    ids = [r["cell_id"] for r in rows]
    # 空动作文本（k=1）**不编码**，直接给零向量 —— 与历史离线口径、以及线上
    # runner._cache_add4_vector 完全一致。编码空串会得到一个非零向量，静默污染第 4 段。
    empty = [i for i, r in enumerate(rows) if not str(r["text"]).strip()]
    vecs: list[list[float] | None] = [None] * len(rows)
    for start in range(0, len(rows), batch):
        idxs = [i for i in range(start, min(start + batch, len(rows)))
                if str(rows[i]["text"]).strip()]
        if idxs:
            got = embed_batch([rows[i]["text"] for i in idxs])
            for i, v in zip(idxs, got):
                vecs[i] = v
        if (start // batch) % 50 == 0:
            print(f"  [C_bare] {min(start+batch, len(rows))}/{len(rows)}", flush=True)
    dim = len(next(v for v in vecs if v is not None))
    vecs = [v if v is not None else [0.0] * dim for v in vecs]
    if empty:
        print(f"  [C_bare] 空文本 {len(empty)} 个 -> 零向量（不调用模型）")
    mat = np.asarray(vecs, dtype=np.float32)
    Path(vectors_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(vectors_path), v=mat)
    np.savez(Path(ids_path), ids=np.asarray(ids, dtype=object))
    print(f"[out] {vectors_path}  {mat.shape}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("texts", help="texts.jsonl -> aw3_emb_v2goal.jsonl")
    p1.add_argument("--texts", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--segment", default="V2_goal_recent")
    p1.add_argument("--batch", type=int, default=8)

    p2 = sub.add_parser("actions", help="actions_texts.jsonl -> emb_C_bare_w10.npz + cell_ids.npz")
    p2.add_argument("--texts", required=True)
    p2.add_argument("--vectors", required=True)
    p2.add_argument("--ids", required=True)
    p2.add_argument("--batch", type=int, default=8)

    args = ap.parse_args(argv)
    if args.cmd == "texts":
        build_text_embeddings(Path(args.texts), Path(args.out), args.segment, args.batch)
    else:
        build_action_embeddings(Path(args.texts), Path(args.vectors), Path(args.ids), args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
