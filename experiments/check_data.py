"""校验作者提供的数据包能不能直接训练。

用法
----
    python3 experiments/check_data.py                 # 默认查 data-package/
    python3 experiments/check_data.py --root /path/to/out
    python3 experiments/check_data.py --fast           # 跳过 sha256（只看存在性与行数）

判据三层，任何一层不过就非零退出：
    1. 文件存在
    2. 字节数一致
    3. sha256 一致（--fast 时跳过）；jsonl 还要行数一致

清单是**写死**的：它就是训出 router/artifact/ 里那份 artifact 的那批文件。
数据变了就应当过不了这一关 —— 那说明你需要重新训，而不是让旧 artifact 假装还成立。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = [
    {"path": "aw3_regen/训练集/aw3_features.jsonl", "bytes": 8783154, "sha256": "d550c5945cfbd3a05a3765cca860c0b31679794275213583bd93086ee10239a5", "rows": 4950},
    {"path": "aw3_regen/训练集/aw3_emb_v2goal.jsonl", "bytes": 68833787, "sha256": "436d6e626bc73b5282d704f9fed57f20ebf6f0195bfa7757616321bc9586a496", "rows": 4950},
    {"path": "aw3_regen/验证集/aw3_features.jsonl", "bytes": 1041766, "sha256": "7f4c668710d87a9e405c30d69b305781a2ff5bb8ded976083a9be18748f0c037", "rows": 588},
    {"path": "aw3_regen/验证集/aw3_emb_v2goal.jsonl", "bytes": 8176134, "sha256": "641f41091da879608f8b0c568c3e6c02d4cbb04fae062a196159f825987e0e0c", "rows": 588},
    {"path": "actions_variants/emb_C_bare_w10.npz", "bytes": 22683904, "sha256": "fd2d201a930c4dd9b38132c7a2d96340487e007602393f0773687d1cee81a68b"},
    {"path": "actions_variants/cell_ids.npz", "bytes": 86511, "sha256": "96e9d162e5517fc4451f2b85ce00105bbe7e265cedaf79d1f796e802724904b0"},
    {"path": "net_saving_label/net_saving_labels.jsonl", "bytes": 1076257, "sha256": "22f78c2d5df39441a1bf9595fae17d6f7e121c5b38ae6caf5478667e6ac9f858", "rows": 5538},
    {"path": "appworld_labels_0920/labels_appworld_0920.jsonl", "bytes": 2319598, "sha256": "f0a4fc4e3898d193ca2696ec4f881d2088790510b677d7b5807cf3394efbf493", "rows": 810},
    {"path": "appworld_labels_待采集训练实例/labels_appworld_0920.jsonl", "bytes": 11818772, "sha256": "10508c21ecae45aefced6532db6fce1b7fcf2133c8d4aa3a9e35ec1a2c619b03", "rows": 4140},
    {"path": "labels_验证集/labels_appworld_0920.jsonl", "bytes": 1697386, "sha256": "d99d57aa1c0ce23dcf369641ee11fb19230349d879b5bd10c1f7bd8219fee9c2", "rows": 588},
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data-package", help="数据包根目录（与清单里的相对路径对应）")
    ap.add_argument("--fast", action="store_true", help="跳过 sha256")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"✗ {root} 不存在")
        return 2

    bad: list[str] = []
    total = 0
    print(f"{'file':<52}{'bytes':>12}{'rows':>8}")
    print("-" * 74)
    for entry in MANIFEST:
        p = root / entry["path"]
        total += entry["bytes"]
        if not p.is_file():
            print(f"{entry['path']:<52}{'MISSING':>12}")
            bad.append(f"缺失: {entry['path']}")
            continue
        size = p.stat().st_size
        rows = rows_of(p) if entry.get("rows") and p.suffix == ".jsonl" else None
        print(f"{entry['path']:<52}{size:>12}{rows if rows is not None else '':>8}")
        if size != entry["bytes"]:
            bad.append(f"字节数不符: {entry['path']}（{size} vs {entry['bytes']}）")
        if entry.get("rows") and rows != entry["rows"]:
            bad.append(f"行数不符: {entry['path']}（{rows} vs {entry['rows']}）")
        if not args.fast and sha256_of(p) != entry["sha256"]:
            bad.append(f"sha256 不符: {entry['path']}")

    print("-" * 74)
    print(f"清单合计 {len(MANIFEST)} 个文件 / {total/1e6:.1f} MB")
    if bad:
        print("\n✗ 校验未通过：")
        for b in bad:
            print("  -", b)
        return 1
    print("✓ 数据包完整，可以直接跑 router.train / select_threshold / export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
