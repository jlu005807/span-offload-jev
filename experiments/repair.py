"""补跑一个臂里缺行或预测器退化的实例（**串行**）。

两个触发条件
------------
* 缺行：并发写 summary 是读-改-写竞态，可能丢行
* 退化：`miss_no_vector > 0`，即该任务里有 span 因为取不到 116 维向量而
  直接走了大模型。实测由并发下的瞬时故障引起（同一任务单跑时 miss=0），
  串行重跑即可修好。

用法
----
    python3 experiments/repair.py --arm offload-active --round 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def load(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else [data]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--instances", default=str(Path(__file__).with_name("instances-90.txt")))
    ap.add_argument("--out-root", default="runs/one_round")
    args = ap.parse_args(argv)

    arm_dir = (REPO / args.out_root / args.arm).resolve()
    out_dir = arm_dir / f"r{args.round}"
    ids = [line.strip() for line in Path(args.instances).read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = []
    for f in out_dir.glob("appworld-*-summary.json"):
        rows.extend(load(f))
    have = {r.get("instance_id") for r in rows}
    degraded = {
        r.get("instance_id")
        for r in rows
        if ((r.get("l2_add4") or {}).get("miss_no_vector") or 0) > 0
    }
    todo = sorted({i for i in ids if i not in have} | degraded)
    print(f"[{args.arm}] 已有 {len(have & set(ids))}/90，待补 {len(todo)}：{todo[:6]}")
    if not todo:
        return 0

    env = dict(os.environ)
    env["APPWORLD_ROOT"] = str(out_dir / "_root")
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.setdefault("MODEL_SERVER_URL", "")
    config = REPO / "configs" / "arms" / f"{args.arm}.yaml"
    for task in todo:
        cmd = [PY, str(REPO / "experiments" / "run_appworld.py"),
               "--config", str(config), "--mode", "classifier_route",
               "--task-id", task, "--output-dir", str(out_dir),
               "--experiment-name", f"{args.arm}-{task}"]
        with (out_dir / "repair.log").open("a", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"  {task}: rc={rc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
