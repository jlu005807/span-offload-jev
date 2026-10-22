"""跑一轮：90 个实例 × 一个臂，按固定顺序、可并发、可续跑。

为什么顺序固定
--------------
M4 的记忆**跨实例累积**，同一个 worker 里实例顺序变了，记忆的命中顺序就变了，
M4 那一行就不可比。所以实例清单是文件（experiments/instances-90.txt），不是集合。

为什么每轮一个 m4.sqlite3
------------------------
一轮 = 一次「冷启动 + 90 个实例」。M4 在轮开始时是空库，这是协议的一部分
（"M4 启动时内部无数据"）；跨轮共享会把第二轮变成热启动。

用法
----
    python3 experiments/run_round.py --arm offload-active --rounds 1 --workers 4
    python3 experiments/run_round.py --arm m4-on --rounds 1 --workers 4
    python3 experiments/run_round.py --arm native-large --rounds 1 --workers 4

产物
----
    runs/<exp>/<arm>/r<K>/appworld-classifier_route-summary.json   （TraceOffload 系）
    runs/<exp>/<arm>/round-<K>/<task>/result.json                  （native 系）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def load_instances(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise SystemExit(f"{path} 是空的")
    return ids


def prepare_root(root: Path) -> None:
    """AppWorld 要求 <root>/experiments/outputs 存在，data 指到 appworld/data。"""
    (root / "experiments" / "outputs").mkdir(parents=True, exist_ok=True)
    data = root / "data"
    target = REPO / "appworld" / "data"
    if not data.exists() and target.is_dir():
        data.symlink_to(target)


def _rows_on_disk(out_dir: Path, native: bool) -> set[str]:
    ids: set[str] = set()
    if native:
        for f in out_dir.glob("*/result.json"):
            ids.update(r.get("instance_id") for r in _load(f))
    else:
        for f in out_dir.glob("appworld-*-summary.json"):
            ids.update(r.get("instance_id") for r in _load(f))
    return {i for i in ids if i}


def _load(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        return [data]
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def jev_enabled_from_config(config_path: Path) -> bool:
    """Read the batch safety switch from the arm config, never its arm name."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    runner = raw.get("runner") if isinstance(raw, dict) else None
    jev = runner.get("jev") if isinstance(runner, dict) else None
    return isinstance(jev, dict) and jev.get("enabled") is True


def _missing_ids(out_dir: Path, ids: list[str], native: bool) -> list[str]:
    """缺行的 + **预测器退化的**。

    退化判据：miss_no_vector > 0 —— 说明这次跑里有 span 因为取不到 116 维向量
    而**直接走了大模型**。实测它由并发下的瞬时故障引起（同一个任务单跑时
    miss=0），不是代码问题，所以串行重跑就能修好。
    """
    have = _rows_on_disk(out_dir, native)
    missing = [i for i in ids if i not in have]
    if native:
        return missing
    degraded = []
    for f in out_dir.glob("appworld-*-summary.json"):
        for row in _load(f):
            block = row.get("l2_add4") or {}
            if (block.get("miss_no_vector") or 0) > 0:
                degraded.append(row.get("instance_id"))
    return sorted(set(missing) | {i for i in degraded if i})


def mem_ok(min_gb: float = 4.0) -> bool:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])
        return info.get("MemAvailable", 0) / 1024 / 1024 >= min_gb
    except Exception:
        return True


def run_one(cmd: list[str], env: dict[str, str], log: Path, timeout: int = 1800) -> tuple[str, int]:
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = 124
    return (cmd[-1], rc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="臂名（= configs/arms/<arm>.yaml 或 native-large/native-small）")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="同时跑几个任务（别超 5）")
    ap.add_argument("--instances", default=str(Path(__file__).with_name("instances-90.txt")))
    ap.add_argument("--out-root", default="runs/one_round")
    ap.add_argument("--timeout", type=int, default=1800, help="单任务超时（秒）")
    args = ap.parse_args(argv)

    ids = load_instances(Path(args.instances))
    out_root = (REPO / args.out_root).resolve()
    arm_dir = out_root / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    native = args.arm.startswith("native-")
    model = args.arm.split("-", 1)[1] if native else None
    # native 臂没有专属配置：模型端点从基础配置里取，臂只由 --model 决定。
    config = (REPO / "configs" / "pilot.yaml") if native else (REPO / "configs" / "arms" / f"{args.arm}.yaml")
    if not config.is_file():
        raise SystemExit(f"找不到配置 {config}")
    jev_enabled = jev_enabled_from_config(config)
    workers = 1 if jev_enabled else args.workers
    if jev_enabled and args.workers != 1:
        print(f"[{args.arm}] Jev enabled; forcing workers=1 to protect summary writes", flush=True)

    for k in range(1, args.rounds + 1):
        if native:
            out_dir = arm_dir / f"round-{k}"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = arm_dir / f"r{k}"
        if jev_enabled and out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(
                f"Jev output directory is nonempty: {out_dir}; use a fresh output directory"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        root = out_dir / "_root"
        prepare_root(root)
        log = out_dir / "round.log"

        env = dict(os.environ)
        env["APPWORLD_ROOT"] = str(root)
        env.setdefault("OPENAI_API_KEY", "dummy")
        env.setdefault("MODEL_SERVER_URL", "")

        def job(task: str) -> tuple[str, int]:
            while not mem_ok():
                time.sleep(30)
            if native:
                cmd = [PY, str(REPO / "experiments" / "run_native.py"),
                       "--model", model, "--config", str(config), "--task-id", task,
                       # native 入口把 result.json 直接写到 --output-dir 下，
                       # 必须一任务一个子目录，否则 90 个任务互相覆盖。
                       "--output-dir", str(out_dir / task), "--dataset-name", "train"]
            else:
                cmd = [PY, str(REPO / "experiments" / "run_appworld.py"),
                       "--config", str(config), "--mode", "classifier_route",
                       "--task-id", task, "--output-dir", str(out_dir),
                       "--experiment-name", f"{args.arm}-{task}"]
            return run_one(cmd, env, log, timeout=args.timeout)

        started = time.time()
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for task, rc in pool.map(job, ids):
                if rc != 0:
                    failures.append((task, rc))
                    print(f"  [fail] {task} rc={rc}", flush=True)
        # 并发写同一张 summary 会丢行（读-改-写竞态）。这里按盘上的行补跑缺的实例，
        # **串行**补 —— 串行不会丢。
        missing = _missing_ids(out_dir, ids, native)
        if jev_enabled:
            if missing:
                print(
                    f"[{args.arm} r{k}] Jev no automatic rerun; "
                    f"missing/degraded={len(missing)} {missing[:8]}", flush=True
                )
            if failures:
                print(
                    f"[{args.arm} r{k}] Jev failures are listed only: "
                    f"{[task for task, _ in failures][:8]}", flush=True
                )
        elif missing:
            print(f"[{args.arm} r{k}] 补跑 {len(missing)} 个缺行（串行）", flush=True)
            for task in missing:
                job(task)
        dur = (time.time() - started) / 60
        print(f"[{args.arm} r{k}] {len(ids) - len(failures)}/{len(ids)} 完成，{dur:.0f} 分钟", flush=True)
        if failures:
            print(f"  失败 {len(failures)} 个：{[t for t, _ in failures][:5]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
