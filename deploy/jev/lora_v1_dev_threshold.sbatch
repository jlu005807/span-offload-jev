#!/bin/bash
# One isolated LoRA-V1 threshold over the 56 official AppWorld dev tasks.
# Submit one job per threshold; use Slurm dependencies to avoid stressing the
# shared Qwen services with concurrent benchmark traffic.
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=10:00:00
#SBATCH --job-name=jev-lora-v1-dev
#SBATCH --output=/share/home/shenwsh3/ozy/span-offload-jev-prompts/runs/jev-lora-v1-dev-%j.log

set -euo pipefail

PROJECT="${PROJECT:-/share/home/shenwsh3/ozy/span-offload-jev-prompts}"
LORA_PROJECT="${LORA_PROJECT:-/share/home/shenwsh3/ozy/jev-appworld-lora}"
JEV_RUN_DIR="${JEV_RUN_DIR:-$LORA_PROJECT/runs/finetune/appworld-v1-context-r16-lr5e-5-seed0-20261107}"
JEV_PREDICTIONS="${JEV_PREDICTIONS:-$LORA_PROJECT/runs/evaluations/finetuned-dev-full-20261107/predictions.jsonl}"
JEV_METRICS="${JEV_METRICS:-$LORA_PROJECT/runs/evaluations/finetuned-dev-full-20261107/metrics.json}"
JEV_PREFLIGHT_MODEL_DIR="${JEV_PREFLIGHT_MODEL_DIR:-$HOME/ozy/models/OpenJev-1.7B}"
CONTAINER="${CONTAINER:-$HOME/singularity/vllm_qwen35_sandbox}"
OPENJEV_VENV="${OPENJEV_VENV:-$LORA_PROJECT/.venv-openjev-container}"
JEV_THRESHOLD="${JEV_THRESHOLD:?Set JEV_THRESHOLD to one of 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8}"
JEV_LARGE_API_BASE="${JEV_LARGE_API_BASE:?Set to the verified Qwen3.8-27B API base}"
JEV_SMALL_API_BASE="${JEV_SMALL_API_BASE:?Set to the verified Qwen3.5-9B API base}"
JEV_COMPARE_BASE="${JEV_COMPARE_BASE:-$PROJECT/configs/arms/jev-active.yaml}"

case "$JEV_THRESHOLD" in
  0.5) PROFILE=lora-v1-t50 ;;
  0.55) PROFILE=lora-v1-t55 ;;
  0.6) PROFILE=lora-v1-t60 ;;
  0.65) PROFILE=lora-v1-t65 ;;
  0.7) PROFILE=lora-v1-t70 ;;
  0.75) PROFILE=lora-v1-t75 ;;
  0.8) PROFILE=lora-v1-t80 ;;
  *) echo "Unsupported JEV_THRESHOLD: $JEV_THRESHOLD" >&2; exit 2 ;;
esac

cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY="127.0.0.1,localhost,192.168.10.0/24"
export no_proxy="$NO_PROXY"
export APPWORLD_ROOT="$PROJECT/appworld"
export PYTHONPATH="$APPWORLD_ROOT/src:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export JEV_MODEL_DIR="$JEV_PREFLIGHT_MODEL_DIR"

RUN_ROOT="$PROJECT/runs/jev-lora-v1-${PROFILE}-${SLURM_JOB_ID}"
mkdir "$RUN_ROOT"
export JEV_COMPARE_SESSION="$RUN_ROOT"
export JEV_COMPARE_CONFIG="$RUN_ROOT/base.yaml"
export JEV_COMPARE_TASKS="$RUN_ROOT/tasks.txt"
export JEV_COMPARE_PROFILE="$PROFILE"
export JEV_COMPARE_OUTPUT="$RUN_ROOT/comparison"
JEV_PORT="${JEV_PORT:-$((19000 + SLURM_JOB_ID % 1000))}"
export JEV_PORT

test -f "$JEV_COMPARE_BASE"
test -f "$JEV_PREDICTIONS"
test -f "$JEV_METRICS"
test -f "$JEV_RUN_DIR/head.pt"
test -f "$JEV_RUN_DIR/adapter_model.safetensors"
test -x "$OPENJEV_VENV/bin/wev"
test -d "$JEV_PREFLIGHT_MODEL_DIR"
test -e "$CONTAINER"

# Reconstruct the task set from the exact prediction artifact and verify that
# every task belongs to AppWorld's official dev split before requesting work.
.venv/bin/python - "$JEV_PREDICTIONS" "$JEV_METRICS" "$JEV_COMPARE_TASKS" <<'PY'
import json
import sys
from pathlib import Path
from appworld.task import load_task_ids

predictions_path, metrics_path, tasks_path = map(Path, sys.argv[1:])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
task_ids = list(dict.fromkeys(row.get("task_id") for row in rows))
if metrics.get("split") != "dev" or metrics.get("sample_count") != len(rows) or len(rows) != 810:
    raise SystemExit("Expected the complete 810-row fine-tuned dev prediction artifact")
if len(task_ids) != 56 or any(not isinstance(task, str) or not task for task in task_ids):
    raise SystemExit(f"Expected 56 unique task IDs, found {len(task_ids)}")
official_dev = set(load_task_ids("dev"))
if not set(task_ids) <= official_dev:
    raise SystemExit("Prediction artifact contains IDs outside the official AppWorld dev split")
tasks_path.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
print(f"VERIFIED_DEV_TASKS {len(task_ids)} prediction_rows={len(rows)}")
PY

# Snapshot code identity and inputs without modifying the source or prediction files.
.venv/bin/python - "$RUN_ROOT" "$JEV_COMPARE_BASE" "$JEV_PREDICTIONS" "$JEV_METRICS" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root, config, predictions, metrics = map(Path, sys.argv[1:])
project = Path.cwd()
files = [Path("experiments/compare_routers.py"), Path("deploy/jev/lora_v1_dev_threshold.sbatch")]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "git_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "working_tree_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
    "source_sha256": {str(name): digest(project / name) for name in files},
    "base_config_sha256": digest(config),
    "dev_predictions_sha256": digest(predictions),
    "dev_metrics_sha256": digest(metrics),
}
(root / "source-snapshot.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
PY

jev_pid=
cleanup() {
  if [[ -n "$jev_pid" ]]; then kill "$jev_pid" 2>/dev/null || true; wait "$jev_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$JEV_PORT/v1/models" >/dev/null 2>&1; then
  echo "Refusing to take ownership of occupied Jev port $JEV_PORT" >&2
  exit 1
fi

singularity exec --nv --cleanenv --bind /share:/share \
  --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env OMP_NUM_THREADS=2 --env CUDA_MODULE_LOADING=LAZY \
  "$CONTAINER" "$OPENJEV_VENV/bin/wev" serve --model "$JEV_RUN_DIR" --device cuda --dtype bf16 \
  --host 127.0.0.1 --port "$JEV_PORT" > "$RUN_ROOT/jev-service.log" 2>&1 &
jev_pid=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "$jev_pid" 2>/dev/null || { echo 'LoRA Jev service exited during startup' >&2; exit 1; }
  if curl --noproxy '*' --connect-timeout 2 --max-time 4 -fsS \
      "http://127.0.0.1:$JEV_PORT/v1/models" > "$RUN_ROOT/jev-models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1

.venv/bin/python - "$JEV_COMPARE_BASE" "$JEV_COMPARE_CONFIG" "$RUN_ROOT/jev-models.json" "$JEV_PORT" "$JEV_THRESHOLD" "$JEV_LARGE_API_BASE" "$JEV_SMALL_API_BASE" <<'PY'
import json, sys
from pathlib import Path
import yaml

base_path, out_path, models_path, port, threshold, large_api, small_api = sys.argv[1:]
config = yaml.safe_load(Path(base_path).read_bytes())
served = json.loads(Path(models_path).read_text(encoding="utf-8"))["data"]
if len(served) != 1 or not served[0].get("id"):
    raise SystemExit("Expected exactly one served LoRA Jev model")
config["models"]["large"]["api_base"] = large_api.rstrip("/")
config["models"]["small"]["api_base"] = small_api.rstrip("/")
config["runner"]["jev"].update(
    enabled=True, mode="active", input_mode="context", prompt_version="v1",
    threshold=float(threshold), endpoint=f"http://127.0.0.1:{port}",
    model_id=served[0]["id"], expected_revision=None,
)
config["runner"].setdefault("m4_memory", {})["enabled"] = False
config["runner"].setdefault("l2_add4", {})["enabled"] = False
config["runner"].setdefault("l2_v4", {})["enabled"] = False
config["runner"]["shadow_route"] = {"enabled": False}
config["runner"].setdefault("m4_memory", {})["guard_enabled"] = True
Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print("CONFIGURED_LORA_MODEL", served[0]["id"], "threshold", threshold)
PY

# compare_routers runs a smoke task first and writes an acceptance receipt;
# it will not start the 56-task batch unless that exact profile passes.
.venv/bin/python -u -m experiments.compare_routers
cp "$JEV_COMPARE_OUTPUT/report.json" "$RUN_ROOT/report.json"
echo "LORA_THRESHOLD_COMPLETE $PROFILE $RUN_ROOT"
