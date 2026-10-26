#!/bin/bash
# Run a four_features single-task gate, then the fixed 90-task batch using
# one shared Jev process inside the user's existing single-GPU SLM allocation.
#SBATCH --partition=x86_64
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --job-name=jev-fourfeat-90
#SBATCH --output=/share/home/shenwsh3/ozy/span-offload-jev/runs/jev-four-features-%j.log

set -euo pipefail
: "${SLM_JOB_ID:?Set SLM_JOB_ID to the newly submitted SLM GPU job}"
service_step_pid=
cleanup() {
  if [[ -n "$service_step_pid" ]]; then
    kill "$service_step_pid" 2>/dev/null || true
    wait "$service_step_pid" 2>/dev/null || true
  fi
  # Set only for a fresh SLM allocation submitted for this experiment.
  # Leave unset when sharing an independently managed SLM service.
  if [[ "${CANCEL_OWN_SLM_ON_EXIT:-0}" == "1" ]]; then
    scancel "$SLM_JOB_ID" || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
PROJECT=/share/home/shenwsh3/ozy/span-offload-jev
cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export PROJECT
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost,192.168.10.25,192.168.10.26
export no_proxy="$NO_PROXY"
export JEV_CONTAINER="$HOME/singularity/vllm_qwen35_sandbox"
export JEV_ENV="$HOME/ozy/jev-service-env"
export JEV_MODEL_DIR="$HOME/ozy/models/OpenJev-1.7B"
export APPWORLD_ROOT="$PROJECT/appworld"

RUN_ROOT="$PROJECT/runs/jev-four-features-90-${SLURM_JOB_ID}"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/service"
export JEV_RUN_CONFIG="$RUN_ROOT/config.yaml"
cp "${JEV_SOURCE_CONFIG:-$PROJECT/runs/setup/four-features-config.yaml}" "$JEV_RUN_CONFIG"
python3 - "$PROJECT" "$RUN_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
project, output = map(Path, sys.argv[1:])
names = ['experiments/task_outcomes.py', 'experiments/run_appworld.py',
         'experiments/report.py', 'deploy/jev/run_single_task.py', 'deploy/jev/run_batch.py']
(output / 'source-hashes.json').write_text(json.dumps({
    'task_failure_policy': 'record-and-continue-v1',
    'sha256': {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in names}
}, indent=2))
PY
export JEV_RUN_OUTPUT="$RUN_ROOT/single"
export JEV_ACCEPTANCE_FILE="$RUN_ROOT/single/acceptance.json"
export JEV_TASK_MANIFEST="$PROJECT/experiments/instances-90.txt"
export JEV_BATCH_ROOT="$RUN_ROOT/batch"
export JEV_SESSION_DIR="$RUN_ROOT/service"
export JEV_RUN_OUTPUT_BATCH="$RUN_ROOT/batch/r1"

test -f "$JEV_RUN_CONFIG"
test -f "$JEV_TASK_MANIFEST"
printf '%s\n' "$RUN_ROOT" > "$PROJECT/runs/setup/current-four-features-root.txt"
# A started Slurm allocation does not mean the vLLM service has loaded yet.
# Wait for SLM before loading Jev and before the integration smoke task.
slm_deadline=$((SECONDS + 1800))
slm_ready=0
while ((SECONDS < slm_deadline)); do
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
    http://192.168.10.26:8800/v1/models > "$JEV_SESSION_DIR/slm-models.json" 2>/dev/null; then
    slm_ready=1
    break
  fi
  sleep 5
done
if [[ "$slm_ready" != "1" ]]; then
  echo 'SLM service did not become ready; stopping the experiment'
  exit 1
fi
if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
  http://192.168.10.26:8900/v1/models >/dev/null 2>&1; then
  echo 'Port 8900 already serves a model; refusing to take ownership of an existing service'
  exit 1
fi

srun --jobid="$SLM_JOB_ID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  bash "$PROJECT/deploy/jev/shared_gpu_service.sh" > "$JEV_SESSION_DIR/jev-server.log" 2>&1 &
service_step_pid=$!

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "$service_step_pid" 2>/dev/null; then
    echo 'Jev service step exited; inspect service log'
    exit 1
  fi
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
    http://192.168.10.26:8900/v1/models > "$JEV_SESSION_DIR/jev-models.json"; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1
curl --noproxy '*' --connect-timeout 3 --max-time 120 -fsS \
  http://192.168.10.26:8900/v1/systemone -H 'Content-Type: application/json' \
  --data-binary @"$PROJECT/runs/setup/jev-smoke-request.json" > "$JEV_SESSION_DIR/warmup.json"

"$PROJECT/.venv/bin/python" - <<'PY'
import os, yaml
from pathlib import Path
from deploy.jev.run_batch import check_services
check_services(yaml.safe_load(Path(os.environ['JEV_RUN_CONFIG']).read_bytes()))
PY

"$PROJECT/.venv/bin/python" -u deploy/jev/run_single_task.py
"$PROJECT/.venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
gate = json.loads(Path(os.environ['JEV_ACCEPTANCE_FILE']).read_text())
if gate.get('ready_for_batch') is not True:
    raise SystemExit('Four-features single-task acceptance failed; full batch not started')
PY

export JEV_RUN_OUTPUT="$JEV_RUN_OUTPUT_BATCH"
"$PROJECT/.venv/bin/python" -u deploy/jev/run_batch.py
