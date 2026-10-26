#!/bin/bash
#SBATCH --partition=x86_64
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --job-name=jev-appworld-90
# CPU partition reports 1MB/node; do not request --mem here.
set -euo pipefail
: "${PROJECT:?Set PROJECT}"
: "${SLM_JOB_ID:?Set SLM_JOB_ID to your own running GPU allocation}"
: "${JEV_SESSION_DIR:?Set a new service session directory}"
cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost,192.168.10.25,192.168.10.26
export no_proxy="$NO_PROXY"
export JEV_CONTAINER="$HOME/singularity/vllm_qwen35_sandbox"
export JEV_ENV="$HOME/ozy/jev-service-env"
export JEV_MODEL_DIR="$HOME/ozy/models/OpenJev-1.7B"
export APPWORLD_ROOT="$PROJECT/appworld"
mkdir "$JEV_SESSION_DIR"
# The single-task gate is checked again by run_batch.py, including config hash.
"$PROJECT/.venv/bin/python" -c 'import json,os; assert json.load(open(os.environ["JEV_ACCEPTANCE_FILE"]))["ready_for_batch"] is True'
if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
  http://192.168.10.26:8900/v1/models >/dev/null 2>&1; then
  echo 'Port 8900 already serves a model; refusing to take ownership of an existing service'
  exit 1
fi
srun --jobid="$SLM_JOB_ID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  bash "$PROJECT/deploy/jev/shared_gpu_service.sh" > "$JEV_SESSION_DIR/jev-server.log" 2>&1 &
service_step_pid=$!
cleanup() {
  kill "$service_step_pid" 2>/dev/null || true
  wait "$service_step_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
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
"$PROJECT/.venv/bin/python" -u deploy/jev/run_batch.py
