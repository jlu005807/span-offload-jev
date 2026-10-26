#!/bin/bash
# Run as one srun step inside an existing GPU allocation. Never on a login node.
set -euo pipefail
: "${JEV_SESSION_DIR:?Set JEV_SESSION_DIR}"
: "${JEV_CONTAINER:?Set JEV_CONTAINER}"
: "${JEV_ENV:?Set JEV_ENV}"
: "${JEV_MODEL_DIR:?Set JEV_MODEL_DIR}"
mkdir -p "$JEV_SESSION_DIR"
singularity exec --nv --cleanenv --bind /share:/share \
  --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env OMP_NUM_THREADS=2 \
  "$JEV_CONTAINER" "$JEV_ENV/bin/wev" serve \
  --model "$JEV_MODEL_DIR" --device cuda --dtype bf16 --host 0.0.0.0 --port 8900 &
server_pid=$!
monitor() {
  while kill -0 "$server_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader >> "$JEV_SESSION_DIR/gpu-memory.csv" || true
    date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$JEV_SESSION_DIR/gpu-process-memory.log"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
      >> "$JEV_SESSION_DIR/gpu-process-memory.log" || true
    sleep 15
  done
}
monitor &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
wait "$server_pid"
