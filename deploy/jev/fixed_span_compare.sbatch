#!/bin/bash
# Only Jev is loaded. No Agent, LLM or SLM inference is performed.
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=01:00:00
#SBATCH --job-name=jev-fixed-span

set -euo pipefail
: "${PROJECT:?Set PROJECT}"
: "${JEV_FIXED_DATASET:?Set JEV_FIXED_DATASET}"
: "${JEV_FIXED_ACCEPTANCE:?Set JEV_FIXED_ACCEPTANCE}"
cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export JEV_MODEL_DIR="${JEV_MODEL_DIR:-$HOME/ozy/models/OpenJev-1.7B}"
export JEV_CONTAINER="${JEV_CONTAINER:-$HOME/singularity/vllm_qwen35_sandbox}"
export JEV_ENV="${JEV_ENV:-$HOME/ozy/jev-service-env}"
export JEV_PORT="${JEV_PORT:-18900}"
export JEV_SESSION_DIR="$PROJECT/runs/jev-fixed-span-${SLURM_JOB_ID}"
export JEV_RUN_OUTPUT="$JEV_SESSION_DIR/inference"
export JEV_RUN_CONFIG="$JEV_SESSION_DIR/config.yaml"
mkdir "$JEV_SESSION_DIR"

# Reject stale acceptance and an occupied port before taking service ownership.
"$PROJECT/.venv/bin/python" - <<'PY'
import json, os, socket
from pathlib import Path
import yaml
from experiments.jev_fixed_span_data import check_acceptance, file_hash
dataset = Path(os.environ['JEV_FIXED_DATASET'])
check_acceptance(dataset, os.environ['JEV_FIXED_ACCEPTANCE'])
port = int(os.environ['JEV_PORT'])
with socket.socket() as sock:
    sock.bind(('127.0.0.1', port))
manifest = json.loads((dataset / 'manifest.json').read_text())
config = manifest['source_config'].copy()
config['endpoint'] = f'http://127.0.0.1:{port}'
Path(os.environ['JEV_RUN_CONFIG']).write_text(yaml.safe_dump({'runner': {'jev': config}}))
names = ['experiments/jev_fixed_span.py', 'experiments/jev_fixed_span_data.py',
         'traceoffload/m2/jev/input_builder.py', 'traceoffload/m2/jev/client.py',
         'traceoffload/m2/jev/types.py', 'deploy/jev/openjev_preflight.py']
Path(os.environ['JEV_SESSION_DIR'], 'source-hashes.json').write_text(json.dumps(
    {name: file_hash(name) for name in names}, indent=2))
PY

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
singularity exec --nv --cleanenv --bind /share:/share \
  --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env OMP_NUM_THREADS=2 \
  "$JEV_CONTAINER" "$JEV_ENV/bin/wev" serve \
  --model "$JEV_MODEL_DIR" --device cuda --dtype bf16 \
  --host 127.0.0.1 --port "$JEV_PORT" > "$JEV_SESSION_DIR/service.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "$server_pid" 2>/dev/null || { echo 'Jev service exited'; exit 1; }
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
    "http://127.0.0.1:${JEV_PORT}/v1/models" > "$JEV_SESSION_DIR/service-models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1

# Warmup is stored separately; the measured calls retain the source 2s timeout.
"$PROJECT/.venv/bin/python" - <<'PY'
from dataclasses import replace
import json, os
from pathlib import Path
import yaml
from experiments.jev_fixed_span_data import load_dataset
from traceoffload.m2.jev.types import load_jev_config
from traceoffload.m2.jev.client import HttpJevClient
from deploy.jev.openjev_preflight import OpenJevPreflight
config = load_jev_config(yaml.safe_load(Path(os.environ['JEV_RUN_CONFIG']).read_bytes()), mode='classifier_route')
models = json.loads(Path(os.environ['JEV_SESSION_DIR'], 'service-models.json').read_text())
assert config.model_id in {m['id'] for m in models['data']}
client = HttpJevClient(replace(config, request_timeout_ms=120000),
    input_preflight=OpenJevPreflight.from_directory(os.environ['JEV_MODEL_DIR']))
sample = load_dataset(os.environ['JEV_FIXED_DATASET'])[0]
response = client.predict(sample['inputs']['combined']['request'], timeout_ms=120000)
Path(os.environ['JEV_SESSION_DIR'], 'warmup.json').write_text(json.dumps(response))
print('JEV_WARMUP_OK', flush=True)
PY

export JEV_FIXED_ACTION=run
"$PROJECT/.venv/bin/python" -u -m experiments.jev_fixed_span
echo 'FIXED_SPAN_EXPERIMENT_COMPLETED'
