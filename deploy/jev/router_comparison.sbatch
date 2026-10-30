#!/bin/bash
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=04:30:00
#SBATCH --job-name=router-compare20

set -euo pipefail
: "${PROJECT:?Set PROJECT}"
: "${JEV_COMPARE_BASE:?Set a validated base configuration}"
: "${JEV_COMPARE_TASKS:?Set the fixed development task list}"
: "${JEV_COMPARE_ACCEPTANCE:?Set the reviewed server-test receipt}"
cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY=127.0.0.1,localhost,192.168.10.24,192.168.10.25,192.168.10.26
export no_proxy="$NO_PROXY"
export APPWORLD_ROOT="$PROJECT/appworld"
export JEV_MODEL_DIR="$HOME/ozy/models/OpenJev-1.7B"
export JEV_COMPARE_SESSION="$PROJECT/runs/router-compare20-${SLURM_JOB_ID}"
export JEV_COMPARE_CONFIG="$JEV_COMPARE_SESSION/base.yaml"
export JEV_COMPARE_OUTPUT="$JEV_COMPARE_SESSION/comparison"
export JEV_PORT=18901 OLLAMA_PORT=11435
CONTAINER="$HOME/singularity/vllm_qwen35_sandbox"
mkdir "$JEV_COMPARE_SESSION"

.venv/bin/python - <<'PY'
import json, os, socket
from pathlib import Path
import yaml
from experiments.jev_fixed_span_data import file_hash
p = Path(os.environ['JEV_COMPARE_SESSION'])
gate = json.loads(Path(os.environ['JEV_COMPARE_ACCEPTANCE']).read_text())
if gate.get('ready_for_server_smoke') is not True:
    raise RuntimeError('Reviewed code acceptance has not passed')
for name, expected in gate['sha256'].items():
    if file_hash(name) != expected:
        raise RuntimeError('Reviewed implementation or inputs changed: ' + name)
for port in [int(os.environ['JEV_PORT']), int(os.environ['OLLAMA_PORT'])]:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
config = yaml.safe_load(Path(os.environ['JEV_COMPARE_BASE']).read_bytes())
config['runner']['jev']['endpoint'] = 'http://127.0.0.1:' + os.environ['JEV_PORT']
config['runner']['m4_memory']['ollama_base_url'] = 'http://127.0.0.1:' + os.environ['OLLAMA_PORT']
config['runner']['m4_memory']['embedding_model'] = 'qwen3-embedding:0.6b'
config['runner']['m4_memory']['embedding_mode'] = 'm1_online'
config['runner']['m4_memory']['template_version'] = 'span-embed-v2'
Path(os.environ['JEV_COMPARE_CONFIG']).write_text(yaml.safe_dump(config, sort_keys=False))
names = ['experiments/compare_routers.py', 'experiments/run_native.py', 'experiments/run_appworld.py',
         'experiments/task_outcomes.py', 'traceoffload/runner.py', 'traceoffload/m1/embedding_client.py',
         'traceoffload/m2/jev/input_builder.py', 'router/artifact/l2_add4_twostage_logit.json']
(p/'source-hashes.json').write_text(json.dumps({n:file_hash(n) for n in names}, indent=2))
(p/'tasks.txt').write_bytes(Path(os.environ['JEV_COMPARE_TASKS']).read_bytes())
PY
export JEV_COMPARE_TASKS="$JEV_COMPARE_SESSION/tasks.txt"

jev_pid= ollama_pid=
cleanup() {
  for pid in "$jev_pid" "$ollama_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

singularity exec --nv --cleanenv --bind /share:/share \
  --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=2 \
  "$CONTAINER" "$HOME/ozy/jev-service-env/bin/wev" serve \
  --model "$JEV_MODEL_DIR" --device cuda --dtype bf16 --host 127.0.0.1 --port "$JEV_PORT" \
  > "$JEV_COMPARE_SESSION/jev-service.log" 2>&1 &
jev_pid=$!
singularity exec --nv --cleanenv --bind /share:/share \
  --env OLLAMA_MODELS="$HOME/ozy/ollama-models" --env OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
  --env OLLAMA_CONTEXT_LENGTH=8192 --env OLLAMA_KEEP_ALIVE=-1 \
  --env OLLAMA_NUM_PARALLEL=1 --env OLLAMA_MAX_LOADED_MODELS=1 \
  "$CONTAINER" "$HOME/ozy/tools/ollama-0.11.11/bin/ollama" serve \
  > "$JEV_COMPARE_SESSION/embedding-service.log" 2>&1 &
ollama_pid=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "$jev_pid" 2>/dev/null && kill -0 "$ollama_pid" 2>/dev/null || { echo 'Owned service exited'; exit 1; }
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$JEV_PORT/v1/models" \
       > "$JEV_COMPARE_SESSION/jev-models.json" 2>/dev/null && \
     curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$OLLAMA_PORT/api/tags" \
       > "$JEV_COMPARE_SESSION/embedding-tags.json" 2>/dev/null; then ready=1; break; fi
  sleep 5
done
test "$ready" -eq 1

# Slurm after:<job> means allocated/started, not that vLLM is serving yet.
SMALL_API=$(.venv/bin/python -c 'import os,yaml; print(yaml.safe_load(open(os.environ["JEV_COMPARE_CONFIG"]))["models"]["small"]["api_base"])')
SLM_DEADLINE=$((SECONDS + 1800))
until curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "${SMALL_API}/models" > "$JEV_COMPARE_SESSION/small-models-ready.json" 2>/dev/null; do
  kill -0 "$jev_pid" 2>/dev/null && kill -0 "$ollama_pid" 2>/dev/null || { echo 'Owned service exited while waiting for SLM'; exit 1; }
  if (( SECONDS >= SLM_DEADLINE )); then echo 'SLM did not become ready within 30 minutes'; exit 1; fi
  sleep 5
done
echo 'SLM_SERVICE_READY'

.venv/bin/python - <<'PY'
from dataclasses import replace
import json, os
from pathlib import Path
import httpx, yaml
from deploy.jev.openjev_preflight import OpenJevPreflight
from traceoffload.m1.embedding_client import OllamaEmbeddingClient
from traceoffload.m2.jev.types import load_jev_config
from traceoffload.m2.jev.client import HttpJevClient
from traceoffload.m2.jev.input_builder import ROUTE_INSTRUCTIONS_V1, ROUTE_CRITERIA_V1
config = yaml.safe_load(Path(os.environ['JEV_COMPARE_CONFIG']).read_bytes())
root = Path(os.environ['JEV_COMPARE_SESSION'])
embedding = OllamaEmbeddingClient(base_url=config['runner']['m4_memory']['ollama_base_url'], timeout_seconds=120)
vector = embedding.embed_one('[GOAL] Check the requested item\n\n')
assert vector.dimension == 1024
(root/'embedding-warmup.json').write_text(json.dumps({'model':vector.model,'dimension':vector.dimension,
    'prompt_eval_count':vector.prompt_eval_count,'client_normalize':True,'server_context_length':8192}))
j = load_jev_config(config, mode='classifier_route')
client = HttpJevClient(replace(j,request_timeout_ms=120000), input_preflight=OpenJevPreflight.from_directory(os.environ['JEV_MODEL_DIR']))
body = {'model':j.model_id,'state':{'model_info':{'large_model':config['models']['large']['model_id'],
    'small_model':config['models']['small']['model_id']},'context':{'goal_text':'Check the requested item'}},
    'questions':{'route':{'type':'choice','instructions':ROUTE_INSTRUCTIONS_V1,'criteria':dict(ROUTE_CRITERIA_V1)}}}
(root/'jev-warmup.json').write_text(json.dumps(client.predict(body,timeout_ms=120000)))
with httpx.Client(trust_env=False,timeout=10) as h:
    for kind in ['large','small']:
        r=h.get(config['models'][kind]['api_base']+'/models');r.raise_for_status()
        (root/(kind+'-models.json')).write_text(r.text)
    (root/'embedding-placement.json').write_text(h.get(embedding._base_url+'/api/ps').text)
print('ALL_SERVICES_WARMED',flush=True)
PY
.venv/bin/python -u -m experiments.compare_routers
