#!/bin/bash
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=05:00:00
#SBATCH --job-name=jev-v1-dev

set -euo pipefail
: "${PROJECT:?Set PROJECT}"
: "${JEV_COMPARE_BASE:?Set a validated base configuration}"
: "${JEV_COMPARE_ACCEPTANCE:?Set the reviewed code acceptance receipt}"
JEV_COMPARE_SESSION_NAME="${JEV_COMPARE_SESSION_NAME:-jev-v1-dev}"
export JEV_COMPARE_PROFILE="${JEV_COMPARE_PROFILE:-v1-t50}"
export JEV_COMPARE_SESSION_NAME
cd "$PROJECT"
module load singularity/4.0.2
module load gcc/9.1.0
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY=127.0.0.1,localhost,192.168.10.24,192.168.10.25,192.168.10.26
export no_proxy="$NO_PROXY"
export APPWORLD_ROOT="$PROJECT/appworld"
export PYTHONPATH="$PROJECT/appworld/src:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export JEV_MODEL_DIR="${JEV_MODEL_DIR:-$HOME/ozy/models/OpenJev-1.7B}"
export JEV_COMPARE_SESSION="$PROJECT/runs/${JEV_COMPARE_SESSION_NAME}-${SLURM_JOB_ID}"
export JEV_COMPARE_CONFIG="$JEV_COMPARE_SESSION/base.yaml"
export JEV_COMPARE_OUTPUT="$JEV_COMPARE_SESSION/comparison"
export JEV_PORT="${JEV_PORT:-18901}" OLLAMA_PORT="${OLLAMA_PORT:-11435}"
CONTAINER="${CONTAINER:-$HOME/singularity/vllm_qwen35_sandbox}"
mkdir "$JEV_COMPARE_SESSION"

.venv/bin/python - <<'PY'
import json, os, socket
from hashlib import sha256
import subprocess
from pathlib import Path
import yaml
from appworld.task import load_task_ids

project = Path(os.environ['PROJECT']).resolve()
receipt = json.loads(Path(os.environ['JEV_COMPARE_ACCEPTANCE']).read_text())
if receipt.get('ready_for_server_smoke') is not True:
    raise RuntimeError('Reviewed code acceptance has not passed')
for name, expected in receipt.get('sha256', {}).items():
    source = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=project)
    actual = sha256(source).hexdigest()
    if actual != expected:
        raise RuntimeError('Acceptance fingerprint mismatch: '+name)
base = Path(os.environ['JEV_COMPARE_BASE']).resolve()
if not base.is_relative_to(project):
    raise RuntimeError('Base config must be inside PROJECT')
task_manifest = project/'experiments/instances-jev-v1-validation-33.txt'
tasks = [line.strip() for line in task_manifest.read_text().splitlines() if line.strip()]
official_splits = {name: set(load_task_ids(name)) for name in ['train', 'dev', 'test_normal', 'test_challenge']}
all_official = set().union(*official_splits.values())
if len(tasks) != 33 or len(set(tasks)) != 33:
    raise RuntimeError('Expected 33 unique validation tasks')
if not set(tasks) <= all_official:
    raise RuntimeError('Validation manifest contains unknown AppWorld task IDs')
previous = set(line.strip() for line in (project/'experiments/instances-90.txt').read_text().splitlines() if line.strip())
if set(tasks) & previous:
    raise RuntimeError('Validation manifest overlaps the 90-task test manifest')
if any(not (Path(os.environ['APPWORLD_ROOT'])/'data/tasks'/task).is_dir() for task in tasks):
    raise RuntimeError('AppWorld task data is missing')
config = yaml.safe_load(base.read_bytes())
config['runner']['jev'].update(endpoint='http://127.0.0.1:'+os.environ['JEV_PORT'], input_mode='context', mode='active')
config['runner']['m4_memory']['ollama_base_url'] = 'http://127.0.0.1:'+os.environ['OLLAMA_PORT']
config['runner']['m4_memory']['embedding_mode'] = 'm1_online'
config['runner']['m4_memory']['template_version'] = 'span-embed-v2'
config['runner']['m4_memory']['guard_enabled'] = True
Path(os.environ['JEV_COMPARE_CONFIG']).write_text(yaml.safe_dump(config, sort_keys=False))
for port in [int(os.environ['JEV_PORT']), int(os.environ['OLLAMA_PORT'])]:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
(Path(os.environ['JEV_COMPARE_SESSION'])/'tasks.txt').write_text('\n'.join(tasks)+'\n')
(Path(os.environ['JEV_COMPARE_SESSION'])/'task-selection.json').write_text(json.dumps({
    'split':'project_validation','count':len(tasks),'task_ids':tasks,
    'manifest':'experiments/instances-jev-v1-validation-33.txt'}, indent=2))
print('SELECTED_VALIDATION_TASKS', len(tasks), flush=True)
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
  "$CONTAINER" "$HOME/ozy/jev-service-env/bin/wev" serve --model "$JEV_MODEL_DIR" --device cuda --dtype bf16 \
  --host 127.0.0.1 --port "$JEV_PORT" > "$JEV_COMPARE_SESSION/jev-service.log" 2>&1 &
jev_pid=$!
singularity exec --nv --cleanenv --bind /share:/share \
  --env OLLAMA_MODELS="$HOME/ozy/ollama-models" --env OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
  --env OLLAMA_CONTEXT_LENGTH=8192 --env OLLAMA_KEEP_ALIVE=-1 --env OLLAMA_NUM_PARALLEL=1 --env OLLAMA_MAX_LOADED_MODELS=1 \
  "$CONTAINER" "$HOME/ozy/tools/ollama-0.11.11/bin/ollama" serve > "$JEV_COMPARE_SESSION/embedding-service.log" 2>&1 &
ollama_pid=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "$jev_pid" 2>/dev/null && kill -0 "$ollama_pid" 2>/dev/null || { echo 'Owned service exited'; exit 1; }
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$JEV_PORT/v1/models" > "$JEV_COMPARE_SESSION/jev-models.json" 2>/dev/null && \
     curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$OLLAMA_PORT/api/tags" > "$JEV_COMPARE_SESSION/embedding-tags.json" 2>/dev/null; then ready=1; break; fi
  sleep 5
done
test "$ready" -eq 1

SMALL_API=$(.venv/bin/python -c 'import os,yaml; print(yaml.safe_load(open(os.environ["JEV_COMPARE_CONFIG"]))["models"]["small"]["api_base"])')
SLM_DEADLINE=$((SECONDS + 1800))
until curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "${SMALL_API}/models" > "$JEV_COMPARE_SESSION/small-models-ready.json" 2>/dev/null; do
  kill -0 "$jev_pid" 2>/dev/null && kill -0 "$ollama_pid" 2>/dev/null || { echo 'Owned service exited while waiting for SLM'; exit 1; }
  if (( SECONDS >= SLM_DEADLINE )); then echo 'SLM did not become ready within 30 minutes'; exit 1; fi
  sleep 5
done
echo SLM_SERVICE_READY

.venv/bin/python - <<'PY'
import json, os
from dataclasses import replace
from pathlib import Path
import httpx, yaml
from deploy.jev.openjev_preflight import OpenJevPreflight
from traceoffload.m1.embedding_client import OllamaEmbeddingClient
from traceoffload.m2.jev.types import load_jev_config
from traceoffload.m2.jev.client import HttpJevClient
from traceoffload.m2.jev.input_builder import ROUTE_CONTEXT_INSTRUCTIONS_V3, ROUTE_CRITERIA_V2
config = yaml.safe_load(Path(os.environ['JEV_COMPARE_CONFIG']).read_bytes())
root = Path(os.environ['JEV_COMPARE_SESSION'])
embedding = OllamaEmbeddingClient(base_url=config['runner']['m4_memory']['ollama_base_url'], timeout_seconds=120)
vector = embedding.embed_one('[GOAL] Check the requested item\n\n')
assert vector.dimension == 1024
(root/'embedding-warmup.json').write_text(json.dumps({'model':vector.model,'dimension':vector.dimension,'prompt_eval_count':vector.prompt_eval_count,'client_normalize':True,'server_context_length':8192}))
j = load_jev_config(config, mode='classifier_route')
client = HttpJevClient(replace(j,request_timeout_ms=120000), input_preflight=OpenJevPreflight.from_directory(os.environ['JEV_MODEL_DIR']))
body = {'model':j.model_id,'state':{'model_info':{'large_model':config['models']['large']['model_id'],'small_model':config['models']['small']['model_id']},'context':{'goal_text':'Check the requested item','goal_available':True}},'questions':{'route':{'type':'choice','instructions':ROUTE_CONTEXT_INSTRUCTIONS_V3,'criteria':dict(ROUTE_CRITERIA_V2)}}}
(root/'jev-warmup.json').write_text(json.dumps(client.predict(body,timeout_ms=120000)))
with httpx.Client(trust_env=False,timeout=10) as h:
  for kind in ['large','small']:
    r=h.get(config['models'][kind]['api_base']+'/models'); r.raise_for_status(); (root/(kind+'-models.json')).write_text(r.text)
print('ALL_SERVICES_WARMED', flush=True)
PY

for profile in v1-t50 v1-t60 v1-t70; do
  export JEV_COMPARE_PROFILE="$profile"
  export JEV_COMPARE_OUTPUT="$JEV_COMPARE_SESSION/$profile/comparison"
  mkdir -p "$JEV_COMPARE_SESSION/$profile"
  .venv/bin/python -u -m experiments.compare_routers
  cp "$JEV_COMPARE_OUTPUT/report.json" "$JEV_COMPARE_SESSION/$profile-report.json"
done
