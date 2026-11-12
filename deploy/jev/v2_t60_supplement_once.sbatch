#!/bin/bash
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=01:30:00
#SBATCH --job-name=jev-v2-t60-supp
#SBATCH --output=/share/home/shenwsh3/ozy/span-offload-jev-prompts/runs/jev-v2-t60-supp-%j.log

set -euo pipefail
: "${PROJECT:?Set PROJECT}"
: "${JEV_COMPARE_ACCEPTANCE:?Set reviewed code receipt}"
TASK_ID=fdc4b74_1
ORIGINAL="$PROJECT/runs/jev-v2-rerun-20261031-573095/v2-t60/comparison"
SESSION="$PROJECT/runs/jev-v2-t60-supplement-${SLURM_JOB_ID}"
CELL="$SESSION/cell"
SOURCE_CONFIG="$ORIGINAL/jev_v2_t60.yaml"
CONFIG="$SESSION/jev_v2_t60.yaml"
JEV_PORT="${JEV_PORT:-28901}"
OLLAMA_PORT="${OLLAMA_PORT:-21435}"
CONTAINER="${CONTAINER:-$HOME/singularity/vllm_qwen35_sandbox}"
JEV_MODEL_DIR="${JEV_MODEL_DIR:-$HOME/ozy/models/OpenJev-1.7B}"

module load singularity/4.0.2
module load gcc/9.1.0
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY=127.0.0.1,localhost,192.168.10.24,192.168.10.25,192.168.10.26
export no_proxy="$NO_PROXY" APPWORLD_ROOT="$PROJECT/appworld"
export PYTHONPATH="$PROJECT/appworld/src:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export JEV_MODEL_DIR
mkdir "$SESSION" "$CELL"

.venv/bin/python - "$SOURCE_CONFIG" "$ORIGINAL" "$CONFIG" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
src,root,out=map(Path,sys.argv[1:]); m=json.loads((root/'manifest.json').read_text()); rows=json.loads((root/'results.json').read_text())
ids=m['task_ids']; seen=[r['instance_id'] for r in rows]
if m.get('experiment_group_id')!='V2-T60' or len(rows)!=32 or set(seen)!=set(ids)-{'fdc4b74_1'} or len(seen)!=len(set(seen)):
    raise SystemExit('Original T60 no longer has exactly the expected missing task')
if m['git_commit_sha']!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():
    raise SystemExit('Server code revision differs from original T60')
if hashlib.sha256(src.read_bytes()).hexdigest()!=m['config_sha256']['jev_v2_t60']:
    raise SystemExit('Original T60 config hash mismatch')
c=yaml.safe_load(src.read_bytes()); c['runner']['jev'].update(endpoint=f"http://127.0.0.1:{os.environ['JEV_PORT']}",input_mode='context',mode='active')
c['runner']['m4_memory']['ollama_base_url']=f"http://127.0.0.1:{os.environ['OLLAMA_PORT']}"
out.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
PY

jev_pid= ollama_pid=
cleanup() { for pid in "$jev_pid" "$ollama_pid"; do if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi; done; }
trap cleanup EXIT
trap 'exit 143' TERM INT
singularity exec --nv --cleanenv --bind /share:/share --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=2 \
  "$CONTAINER" "$HOME/ozy/jev-service-env/bin/wev" serve --model "$JEV_MODEL_DIR" --device cuda --dtype bf16 --host 127.0.0.1 --port "$JEV_PORT" > "$SESSION/jev-service.log" 2>&1 &
jev_pid=$!
singularity exec --nv --cleanenv --bind /share:/share --env OLLAMA_MODELS="$HOME/ozy/ollama-models" --env OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
  --env OLLAMA_CONTEXT_LENGTH=8192 --env OLLAMA_KEEP_ALIVE=-1 --env OLLAMA_NUM_PARALLEL=1 --env OLLAMA_MAX_LOADED_MODELS=1 \
  "$CONTAINER" "$HOME/ozy/tools/ollama-0.11.11/bin/ollama" serve > "$SESSION/embedding-service.log" 2>&1 &
ollama_pid=$!
ready=0
for _ in $(seq 1 240); do
  kill -0 "$jev_pid" 2>/dev/null && kill -0 "$ollama_pid" 2>/dev/null || { echo 'Owned service exited'; exit 1; }
  if curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$JEV_PORT/v1/models" > "$SESSION/jev-models.json" 2>/dev/null && \
     curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:$OLLAMA_PORT/api/tags" > "$SESSION/embedding-tags.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1

.venv/bin/python - "$CONFIG" "$SESSION" <<'PY'
import json,os,sys,time
from dataclasses import replace
from pathlib import Path
import httpx,yaml
from deploy.jev.openjev_preflight import OpenJevPreflight
from traceoffload.m1.embedding_client import OllamaEmbeddingClient
from traceoffload.m2.jev.client import HttpJevClient
from traceoffload.m2.jev.input_builder import ROUTE_CONTEXT_INSTRUCTIONS_V3,ROUTE_CRITERIA_V2
from traceoffload.m2.jev.types import load_jev_config
c=yaml.safe_load(Path(sys.argv[1]).read_bytes()); root=Path(sys.argv[2])
for kind in ('large','small'):
    url=c['models'][kind]['api_base'].rstrip('/')+'/models'; expected=c['models'][kind].get('served_model_name') or c['models'][kind]['model_id']
    with httpx.Client(trust_env=False,timeout=60) as h: r=h.get(url); r.raise_for_status()
    if expected not in {x.get('id') for x in r.json().get('data',[])}: raise RuntimeError(kind+' model ID mismatch')
    (root/(kind+'-models.json')).write_text(r.text,encoding='utf-8')
v=OllamaEmbeddingClient(base_url=c['runner']['m4_memory']['ollama_base_url'],timeout_seconds=120).embed_one('[GOAL] Check requested item\n\n')
if v.dimension!=1024: raise RuntimeError('Embedding dimension mismatch')
j=load_jev_config(c,mode='classifier_route'); client=HttpJevClient(replace(j,request_timeout_ms=120000),input_preflight=OpenJevPreflight.from_directory(os.environ['JEV_MODEL_DIR']))
b={'model':j.model_id,'state':{'model_info':{'large_model':c['models']['large']['model_id'],'small_model':c['models']['small']['model_id']},'context':{'goal_text':'Check the requested item','goal_available':True}},'questions':{'route':{'type':'choice','instructions':ROUTE_CONTEXT_INSTRUCTIONS_V3,'criteria':dict(ROUTE_CRITERIA_V2)}}}
w=client.predict(b,timeout_ms=120000)
if w.get('status')!='ok': raise RuntimeError('Jev warmup failed')
(root/'jev-warmup.json').write_text(json.dumps({'status':w['status'],'model':w.get('model')}),encoding='utf-8')
print('ALL_SERVICES_WARMED',flush=True)
PY

export JEV_MODEL_DIR JEV_COMPARE_ACTION=worker JEV_COMPARE_GROUP=jev_v2_t60 JEV_COMPARE_TASK="$TASK_ID"
export JEV_COMPARE_CONFIG="$CONFIG" JEV_COMPARE_CELL="$CELL" JEV_COMPARE_SPLIT=test_challenge
export JEV_COMPARE_EXPERIMENT="routercmp-v2-t60-supplement-${TASK_ID}-${SLURM_JOB_ID}"
timeout --signal=TERM --kill-after=30s 930s .venv/bin/python -u -m experiments.compare_routers
.venv/bin/python - "$CELL/result.json" <<'PY'
import json,sys
from pathlib import Path
from experiments.compare_routers import acceptance_errors,write_json
p=Path(sys.argv[1]); row=json.loads(p.read_text(encoding='utf-8')); errors=acceptance_errors('jev_v2_t60',row)
write_json(p.parent/'acceptance.json',{'acceptance_kind':'single_task_supplement','ready_for_comparison':not errors,'errors':errors,'task_id':row.get('instance_id')})
if errors: raise SystemExit('Supplement measurement failed acceptance; original T60 remains unchanged')
print('SUPPLEMENT_MEASUREMENT_ACCEPTED',row.get('task_success'),flush=True)
PY

.venv/bin/python - "$CELL/result.json" "$ORIGINAL" "$SESSION" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from experiments.compare_routers import _estimated_execution_cost_usd, summarize, write_json

result_path, comparison, supplement = map(Path, sys.argv[1:])
manifest = json.loads((comparison / "manifest.json").read_text(encoding="utf-8"))
original = json.loads((comparison / "results.json").read_text(encoding="utf-8"))
row = json.loads(result_path.read_text(encoding="utf-8"))
expected_ids = set(manifest["task_ids"])
present = [item["instance_id"] for item in original]
if manifest.get("experiment_group_id") != "V2-T60":
    raise SystemExit("Source manifest is not V2-T60")
if len(original) != 32 or len(present) != len(set(present)) or set(present) != expected_ids - {"fdc4b74_1"}:
    raise SystemExit("Original T60 results changed; refusing to merge supplement")
if row.get("instance_id") != "fdc4b74_1" or row.get("valid_for_comparison") is not True:
    raise SystemExit("Supplement row identity or acceptance is invalid")
row.update(group="jev_v2_t60", stage="batch", run_id=manifest["run_id"],
    git_commit_sha=manifest["git_commit_sha"], experiment_group_id="V2-T60", prompt_version="v2",
    prompt_sha256=(row.get("jev_audit") or {}).get("prompt_sha256"),
    price_table_version=manifest["price_table_version"],
    config_sha256=manifest["config_sha256"]["jev_v2_t60"],
    estimated_execution_cost_usd=_estimated_execution_cost_usd(row),
    supplemental_attempt={"job_id": os.environ["SLURM_JOB_ID"],
        "output_dir": str(supplement), "source_run_id": manifest["run_id"],
        "effective_config_sha256": hashlib.sha256((supplement / "jev_v2_t60.yaml").read_bytes()).hexdigest()})
merged = original + [row]
if len(merged) != 33 or {item["instance_id"] for item in merged} != expected_ids:
    raise SystemExit("Merged T60 would not match the original 33-task manifest")

for name in ("results.json", "report.json", "progress.jsonl", "batch-error.json"):
    src = comparison / name
    if src.exists():
        shutil.copy2(src, supplement / ("premerge-" + name))
top_report = comparison.parents[1] / "v2-t60-report.json"
if top_report.exists():
    shutil.copy2(top_report, supplement / "premerge-v2-t60-report.json")

write_json(comparison / "results.json", merged)
with (comparison / "progress.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"completed": 33, "total": 33, "task_id": "fdc4b74_1",
        "group": "jev_v2_t60", "success": row["task_success"], "supplemental": True}) + "\n")
report = summarize(merged, groups=["jev_v2_t60"], profile="v2-t60")
report["complete"] = True
report["supplemental_completion"] = {"task_id": "fdc4b74_1", "job_id": os.environ["SLURM_JOB_ID"],
    "source_run_id": manifest["run_id"]}
write_json(comparison / "report.json", report)
write_json(top_report, report)
write_json(supplement / "continuation.json", {"merged": True, "task_id": "fdc4b74_1",
    "original_comparison": str(comparison), "result_count": len(merged), "report_complete": True})
print("V2_T60_SUPPLEMENT_MERGED", row["task_success"], flush=True)
PY
