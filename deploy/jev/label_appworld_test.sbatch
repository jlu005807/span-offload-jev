#!/bin/bash
# Label existing test counterfactual traces; no model or AppWorld execution.
#SBATCH --partition=x86_64
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --job-name=jev-label-test
#SBATCH --output=/share/home/shenwsh3/ozy/jev-appworld-lora/runs/label-test-%j.log

set -euo pipefail

# CPU nodes advertise RealMemory=1 MB in Slurm despite ample physical RAM.
# Follow the established train/dev labeling jobs without a memory reservation.

export JEV_LABEL_PREFLIGHT="${JEV_LABEL_PREFLIGHT:-0}"
export JEV_LABEL_EXPECTED_MANIFEST_SHA256="${JEV_LABEL_EXPECTED_MANIFEST_SHA256:?Provide the verified test MANIFEST.json hash}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LABEL_PYTHON=/share/home/shenwsh3/.conda/envs/span-offload/bin/python
if [[ "$JEV_LABEL_PREFLIGHT" != 1 ]]; then
  export JEV_LABEL_RUN_ROOT="/share/home/shenwsh3/ozy/jev-appworld-lora/runs/label-test-${SLURM_JOB_ID:?}"
  mkdir "$JEV_LABEL_RUN_ROOT"
  cp "${BASH_SOURCE[0]}" "$JEV_LABEL_RUN_ROOT/submitted-job.sbatch"
fi

"$LABEL_PYTHON" - <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

base = Path('/share/home/shenwsh3/ozy')
lora = base / 'jev-appworld-lora'
project = base / 'trace-offload-orginal'
appworld = base / 'span-offload-jev-prompts' / 'appworld'
data = Path('/share/home/shenwsh3/qwen36_vllm/AppWorldData_1102/测试集')
output = lora / 'labels/test/appworld_labels_0920'
script = project / 'classfierTrain/02_数据标注/label_spans_appworld_0920.py'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = {
    script: 'bb03f3c6222930df51ec435bd8284ceec5a2eba17ad77e98208d7d2455369686',
    script.with_name('label_spans_appworld_l123.py'): '8a68f43ecc81c2e52fb89f90be12489e68037d538dc27c244f95b837dcd17222',
    project / 'classfierTrain/00_共享库/aw3_build.py': 'b55691a19ddfd8e7769e6e89605897c7c4e90b31d5fabb681eb9c0490ca8785c',
}
for path, expected in sources.items():
    if digest(path) != expected:
        raise RuntimeError(f'Label source changed: {path}')
if digest(data / 'MANIFEST.json') != os.environ['JEV_LABEL_EXPECTED_MANIFEST_SHA256']:
    raise RuntimeError('Test manifest changed after submission')
if output.exists():
    raise FileExistsError(f'Refusing to overwrite label output: {output}')

sys.path.insert(0, str(appworld / 'src'))
spec = importlib.util.spec_from_file_location('test_labeler', script)
labeler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labeler)
labeler.aw3.API_DOCS = appworld / 'data/api_docs/openapi'
labeler.API_METHODS = labeler.aw3.load_api_methods()
if len(labeler.API_METHODS) != 473:
    raise RuntimeError(f'Unexpected API method table size: {len(labeler.API_METHODS)}')
points = labeler.discover_points(data)
if len(points) != 4757 or len(set(points)) != len(points):
    raise RuntimeError(f'Unexpected candidate points: {len(points)}')
manifest = json.loads((data / 'MANIFEST.json').read_text(encoding='utf-8'))
task_ids = {task['task_id'] for task in manifest['tasks']}
if len(task_ids) != 243:
    raise RuntimeError('Expected the verified 243-task test subset')
train_dev_hashes = {}
for split in ['train', 'dev']:
    path = lora / f'labels/{split}/appworld_labels_0920/labels_appworld_0920.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    overlap = task_ids & {row['task_id'] for row in rows}
    if overlap:
        raise RuntimeError(f'Test overlaps {split}: {sorted(overlap)}')
    train_dev_hashes[split] = digest(path)

receipt = {
    'started_at': datetime.now(timezone.utc).isoformat(),
    'status': 'running', 'job_id': os.environ.get('SLURM_JOB_ID'),
    'python': sys.executable, 'data': str(data), 'output': str(output),
    'parameters': {'bootstrap': 2000, 'seed': 20260920, 'expected_reps': 10, 'legacy_global_rng': False},
    'label_version': labeler.LABEL_VERSION,
    'source_sha256': {str(p): expected for p, expected in sources.items()},
    'trace_messages_sha256': digest(project / 'spanOffload/trace_messages.py'),
    'input_sha256': {name: digest(data / name) for name in ['MANIFEST.json', 'TASKS.csv', 'MISSING.csv']},
    'api_docs': str(labeler.aw3.API_DOCS), 'api_method_count': len(labeler.API_METHODS),
    'api_docs_sha256': {p.name: digest(p) for p in sorted(labeler.aw3.API_DOCS.glob('*.json'))},
    'train_dev_label_sha256': train_dev_hashes,
    'candidate_points': len(points), 'tasks_with_points': len({t for t, k in points}),
    'tasks_without_points': sorted(task_ids - {t for t, k in points}),
    'train_dev_task_overlap': 0,
}
print('LABEL_PREFLIGHT_OK', json.dumps({k: receipt[k] for k in ['candidate_points', 'tasks_with_points', 'api_method_count', 'train_dev_task_overlap']}), flush=True)
if os.environ['JEV_LABEL_PREFLIGHT'] == '1':
    raise SystemExit(0)

run_root = Path(os.environ['JEV_LABEL_RUN_ROOT'])
receipt_path = run_root / 'label-generation-receipt.json'
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
# Claim the fresh output directory atomically; the original script permits it.
output.mkdir(parents=True, exist_ok=False)
exit_code = labeler.main([
    '--data', str(data), '--out', str(output),
    '--bootstrap', '2000', '--expected-reps', '10', '--seed', '20260920',
])
if exit_code not in (0, None):
    raise RuntimeError(f'Labeler exited with {exit_code}')
labels = output / 'labels_appworld_0920.jsonl'
summary_path = output / 'summary_appworld_0920.json'
rows = [json.loads(line) for line in labels.read_text(encoding='utf-8').splitlines() if line.strip()]
summary = json.loads(summary_path.read_text(encoding='utf-8'))
if len(rows) != len(points) or {(r['task_id'], r['k']) for r in rows} != set(points):
    raise RuntimeError('Label output does not cover every candidate exactly once')
if summary['n_points'] != len(rows) or summary['n_tasks'] != len({r['task_id'] for r in rows}):
    raise RuntimeError('Label summary does not match task-level output')
for row in rows:
    if not ((row['mask'] == 0 and row['label'] == 'unknown') or
            (row['mask'] == 1 and type(row['label']) is int and row['label'] in (0, 1, 2))):
        raise RuntimeError('Invalid label/mask pair')
receipt.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
               label_rows=len(rows), label_distribution=dict(Counter(str(r['label']) for r in rows)),
               output_sha256={p.name: digest(p) for p in [labels, summary_path]})
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('LABEL_TEST_COMPLETE', json.dumps(receipt['label_distribution']), str(output), flush=True)
PY
