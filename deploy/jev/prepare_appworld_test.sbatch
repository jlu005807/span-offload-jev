#!/bin/bash
# Submit with --dependency=afterok:<label job>; CPU-only conversion.
#SBATCH --partition=x86_64
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=jev-convert-test
#SBATCH --output=/share/home/shenwsh3/ozy/jev-appworld-lora/runs/convert-test-%j.log

set -euo pipefail

# x86_64 advertises RealMemory=1 MB: follow established CPU jobs without --mem.
LORA_PROJECT=/share/home/shenwsh3/ozy/jev-appworld-lora
CODE_DIR="${JEV_TEST_CODE_DIR:?Set to the immutable conversion-code directory}"
PROJECT=/share/home/shenwsh3/ozy/span-offload-jev-prompts
LABEL_JOB="${JEV_TEST_LABEL_JOB:?Set to the validated upstream label job}"
[[ "$LABEL_JOB" =~ ^[0-9]+$ ]]
OUT="$LORA_PROJECT/data/openjev-v1-context-test-labeljob$LABEL_JOB"
RUN_ROOT="$LORA_PROJECT/runs/convert-test-${SLURM_JOB_ID:?}"
APPWORLD_PYTHON=/share/home/shenwsh3/ozy/span-offload-jev-prompts/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

test ! -e "$OUT"
mkdir "$RUN_ROOT"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/submitted-job.sbatch"
cd "$PROJECT"
test "$(git rev-parse HEAD)" = "${JEV_TEST_CODE_COMMIT:?Set the frozen code commit}"
cd "$CODE_DIR"
sha256sum -c test-preparation-source.sha256
cp test-preparation-source.sha256 "$RUN_ROOT/"

"$APPWORLD_PYTHON" "$CODE_DIR/prepare_appworld_test.py" \
  --data /share/home/shenwsh3/qwen36_vllm/AppWorldData_1102/测试集 \
  --labels "$LORA_PROJECT/labels/test/appworld_labels_0920/labels_appworld_0920.jsonl" \
  --label-summary "$LORA_PROJECT/labels/test/appworld_labels_0920/summary_appworld_0920.json" \
  --label-receipt "$LORA_PROJECT/runs/label-test-$LABEL_JOB/label-generation-receipt.json" \
  --train-file "$LORA_PROJECT/data/openjev-v1-context/train.jsonl" \
  --dev-file "$LORA_PROJECT/data/openjev-v1-context/dev.jsonl" \
  --expected-label-job "$LABEL_JOB" --out "$OUT"

cp "$OUT/conversion_summary.json" "$RUN_ROOT/"
echo "TEST_FORMAT_CONVERSION_COMPLETE $OUT"
# The GPU scoring job will run the native OpenJev loader and record its
# accepted/dropped sample counts before producing P(SLM), AUC and AP.
