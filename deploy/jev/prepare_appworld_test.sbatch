#!/bin/bash
# Submit with --dependency=afterok:<label job>; CPU-only conversion and encoding.
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
CONTAINER=/share/home/shenwsh3/singularity/vllm_qwen35_sandbox
OPENJEV_PYTHON="$LORA_PROJECT/.venv-openjev-container/bin/python"
MODEL_RUN="$LORA_PROJECT/runs/finetune/appworld-v1-context-r16-lr5e-5-seed0-20261107"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

test ! -e "$OUT"
mkdir "$RUN_ROOT"
cp "${BASH_SOURCE[0]}" "$RUN_ROOT/submitted-job.sbatch"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "${JEV_TEST_CODE_COMMIT:?Set the frozen code commit}"
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

module load singularity/4.0.2
module load gcc/9.1.0
# No --nv and no wev.inference.load: only tokenize and check the real loader.
singularity exec --cleanenv --bind /share:/share \
  --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  "$CONTAINER" "$OPENJEV_PYTHON" "$CODE_DIR/prepare_appworld_test.py" \
  --validate-loader --model-run "$MODEL_RUN" --out "$OUT"

cp "$OUT/conversion_summary.json" "$OUT/loader_acceptance.json" \
  "$OUT/loader_model_identity.json" "$RUN_ROOT/"
echo "TEST_DATA_READY_FOR_EVALUATION $OUT"
