#!/bin/bash
# Run the complete LoRA-V1 dev threshold round in one serial GPU allocation.
# Each threshold sources the established single-threshold workflow, preserving
# its smoke gate, result format, and per-threshold output directory.
#SBATCH --partition=gpu_ai
#SBATCH --account=zhangyc29
#SBATCH --qos=32gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=24:00:00
#SBATCH --job-name=jev-lora-v1-round2
#SBATCH --output=/share/home/shenwsh3/ozy/span-offload-jev-prompts/runs/jev-lora-v1-round2-%j.log

set -euo pipefail

PROJECT="${PROJECT:-/share/home/shenwsh3/ozy/span-offload-jev-prompts}"
JEV_ROUND_INDEX="${JEV_ROUND_INDEX:-2}"
[[ "$JEV_ROUND_INDEX" =~ ^[1-9][0-9]*$ ]]
THRESHOLD_SCRIPT="$PROJECT/deploy/jev/lora_v1_dev_threshold.sbatch"
JEV_COMPARE_BASE="${JEV_COMPARE_BASE:?Set to the frozen round configuration}"
JEV_LARGE_API_BASE="${JEV_LARGE_API_BASE:?Set to the verified Qwen3.8-27B API base}"
JEV_SMALL_API_BASE="${JEV_SMALL_API_BASE:?Set to the verified Qwen3.5-9B API base}"
THRESHOLDS=(0.5 0.55 0.6 0.65 0.7 0.75 0.8)

test -f "$THRESHOLD_SCRIPT"
test -f "$JEV_COMPARE_BASE"
cd "$PROJECT"

index=0
for threshold in "${THRESHOLDS[@]}"; do
  export JEV_THRESHOLD="$threshold"
  export JEV_COMPARE_BASE JEV_LARGE_API_BASE JEV_SMALL_API_BASE
  export JEV_PORT="$((19000 + SLURM_JOB_ID % 900 + index))"
  echo "ROUND${JEV_ROUND_INDEX}_THRESHOLD_START $JEV_THRESHOLD port=$JEV_PORT"

  source "$THRESHOLD_SCRIPT"
  cleanup
  jev_pid=
  trap - EXIT TERM INT
  echo "ROUND${JEV_ROUND_INDEX}_THRESHOLD_FINISHED $JEV_THRESHOLD"
  index=$((index + 1))
done

echo "LORA_THRESHOLD_ROUND_COMPLETE round=$JEV_ROUND_INDEX job=$SLURM_JOB_ID thresholds=${#THRESHOLDS[@]}"
