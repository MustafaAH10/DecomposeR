#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-decomposer_v4/configs/sft_shared_turnlevel_trl_qwen3_8b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/qwen3_8b_turnlevel}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${RESUME_FROM_CHECKPOINT}" && -d "${OUTPUT_DIR}" ]]; then
  latest_checkpoint="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    RESUME_FROM_CHECKPOINT="${latest_checkpoint}"
    echo "Resuming from ${RESUME_FROM_CHECKPOINT}"
  fi
fi

cmd=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port "${MASTER_PORT}" \
  decomposer_v4/train/sft/train_trl.py \
  --config "${CONFIG}")

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  cmd+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${cmd[@]}"
