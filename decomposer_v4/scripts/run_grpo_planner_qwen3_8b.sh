#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_EXE="${PYTHON_EXE:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="${PYTHON_EXE_FALLBACK:-python3}"
fi

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-decomposer_v4/configs/grpo_planner_qwen3_8b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRAIN_SOURCE="${TRAIN_SOURCE:-}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-decomposer-rl}"
export WANDB_NAME="${WANDB_NAME:-planner-grpo-qwen3-8b}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-planner-qwen3-8b}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"

ulimit -n 65535 || true

if [[ -f "${REPO_ROOT}/verl/verl/trainer/ppo/ray_trainer.py" ]]; then
  "${PYTHON_EXE}" decomposer_v4/scripts/patch_verl_reward_extra_metrics.py
fi

cmd=(
  "${PYTHON_EXE}"
  decomposer_v4/train/rl/planner/train_verl.py
  --config "${CONFIG}"
  --num-gpus "${NUM_GPUS}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${TRAIN_SOURCE}" ]]; then
  cmd+=(--train-source "${TRAIN_SOURCE}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

"${cmd[@]}"
