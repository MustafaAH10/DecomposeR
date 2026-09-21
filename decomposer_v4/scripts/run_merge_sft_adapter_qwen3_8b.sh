#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
ADAPTER_SOURCE="${ADAPTER_SOURCE:-outputs/sft/qwen3_8b_turnlevel}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/qwen3_8b_sft_merged}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ADAPTER_DEVICE="${ADAPTER_DEVICE:-cpu}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-1}"

PYTHON_EXE="${PYTHON_EXE:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="${PYTHON_EXE_FALLBACK:-python3}"
fi

cmd=(
  "${PYTHON_EXE}" decomposer_v4/train/sft/merge_adapter.py
  --base-model "${BASE_MODEL}"
  --adapter-source "${ADAPTER_SOURCE}"
  --output-dir "${OUTPUT_DIR}"
  --torch-dtype "${TORCH_DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --adapter-device "${ADAPTER_DEVICE}"
  --max-shard-size "${MAX_SHARD_SIZE}"
)

if [[ "${SAFE_SERIALIZATION}" == "1" ]]; then
  cmd+=(--safe-serialization)
fi

"${cmd[@]}"
