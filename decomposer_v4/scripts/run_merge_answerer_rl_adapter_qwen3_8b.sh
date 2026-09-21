#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-outputs/models/qwen3_8b_planner_rl_merged}"
RUN_DIR="${RUN_DIR:-outputs/rl/answerer_grpo}"
ADAPTER_SOURCE="${ADAPTER_SOURCE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/models/qwen3_8b_final_rl_merged}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-1}"

if [[ -z "${ADAPTER_SOURCE}" ]]; then
  latest_step_dir="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
  if [[ -z "${latest_step_dir}" ]]; then
    echo "No VERL checkpoint found under ${RUN_DIR}" >&2
    exit 1
  fi
  for candidate in "${latest_step_dir}/actor/lora_adapter" "${latest_step_dir}/lora_adapter"; do
    if [[ -f "${candidate}/adapter_config.json" ]]; then
      ADAPTER_SOURCE="${candidate}"
      break
    fi
  done
  if [[ -z "${ADAPTER_SOURCE}" && -d "${latest_step_dir}/actor" ]]; then
    echo "No exported LoRA adapter found; exporting from VERL FSDP actor checkpoint at ${latest_step_dir}/actor"
    python3 -m decomposer_v4.train.rl.verl.export_lora_adapter \
      --actor-checkpoint-dir "${latest_step_dir}/actor"
    ADAPTER_SOURCE="${latest_step_dir}/actor/lora_adapter"
  fi
fi

if [[ ! -f "${ADAPTER_SOURCE}/adapter_config.json" ]]; then
  echo "Expected VERL LoRA adapter under ${latest_step_dir:-${RUN_DIR}}" >&2
  find "${latest_step_dir:-${RUN_DIR}}" -maxdepth 4 -type f -name adapter_config.json -print >&2 || true
  exit 1
fi

cmd=(
  python3 decomposer_v4/train/sft/merge_adapter.py
  --base-model "${BASE_MODEL}"
  --adapter-source "${ADAPTER_SOURCE}"
  --output-dir "${OUTPUT_DIR}"
  --torch-dtype "${TORCH_DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --max-shard-size "${MAX_SHARD_SIZE}"
)

if [[ "${SAFE_SERIALIZATION}" == "1" ]]; then
  cmd+=(--safe-serialization)
fi

"${cmd[@]}"
