#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_EXE="${PYTHON_EXE:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="${PYTHON_EXE_FALLBACK:-python3}"
fi

INPUT_SOURCE="${INPUT_SOURCE:-data/rl/planner}"
HF_MODEL_REPO="${HF_MODEL_REPO:-}"
MODEL_DIR="${MODEL_DIR:-outputs/models/qwen3_8b_planner_rl_merged}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
FORCE_MODEL_DOWNLOAD="${FORCE_MODEL_DOWNLOAD:-0}"
BASE_MODEL="${BASE_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/planner_regenerated}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
QUERY_ID="${QUERY_ID:-}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
NUM_RESULTS_PER_SEARCH="${NUM_RESULTS_PER_SEARCH:-5}"
MAX_FETCHES="${MAX_FETCHES:-4}"
MAX_FETCH_CHARACTERS="${MAX_FETCH_CHARACTERS:-6000}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-8}"
MAX_CONCURRENT_FETCHES="${MAX_CONCURRENT_FETCHES:-4}"
MIN_SEARCH_SUCCESS_RATE="${MIN_SEARCH_SUCCESS_RATE:-1.0}"
MIN_FETCH_REQUESTS="${MIN_FETCH_REQUESTS:-1}"
MIN_FETCH_SUCCESSES="${MIN_FETCH_SUCCESSES:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
PLANNER_INITIAL_MAX_NEW_TOKENS="${PLANNER_INITIAL_MAX_NEW_TOKENS:-2000}"
PLANNER_REVISION_MAX_NEW_TOKENS="${PLANNER_REVISION_MAX_NEW_TOKENS:-2000}"
MAX_QUERY_ATTEMPTS="${MAX_QUERY_ATTEMPTS:-3}"
BF16="${BF16:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
  if [[ -z "${HF_MODEL_REPO}" ]]; then
    echo "HF_MODEL_REPO must be set when DOWNLOAD_MODEL=1" >&2
    exit 1
  fi
  export HF_MODEL_REPO MODEL_DIR
  if [[ -n "${HF_TOKEN:-}" ]] && command -v hf >/dev/null 2>&1; then
    hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
  fi
  if [[ "${FORCE_MODEL_DOWNLOAD}" == "1" || ! -f "${MODEL_DIR}/config.json" ]]; then
    "${PYTHON_EXE}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    resume_download=True,
)
print(local_dir)
PY
  fi
  BASE_MODEL="${MODEL_DIR}"
fi

if [[ -z "${BASE_MODEL}" ]]; then
  BASE_MODEL="${MODEL_DIR}"
fi
if [[ -z "${TOKENIZER_NAME}" ]]; then
  TOKENIZER_NAME="${BASE_MODEL}"
fi

cmd=("${PYTHON_EXE}" decomposer_v4/train/rl/planner/regenerate_graphs.py \
  --input-source "${INPUT_SOURCE}" \
  --base-model "${BASE_MODEL}" \
  --tokenizer-name "${TOKENIZER_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --num-results-per-search "${NUM_RESULTS_PER_SEARCH}" \
  --max-fetches "${MAX_FETCHES}" \
  --max-fetch-characters "${MAX_FETCH_CHARACTERS}" \
  --max-concurrent-searches "${MAX_CONCURRENT_SEARCHES}" \
  --max-concurrent-fetches "${MAX_CONCURRENT_FETCHES}" \
  --min-search-success-rate "${MIN_SEARCH_SUCCESS_RATE}" \
  --min-fetch-requests "${MIN_FETCH_REQUESTS}" \
  --min-fetch-successes "${MIN_FETCH_SUCCESSES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --planner-initial-max-new-tokens "${PLANNER_INITIAL_MAX_NEW_TOKENS}" \
  --planner-revision-max-new-tokens "${PLANNER_REVISION_MAX_NEW_TOKENS}" \
  --max-query-attempts "${MAX_QUERY_ATTEMPTS}")

if [[ -n "${QUERY_ID}" ]]; then
  cmd+=(--query-id "${QUERY_ID}")
fi
if [[ "${BF16}" == "1" ]]; then
  cmd+=(--bf16)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

exec "${cmd[@]}"
