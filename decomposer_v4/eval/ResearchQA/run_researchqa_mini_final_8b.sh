#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

MODEL="${MODEL:-checkpoints/decomposer_qwen3_8b_final_merged}"
RUN_NAME="${RUN_NAME:-decomposer-qwen3-8b-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/researchqa/ResearchQA-mini/${RUN_NAME}}"

MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-6144}"
ANSWER_EXECUTION_MAX_COMPLETION_TOKENS="${ANSWER_EXECUTION_MAX_COMPLETION_TOKENS:-6144}"
ANSWER_FINAL_MAX_COMPLETION_TOKENS="${ANSWER_FINAL_MAX_COMPLETION_TOKENS:-6144}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.7}"
ANSWERER_TEMPERATURE="${ANSWERER_TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-17}"

NUM_RESULTS_PER_SEARCH="${NUM_RESULTS_PER_SEARCH:-5}"
MAX_FETCHES="${MAX_FETCHES:-4}"
MAX_FETCH_CHARACTERS="${MAX_FETCH_CHARACTERS:-6000}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-4}"
MAX_CONCURRENT_FETCHES="${MAX_CONCURRENT_FETCHES:-4}"

RUN_COVERAGE="${RUN_COVERAGE:-1}"
COVERAGE_MODEL="${COVERAGE_MODEL:-gpt-4.1-mini}"
COVERAGE_BATCH_SIZE="${COVERAGE_BATCH_SIZE:-8}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"

args=(
  decomposer_v4/eval/ResearchQA/run_researchqa_end_to_end.py
  --split test_mini
  --limit "${LIMIT}"
  --offset "${OFFSET}"
  --run-name "${RUN_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --dataset-json "${OUTPUT_ROOT}/test_mini.json"
  --planner-model "${MODEL}"
  --answerer-model "${MODEL}"
  --planner-backend "${MODEL_BACKEND}"
  --answerer-backend "${MODEL_BACKEND}"
  --planner-temperature "${PLANNER_TEMPERATURE}"
  --answerer-temperature "${ANSWERER_TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --seed "${SEED}"
  --num-results-per-search "${NUM_RESULTS_PER_SEARCH}"
  --max-fetches "${MAX_FETCHES}"
  --max-fetch-characters "${MAX_FETCH_CHARACTERS}"
  --max-concurrent-searches "${MAX_CONCURRENT_SEARCHES}"
  --max-concurrent-fetches "${MAX_CONCURRENT_FETCHES}"
  --answerer-payload-variant compact
  --answerer-prompt-variant compact
  --answer-execution-max-completion-tokens "${ANSWER_EXECUTION_MAX_COMPLETION_TOKENS}"
  --answer-final-max-completion-tokens "${ANSWER_FINAL_MAX_COMPLETION_TOKENS}"
  --coverage-model "${COVERAGE_MODEL}"
  --coverage-batch-size "${COVERAGE_BATCH_SIZE}"
)

if [[ "${RUN_COVERAGE}" == "1" ]]; then
  args+=(--run-coverage)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite-existing)
fi

"${PYTHON_BIN}" "${args[@]}" "$@"
