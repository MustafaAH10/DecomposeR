#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

MODEL="${MODEL:-checkpoints/decomposer_qwen3_8b_final_merged}"
RUN_NAME="${RUN_NAME:-decomposer-qwen3-8b-final}"
SUBSET="${SUBSET:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/healthbench/${RUN_NAME}}"

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

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
# Reproduce the paper's HealthBench subset by default: 500 examples with 20%
# drawn from the official hard subset, using a fixed sampling seed. Override any
# of SAMPLE / HARD_FRACTION / PREP_SEED to change the subset.
SAMPLE="${SAMPLE:-500}"
HARD_FRACTION="${HARD_FRACTION:-0.2}"
PREP_SEED="${PREP_SEED:-17}"
RUN_GRADE="${RUN_GRADE:-1}"
GRADER_MODEL="${GRADER_MODEL:-gpt-4.1-2025-04-14}"
N_THREADS="${N_THREADS:-16}"

QUERY_JSONL="data/eval/healthbench/${SUBSET}_decomposer_queries.jsonl"
SELECTED_JSONL="data/eval/healthbench/${SUBSET}_selected_healthbench.jsonl"
REPORTS_JSONL="${OUTPUT_ROOT}/reports.jsonl"
TRAJECTORY_DIR="${OUTPUT_ROOT}/trajectories"
PRECOMPUTED_JSONL="${OUTPUT_ROOT}/healthbench_precomputed.jsonl"

# --hard-fraction is only valid with --subset all; only forward it there.
PREP_HARD_ARGS=()
if [[ "$SUBSET" == "all" ]]; then
  PREP_HARD_ARGS=(--hard-fraction "$HARD_FRACTION")
fi

"$PYTHON_BIN" decomposer_v4/eval/prepare_healthbench_queries.py \
  --subset "$SUBSET" \
  --output-path "$QUERY_JSONL" \
  --selected-healthbench-output-path "$SELECTED_JSONL" \
  --summary-json "${OUTPUT_ROOT}/query_prep_summary.json" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --sample "$SAMPLE" \
  --seed "$PREP_SEED" \
  "${PREP_HARD_ARGS[@]}"

"$PYTHON_BIN" decomposer_v4/eval/run_end_to_end_reports.py \
  --input-path "$QUERY_JSONL" \
  --planner-model "$MODEL" \
  --answerer-model "$MODEL" \
  --planner-backend "$MODEL_BACKEND" \
  --answerer-backend "$MODEL_BACKEND" \
  --planner-temperature "$PLANNER_TEMPERATURE" \
  --answerer-temperature "$ANSWERER_TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --seed "$SEED" \
  --answerer-payload-variant compact \
  --answerer-prompt-variant compact \
  --answer-execution-max-completion-tokens "$ANSWER_EXECUTION_MAX_COMPLETION_TOKENS" \
  --answer-final-max-completion-tokens "$ANSWER_FINAL_MAX_COMPLETION_TOKENS" \
  --benchmark-format generic \
  --report-answer-field final_answer \
  --output-jsonl "$REPORTS_JSONL" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --summary-json "${OUTPUT_ROOT}/summary.json" \
  --num-results-per-search "$NUM_RESULTS_PER_SEARCH" \
  --max-fetches "$MAX_FETCHES" \
  --max-fetch-characters "$MAX_FETCH_CHARACTERS" \
  --max-concurrent-searches "$MAX_CONCURRENT_SEARCHES" \
  --max-concurrent-fetches "$MAX_CONCURRENT_FETCHES"

"$PYTHON_BIN" decomposer_v4/eval/build_healthbench_precomputed.py \
  --subset "$SUBSET" \
  --healthbench-input-path "$SELECTED_JSONL" \
  --reports-jsonl "$REPORTS_JSONL" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --completion-field final_answer \
  --output-jsonl "$PRECOMPUTED_JSONL" \
  --summary-json "${OUTPUT_ROOT}/precomputed_summary.json"

if [[ "$RUN_GRADE" == "1" ]]; then
  "$PYTHON_BIN" decomposer_v4/eval/grade_healthbench_precomputed.py \
    --input-path "$PRECOMPUTED_JSONL" \
    --output-json "${OUTPUT_ROOT}/healthbench_score.json" \
    --grader-model "$GRADER_MODEL" \
    --n-threads "$N_THREADS"
fi
