#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JINA_API_KEY:?Set JINA_API_KEY in the environment}"
: "${SERPER_API_KEY:?Set SERPER_API_KEY in the environment}"
MODEL="${MODEL:-mustafaah/qwen3-8b-decomposer-v4-planner-answerer-rl-step358-merged}"
python3 decomposer_v4/eval/run_end_to_end_reports.py \
  --input-path examples/research_questions.jsonl \
  --planner-model "$MODEL" --answerer-model "$MODEL" \
  --planner-backend vllm --answerer-backend vllm \
  --planner-temperature 0.2 --answerer-temperature 0.2 \
  --max-completion-tokens 6144 --max-model-len 32768 \
  --answer-execution-max-completion-tokens 6144 \
  --answer-final-max-completion-tokens 6144 \
  --answerer-payload-variant compact --answerer-prompt-variant compact \
  --num-results-per-search 5 --max-fetches 4 \
  --max-fetch-characters 6000 --max-concurrent-searches 4 \
  --max-concurrent-fetches 4 --seed 17 \
  --benchmark-format generic \
  --output-jsonl demo_runs/reports.jsonl \
  --trajectory-dir demo_runs/trajectories \
  --summary-json demo_runs/summary.json "$@"
