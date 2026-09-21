#!/usr/bin/env bash
set -euo pipefail

# End-to-end Deep Research Bench runner:
# 1. prepare English/Chinese query splits
# 2. generate DecomposeR reports for English and Chinese
# 3. concatenate a combined raw-data file
# 4. run DRB RACE and FACT for English, Chinese, and combined outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRB_DIR="${DRB_DIR:-deep_research_bench}"
MODEL="${MODEL:-checkpoints/decomposer_qwen3_8b_final_merged}"
RUN_NAME="${RUN_NAME:-decomposer-qwen3-8b-final}"

PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.7}"
ANSWERER_TEMPERATURE="${ANSWERER_TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-6144}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-17}"
ANSWER_EXECUTION_MAX_COMPLETION_TOKENS="${ANSWER_EXECUTION_MAX_COMPLETION_TOKENS:-6144}"
ANSWER_FINAL_MAX_COMPLETION_TOKENS="${ANSWER_FINAL_MAX_COMPLETION_TOKENS:-6144}"

NUM_RESULTS_PER_SEARCH="${NUM_RESULTS_PER_SEARCH:-5}"
MAX_FETCHES="${MAX_FETCHES:-4}"
MAX_FETCH_CHARACTERS="${MAX_FETCH_CHARACTERS:-6000}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-4}"
MAX_CONCURRENT_FETCHES="${MAX_CONCURRENT_FETCHES:-4}"

N_TOTAL_PROCESS="${N_TOTAL_PROCESS:-10}"
OVERWRITE="${OVERWRITE:-1}"

QUERY_DIR="$DRB_DIR/data/prompt_data"
RAW_DATA_DIR="$DRB_DIR/data/test_data/raw_data"
RESULTS_DIR="$DRB_DIR/results"

mkdir -p "$RAW_DATA_DIR" "outputs/eval/deep_research_bench"

echo "==> Preparing DRB language splits"
python3 decomposer_v4/eval/prepare_deep_research_bench.py \
  --input-path "$QUERY_DIR/query.jsonl" \
  --output-path "$QUERY_DIR/query_en.jsonl" \
  --language en

python3 decomposer_v4/eval/prepare_deep_research_bench.py \
  --input-path "$QUERY_DIR/query.jsonl" \
  --output-path "$QUERY_DIR/query_zh.jsonl" \
  --language zh

generate_split() {
  local split="$1"
  local query_file="$QUERY_DIR/query_${split}.jsonl"
  local target="${RUN_NAME}-${split}"

  echo "==> Generating DRB ${split} reports for ${target}"
  python3 decomposer_v4/eval/run_end_to_end_reports.py \
    --input-path "$query_file" \
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
    --benchmark-format deep_research_bench \
    --output-jsonl "$RAW_DATA_DIR/${target}.jsonl" \
    --trajectory-dir "outputs/eval/deep_research_bench/${target}/trajectories" \
    --summary-json "outputs/eval/deep_research_bench/${target}/summary.json" \
    --num-results-per-search "$NUM_RESULTS_PER_SEARCH" \
    --max-fetches "$MAX_FETCHES" \
    --max-fetch-characters "$MAX_FETCH_CHARACTERS" \
    --max-concurrent-searches "$MAX_CONCURRENT_SEARCHES" \
    --max-concurrent-fetches "$MAX_CONCURRENT_FETCHES"
}

generate_split en
generate_split zh

echo "==> Building combined DRB raw-data file"
cat \
  "$RAW_DATA_DIR/${RUN_NAME}-en.jsonl" \
  "$RAW_DATA_DIR/${RUN_NAME}-zh.jsonl" \
  > "$RAW_DATA_DIR/${RUN_NAME}-all.jsonl"

run_drb_eval() {
  local target="$1"
  local lang_flag=""
  if [[ "$target" == *"-en" ]]; then
    lang_flag="--only_en"
  elif [[ "$target" == *"-zh" ]]; then
    lang_flag="--only_zh"
  fi

  local race_dir="$RESULTS_DIR/race/$target"
  local fact_dir="$RESULTS_DIR/fact/$target"

  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$race_dir" "$fact_dir"
  fi
  mkdir -p "$race_dir" "$fact_dir"

  echo "==> Running RACE for $target"
  (
    cd "$DRB_DIR"
    python -u deepresearch_bench_race.py "$target" \
      --raw_data_dir "data/test_data/raw_data" \
      --max_workers "$N_TOTAL_PROCESS" \
      --query_file "data/prompt_data/query.jsonl" \
      --output_dir "results/race/$target" \
      $lang_flag --force
  ) | tee "$race_dir/run.log"

  echo "==> Running FACT extraction for $target"
  (
    cd "$DRB_DIR"
    python -u -m utils.extract \
      --raw_data_path "data/test_data/raw_data/$target.jsonl" \
      --output_path "results/fact/$target/extracted.jsonl" \
      --query_data_path "data/prompt_data/query.jsonl" \
      --n_total_process "$N_TOTAL_PROCESS"
  ) | tee "$fact_dir/extract.log"

  echo "==> Running FACT deduplication for $target"
  (
    cd "$DRB_DIR"
    python -u -m utils.deduplicate \
      --raw_data_path "results/fact/$target/extracted.jsonl" \
      --output_path "results/fact/$target/deduplicated.jsonl" \
      --query_data_path "data/prompt_data/query.jsonl" \
      --n_total_process "$N_TOTAL_PROCESS"
  ) | tee "$fact_dir/deduplicate.log"

  echo "==> Running FACT scraping for $target"
  (
    cd "$DRB_DIR"
    python -u -m utils.scrape \
      --raw_data_path "results/fact/$target/deduplicated.jsonl" \
      --output_path "results/fact/$target/scraped.jsonl" \
      --n_total_process "$N_TOTAL_PROCESS"
  ) | tee "$fact_dir/scrape.log"

  echo "==> Running FACT validation for $target"
  (
    cd "$DRB_DIR"
    python -u -m utils.validate \
      --raw_data_path "results/fact/$target/scraped.jsonl" \
      --output_path "results/fact/$target/validated.jsonl" \
      --query_data_path "data/prompt_data/query.jsonl" \
      --n_total_process "$N_TOTAL_PROCESS"
  ) | tee "$fact_dir/validate.log"

  echo "==> Writing FACT stats for $target"
  (
    cd "$DRB_DIR"
    python -u -m utils.stat \
      --input_path "results/fact/$target/validated.jsonl" \
      --output_path "results/fact/$target/fact_result.txt"
  )
}

run_drb_eval "${RUN_NAME}-en"
run_drb_eval "${RUN_NAME}-zh"
run_drb_eval "${RUN_NAME}-all"

cat <<EOF

Done.

RACE:
  $RESULTS_DIR/race/${RUN_NAME}-en/race_result.txt
  $RESULTS_DIR/race/${RUN_NAME}-zh/race_result.txt
  $RESULTS_DIR/race/${RUN_NAME}-all/race_result.txt

FACT:
  $RESULTS_DIR/fact/${RUN_NAME}-en/fact_result.txt
  $RESULTS_DIR/fact/${RUN_NAME}-zh/fact_result.txt
  $RESULTS_DIR/fact/${RUN_NAME}-all/fact_result.txt
EOF
