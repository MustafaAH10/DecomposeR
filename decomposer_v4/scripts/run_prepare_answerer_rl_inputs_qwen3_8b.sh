#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_EXE="${PYTHON_EXE:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="${PYTHON_EXE_FALLBACK:-python3}"
fi

PLANNER_DIR="${PLANNER_DIR:-outputs/rl/planner_regenerated}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl/answerer_prepared}"
QUERY_ID="${QUERY_ID:-}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
MAX_FETCH_CHARACTERS="${MAX_FETCH_CHARACTERS:-6000}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-8}"
MAX_CONCURRENT_FETCHES="${MAX_CONCURRENT_FETCHES:-8}"
OVERWRITE="${OVERWRITE:-0}"

cmd=("${PYTHON_EXE}" decomposer_v4/train/rl/answerer/prepare_inputs.py \
  --planner-dir "${PLANNER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --max-fetch-characters "${MAX_FETCH_CHARACTERS}" \
  --max-concurrent-searches "${MAX_CONCURRENT_SEARCHES}" \
  --max-concurrent-fetches "${MAX_CONCURRENT_FETCHES}")

if [[ -n "${QUERY_ID}" ]]; then
  cmd+=(--query-id "${QUERY_ID}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

exec "${cmd[@]}"
