#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p data/sft

./.venv/bin/python decomposer_v4/data/builders/build_turn_level_sft_corpus.py \
  --planner-input-path data/sft/raw_planner_trajectories \
  --answerer-input-path data/sft/raw_answerer_trajectories \
  --output-dir data/sft \
  --val-ratio 0.02
