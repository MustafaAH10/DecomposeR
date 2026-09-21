#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-decomposer_v4/configs/sft_shared_turnlevel_trl_qwen3_8b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/qwen3_8b_turnlevel}"

export CONFIG
export OUTPUT_DIR

exec ./decomposer_v4/scripts/run_shared_turnlevel_sft_trl.sh
