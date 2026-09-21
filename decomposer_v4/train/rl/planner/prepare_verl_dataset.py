#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.train.rl.common import make_raw_prompt, serialize_json_field, serialize_rubrics, write_parquet_rows
from decomposer_v4.train.rl.runtime import load_rl_rows
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare planner RL parquet for VERL.")
    parser.add_argument(
        "--input-source",
        default="data/rl/planner",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/rl/planner_grpo/train.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rl_rows(args.input_source)
    if not rows:
        raise RuntimeError(f"No planner RL rows found in {args.input_source}")

    dataset_rows = []
    for index, row in enumerate(rows):
        query = str(row.get("query", ""))
        query_id = str(row.get("query_id", "")).strip()
        dataset_rows.append(
            {
                "prompt": make_raw_prompt(query),
                "data_source": "decomposer_planner",
                "reward_model": {
                    "style": "decomposer_planner",
                    "ground_truth": query_id or query,
                },
                "extra_info": {"index": index},
                "record_id": str(row.get("record_id", query_id)),
                "query_id": query_id,
                "query": query,
                "rubrics": serialize_rubrics(row.get("rubrics", ())),
                "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
                "metadata_json": serialize_json_field(
                    row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
                    {},
                ),
            }
        )

    output_path = write_parquet_rows(ROOT / args.output_path, dataset_rows)
    summary = {
        "count": len(dataset_rows),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
