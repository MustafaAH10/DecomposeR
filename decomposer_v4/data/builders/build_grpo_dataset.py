#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.train.rl.runtime import load_rl_rows
from decomposer_v4.utils.io import ROOT, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GRPO training manifest from rubric JSON files.")
    parser.add_argument(
        "--rubric-source",
        default="outputs/rubrics/rl_queries",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/rl/decomposer_v4_rl_grpo_manifest.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/rl_queries_with_rubrics_success_only.csv",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    rows = load_rl_rows(args.rubric_source)
    if not rows:
        raise RuntimeError(f"No rubric-backed RL rows found in {args.rubric_source}")

    output_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    for row in rows:
        output_rows.append(
            {
                "prompt": str(row.get("query", "")),
                "query_id": str(row.get("query_id", "")),
                "query": str(row.get("query", "")),
                "rubrics_json": json.dumps(
                    [
                        {
                            "title": rubric.title,
                            "description": rubric.description,
                            "weight": rubric.weight,
                        }
                        for rubric in row.get("rubrics", ())
                    ],
                    ensure_ascii=False,
                ),
                "task_type": str(row.get("task_type", "deep_research")),
                "metadata_json": json.dumps(row.get("metadata", {}), ensure_ascii=False),
            }
        )
        csv_rows.append(
            {
                "query_id": str(row.get("query_id", "")),
                "query": str(row.get("query", "")),
            }
        )

    output_jsonl = _resolve_path(args.output_jsonl)
    output_csv = _resolve_path(args.output_csv)
    write_jsonl(output_jsonl, output_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        json.dumps(
            {
                "count": len(output_rows),
                "jsonl": str(output_jsonl),
                "csv": str(output_csv),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
