#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.train.rl.common import make_raw_prompt, serialize_json_field, write_parquet_rows
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare answerer RL parquet for VERL.")
    parser.add_argument(
        "--prepared-dir",
        default="data/rl/answerer_prepared",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/rl/answerer_grpo/train.parquet",
    )
    return parser.parse_args()


def _prepared_files(prepared_dir: Path) -> list[Path]:
    return sorted(path for path in prepared_dir.glob("*.json") if path.is_file() and not path.name.startswith("_"))


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    if not prepared_dir.is_absolute():
        prepared_dir = (ROOT / prepared_dir).resolve()
    files = _prepared_files(prepared_dir)
    if not files:
        raise RuntimeError(f"No prepared answerer artifacts found in {prepared_dir}")

    dataset_rows = []
    for index, path in enumerate(files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        query = str(payload.get("query", ""))
        query_id = str(payload.get("query_id", path.stem)).strip() or path.stem
        dataset_rows.append(
            {
                "prompt": make_raw_prompt(query),
                "data_source": "decomposer_answerer",
                "reward_model": {
                    "style": "decomposer_answerer",
                    "ground_truth": query_id or query,
                },
                "extra_info": {"index": index},
                "record_id": str(payload.get("record_id", query_id)),
                "query_id": query_id,
                "query": query,
                "task_type": str(payload.get("task_type", "deep_research") or "deep_research"),
                "metadata_json": serialize_json_field(
                    payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
                    {},
                ),
                "artifact_path": str(path),
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
