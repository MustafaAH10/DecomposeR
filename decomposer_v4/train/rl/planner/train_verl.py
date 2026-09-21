#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.train.rl.common import resolve_path
from decomposer_v4.train.rl.runtime import load_rl_rows
from decomposer_v4.train.rl.verl.launch import prepare_verl_launch, run_verl
from decomposer_v4.utils.io import ROOT, load_yaml
from decomposer_v4.train.rl.common import make_raw_prompt, serialize_json_field, serialize_rubrics, write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch planner GRPO via VERL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-source", default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepare_dataset(train_source: str, output_dir: Path) -> Path:
    rows = load_rl_rows(train_source)
    if not rows:
        raise RuntimeError(f"No planner RL rows found in {train_source}")
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
    return write_parquet_rows(output_dir / "train.parquet", dataset_rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    output_dir = Path(
        resolve_path(args.output_dir or config.get("output_dir", "outputs/rl/planner_grpo"))
        or ""
    )
    train_source = str(
        args.train_source
        or config.get("train_source", "data/rl/planner")
    )
    num_gpus = int(args.num_gpus or int(config.get("num_gpus", 0) or 0) or int(os.environ.get("NUM_GPUS", "1")))
    num_gpus = max(num_gpus, 1)

    train_file = _prepare_dataset(train_source, output_dir)
    verl_config_path, agent_loop_config_path = prepare_verl_launch(
        stage="planner",
        stage_cfg=config,
        train_file=train_file,
        output_dir=output_dir,
        num_gpus=num_gpus,
    )

    summary = {
        "stage": "planner",
        "train_file": str(train_file),
        "verl_config_path": str(verl_config_path),
        "agent_loop_config_path": str(agent_loop_config_path),
        "output_dir": str(output_dir),
        "num_gpus": num_gpus,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    exit_code = run_verl(verl_config_path, dry_run=args.dry_run)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
