#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.train.rl.common import make_raw_prompt, resolve_path, serialize_json_field, write_parquet_rows
from decomposer_v4.train.rl.verl.launch import prepare_verl_launch, run_verl
from decomposer_v4.utils.io import ROOT, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch answerer GRPO via VERL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-source", default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepared_files(prepared_dir: Path) -> list[Path]:
    return sorted(path for path in prepared_dir.glob("*.json") if path.is_file() and not path.name.startswith("_"))


def _prepare_dataset(train_source: str, output_dir: Path) -> Path:
    prepared_dir = Path(train_source)
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
    return write_parquet_rows(output_dir / "train.parquet", dataset_rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    output_dir = Path(
        resolve_path(args.output_dir or config.get("output_dir", "outputs/rl/answerer_grpo"))
        or ""
    )
    train_source = str(
        args.train_source
        or config.get("train_source", "data/rl/answerer_prepared")
    )
    num_gpus = int(args.num_gpus or int(config.get("num_gpus", 0) or 0) or int(os.environ.get("NUM_GPUS", "1")))

    train_file = _prepare_dataset(train_source, output_dir)
    verl_config_path, agent_loop_config_path = prepare_verl_launch(
        stage="answerer",
        stage_cfg=config,
        train_file=train_file,
        output_dir=output_dir,
        num_gpus=max(num_gpus, 1),
    )

    summary = {
        "stage": "answerer",
        "train_file": str(train_file),
        "verl_config_path": str(verl_config_path),
        "agent_loop_config_path": str(agent_loop_config_path),
        "output_dir": str(output_dir),
        "num_gpus": max(num_gpus, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    exit_code = run_verl(verl_config_path, dry_run=args.dry_run)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
