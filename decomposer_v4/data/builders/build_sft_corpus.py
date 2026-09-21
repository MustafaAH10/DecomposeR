#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.utils.io import ROOT, read_json, read_jsonl, train_val_split, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build planner/answerer/shared SFT corpora from full v3 trajectories.")
    parser.add_argument("--input-path", required=True, nargs="+")
    parser.add_argument("--planner-train-output-jsonl", required=True)
    parser.add_argument("--planner-val-output-jsonl", required=True)
    parser.add_argument("--answerer-train-output-jsonl", required=True)
    parser.add_argument("--answerer-val-output-jsonl", required=True)
    parser.add_argument("--shared-train-output-jsonl", required=True)
    parser.add_argument("--shared-val-output-jsonl", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    return parser.parse_args()


def _load_records(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = ROOT / raw_path
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_dir():
                    continue
                if child.name.startswith("_"):
                    continue
                if child.suffix == ".jsonl":
                    try:
                        rows.extend(read_jsonl(child))
                    except Exception as exc:
                        print(f"Skipping unreadable file {child}: {exc}", file=sys.stderr)
                    continue
                if child.suffix == ".json":
                    try:
                        payload = read_json(child)
                    except Exception as exc:
                        print(f"Skipping unreadable file {child}: {exc}", file=sys.stderr)
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
                    elif isinstance(payload, list):
                        rows.extend(item for item in payload if isinstance(item, dict))
            continue
        if path.suffix == ".jsonl":
            try:
                rows.extend(read_jsonl(path))
            except Exception as exc:
                print(f"Skipping unreadable file {path}: {exc}", file=sys.stderr)
            continue
        try:
            payload = read_json(path)
        except Exception as exc:
            print(f"Skipping unreadable file {path}: {exc}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        elif isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))
    return rows


def _planner_record(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    messages = row.get("planner_messages")
    query = str(row.get("query", "")).strip()
    if not isinstance(messages, list) or not messages or not query:
        return None
    record_id = str(row.get("record_id") or row.get("query_id") or f"planner-{index}")
    return {
        "record_id": record_id,
        "query": query,
        "stage": "planner",
        "messages": messages,
        "metadata": {
            "query_id": str(row.get("query_id", "")),
            "task_type": str(row.get("task_type", "deep_research")),
            "num_turns": str(
                len([message for message in messages if isinstance(message, dict) and str(message.get("role", "")) == "assistant"])
            ),
        },
    }


def _answerer_record(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    messages = row.get("answerer_messages")
    query = str(row.get("query", "")).strip()
    if not isinstance(messages, list) or not messages or not query:
        return None
    record_id = str(row.get("record_id") or row.get("query_id") or f"answerer-{index}")
    assistant_turns = len(
        [message for message in messages if isinstance(message, dict) and str(message.get("role", "")) == "assistant"]
    )
    if assistant_turns <= 0:
        return None
    return {
        "record_id": f"{record_id}-answerer",
        "query": query,
        "stage": "answerer",
        "messages": messages,
        "metadata": {
            "query_id": str(row.get("query_id", "")),
            "task_type": str(row.get("task_type", "deep_research")),
            "num_turns": str(assistant_turns),
        },
    }


def main() -> None:
    args = parse_args()
    trajectories = _load_records(args.input_path)
    planner_rows_by_id: dict[str, dict[str, Any]] = {}
    answerer_rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(trajectories):
        planner = _planner_record(row, index)
        if planner is not None:
            planner_rows_by_id.setdefault(str(planner["record_id"]), planner)
        answerer = _answerer_record(row, index)
        if answerer is not None:
            answerer_rows_by_id.setdefault(str(answerer["record_id"]), answerer)

    planner_rows = list(planner_rows_by_id.values())
    answerer_rows = list(answerer_rows_by_id.values())
    shared_rows = planner_rows + answerer_rows

    planner_train, planner_val = train_val_split(planner_rows, args.val_ratio)
    answerer_train, answerer_val = train_val_split(answerer_rows, args.val_ratio)
    shared_train, shared_val = train_val_split(shared_rows, args.val_ratio)

    write_jsonl(ROOT / args.planner_train_output_jsonl, planner_train)
    write_jsonl(ROOT / args.planner_val_output_jsonl, planner_val)
    write_jsonl(ROOT / args.answerer_train_output_jsonl, answerer_train)
    write_jsonl(ROOT / args.answerer_val_output_jsonl, answerer_val)
    write_jsonl(ROOT / args.shared_train_output_jsonl, shared_train)
    write_jsonl(ROOT / args.shared_val_output_jsonl, shared_val)

    print(f"Planner rows: train={len(planner_train)} val={len(planner_val)}")
    print(f"Answerer rows: train={len(answerer_train)} val={len(answerer_val)}")
    print(f"Shared rows: train={len(shared_train)} val={len(shared_val)}")


if __name__ == "__main__":
    main()
