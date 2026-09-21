#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.utils.io import ROOT, read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build turn-level SFT corpora for shared planner+answerer training.")
    parser.add_argument("--planner-input-path", required=True)
    parser.add_argument("--answerer-input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    return parser.parse_args()


def _load_records(raw_path: str) -> list[dict[str, Any]]:
    path = ROOT / raw_path
    rows: list[dict[str, Any]] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_dir():
                continue
            if child.name.startswith("_"):
                continue
            try:
                if child.suffix == ".jsonl":
                    rows.extend(read_jsonl(child))
                elif child.suffix == ".json":
                    payload = read_json(child)
                    if isinstance(payload, dict):
                        rows.append(payload)
                    elif isinstance(payload, list):
                        rows.extend(item for item in payload if isinstance(item, dict))
            except Exception as exc:
                print(f"Skipping unreadable file {child}: {exc}", file=sys.stderr)
        return rows
    try:
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        payload = read_json(path)
    except Exception as exc:
        print(f"Skipping unreadable file {path}: {exc}", file=sys.stderr)
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _stable_is_val(query_id: str, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    key = query_id or "unknown"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < val_ratio


def _split_rows(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("query_id") or row.get("metadata", {}).get("query_id") or row.get("record_id") or "")
        if _stable_is_val(query_id, val_ratio):
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _make_turn_row(
    *,
    parent_row: dict[str, Any],
    stage: str,
    turn: dict[str, Any],
) -> dict[str, Any] | None:
    prompt_messages = turn.get("messages")
    completion_text = str(turn.get("completion", ""))
    query = str(parent_row.get("query", "")).strip()
    if not isinstance(prompt_messages, list) or not prompt_messages or not completion_text.strip() or not query:
        return None
    completion_messages = [{"role": "assistant", "content": completion_text}]
    query_id = str(parent_row.get("query_id", "")).strip()
    record_id = str(parent_row.get("record_id") or query_id or "unknown")
    span_id = str(turn.get("span_id") or turn.get("turn_type") or "turn")
    turn_type = str(turn.get("turn_type") or "")
    return {
        "record_id": f"{record_id}:{span_id}",
        "query_id": query_id,
        "query": query,
        "stage": stage,
        "turn_type": turn_type,
        "prompt": prompt_messages,
        "completion": completion_messages,
        "metadata": {
            "query_id": query_id,
            "task_type": str(parent_row.get("task_type", "deep_research")),
            "span_id": span_id,
            "turn_type": turn_type,
            "prompt_message_count": len(prompt_messages),
            "completion_chars": len(completion_text),
            "target_node_ids": list(turn.get("target_node_ids", [])) if isinstance(turn.get("target_node_ids"), list) else [],
        },
    }


def _planner_turn_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        for turn in record.get("planner_turns", []):
            if not isinstance(turn, dict):
                continue
            row = _make_turn_row(parent_row=record, stage="planner", turn=turn)
            if row is None:
                continue
            record_id = str(row["record_id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            rows.append(row)
    return sorted(rows, key=lambda item: str(item["record_id"]))


def _answerer_turn_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        for turn in record.get("answerer_turns", []):
            if not isinstance(turn, dict):
                continue
            row = _make_turn_row(parent_row=record, stage="answerer", turn=turn)
            if row is None:
                continue
            record_id = str(row["record_id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            rows.append(row)
    return sorted(rows, key=lambda item: str(item["record_id"]))


def _stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_turn_type: dict[str, int] = {}
    for row in rows:
        key = str(row.get("turn_type", "unknown"))
        by_turn_type[key] = by_turn_type.get(key, 0) + 1
    completion_chars = [int(row.get("metadata", {}).get("completion_chars", 0)) for row in rows if int(row.get("metadata", {}).get("completion_chars", 0)) > 0]
    return {
        "count": len(rows),
        "mean_completion_chars": round(sum(completion_chars) / len(completion_chars), 1) if completion_chars else 0.0,
        "turn_type_counts": by_turn_type,
    }


def main() -> None:
    args = parse_args()
    planner_records = _load_records(args.planner_input_path)
    answerer_records = _load_records(args.answerer_input_path)

    planner_rows = _planner_turn_rows(planner_records)
    answerer_rows = _answerer_turn_rows(answerer_records)
    shared_rows = planner_rows + answerer_rows

    planner_train, planner_val = _split_rows(planner_rows, args.val_ratio)
    answerer_train, answerer_val = _split_rows(answerer_rows, args.val_ratio)
    shared_train, shared_val = _split_rows(shared_rows, args.val_ratio)

    output_dir = ROOT / args.output_dir
    write_jsonl(output_dir / "planner_turn_train.jsonl", planner_train)
    write_jsonl(output_dir / "planner_turn_val.jsonl", planner_val)
    write_jsonl(output_dir / "answerer_turn_train.jsonl", answerer_train)
    write_jsonl(output_dir / "answerer_turn_val.jsonl", answerer_val)
    write_jsonl(output_dir / "shared_turn_train.jsonl", shared_train)
    write_jsonl(output_dir / "shared_turn_val.jsonl", shared_val)

    stats = {
        "planner_all": _stats_for_rows(planner_rows),
        "answerer_all": _stats_for_rows(answerer_rows),
        "shared_all": _stats_for_rows(shared_rows),
        "splits": {
            "planner_train": len(planner_train),
            "planner_val": len(planner_val),
            "answerer_train": len(answerer_train),
            "answerer_val": len(answerer_val),
            "shared_train": len(shared_train),
            "shared_val": len(shared_val),
        },
    }
    write_json(output_dir / "turn_level_stats.json", stats)

    print(f"Planner turns: train={len(planner_train)} val={len(planner_val)}")
    print(f"Answerer turns: train={len(answerer_train)} val={len(answerer_val)}")
    print(f"Shared turns: train={len(shared_train)} val={len(shared_val)}")


if __name__ == "__main__":
    main()
