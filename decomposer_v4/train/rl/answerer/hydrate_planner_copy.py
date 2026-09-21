#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.utils.io import ROOT


SYNC_KEYS = (
    "parsed_graph",
    "search_node_outputs",
    "search_results",
    "fetch_requests",
    "fetch_blocks",
    "fetched_results",
    "citation_catalog",
    "timings",
    "search_executions",
    "fetch_executions",
    "evidence_pool",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync fetched/prepared answerer fields back into a copied planner artifact directory."
    )
    parser.add_argument("--planner-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--query-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def _planner_files(planner_dir: Path) -> list[Path]:
    return sorted(path for path in planner_dir.glob("*.json") if path.is_file() and not path.name.startswith("_"))


def _select_files(paths: list[Path], query_id: str, offset: int, limit: int) -> list[Path]:
    if query_id:
        return [path for path in paths if path.stem == query_id]
    sliced = paths[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    planner_dir = ROOT / args.planner_dir
    prepared_dir = ROOT / args.prepared_dir

    planner_files = _select_files(_planner_files(planner_dir), args.query_id, args.offset, args.limit)
    if args.query_id and not planner_files:
        raise ValueError(f"Planner artifact not found for query_id={args.query_id}")

    updated = 0
    missing_prepared = 0

    for planner_path in planner_files:
        prepared_path = prepared_dir / planner_path.name
        if not prepared_path.exists():
            missing_prepared += 1
            continue

        planner_payload = _load_json(planner_path)
        prepared_payload = _load_json(prepared_path)

        for key in SYNC_KEYS:
            if key in prepared_payload:
                planner_payload[key] = prepared_payload[key]

        planner_payload["prepared_artifact_path"] = str(prepared_path)
        _write_json(planner_path, planner_payload)
        updated += 1

    print(
        json.dumps(
            {
                "updated": updated,
                "missing_prepared": missing_prepared,
                "planner_dir": str(planner_dir),
                "prepared_dir": str(prepared_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
