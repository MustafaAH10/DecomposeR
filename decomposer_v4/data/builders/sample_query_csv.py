#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.utils.io import ROOT, read_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a query-only CSV from a wider source CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _query_text(row: dict[str, str]) -> str:
    for key in ("query", "question", "_question_text"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _query_id(row: dict[str, str], fallback_query: str) -> str:
    value = str(row.get("query_id") or row.get("id") or "").strip()
    if value:
        return value
    return fallback_query


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(ROOT / args.input_csv)
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        query = _query_text(row)
        if not query:
            continue
        query_id = _query_id(row, query)
        if not query_id:
            continue
        deduped.setdefault(query_id, {"query_id": query_id, "query": query})

    candidates = list(deduped.values())
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if len(candidates) < args.sample_size:
        raise ValueError(
            f"Requested sample_size={args.sample_size}, but only {len(candidates)} unique valid rows are available."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(candidates, args.sample_size)

    target = ROOT / args.output_csv
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query"])
        writer.writeheader()
        writer.writerows(sampled)

    print(
        f"Sampled {len(sampled)} queries from {len(candidates)} candidates "
        f"using seed={args.seed} -> {target}"
    )


if __name__ == "__main__":
    main()
