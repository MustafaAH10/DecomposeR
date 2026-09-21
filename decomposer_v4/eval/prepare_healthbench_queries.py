#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer_v4.utils.io import ROOT, write_json, write_jsonl


HEALTHBENCH_URLS = {
    "all": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HealthBench chat prompts into DecomposeR query JSONL rows."
    )
    parser.add_argument("--subset", choices=sorted(HEALTHBENCH_URLS), default="all")
    parser.add_argument(
        "--input-path",
        default="",
        help="Optional local/URL HealthBench JSONL. If omitted, uses the official subset URL.",
    )
    parser.add_argument(
        "--output-path",
        default="data/eval/healthbench/healthbench_all_decomposer_queries.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/eval/healthbench/healthbench_query_prep_summary.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample this many rows after offset/limit filtering.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=0.0,
        help=(
            "If >0 with --subset all and --sample, draw this fraction from the "
            "official HealthBench hard subset prompt ids and the rest from non-hard rows."
        ),
    )
    parser.add_argument(
        "--hard-input-path",
        default="",
        help="Optional local/URL HealthBench Hard JSONL used for stratified sampling.",
    )
    parser.add_argument(
        "--selected-healthbench-output-path",
        default="",
        help="Optional HealthBench-format JSONL containing the same selected original rows for grading precomputed completions.",
    )
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _read_jsonl(path_or_url: str) -> list[dict[str, Any]]:
    if path_or_url.startswith(("http://", "https://")):
        rows: list[dict[str, Any]] = []
        with urllib.request.urlopen(path_or_url) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    rows = []
    for line in _resolve_path(path_or_url).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _format_conversation(messages: list[dict[str, Any]]) -> str:
    role_mapping = {"user": "Patient", "assistant": "Physician"}
    return "\n\n".join(
        f"{role_mapping.get(str(item.get('role', '')), str(item.get('role', '')))}: {item.get('content', '')}"
        for item in messages
    )


def _stringify_healthbench_prompt(prompt: list[dict[str, Any]]) -> str:
    if len(prompt) == 1:
        return (
            "Can you answer the question from a patient about a medical condition "
            f"or concern they have: {prompt[0].get('content', '')}"
        )
    if len(prompt) > 1:
        return (
            "Here is a conversation between a patient and a physician. The patient is asking a question "
            "about a medical condition or concern they have, and the conversation contains necessary "
            "background information about the patient:\n\n"
            f"{_format_conversation(prompt[:-1])}\n\n"
            "Can you search for needed information and answer the patient's question: "
            f"{prompt[-1].get('content', '')}"
        )
    raise ValueError("HealthBench prompt must contain at least one message.")


def main() -> None:
    args = parse_args()
    input_path = args.input_path.strip() or HEALTHBENCH_URLS[args.subset]
    rows = _read_jsonl(input_path)
    selected_hard_ids: set[str] | None = None

    start = max(int(args.offset), 0)
    if int(args.limit) > 0:
        rows = rows[start : start + int(args.limit)]
    else:
        rows = rows[start:]
    if int(args.sample) > 0 and int(args.sample) < len(rows):
        rng = random.Random(int(args.seed))
        if float(args.hard_fraction) > 0:
            if args.subset != "all":
                raise ValueError("--hard-fraction is only supported with --subset all.")
            hard_input_path = args.hard_input_path.strip() or HEALTHBENCH_URLS["hard"]
            hard_rows = _read_jsonl(hard_input_path)
            hard_ids = {str(row.get("prompt_id", "")).strip() for row in hard_rows}
            selected_hard_ids = hard_ids
            hard_pool = [row for row in rows if str(row.get("prompt_id", "")).strip() in hard_ids]
            nonhard_pool = [row for row in rows if str(row.get("prompt_id", "")).strip() not in hard_ids]
            hard_target = round(int(args.sample) * float(args.hard_fraction))
            hard_target = min(max(hard_target, 0), len(hard_pool))
            nonhard_target = int(args.sample) - hard_target
            if nonhard_target > len(nonhard_pool):
                raise RuntimeError(
                    f"Not enough non-hard rows for requested sample: requested {nonhard_target}, "
                    f"available {len(nonhard_pool)}."
                )
            rows = rng.sample(hard_pool, hard_target) + rng.sample(nonhard_pool, nonhard_target)
            rng.shuffle(rows)
        else:
            rows = rng.sample(rows, int(args.sample))

    converted: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            raise ValueError(f"Expected list prompt for prompt_id={row.get('prompt_id')}")
        prompt_id = str(row.get("prompt_id") or "").strip()
        if not prompt_id:
            raise ValueError("HealthBench row missing prompt_id.")
        converted.append(
            {
                "id": prompt_id,
                "query_id": prompt_id,
                "query": _stringify_healthbench_prompt(prompt),
                "task_type": "healthbench",
                "healthbench_subset": args.subset,
                "example_tags": row.get("example_tags", []),
            }
        )

    output_path = _resolve_path(args.output_path)
    summary_json = _resolve_path(args.summary_json)
    write_jsonl(output_path, converted)
    selected_output_path = ""
    if args.selected_healthbench_output_path.strip():
        selected_path = _resolve_path(args.selected_healthbench_output_path)
        write_jsonl(selected_path, rows)
        selected_output_path = str(selected_path)
    if selected_hard_ids is None and float(args.hard_fraction) > 0:
        hard_rows = _read_jsonl(args.hard_input_path.strip() or HEALTHBENCH_URLS["hard"])
        selected_hard_ids = {str(row.get("prompt_id", "")).strip() for row in hard_rows}
    hard_count = (
        sum(1 for row in rows if str(row.get("prompt_id", "")).strip() in selected_hard_ids)
        if selected_hard_ids is not None
        else None
    )
    write_json(
        summary_json,
        {
            "subset": args.subset,
            "input_path": input_path,
            "output_path": str(output_path),
            "selected_healthbench_output_path": selected_output_path,
            "count": len(converted),
            "offset": int(args.offset),
            "limit": int(args.limit),
            "sample": int(args.sample),
            "seed": int(args.seed),
            "hard_fraction": float(args.hard_fraction),
            "hard_count": hard_count,
        },
    )
    print(
        json.dumps(
            {
                "count": len(converted),
                "hard_count": hard_count,
                "output_path": str(output_path),
                "selected_healthbench_output_path": selected_output_path,
                "summary_json": str(summary_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
