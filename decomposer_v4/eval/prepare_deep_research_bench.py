#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer_v4.utils.io import read_jsonl, write_jsonl


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Deep Research Bench prompts by language and emit an English-only JSONL split."
    )
    parser.add_argument(
        "--input-path",
        default="deep_research_bench/data/prompt_data/query.jsonl",
        help="Path to the original Deep Research Bench query.jsonl file.",
    )
    parser.add_argument(
        "--output-path",
        default="deep_research_bench/data/prompt_data/query_en.jsonl",
        help="Path to write the filtered JSONL file.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code to keep. For Deep Research Bench this should be `en`.",
    )
    return parser.parse_args()


def _looks_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def _normalize_language(row: dict[str, Any]) -> str:
    for key in ("language", "lang", "locale"):
        value = str(row.get(key, "")).strip().lower()
        if value:
            return value
    prompt = str(row.get("prompt", row.get("query", ""))).strip()
    return "zh" if _looks_cjk(prompt) else "en"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    target_language = str(args.language or "en").strip().lower()

    rows = read_jsonl(input_path)
    kept = [row for row in rows if _normalize_language(row) == target_language]

    write_jsonl(output_path, kept)

    print(f"Input rows: {len(rows)}")
    print(f"Kept rows ({target_language}): {len(kept)}")
    print(f"Wrote: {output_path.resolve()}")


if __name__ == "__main__":
    main()
