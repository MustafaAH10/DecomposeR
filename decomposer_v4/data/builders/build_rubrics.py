#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient
from decomposer_v4.utils.io import (
    ROOT,
    extract_json_object,
    read_csv_rows,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/rubric_system_prompt.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 8-10 rubrics for deep research queries.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--model-profile",
        default="decomposer_v4/configs/model_profiles/openai_mini_nonreasoning.yaml",
    )
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--id-field", default="query_id")
    parser.add_argument("--with-search-context", action="store_true")
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return parser.parse_args()


def _resolve_root_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _load_rows(path: str) -> list[dict[str, Any]]:
    if path.endswith(".csv"):
        return read_csv_rows(_resolve_root_path(path))
    return read_jsonl(_resolve_root_path(path))


def _normalize_rubrics(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("rubrics")
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title", "")).split()).strip()
        description = " ".join(str(item.get("description", "")).split()).strip()
        if title and description:
            rows.append({"title": title, "description": description})
    return rows


def _search_context(query: str, num_results: int) -> str:
    client = JinaSearchClient.from_env()
    results = client.search(query, num_results=num_results)
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        lines.append(
            "\n".join(
                [
                    f"[R{idx}] {result.title}",
                    f"URL: {result.url}",
                    f"Description: {result.text}",
                ]
            )
        )
    return "\n\n".join(lines).strip()


def _record_id(row: dict[str, Any], id_field: str) -> str:
    record_id = str(row.get(id_field) or row.get("id") or row.get("record_id") or "").strip()
    if record_id:
        return record_id
    query = str(row.get("query", "")).strip()
    if query:
        import hashlib

        return hashlib.sha256(query.encode("utf-8")).hexdigest()
    raise ValueError("Could not determine record id for rubric row.")


def _build_output_row(
    *,
    row: dict[str, Any],
    id_field: str,
    query_field: str,
    rubrics: list[dict[str, Any]],
) -> dict[str, Any]:
    query = str(row.get(query_field, "")).strip()
    query_id = _record_id(row, id_field)
    return {
        "query_id": query_id,
        "query": query,
        "rubrics": rubrics,
        "metadata": {"source": str(row.get("source", row.get("dataset", "")))},
    }


def _load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "input_path": args.input_path,
        "output_jsonl": args.output_jsonl,
        "output_dir": str(output_dir),
        "model_profile": args.model_profile,
        "query_field": args.query_field,
        "id_field": args.id_field,
        "with_search_context": args.with_search_context,
        "search_results": args.search_results,
        "limit": args.limit,
        "skip_existing": args.skip_existing,
    }
    write_json(output_dir / "_run_config.json", payload)


def _validate_args(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    output_jsonl = _resolve_root_path(args.output_jsonl) if args.output_jsonl.strip() else None
    output_dir = _resolve_root_path(args.output_dir) if args.output_dir.strip() else None
    if output_jsonl is None and output_dir is None:
        raise SystemExit("Provide at least one of --output-jsonl or --output-dir.")
    return output_jsonl, output_dir


def main() -> None:
    args = parse_args()
    from decomposer_v4.train.model_backends import build_chat_backend

    output_jsonl, output_dir = _validate_args(args)
    rows = _load_rows(args.input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "_errors").mkdir(parents=True, exist_ok=True)
        _write_run_config(output_dir, args)

    backend = build_chat_backend(str(ROOT / args.model_profile))
    output_rows: list[dict[str, Any]] = []
    if output_dir is not None:
        output_rows.extend(_load_existing_rows(output_dir))

    existing_ids = {str(row.get("query_id", "")).strip() for row in output_rows}

    for row in rows:
        query = str(row.get(args.query_field, "")).strip()
        if not query:
            continue
        query_id = _record_id(row, args.id_field)

        if output_dir is not None and args.skip_existing and query_id in existing_ids:
            continue

        user_prompt = f"Query:\n{query}\n"
        try:
            if args.with_search_context:
                context = _search_context(query, args.search_results)
                if context:
                    user_prompt += f"\nSearch context:\n{context}\n"

            content = backend.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format_json=True,
            )
            rubrics = _normalize_rubrics(extract_json_object(content))
            output_row = _build_output_row(
                row=row,
                id_field=args.id_field,
                query_field=args.query_field,
                rubrics=rubrics,
            )
            if output_dir is not None:
                write_json(output_dir / f"{query_id}.json", output_row)
                existing_ids.add(query_id)
            output_rows.append(output_row)
        except Exception as exc:
            if output_dir is None:
                raise
            error_payload = {
                "query_id": query_id,
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(output_dir / "_errors" / f"{query_id}.json", error_payload)

    if output_jsonl is not None:
        rows_to_write = output_rows if output_dir is None else _load_existing_rows(output_dir)
        rows_to_write.sort(key=lambda item: str(item.get("query_id", "")))
        write_jsonl(output_jsonl, rows_to_write)


if __name__ == "__main__":
    main()
