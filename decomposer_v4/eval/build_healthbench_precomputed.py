#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer_v4.eval.prepare_healthbench_queries import HEALTHBENCH_URLS
from decomposer_v4.utils.io import ROOT, read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach DecomposeR completions to HealthBench rows so the official "
            "rubric grader can score precomputed generations."
        )
    )
    parser.add_argument("--subset", choices=sorted(HEALTHBENCH_URLS), default="all")
    parser.add_argument(
        "--healthbench-input-path",
        default="",
        help="Optional local/URL HealthBench JSONL. If omitted, uses the official subset URL.",
    )
    parser.add_argument(
        "--reports-jsonl",
        default="",
        help="Optional DecomposeR report JSONL fallback with id/article fields.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default="",
        help="Directory containing one full DecomposeR trajectory JSON per HealthBench prompt_id.",
    )
    parser.add_argument(
        "--completion-field",
        choices=("final_answer", "final_answer_with_citations", "article"),
        default="final_answer",
        help=(
            "Use final_answer for HealthBench scoring by default. Keep full citation "
            "answers in the trajectories for inspection."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="HealthBench-format JSONL with completion_to_trial attached.",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/eval/healthbench/healthbench_precomputed_summary.json",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _read_jsonl_path_or_url(path_or_url: str) -> list[dict[str, Any]]:
    if path_or_url.startswith(("http://", "https://")):
        rows: list[dict[str, Any]] = []
        with urllib.request.urlopen(path_or_url) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return read_jsonl(_resolve_path(path_or_url))


def _load_reports(path: str) -> dict[str, dict[str, Any]]:
    if not path.strip():
        return {}
    rows = read_jsonl(_resolve_path(path))
    return {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}


def _completion_from_trajectory(path: Path, field: str) -> str:
    trajectory = read_json(path)
    if field == "article":
        value = trajectory.get("final_answer_with_citations") or trajectory.get("final_answer") or ""
    else:
        value = trajectory.get(field) or ""
    return str(value).strip()


def main() -> None:
    args = parse_args()
    healthbench_input = args.healthbench_input_path.strip() or HEALTHBENCH_URLS[args.subset]
    original_rows = _read_jsonl_path_or_url(healthbench_input)
    reports_by_id = _load_reports(args.reports_jsonl)
    trajectory_dir = _resolve_path(args.trajectory_dir) if args.trajectory_dir.strip() else None

    output_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in original_rows:
        prompt_id = str(row.get("prompt_id", "")).strip()
        completion = ""
        trajectory_path = None
        if trajectory_dir is not None:
            candidate = trajectory_dir / f"{prompt_id}.json"
            if candidate.exists():
                trajectory_path = candidate
                completion = _completion_from_trajectory(candidate, args.completion_field)
        if not completion:
            report = reports_by_id.get(prompt_id, {})
            completion = str(report.get("article") or report.get("completion_to_trial") or "").strip()
        if not completion:
            missing.append(prompt_id)
            if not args.allow_missing:
                continue
        enriched = dict(row)
        enriched["completion_to_trial"] = completion
        enriched["decomposer_generation_metadata"] = {
            "trajectory_path": str(trajectory_path) if trajectory_path is not None else "",
            "completion_field": args.completion_field,
        }
        output_rows.append(enriched)

    if missing and not args.allow_missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing completions for {len(missing)} rows. First missing ids: {preview}")

    output_jsonl = _resolve_path(args.output_jsonl)
    summary_json = _resolve_path(args.summary_json)
    write_jsonl(output_jsonl, output_rows)
    write_json(
        summary_json,
        {
            "subset": args.subset,
            "healthbench_input_path": healthbench_input,
            "output_jsonl": str(output_jsonl),
            "completion_field": args.completion_field,
            "input_count": len(original_rows),
            "completed_count": len([row for row in output_rows if row.get("completion_to_trial")]),
            "missing_count": len(missing),
            "missing_prompt_ids": missing[:100],
        },
    )
    print(
        json.dumps(
            {
                "input_count": len(original_rows),
                "completed_count": len([row for row in output_rows if row.get("completion_to_trial")]),
                "missing_count": len(missing),
                "output_jsonl": str(output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
