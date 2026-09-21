#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from decomposer_v4.data.generation.generate_live_trajectory import PlannerSession
from decomposer_v4.data.generation.scaffold import (
    execute_revision_retrievals,
    execute_search_nodes_timed,
    hydrate_graph_with_evidence,
    normalize_fetch_results,
    normalize_search_results,
)
from decomposer_v4.train.rl.runtime import LocalChatBackend, LocalChatSpanGenerator, load_local_policy_model, load_rl_rows
from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate planner graphs from a merged planner checkpoint.")
    parser.add_argument(
        "--input-source",
        default="data/rl/planner",
    )
    parser.add_argument("--base-model", default="outputs/models/qwen3_8b_planner_rl_merged")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--output-dir", default="outputs/rl/planner_regenerated")
    parser.add_argument("--query-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetches", type=int, default=4)
    parser.add_argument("--max-fetch-characters", type=int, default=6000)
    parser.add_argument("--max-concurrent-searches", type=int, default=8)
    parser.add_argument("--max-concurrent-fetches", type=int, default=4)
    parser.add_argument(
        "--min-search-success-rate",
        type=float,
        default=1.0,
        help="Retry/error if the fraction of search nodes with at least one result is below this value.",
    )
    parser.add_argument(
        "--min-fetch-requests",
        type=int,
        default=1,
        help="Retry/error if the planner requests fewer fetches than this after search.",
    )
    parser.add_argument(
        "--min-fetch-successes",
        type=int,
        default=1,
        help="Retry/error if fewer than this many requested fetches return page text.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--planner-initial-max-new-tokens", type=int, default=1400)
    parser.add_argument("--planner-revision-max-new-tokens", type=int, default=1400)
    parser.add_argument("--max-query-attempts", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _serialize_rubrics(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "title": rubric.title,
            "description": rubric.description,
            "weight": float(rubric.weight),
        }
        for rubric in row.get("rubrics", ())
    ]


def _select_rows(rows: list[dict[str, Any]], query_id: str, offset: int, limit: int) -> list[dict[str, Any]]:
    if query_id:
        return [row for row in rows if str(row.get("query_id", "")).strip() == query_id]
    sliced = rows[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def _search_health(search_timings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(search_timings)
    nonempty = sum(1 for item in search_timings.values() if int(item.get("result_count", 0) or 0) > 0)
    empty_node_ids = [
        str(node_id)
        for node_id, item in search_timings.items()
        if int(item.get("result_count", 0) or 0) <= 0
    ]
    return {
        "search_node_count": total,
        "search_nonempty_count": nonempty,
        "search_empty_count": total - nonempty,
        "search_success_rate": float(nonempty / total) if total else 1.0,
        "search_empty_node_ids": empty_node_ids,
    }


def _fetch_health(
    fetch_requests: list[dict[str, str]],
    fetch_timings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    successes = sum(1 for item in fetch_timings.values() if bool(item.get("succeeded", False)))
    failed = sum(1 for item in fetch_timings.values() if not bool(item.get("succeeded", False)))
    return {
        "fetch_request_count": len(fetch_requests),
        "fetch_attempt_count": len(fetch_timings),
        "fetch_success_count": successes,
        "fetch_failed_count": failed,
        "fetch_success_rate": float(successes / len(fetch_timings)) if fetch_timings else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = _select_rows(load_rl_rows(args.input_source), args.query_id, args.offset, args.limit)
    if args.query_id and not rows:
        raise ValueError(f"Query id not found in {args.input_source}: {args.query_id}")

    _, tokenizer, model = load_local_policy_model(
        base_model=args.base_model,
        tokenizer_name=args.tokenizer_name,
        bf16=args.bf16,
    )
    generator = LocalChatSpanGenerator(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
    )
    backend = LocalChatBackend(
        generator=generator,
        default_temperature=args.temperature,
        default_top_p=args.top_p,
        default_max_completion_tokens=max(
            args.planner_initial_max_new_tokens,
            args.planner_revision_max_new_tokens,
        ),
    )
    search_backend = MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )

    output_dir = ROOT / args.output_dir
    error_dir = output_dir / "_errors"

    completed = 0
    skipped = 0
    errors = 0
    failed_query_ids: list[str] = []
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            errors += 1
            continue

        target_path = output_dir / f"{query_id}.json"
        error_path = error_dir / f"{query_id}.json"
        if target_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(rows)}] skip {query_id}")
            continue

        success_payload: dict[str, Any] | None = None
        attempt_summaries: list[dict[str, Any]] = []
        last_error = ""
        last_planner_messages: list[dict[str, Any]] = []
        last_planner_spans: list[dict[str, Any]] = []

        for query_attempt in range(1, max(1, int(args.max_query_attempts)) + 1):
            timings: dict[str, Any] = {"search_timings_initial": {}, "search_timings_post_revision": {}}
            total_started = time.perf_counter()
            planner = PlannerSession(backend, query)
            try:
                stage_started = time.perf_counter()
                planner_initial_output, initial_graph = planner.initial_plan()
                timings["planner_initial_seconds"] = time.perf_counter() - stage_started
                ok, meta = validate_graph(initial_graph)
                if not ok:
                    raise RuntimeError(f"initial_graph_invalid:{meta}")

                stage_started = time.perf_counter()
                search_node_outputs, initial_search_executions, search_timings_initial = execute_search_nodes_timed(
                    graph=initial_graph,
                    search_backend=search_backend,
                    num_results_per_search=args.num_results_per_search,
                    max_concurrent_searches=args.max_concurrent_searches,
                )
                timings["search_seconds_initial"] = time.perf_counter() - stage_started
                timings["search_timings_initial"] = search_timings_initial
                search_results_initial = normalize_search_results(initial_search_executions)
                initial_search_health = _search_health(search_timings_initial)
                timings["search_health_initial"] = initial_search_health
                if initial_search_health["search_success_rate"] < float(args.min_search_success_rate):
                    raise RuntimeError(f"initial_search_unhealthy:{initial_search_health}")

                stage_started = time.perf_counter()
                planner_revision_output, planner_action, final_graph, fetch_requests = planner.revise_and_fetch(
                    current_graph=initial_graph,
                    search_results=search_results_initial,
                    max_fetches=args.max_fetches,
                )
                timings["planner_revision_seconds"] = time.perf_counter() - stage_started
                ok, meta = validate_graph(final_graph)
                if not ok:
                    raise RuntimeError(f"final_graph_invalid:{meta}")

                stage_started = time.perf_counter()
                (
                    _,
                    final_search_executions,
                    search_timings_post_revision,
                    _,
                    fetch_executions,
                    fetch_timings,
                ) = execute_revision_retrievals(
                    graph=final_graph,
                    search_backend=search_backend,
                    num_results_per_search=args.num_results_per_search,
                    existing_search_executions=initial_search_executions,
                    fetch_requests=fetch_requests,
                    max_fetch_characters=args.max_fetch_characters,
                    max_concurrent_searches=args.max_concurrent_searches,
                    max_concurrent_fetches=args.max_concurrent_fetches,
                )
                timings["search_seconds_post_revision"] = time.perf_counter() - stage_started
                timings["search_timings_post_revision"] = search_timings_post_revision
                timings["fetch_timings"] = fetch_timings
                search_results = normalize_search_results(final_search_executions)
                fetched_results = normalize_fetch_results(fetch_requests, fetch_executions)
                final_search_health = _search_health(search_timings_post_revision)
                final_fetch_health = _fetch_health(fetch_requests, fetch_timings)
                timings["search_health_post_revision"] = final_search_health
                timings["fetch_health"] = final_fetch_health
                if final_search_health["search_success_rate"] < float(args.min_search_success_rate):
                    raise RuntimeError(f"post_revision_search_unhealthy:{final_search_health}")
                if final_fetch_health["fetch_request_count"] < int(args.min_fetch_requests):
                    raise RuntimeError(f"insufficient_fetch_requests:{final_fetch_health}")
                if final_fetch_health["fetch_success_count"] < int(args.min_fetch_successes):
                    raise RuntimeError(f"insufficient_fetch_successes:{final_fetch_health}")
                parsed_graph = hydrate_graph_with_evidence(
                    final_graph,
                    search_results=search_results,
                    fetched_results=fetched_results,
                )
                timings["total_seconds"] = time.perf_counter() - total_started

                attempt_summaries.append(
                    {
                        "attempt": query_attempt,
                        "error": "",
                        "timings": timings,
                    }
                )
                success_payload = {
                    "record_id": str(row.get("record_id", query_id)),
                    "query_id": query_id,
                    "query": query,
                    "rubrics": _serialize_rubrics(row),
                    "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
                    "metadata": row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
                    "planner_messages": planner.messages,
                    "planner_turns": planner.spans,
                    "planner_spans": planner.spans,
                    "planner_initial_output": planner_initial_output,
                    "planner_revision_output": planner_revision_output,
                    "planner_action": planner_action,
                    "initial_graph": initial_graph,
                    "final_graph": final_graph,
                    "parsed_graph": parsed_graph,
                    "search_node_outputs": search_node_outputs,
                    "search_results": search_results,
                    "fetch_requests": fetch_requests,
                    "fetched_results": fetched_results,
                    "timings": timings,
                    "generation_attempt": query_attempt,
                    "generation_attempts": attempt_summaries,
                    "search_executions": [
                        {
                            "node_id": execution.node_id,
                            "query": execution.query,
                            "passages": [asdict(passage) for passage in execution.passages],
                        }
                        for execution in final_search_executions
                    ],
                    "fetch_executions": [
                        {
                            "node_id": execution.node_id,
                            "query": execution.query,
                            "passages": [asdict(passage) for passage in execution.passages],
                        }
                        for execution in fetch_executions
                    ],
                    "error": None,
                }
                break
            except Exception as exc:
                timings["total_seconds"] = time.perf_counter() - total_started
                last_error = str(exc)
                last_planner_messages = planner.messages
                last_planner_spans = planner.spans
                attempt_summaries.append(
                    {
                        "attempt": query_attempt,
                        "error": last_error,
                        "timings": timings,
                    }
                )
                if query_attempt < max(1, int(args.max_query_attempts)):
                    print(
                        f"[{index}/{len(rows)}] retry {query_id} "
                        f"(attempt {query_attempt}/{int(args.max_query_attempts)} failed: {last_error})"
                    )

        if success_payload is not None:
            _atomic_write_json(target_path, success_payload)
            if error_path.exists():
                error_path.unlink()
            completed += 1
            timings = success_payload.get("timings", {})
            search_health = timings.get("search_health_post_revision", {})
            fetch_health = timings.get("fetch_health", {})
            print(
                f"[{index}/{len(rows)}] ok {query_id} "
                f"on attempt {int(success_payload.get('generation_attempt', 1))} "
                f"search={int(search_health.get('search_nonempty_count', 0))}/"
                f"{int(search_health.get('search_node_count', 0))} "
                f"fetch={int(fetch_health.get('fetch_success_count', 0))}/"
                f"{int(fetch_health.get('fetch_request_count', 0))} "
                f"in {float(timings.get('total_seconds', 0.0)):.2f}s"
            )
            continue

        errors += 1
        failed_query_ids.append(query_id)
        payload = {
            "record_id": str(row.get("record_id", query_id)),
            "query_id": query_id,
            "query": query,
            "rubrics": _serialize_rubrics(row),
            "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
            "metadata": row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
            "planner_messages": last_planner_messages,
            "planner_turns": last_planner_spans,
            "planner_spans": last_planner_spans,
            "generation_attempts": attempt_summaries,
            "error": last_error or "unknown_error",
        }
        _atomic_write_json(error_path, payload)
        print(f"[{index}/{len(rows)}] error {query_id}: {payload['error']}")

    elapsed = time.perf_counter() - started
    summary = {
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "failed_query_ids": failed_query_ids,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
