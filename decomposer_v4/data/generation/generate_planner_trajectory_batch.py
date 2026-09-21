#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.data.generation.generate_live_trajectory import PlannerSession, _iter_rows, _load_rows, _query_text
from decomposer_v4.data.generation.scaffold import (
    execute_search_nodes_timed,
    hydrate_graph_with_evidence,
    normalize_search_results,
)
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable planner-only trajectories for SFT.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument(
        "--planner-profile",
        default="decomposer_v4/configs/model_profiles/openai_mini_nonreasoning.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-id", default="")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetches", type=int, default=4)
    parser.add_argument("--max-concurrent-searches", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _planner_partial_trajectory(
    *,
    query_id: str,
    row: dict[str, Any],
    query: str,
    planner: PlannerSession,
    timings: dict[str, Any],
    planner_initial_output: str,
    planner_revision_output: str,
    planner_action: str,
    initial_graph: dict[str, Any],
    final_graph: dict[str, Any],
    search_node_outputs: dict[str, str],
    search_results: dict[str, list[dict[str, Any]]],
    fetch_requests: list[dict[str, str]],
    parsed_graph: dict[str, Any],
    search_executions: tuple[Any, ...],
    error: str,
) -> dict[str, Any]:
    return {
        "record_id": query_id,
        "query_id": query_id,
        "query": query,
        "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
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
        "fetched_results": [],
        "timings": timings,
        "search_executions": [
            {
                "node_id": execution.node_id,
                "query": execution.query,
                "passages": [asdict(passage) for passage in execution.passages],
            }
            for execution in search_executions
        ],
        "flat_turns": [
            {
                "stage": "planner",
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in planner.messages
        ],
        "error": error,
    }


def build_planner_trajectory(
    *,
    row: dict[str, Any],
    planner_backend: Any,
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    max_fetches: int,
    max_concurrent_searches: int = 4,
) -> dict[str, Any]:
    query = _query_text(row)
    query_id = str(row.get("query_id") or row.get("id") or hashlib.md5(query.encode("utf-8")).hexdigest()[:12])
    timings: dict[str, Any] = {"search_timings_initial": {}}
    total_started = time.perf_counter()

    planner = PlannerSession(planner_backend, query)
    planner_initial_output = ""
    planner_revision_output = ""
    planner_action = ""
    initial_graph: dict[str, Any] = {}
    final_graph: dict[str, Any] = {}
    search_node_outputs: dict[str, str] = {}
    search_results: dict[str, list[dict[str, Any]]] = {}
    fetch_requests: list[dict[str, str]] = []
    parsed_graph: dict[str, Any] = {}
    search_executions: tuple[Any, ...] = ()

    try:
        stage_started = time.perf_counter()
        planner_initial_output, initial_graph = planner.initial_plan()
        timings["planner_initial_seconds"] = time.perf_counter() - stage_started
        ok, meta = validate_graph(initial_graph)
        if not ok:
            raise RuntimeError(f"initial_graph_invalid:{meta}")

        stage_started = time.perf_counter()
        search_node_outputs, search_executions, search_timings_initial = execute_search_nodes_timed(
            graph=initial_graph,
            search_backend=search_backend,
            num_results_per_search=num_results_per_search,
            max_concurrent_searches=max_concurrent_searches,
        )
        timings["search_seconds"] = time.perf_counter() - stage_started
        timings["search_timings_initial"] = search_timings_initial
        search_results = normalize_search_results(search_executions)

        stage_started = time.perf_counter()
        planner_revision_output, planner_action, final_graph, fetch_requests = planner.revise_and_fetch(
            current_graph=initial_graph,
            search_results=search_results,
            max_fetches=max_fetches,
        )
        timings["planner_revision_seconds"] = time.perf_counter() - stage_started
        ok, meta = validate_graph(final_graph)
        if not ok:
            raise RuntimeError(f"final_graph_invalid:{meta}")

        parsed_graph = hydrate_graph_with_evidence(
            final_graph,
            search_results=search_results,
            fetched_results=[],
        )
        timings["total_seconds"] = time.perf_counter() - total_started
        return {
            "record_id": query_id,
            "query_id": query_id,
            "query": query,
            "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
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
            "fetched_results": [],
            "timings": timings,
            "search_executions": [
                {
                    "node_id": execution.node_id,
                    "query": execution.query,
                    "passages": [asdict(passage) for passage in execution.passages],
                }
                for execution in search_executions
            ],
            "flat_turns": [
                {
                    "stage": "planner",
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
                for message in planner.messages
            ],
            "error": None,
        }
    except Exception as exc:
        timings["total_seconds"] = time.perf_counter() - total_started
        return _planner_partial_trajectory(
            query_id=query_id,
            row=row,
            query=query,
            planner=planner,
            timings=timings,
            planner_initial_output=planner_initial_output,
            planner_revision_output=planner_revision_output,
            planner_action=planner_action,
            initial_graph=initial_graph,
            final_graph=final_graph,
            search_node_outputs=search_node_outputs,
            search_results=search_results,
            fetch_requests=fetch_requests,
            parsed_graph=parsed_graph,
            search_executions=search_executions,
            error=str(exc),
        )


def main() -> None:
    args = parse_args()
    from decomposer_v4.train.model_backends import build_chat_backend

    rows = _iter_rows(_load_rows(args.input_path), args.query_id, args.offset, args.limit)
    planner_backend = build_chat_backend(str(ROOT / args.planner_profile))
    search_backend = MultiSearchClient(clients=(JinaSearchClient.from_env(),))

    output_dir = ROOT / args.output_dir
    error_dir = output_dir / "_errors"
    config_path = output_dir / "_run_config.json"
    _atomic_write_json(
        config_path,
        {
            "input_path": args.input_path,
            "planner_profile": args.planner_profile,
            "output_dir": args.output_dir,
            "query_id": args.query_id,
            "limit": args.limit,
            "offset": args.offset,
            "num_results_per_search": args.num_results_per_search,
            "max_fetches": args.max_fetches,
            "max_concurrent_searches": args.max_concurrent_searches,
            "overwrite": args.overwrite,
            "planned_rows": len(rows),
        },
    )

    completed = 0
    skipped = 0
    errors = 0
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        query = _query_text(row)
        query_id = str(row.get("query_id") or row.get("id") or hashlib.md5(query.encode("utf-8")).hexdigest()[:12])
        if not query_id or not query:
            errors += 1
            continue

        target_path = output_dir / f"{query_id}.json"
        error_path = error_dir / f"{query_id}.json"
        if target_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(rows)}] skip {query_id}")
            continue

        trajectory = build_planner_trajectory(
            row=row,
            planner_backend=planner_backend,
            search_backend=search_backend,
            num_results_per_search=args.num_results_per_search,
            max_fetches=args.max_fetches,
            max_concurrent_searches=args.max_concurrent_searches,
        )
        if trajectory.get("error"):
            errors += 1
            _atomic_write_json(error_path, trajectory)
            print(f"[{index}/{len(rows)}] error {query_id}: {trajectory['error']}")
            continue

        _atomic_write_json(target_path, trajectory)
        if error_path.exists():
            error_path.unlink()
        completed += 1
        planner_seconds = float(trajectory.get("timings", {}).get("total_seconds", 0.0))
        print(f"[{index}/{len(rows)}] ok {query_id} in {planner_seconds:.2f}s")

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "errors": errors,
                "elapsed_seconds": elapsed,
                "output_dir": str(output_dir),
                "error_dir": str(error_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
