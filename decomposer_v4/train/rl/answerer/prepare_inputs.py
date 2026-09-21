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

from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.schemas.trajectory import SearchExecution, SearchPassage
from decomposer_v4.data.generation.scaffold import (
    build_citation_catalog,
    evidence_pool_from_executions,
    execute_revision_retrievals,
    hydrate_graph_with_evidence,
    normalize_fetch_results,
    normalize_search_results,
)
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare retrieval-complete answerer RL inputs from regenerated planners.")
    parser.add_argument("--planner-dir", default="outputs/rl/planner_regenerated")
    parser.add_argument("--output-dir", default="data/rl/answerer_prepared")
    parser.add_argument("--query-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetch-characters", type=int, default=6000)
    parser.add_argument("--max-concurrent-searches", type=int, default=8)
    parser.add_argument("--max-concurrent-fetches", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _planner_files(planner_dir: Path) -> list[Path]:
    return sorted(path for path in planner_dir.glob("*.json") if path.is_file() and not path.name.startswith("_"))


def _select_files(paths: list[Path], query_id: str, offset: int, limit: int) -> list[Path]:
    if query_id:
        return [path for path in paths if path.stem == query_id]
    sliced = paths[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def _search_execution_from_dict(payload: dict[str, Any]) -> SearchExecution:
    passages = tuple(
        SearchPassage(
            passage_id=str(passage.get("passage_id", "")).strip(),
            node_id=str(passage.get("node_id", "")).strip(),
            title=str(passage.get("title", "")).strip(),
            url=str(passage.get("url", "")).strip(),
            text=str(passage.get("text", "")).strip(),
            score=float(passage.get("score", 0.0) or 0.0),
            source=str(passage.get("source", "")).strip(),
            published_date=str(passage.get("published_date", "")).strip(),
            author=str(passage.get("author", "")).strip(),
            language=str(passage.get("language", "")).strip(),
            language_score=float(passage.get("language_score", 0.0) or 0.0),
            token_count=int(passage.get("token_count", 0) or 0),
        )
        for passage in payload.get("passages", [])
        if isinstance(passage, dict)
    )
    return SearchExecution(
        node_id=str(payload.get("node_id", "")).strip(),
        query=str(payload.get("query", "")).strip(),
        passages=passages,
    )


def _fetch_execution_from_result_dict(payload: dict[str, Any]) -> SearchExecution:
    node_id = str(payload.get("source_node_id", payload.get("node_id", ""))).strip()
    url = str(payload.get("url", "")).strip()
    passage = SearchPassage(
        passage_id=str(payload.get("passage_id", "")).strip(),
        node_id=node_id,
        title=str(payload.get("title", "")).strip(),
        url=url,
        text=str(payload.get("text", "")).strip(),
        score=float(payload.get("score", 0.0) or 0.0),
        source=str(payload.get("source", "serper_fetch")).strip(),
        published_date=str(payload.get("published_date", "")).strip(),
        author=str(payload.get("author", "")).strip(),
        language=str(payload.get("language", "")).strip(),
        language_score=float(payload.get("language_score", 0.0) or 0.0),
        token_count=int(payload.get("token_count", 0) or 0),
    )
    return SearchExecution(node_id=node_id, query=url, passages=(passage,))


def main() -> None:
    args = parse_args()
    planner_dir = ROOT / args.planner_dir
    output_dir = ROOT / args.output_dir
    error_dir = output_dir / "_errors"

    files = _select_files(_planner_files(planner_dir), args.query_id, args.offset, args.limit)
    if args.query_id and not files:
        raise ValueError(f"Planner artifact not found for query_id={args.query_id}")
    if not files:
        raise ValueError(
            f"No planner JSON artifacts found in {planner_dir}. "
            "Check PLANNER_DIR; it should contain one JSON file per regenerated query."
        )

    search_backend: MultiSearchClient | None = None

    completed = 0
    skipped = 0
    errors = 0
    started = time.perf_counter()

    for index, planner_path in enumerate(files, start=1):
        query_id = planner_path.stem
        target_path = output_dir / f"{query_id}.json"
        error_path = error_dir / f"{query_id}.json"
        if target_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(files)}] skip {query_id}")
            continue

        planner_record = json.loads(planner_path.read_text(encoding="utf-8"))
        final_graph = planner_record.get("final_graph", {}) if isinstance(planner_record.get("final_graph"), dict) else {}
        timings = dict(planner_record.get("timings", {})) if isinstance(planner_record.get("timings"), dict) else {}
        timings.setdefault("search_timings_post_revision", {})
        timings.setdefault("fetch_timings", {})

        try:
            ok, meta = validate_graph(final_graph)
            if not ok:
                raise RuntimeError(f"final_graph_invalid:{meta}")

            existing_search_executions = tuple(
                _search_execution_from_dict(item)
                for item in planner_record.get("search_executions", [])
                if isinstance(item, dict)
            )
            existing_fetch_executions = tuple(
                _search_execution_from_dict(item)
                for item in planner_record.get("fetch_executions", [])
                if isinstance(item, dict)
            )
            if not existing_fetch_executions:
                existing_fetch_executions = tuple(
                    _fetch_execution_from_result_dict(item)
                    for item in planner_record.get("fetched_results", [])
                    if isinstance(item, dict) and str(item.get("text", "")).strip()
                )
            fetch_requests = [
                {
                    "source_node_id": str(item.get("source_node_id", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                }
                for item in planner_record.get("fetch_requests", [])
                if isinstance(item, dict)
            ]

            stage_started = time.perf_counter()
            can_reuse_existing_retrieval = bool(existing_search_executions) and (
                bool(existing_fetch_executions) or not fetch_requests
            )
            if can_reuse_existing_retrieval:
                search_node_outputs = (
                    planner_record.get("search_node_outputs", {})
                    if isinstance(planner_record.get("search_node_outputs"), dict)
                    else {}
                )
                search_executions = existing_search_executions
                search_timings_post_revision = timings.get("search_timings_post_revision", {})
                fetch_blocks = (
                    planner_record.get("fetch_blocks", {})
                    if isinstance(planner_record.get("fetch_blocks"), dict)
                    else {}
                )
                fetch_executions = existing_fetch_executions
                fetch_timings = timings.get("fetch_timings", {})
            else:
                if search_backend is None:
                    search_backend = MultiSearchClient(
                        clients=(JinaSearchClient.from_env(),),
                        fetch_clients=(SerperFetchClient.from_env(),),
                    )
                (
                    search_node_outputs,
                    search_executions,
                    search_timings_post_revision,
                    fetch_blocks,
                    fetch_executions,
                    fetch_timings,
                ) = execute_revision_retrievals(
                    graph=final_graph,
                    search_backend=search_backend,
                    num_results_per_search=args.num_results_per_search,
                    existing_search_executions=existing_search_executions,
                    fetch_requests=fetch_requests,
                    max_fetch_characters=args.max_fetch_characters,
                    max_concurrent_searches=args.max_concurrent_searches,
                    max_concurrent_fetches=args.max_concurrent_fetches,
                )
            timings["post_revision_retrieval_seconds"] = time.perf_counter() - stage_started
            timings["post_revision_retrieval_reused"] = can_reuse_existing_retrieval
            timings["search_timings_post_revision"] = search_timings_post_revision
            timings["fetch_timings"] = fetch_timings

            search_results = normalize_search_results(search_executions)
            fetched_results = normalize_fetch_results(fetch_requests, fetch_executions)
            citation_catalog = build_citation_catalog(
                search_results=search_results,
                fetched_results=fetched_results,
            )
            parsed_graph = hydrate_graph_with_evidence(
                final_graph,
                search_results=search_results,
                fetched_results=fetched_results,
            )
            all_executions = tuple(list(search_executions) + list(fetch_executions))

            payload = {
                "record_id": str(planner_record.get("record_id", query_id)),
                "query_id": str(planner_record.get("query_id", query_id)),
                "query": str(planner_record.get("query", "")),
                "rubrics": planner_record.get("rubrics", []) if isinstance(planner_record.get("rubrics"), list) else [],
                "task_type": str(planner_record.get("task_type", "deep_research") or "deep_research"),
                "metadata": planner_record.get("metadata", {}) if isinstance(planner_record.get("metadata"), dict) else {},
                "planner_artifact_path": str(planner_path),
                "planner_messages": planner_record.get("planner_messages", []),
                "planner_turns": planner_record.get("planner_turns", []),
                "planner_spans": planner_record.get("planner_spans", planner_record.get("planner_turns", [])),
                "planner_initial_output": str(planner_record.get("planner_initial_output", "")),
                "planner_revision_output": str(planner_record.get("planner_revision_output", "")),
                "planner_action": str(planner_record.get("planner_action", "")),
                "initial_graph": planner_record.get("initial_graph", {}),
                "final_graph": final_graph,
                "parsed_graph": parsed_graph,
                "search_node_outputs": search_node_outputs,
                "search_results": search_results,
                "fetch_requests": fetch_requests,
                "fetch_blocks": fetch_blocks,
                "fetched_results": fetched_results,
                "citation_catalog": citation_catalog,
                "timings": timings,
                "search_executions": [
                    {
                        "node_id": execution.node_id,
                        "query": execution.query,
                        "passages": [asdict(passage) for passage in execution.passages],
                    }
                    for execution in search_executions
                ],
                "fetch_executions": [
                    {
                        "node_id": execution.node_id,
                        "query": execution.query,
                        "passages": [asdict(passage) for passage in execution.passages],
                    }
                    for execution in fetch_executions
                ],
                "evidence_pool": evidence_pool_from_executions(all_executions),
                "error": None,
            }
            _atomic_write_json(target_path, payload)
            if error_path.exists():
                error_path.unlink()
            completed += 1
            print(f"[{index}/{len(files)}] ok {query_id}")
        except Exception as exc:
            errors += 1
            payload = {
                "query_id": query_id,
                "planner_artifact_path": str(planner_path),
                "timings": timings,
                "error": str(exc),
            }
            _atomic_write_json(error_path, payload)
            print(f"[{index}/{len(files)}] error {query_id}: {exc}")

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "errors": errors,
                "elapsed_seconds": elapsed,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
