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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.schemas.trajectory import SearchExecution, SearchPassage
from decomposer_v4.data.generation.generate_live_trajectory import (
    AnswererSession,
    _parse_answer_content,
)
from decomposer_v4.data.generation.scaffold import (
    aggregate_execution_waves,
    append_citation_block,
    apply_citations_to_node_outputs,
    build_citation_block,
    build_citation_catalog,
    build_execution_turn_payload,
    build_final_answer_payload,
    evidence_pool_from_executions,
    execute_revision_retrievals,
    hydrate_graph_with_evidence,
    normalize_fetch_results,
    normalize_search_results,
)
from decomposer_v4.utils.io import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable answerer trajectories from planner artifacts.")
    parser.add_argument("--planner-dir", required=True)
    parser.add_argument(
        "--answerer-profile",
        default="decomposer_v4/configs/model_profiles/openai_mini_nonreasoning.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetch-characters", type=int, default=6000)
    parser.add_argument("--max-concurrent-searches", type=int, default=4)
    parser.add_argument("--max-concurrent-fetches", type=int, default=4)
    parser.add_argument("--omit-answerer-system-prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _planner_files(planner_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in planner_dir.glob("*.json")
        if path.is_file() and not path.name.startswith("_")
    )


def _select_planner_files(paths: list[Path], query_id: str, offset: int, limit: int) -> list[Path]:
    if query_id:
        for path in paths:
            if path.stem == query_id:
                return [path]
        raise ValueError(f"Planner artifact not found for query_id={query_id}")
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


def _flatten_turns(planner_messages: list[dict[str, str]], answerer: AnswererSession) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage, messages in (("planner", planner_messages), ("answerer", answerer.messages)):
        for message in messages:
            rows.append(
                {
                    "stage": stage,
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
            )
    return rows


def _partial_answerer_trajectory(
    *,
    planner_record: dict[str, Any],
    planner_path: Path,
    answerer: AnswererSession,
    timings: dict[str, Any],
    search_node_outputs: dict[str, str],
    search_results: dict[str, list[dict[str, Any]]],
    fetch_blocks: dict[str, list[str]],
    fetched_results: list[dict[str, Any]],
    parsed_graph: dict[str, Any],
    produced_outputs: dict[str, str],
    citation_catalog: dict[str, dict[str, Any]],
    node_output_citation_blocks: dict[str, str],
    node_outputs_with_citations: dict[str, str],
    branch_output_citation_blocks: dict[str, str],
    final_answer_with_citations: str,
    final_answer_citation_block: str,
    all_executions: tuple[SearchExecution, ...],
    error: str,
) -> dict[str, Any]:
    final_graph = planner_record.get("final_graph", {}) if isinstance(planner_record.get("final_graph"), dict) else {}
    final_node_id = next(
        (
            str(node.get("id", ""))
            for node in final_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("type", "")) == "answer"
        ),
        "",
    )
    return {
        "record_id": str(planner_record.get("record_id") or planner_record.get("query_id") or planner_path.stem),
        "query_id": str(planner_record.get("query_id") or planner_path.stem),
        "query": str(planner_record.get("query", "")).strip(),
        "task_type": str(planner_record.get("task_type", "deep_research") or "deep_research"),
        "planner_artifact_path": str(planner_path),
        "planner_messages": planner_record.get("planner_messages", []),
        "planner_turns": planner_record.get("planner_turns", []),
        "planner_spans": planner_record.get("planner_spans", planner_record.get("planner_turns", [])),
        "answerer_messages": answerer.messages,
        "answerer_turns": answerer.spans,
        "answerer_spans": answerer.spans,
        "planner_initial_output": str(planner_record.get("planner_initial_output", "")),
        "planner_revision_output": str(planner_record.get("planner_revision_output", "")),
        "planner_action": str(planner_record.get("planner_action", "")),
        "initial_graph": planner_record.get("initial_graph", {}),
        "final_graph": final_graph,
        "parsed_graph": parsed_graph,
        "search_node_outputs": search_node_outputs,
        "search_results": search_results,
        "fetch_requests": planner_record.get("fetch_requests", []),
        "fetch_blocks": fetch_blocks,
        "fetched_results": fetched_results,
        "citation_catalog": citation_catalog,
        "node_outputs": produced_outputs,
        "node_output_citation_blocks": node_output_citation_blocks,
        "node_outputs_with_citations": node_outputs_with_citations,
        "branch_outputs": {
            key: value
            for key, value in produced_outputs.items()
            if key != final_node_id
        },
        "branch_output_citation_blocks": branch_output_citation_blocks,
        "branch_outputs_with_citations": {
            key: value
            for key, value in node_outputs_with_citations.items()
            if key != final_node_id
        },
        "final_answer": str(produced_outputs.get(final_node_id, "")),
        "final_answer_citation_block": final_answer_citation_block,
        "final_answer_with_citations": final_answer_with_citations,
        "timings": timings,
        "search_executions": [
            {
                "node_id": execution.node_id,
                "query": execution.query,
                "passages": [asdict(passage) for passage in execution.passages],
            }
            for execution in all_executions
        ],
        "evidence_pool": evidence_pool_from_executions(all_executions),
        "flat_turns": _flatten_turns(planner_record.get("planner_messages", []), answerer),
        "error": error,
    }


def build_answerer_trajectory_from_planner(
    *,
    planner_record: dict[str, Any],
    planner_path: Path,
    answerer_backend: Any,
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    max_fetch_characters: int,
    max_concurrent_searches: int = 4,
    max_concurrent_fetches: int = 4,
    include_answerer_system_prompt: bool = True,
) -> dict[str, Any]:
    query = str(planner_record.get("query", "")).strip()
    query_id = str(planner_record.get("query_id") or planner_path.stem)
    final_graph = planner_record.get("final_graph", {}) if isinstance(planner_record.get("final_graph"), dict) else {}
    timings = dict(planner_record.get("timings", {})) if isinstance(planner_record.get("timings"), dict) else {}
    timings.setdefault("answer_execution_turns", {})
    timings.setdefault("search_timings_post_revision", {})
    timings.setdefault("fetch_timings", {})
    timings["planner_total_seconds"] = float(timings.get("total_seconds", 0.0) or 0.0)

    answerer = AnswererSession(
        answerer_backend,
        query,
        include_system_prompt=include_answerer_system_prompt,
    )
    answerer_started = time.perf_counter()

    search_node_outputs = dict(planner_record.get("search_node_outputs", {})) if isinstance(planner_record.get("search_node_outputs"), dict) else {}
    search_results = dict(planner_record.get("search_results", {})) if isinstance(planner_record.get("search_results"), dict) else {}
    fetch_blocks: dict[str, list[str]] = {}
    fetched_results: list[dict[str, Any]] = []
    parsed_graph: dict[str, Any] = {}
    produced_outputs: dict[str, str] = {}
    citation_catalog: dict[str, dict[str, Any]] = {}
    node_output_citation_blocks: dict[str, str] = {}
    node_outputs_with_citations: dict[str, str] = {}
    branch_output_citation_blocks: dict[str, str] = {}
    final_answer_with_citations = ""
    final_answer_citation_block = ""

    existing_search_executions = tuple(
        _search_execution_from_dict(item)
        for item in planner_record.get("search_executions", [])
        if isinstance(item, dict)
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

    search_executions: tuple[SearchExecution, ...] = existing_search_executions
    fetch_executions: tuple[SearchExecution, ...] = ()

    try:
        ok, meta = validate_graph(final_graph)
        if not ok:
            raise RuntimeError(f"final_graph_invalid:{meta}")

        stage_started = time.perf_counter()
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
            num_results_per_search=num_results_per_search,
            existing_search_executions=existing_search_executions,
            fetch_requests=fetch_requests,
            max_fetch_characters=max_fetch_characters,
            max_concurrent_searches=max_concurrent_searches,
            max_concurrent_fetches=max_concurrent_fetches,
        )
        timings["post_revision_retrieval_seconds"] = time.perf_counter() - stage_started
        timings["search_timings_post_revision"] = search_timings_post_revision
        timings["fetch_timings"] = fetch_timings

        search_results = normalize_search_results(search_executions)
        fetched_results = normalize_fetch_results(fetch_requests, fetch_executions)
        parsed_graph = hydrate_graph_with_evidence(
            final_graph,
            search_results=search_results,
            fetched_results=fetched_results,
        )

        for wave_index, node_ids in enumerate(aggregate_execution_waves(final_graph), start=1):
            wave_started = time.perf_counter()
            payload = build_execution_turn_payload(
                query=query,
                graph=final_graph,
                node_ids=node_ids,
                search_results=search_results,
                fetched_results=fetched_results,
                produced_outputs=produced_outputs,
                wave_index=wave_index,
            )
            completion = answerer.run_turn(
                turn_type="answer_execution",
                payload=payload,
                expected_node_ids=node_ids,
            )
            produced_outputs.update(_parse_answer_content(completion, node_ids))
            timings["answer_execution_turns"][f"wave_{wave_index}"] = {
                "node_ids": list(node_ids),
                "seconds": time.perf_counter() - wave_started,
            }

        stage_started = time.perf_counter()
        final_payload = build_final_answer_payload(
            query=query,
            graph=final_graph,
            produced_outputs=produced_outputs,
        )
        final_node_id = next(
            str(node.get("id", ""))
            for node in final_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("type", "")) == "answer"
        )
        final_completion = answerer.run_turn(
            turn_type="answer_final",
            payload=final_payload,
            expected_node_ids=[final_node_id],
        )
        final_answer = _parse_answer_content(final_completion, [final_node_id])[final_node_id]
        produced_outputs[final_node_id] = final_answer

        citation_catalog = build_citation_catalog(
            search_results=search_results,
            fetched_results=fetched_results,
        )
        node_output_citation_blocks = {
            str(node_id): build_citation_block(str(content), citation_catalog)
            for node_id, content in produced_outputs.items()
        }
        node_outputs_with_citations = apply_citations_to_node_outputs(produced_outputs, citation_catalog)
        branch_output_citation_blocks = {
            str(node_id): citation_block
            for node_id, citation_block in node_output_citation_blocks.items()
            if str(node_id) != final_node_id
        }
        final_answer_citation_block = build_citation_block(final_answer, citation_catalog)
        final_answer_with_citations = append_citation_block(final_answer, citation_catalog)
        timings["answer_final_seconds"] = time.perf_counter() - stage_started
        timings["citation_count"] = len(citation_catalog)
        timings["answerer_stage_seconds"] = time.perf_counter() - answerer_started
        timings["total_seconds"] = float(timings.get("planner_total_seconds", 0.0)) + float(timings["answerer_stage_seconds"])

        all_executions = tuple(list(search_executions) + list(fetch_executions))
        return {
            "record_id": str(planner_record.get("record_id") or query_id),
            "query_id": query_id,
            "query": query,
            "task_type": str(planner_record.get("task_type", "deep_research") or "deep_research"),
            "planner_artifact_path": str(planner_path),
            "planner_messages": planner_record.get("planner_messages", []),
            "planner_turns": planner_record.get("planner_turns", []),
            "planner_spans": planner_record.get("planner_spans", planner_record.get("planner_turns", [])),
            "answerer_messages": answerer.messages,
            "answerer_turns": answerer.spans,
            "answerer_spans": answerer.spans,
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
            "node_outputs": produced_outputs,
            "node_output_citation_blocks": node_output_citation_blocks,
            "node_outputs_with_citations": node_outputs_with_citations,
            "branch_outputs": {
                key: value
                for key, value in produced_outputs.items()
                if key != final_node_id
            },
            "branch_output_citation_blocks": branch_output_citation_blocks,
            "branch_outputs_with_citations": {
                key: value
                for key, value in node_outputs_with_citations.items()
                if key != final_node_id
            },
            "final_answer": final_answer,
            "final_answer_citation_block": final_answer_citation_block,
            "final_answer_with_citations": final_answer_with_citations,
            "timings": timings,
            "search_executions": [
                {
                    "node_id": execution.node_id,
                    "query": execution.query,
                    "passages": [asdict(passage) for passage in execution.passages],
                }
                for execution in all_executions
            ],
            "evidence_pool": evidence_pool_from_executions(all_executions),
            "flat_turns": _flatten_turns(planner_record.get("planner_messages", []), answerer),
            "error": None,
        }
    except Exception as exc:
        timings["answerer_stage_seconds"] = time.perf_counter() - answerer_started
        timings["total_seconds"] = float(timings.get("planner_total_seconds", 0.0)) + float(timings["answerer_stage_seconds"])
        all_executions = tuple(list(search_executions) + list(fetch_executions))
        return _partial_answerer_trajectory(
            planner_record=planner_record,
            planner_path=planner_path,
            answerer=answerer,
            timings=timings,
            search_node_outputs=search_node_outputs,
            search_results=search_results,
            fetch_blocks=fetch_blocks,
            fetched_results=fetched_results,
            parsed_graph=parsed_graph,
            produced_outputs=produced_outputs,
            citation_catalog=citation_catalog,
            node_output_citation_blocks=node_output_citation_blocks,
            node_outputs_with_citations=node_outputs_with_citations,
            branch_output_citation_blocks=branch_output_citation_blocks,
            final_answer_with_citations=final_answer_with_citations,
            final_answer_citation_block=final_answer_citation_block,
            all_executions=all_executions,
            error=str(exc),
        )


def main() -> None:
    args = parse_args()
    from decomposer_v4.train.model_backends import build_chat_backend

    planner_dir = ROOT / args.planner_dir
    paths = _select_planner_files(_planner_files(planner_dir), args.query_id, args.offset, args.limit)
    answerer_backend = build_chat_backend(str(ROOT / args.answerer_profile))
    search_backend = MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )

    output_dir = ROOT / args.output_dir
    error_dir = output_dir / "_errors"
    _atomic_write_json(
        output_dir / "_run_config.json",
        {
            "planner_dir": args.planner_dir,
            "answerer_profile": args.answerer_profile,
            "output_dir": args.output_dir,
            "query_id": args.query_id,
            "limit": args.limit,
            "offset": args.offset,
            "num_results_per_search": args.num_results_per_search,
            "max_fetch_characters": args.max_fetch_characters,
            "max_concurrent_searches": args.max_concurrent_searches,
            "max_concurrent_fetches": args.max_concurrent_fetches,
            "omit_answerer_system_prompt": args.omit_answerer_system_prompt,
            "overwrite": args.overwrite,
            "planned_rows": len(paths),
        },
    )

    completed = 0
    skipped = 0
    errors = 0
    started = time.perf_counter()

    for index, planner_path in enumerate(paths, start=1):
        query_id = planner_path.stem
        target_path = output_dir / f"{query_id}.json"
        error_path = error_dir / f"{query_id}.json"
        if target_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(paths)}] skip {query_id}")
            continue

        planner_record = json.loads(planner_path.read_text(encoding="utf-8"))
        if str(planner_record.get("error") or "").strip():
            errors += 1
            _atomic_write_json(
                error_path,
                {
                    "planner_artifact_path": str(planner_path),
                    "query_id": query_id,
                    "error": f"planner_artifact_error:{planner_record.get('error')}",
                },
            )
            print(f"[{index}/{len(paths)}] error {query_id}: planner artifact has error")
            continue

        trajectory = build_answerer_trajectory_from_planner(
            planner_record=planner_record,
            planner_path=planner_path,
            answerer_backend=answerer_backend,
            search_backend=search_backend,
            num_results_per_search=args.num_results_per_search,
            max_fetch_characters=args.max_fetch_characters,
            max_concurrent_searches=args.max_concurrent_searches,
            max_concurrent_fetches=args.max_concurrent_fetches,
            include_answerer_system_prompt=not args.omit_answerer_system_prompt,
        )
        if trajectory.get("error"):
            errors += 1
            _atomic_write_json(error_path, trajectory)
            print(f"[{index}/{len(paths)}] error {query_id}: {trajectory['error']}")
            continue

        _atomic_write_json(target_path, trajectory)
        if error_path.exists():
            error_path.unlink()
        completed += 1
        answerer_seconds = float(trajectory.get("timings", {}).get("answerer_stage_seconds", 0.0))
        print(f"[{index}/{len(paths)}] ok {query_id} in {answerer_seconds:.2f}s")

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
