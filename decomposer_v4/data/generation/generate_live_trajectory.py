#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.data.generation.scaffold import (
    aggregate_execution_waves,
    append_citation_block,
    apply_citations_to_node_outputs,
    build_after_search_payload,
    build_citation_catalog,
    build_citation_block,
    build_execution_turn_payload,
    build_final_answer_payload,
    evidence_pool_from_executions,
    execute_revision_retrievals,
    execute_search_nodes_timed,
    hydrate_graph_with_evidence,
    normalize_fetch_results,
    normalize_search_results,
    parse_answerer_node_outputs,
    parse_planner_initial_output,
    parse_planner_revision_output,
)
from decomposer_v4.data.generation.answerer_compact_scaffold import (
    build_execution_turn_payload_compact,
    build_final_answer_payload_compact,
    search_dependency_ids_for_wave,
)
from decomposer_v4.utils.io import ROOT, read_csv_rows, read_jsonl, read_text, write_json, write_jsonl


PLANNER_SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/planner_system_prompt.txt")
ANSWERER_SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/answerer_system_prompt.txt")
ANSWERER_SYSTEM_PROMPT_COMPACT = read_text(ROOT / "decomposer_v4/prompts/answerer_system_prompt_compact.txt")


def load_planner_system_prompt(variant: str = "default") -> str:
    normalized = str(variant or "default").strip().lower()
    if normalized in {"", "default", "standard"}:
        return PLANNER_SYSTEM_PROMPT
    path = ROOT / f"decomposer_v4/prompts/planner_system_prompt_{normalized}.txt"
    if not path.exists():
        raise ValueError(f"Unknown planner system prompt variant: {variant}")
    return read_text(path)


def load_answerer_system_prompt(variant: str = "default") -> str:
    normalized = str(variant or "default").strip().lower()
    if normalized in {"", "default", "standard"}:
        return ANSWERER_SYSTEM_PROMPT
    if normalized == "compact":
        return ANSWERER_SYSTEM_PROMPT_COMPACT
    path = ROOT / f"decomposer_v4/prompts/answerer_system_prompt_{normalized}.txt"
    if not path.exists():
        raise ValueError(f"Unknown answerer system prompt variant: {variant}")
    return read_text(path)


def _format_exception(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _format_traceback(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate planner + answerer trajectories for decomposer_v4.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--query-id", default="")
    parser.add_argument(
        "--planner-profile",
        default="decomposer_v4/configs/model_profiles/openai_mini_nonreasoning.yaml",
    )
    parser.add_argument(
        "--answerer-profile",
        default="decomposer_v4/configs/model_profiles/openai_mini_nonreasoning.yaml",
    )
    parser.add_argument("--output-json", default="outputs/smoke/live_trajectory.json")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetches", type=int, default=4)
    parser.add_argument("--max-fetch-characters", type=int, default=6000)
    parser.add_argument("--max-concurrent-searches", type=int, default=4)
    parser.add_argument("--max-concurrent-fetches", type=int, default=4)
    parser.add_argument("--omit-answerer-system-prompt", action="store_true")
    parser.add_argument("--planner-prompt-variant", default="default")
    parser.add_argument("--answerer-payload-variant", choices=("default", "compact"), default="default")
    parser.add_argument("--answerer-prompt-variant", default="default")
    parser.add_argument("--answer-execution-max-completion-tokens", type=int, default=0)
    parser.add_argument("--answer-final-max-completion-tokens", type=int, default=0)
    parser.add_argument("--disable-citation-appendix", action="store_true")
    return parser.parse_args()


def _load_rows(path: str) -> list[dict[str, Any]]:
    if path.endswith(".csv"):
        return read_csv_rows(ROOT / path)
    return read_jsonl(ROOT / path)


def _query_text(row: dict[str, Any]) -> str:
    for key in ("query", "question", "_question_text"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _iter_rows(rows: list[dict[str, Any]], query_id: str, offset: int, limit: int) -> list[dict[str, Any]]:
    if query_id:
        for row in rows:
            if str(row.get("query_id", "")) == query_id or str(row.get("id", "")) == query_id:
                return [row]
        raise ValueError(f"Query id not found: {query_id}")
    filtered = [row for row in rows if _query_text(row)]
    return filtered[offset : offset + limit if limit > 0 else None]


def _answerer_completion_is_usable(text: str, expected_node_ids: list[str] | tuple[str, ...] | None = None) -> bool:
    try:
        parsed = parse_answerer_node_outputs(text, expected_node_ids=expected_node_ids)
    except Exception:
        return False
    return all(str(content or "").strip() for content in parsed.values())


def _answerer_retry_prompt(expected_node_ids: list[str], turn_type: str) -> str:
    ids = ", ".join(expected_node_ids)
    return (
        f"Your previous {turn_type} response was empty, malformed, or targeted the wrong node. "
        "Return a very brief `<think>` block followed by XML node blocks only. "
        f"Emit exactly these node ids: {ids}. "
        "Use the form `<node id=\"N3\">...</node>` with non-empty content for every required node. "
        "Do not add commentary outside `<think>` and `<node>` blocks."
    )


def _planner_initial_retry_prompt() -> str:
    return (
        "Your previous planner response was empty or invalid. "
        "Return a very brief `<think>` block followed by JSON only with a top-level `graph` object, "
        "or a raw graph object, that matches the required schema exactly."
    )


def _planner_revision_retry_prompt() -> str:
    return (
        "Your previous planner-after-search response was empty or invalid. "
        "Return a very brief `<think>` block followed by JSON only with `action`, `revised_graph` (or `graph`), "
        "and `fetch_requests`."
    )


class PlannerSession:
    def __init__(self, backend: Any, query: str, *, system_prompt: str | None = None) -> None:
        self.backend = backend
        self.query = query
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt or PLANNER_SYSTEM_PROMPT},
        ]
        self.spans: list[dict[str, Any]] = []

    def initial_plan(self) -> tuple[str, dict[str, Any]]:
        clean_prompt = [*self.messages, {"role": "user", "content": self.query}]
        completion = ""
        parsed_graph: dict[str, Any] | None = None
        attempt_records: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            if attempt == 1:
                attempt_messages = [dict(item) for item in clean_prompt]
            else:
                attempt_messages = [
                    *clean_prompt,
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": _planner_initial_retry_prompt()},
                ]
            completion = self.backend.complete(messages=attempt_messages, response_format_json=False)
            attempt_records.append(
                {
                    "attempt": attempt,
                    "messages": [dict(item) for item in attempt_messages],
                    "completion": completion,
                }
            )
            try:
                parsed_graph = parse_planner_initial_output(completion)
                break
            except Exception:
                continue
        self.messages.extend(
            [
                {"role": "user", "content": self.query},
                {"role": "assistant", "content": completion},
            ]
        )
        self.spans.append(
            {
                "span_id": "planner_initial",
                "role": "planner",
                "turn_type": "planner_initial",
                "messages": [dict(item) for item in clean_prompt],
                "completion": completion,
                "attempts": attempt_records,
            }
        )
        if parsed_graph is None:
            return completion, parse_planner_initial_output(completion)
        return completion, parsed_graph

    def revise_and_fetch(
        self,
        *,
        current_graph: dict[str, Any],
        search_results: dict[str, list[dict[str, Any]]],
        max_fetches: int,
    ) -> tuple[str, str, dict[str, Any], list[dict[str, str]]]:
        user_payload = build_after_search_payload(self.query, current_graph, search_results, max_fetches)
        clean_prompt = [
            *self.messages,
            {"role": "user", "content": user_payload},
        ]
        completion = ""
        parsed_result: tuple[str, dict[str, Any], list[dict[str, str]]] | None = None
        attempt_records: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            if attempt == 1:
                attempt_messages = [dict(item) for item in clean_prompt]
            else:
                attempt_messages = [
                    *clean_prompt,
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": _planner_revision_retry_prompt()},
                ]
            completion = self.backend.complete(messages=attempt_messages, response_format_json=False)
            attempt_records.append(
                {
                    "attempt": attempt,
                    "messages": [dict(item) for item in attempt_messages],
                    "completion": completion,
                }
            )
            try:
                parsed_result = parse_planner_revision_output(
                    completion,
                    current_graph=current_graph,
                    available_search_results=search_results,
                    max_fetches=max_fetches,
                )
                break
            except Exception:
                continue
        self.messages.extend(
            [
                {"role": "user", "content": user_payload},
                {"role": "assistant", "content": completion},
            ]
        )
        self.spans.append(
            {
                "span_id": "planner_revision",
                "role": "planner",
                "turn_type": "planner_revision",
                "messages": [dict(item) for item in clean_prompt],
                "completion": completion,
                "attempts": attempt_records,
            }
        )
        if parsed_result is None:
            parsed_result = parse_planner_revision_output(
                completion,
                current_graph=current_graph,
                available_search_results=search_results,
                max_fetches=max_fetches,
            )
        action, final_graph, fetch_requests = parsed_result
        return completion, action, final_graph, fetch_requests


class AnswererSession:
    def __init__(
        self,
        backend: Any,
        query: str,
        *,
        include_system_prompt: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.query = query
        self.include_system_prompt = include_system_prompt
        self.system_message = {"role": "system", "content": system_prompt or ANSWERER_SYSTEM_PROMPT}
        self.messages: list[dict[str, str]] = [dict(self.system_message)] if include_system_prompt else []
        self.spans: list[dict[str, Any]] = []

    def run_turn(
        self,
        *,
        turn_type: str,
        payload: str,
        expected_node_ids: list[str],
        max_completion_tokens: int | None = None,
    ) -> str:
        clean_prompt = [*self.messages, {"role": "user", "content": payload}]
        completion = ""
        attempt_records: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            if attempt == 1:
                attempt_messages = [dict(item) for item in clean_prompt]
            else:
                attempt_messages = [
                    *clean_prompt,
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": _answerer_retry_prompt(expected_node_ids, turn_type)},
                ]
            completion = self.backend.complete(
                messages=attempt_messages,
                response_format_json=False,
                max_completion_tokens=max_completion_tokens,
            )
            attempt_records.append(
                {
                    "attempt": attempt,
                    "messages": [dict(item) for item in attempt_messages],
                    "completion": completion,
                }
            )
            if _answerer_completion_is_usable(completion, expected_node_ids):
                break
        self.messages.append({"role": "user", "content": payload})
        self.messages.append({"role": "assistant", "content": completion})
        self.spans.append(
            {
                "span_id": f"{turn_type}_{len(self.spans) + 1}",
                "role": "answerer",
                "turn_type": turn_type,
                "messages": [dict(item) for item in clean_prompt],
                "completion": completion,
                "attempts": attempt_records,
                "target_node_ids": list(expected_node_ids),
            }
        )
        return completion


def _parse_answer_content(text: str, expected_node_ids: list[str]) -> dict[str, str]:
    return parse_answerer_node_outputs(text, expected_node_ids=expected_node_ids)


def _flatten_turns(planner: PlannerSession, answerer: AnswererSession) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage, messages in (("planner", planner.messages), ("answerer", answerer.messages)):
        for message in messages:
            rows.append(
                {
                    "stage": stage,
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
            )
    return rows


def _partial_trajectory(
    *,
    query_id: str,
    row: dict[str, Any],
    query: str,
    planner: PlannerSession,
    answerer: AnswererSession,
    timings: dict[str, Any],
    planner_initial_output: str,
    planner_revision_output: str,
    planner_action: str,
    initial_graph: dict[str, Any],
    final_graph: dict[str, Any],
    search_node_outputs: dict[str, str],
    search_results: dict[str, list[dict[str, Any]]],
    fetch_requests: list[dict[str, str]],
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
    all_executions: tuple[Any, ...],
    error: str,
    error_traceback: str = "",
) -> dict[str, Any]:
    evidence_pool = evidence_pool_from_executions(all_executions)
    final_node_id = next(
        (
            str(node.get("id", ""))
            for node in final_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("type", "")) == "answer"
        ),
        "",
    )
    return {
        "record_id": query_id,
        "query_id": query_id,
        "query": query,
        "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
        "planner_messages": planner.messages,
        "planner_turns": planner.spans,
        "planner_spans": planner.spans,
        "answerer_messages": answerer.messages,
        "answerer_turns": answerer.spans,
        "answerer_spans": answerer.spans,
        "planner_initial_output": planner_initial_output,
        "planner_revision_output": planner_revision_output,
        "planner_action": planner_action,
        "initial_graph": initial_graph,
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
        "evidence_pool": evidence_pool,
        "flat_turns": _flatten_turns(planner, answerer),
        "error": error,
        "error_traceback": error_traceback,
    }


def build_trajectory(
    *,
    row: dict[str, Any],
    planner_backend: Any,
    answerer_backend: Any,
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    max_fetches: int,
    max_fetch_characters: int,
    max_concurrent_searches: int = 4,
    max_concurrent_fetches: int = 4,
    include_answerer_system_prompt: bool = True,
    planner_prompt_variant: str = "default",
    planner_system_prompt: str | None = None,
    answerer_payload_variant: str = "default",
    answerer_prompt_variant: str = "default",
    answerer_system_prompt: str | None = None,
    answer_execution_max_completion_tokens: int | None = None,
    answer_final_max_completion_tokens: int | None = None,
    append_final_citation_block: bool = True,
) -> dict[str, Any]:
    query = _query_text(row)
    query_id = str(row.get("query_id") or row.get("id") or hashlib.md5(query.encode("utf-8")).hexdigest()[:12])
    timings: dict[str, Any] = {
        "answer_execution_turns": {},
        "search_timings_initial": {},
        "search_timings_post_revision": {},
        "fetch_timings": {},
    }
    total_started = time.perf_counter()

    if planner_system_prompt is None:
        planner_system_prompt = load_planner_system_prompt(planner_prompt_variant)
    planner = PlannerSession(planner_backend, query, system_prompt=planner_system_prompt)
    normalized_payload_variant = str(answerer_payload_variant or "default").strip().lower()
    normalized_prompt_variant = str(answerer_prompt_variant or "default").strip().lower()
    if answerer_system_prompt is None:
        answerer_system_prompt = load_answerer_system_prompt(normalized_prompt_variant)
    answerer = AnswererSession(
        answerer_backend,
        query,
        include_system_prompt=include_answerer_system_prompt,
        system_prompt=answerer_system_prompt,
    )
    planner_initial_output = ""
    planner_revision_output = ""
    planner_action = ""
    initial_graph: dict[str, Any] = {}
    final_graph: dict[str, Any] = {}
    search_node_outputs: dict[str, str] = {}
    search_results: dict[str, list[dict[str, Any]]] = {}
    fetch_requests: list[dict[str, str]] = []
    fetch_blocks: dict[str, list[str]] = {}
    fetched_results: list[dict[str, Any]] = []
    search_executions: tuple[Any, ...] = ()
    fetch_executions: tuple[Any, ...] = ()
    parsed_graph: dict[str, Any] = {}
    produced_outputs: dict[str, str] = {}
    final_answer = ""
    citation_catalog: dict[str, dict[str, Any]] = {}
    node_output_citation_blocks: dict[str, str] = {}
    node_outputs_with_citations: dict[str, str] = {}
    branch_output_citation_blocks: dict[str, str] = {}
    final_answer_with_citations = ""
    final_answer_citation_block = ""

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
        timings["initial_search_seconds"] = timings["search_seconds"]
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
            existing_search_executions=search_executions,
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

        introduced_search_parent_ids: set[str] = set()
        for wave_index, node_ids in enumerate(aggregate_execution_waves(final_graph), start=1):
            wave_started = time.perf_counter()
            if normalized_payload_variant == "compact":
                payload = build_execution_turn_payload_compact(
                    query=query,
                    graph=final_graph,
                    node_ids=node_ids,
                    search_results=search_results,
                    fetched_results=fetched_results,
                    produced_outputs=produced_outputs,
                    wave_index=wave_index,
                    introduced_search_parent_ids=introduced_search_parent_ids,
                )
            else:
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
                max_completion_tokens=answer_execution_max_completion_tokens,
            )
            produced_outputs.update(_parse_answer_content(completion, node_ids))
            if normalized_payload_variant == "compact":
                introduced_search_parent_ids.update(search_dependency_ids_for_wave(final_graph, node_ids))
            timings["answer_execution_turns"][f"wave_{wave_index}"] = {
                "node_ids": list(node_ids),
                "seconds": time.perf_counter() - wave_started,
            }

        stage_started = time.perf_counter()
        if normalized_payload_variant == "compact":
            final_payload = build_final_answer_payload_compact(
                query=query,
                graph=final_graph,
                produced_outputs=produced_outputs,
            )
        else:
            final_payload = build_final_answer_payload(query=query, graph=final_graph, produced_outputs=produced_outputs)
        final_node_id = next(
            str(node.get("id", ""))
            for node in final_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("type", "")) == "answer"
        )
        final_completion = answerer.run_turn(
            turn_type="answer_final",
            payload=final_payload,
            expected_node_ids=[final_node_id],
            max_completion_tokens=answer_final_max_completion_tokens,
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
        if append_final_citation_block:
            final_answer_with_citations = append_citation_block(final_answer, citation_catalog)
        else:
            final_answer_with_citations = final_answer
        timings["answer_final_seconds"] = time.perf_counter() - stage_started
        timings["citation_count"] = len(citation_catalog)
        timings["total_seconds"] = time.perf_counter() - total_started

        all_executions = tuple(list(search_executions) + list(fetch_executions))
        evidence_pool = evidence_pool_from_executions(all_executions)

        return {
            "record_id": query_id,
            "query_id": query_id,
            "query": query,
            "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
            "planner_messages": planner.messages,
            "planner_turns": planner.spans,
            "planner_spans": planner.spans,
            "answerer_messages": answerer.messages,
            "answerer_turns": answerer.spans,
            "answerer_spans": answerer.spans,
            "planner_initial_output": planner_initial_output,
            "planner_revision_output": planner_revision_output,
            "planner_action": planner_action,
            "initial_graph": initial_graph,
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
            "evidence_pool": evidence_pool,
            "flat_turns": _flatten_turns(planner, answerer),
        }
    except Exception as exc:
        timings["total_seconds"] = time.perf_counter() - total_started
        all_executions = tuple(list(search_executions) + list(fetch_executions))
        return _partial_trajectory(
            query_id=query_id,
            row=row,
            query=query,
            planner=planner,
            answerer=answerer,
            timings=timings,
            planner_initial_output=planner_initial_output,
            planner_revision_output=planner_revision_output,
            planner_action=planner_action,
            initial_graph=initial_graph,
            final_graph=final_graph,
            search_node_outputs=search_node_outputs,
            search_results=search_results,
            fetch_requests=fetch_requests,
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
            error=_format_exception(exc),
            error_traceback=_format_traceback(exc),
        )


def main() -> None:
    args = parse_args()
    from decomposer_v4.train.model_backends import build_chat_backend

    rows = _iter_rows(_load_rows(args.input_path), args.query_id, args.offset, args.limit)
    planner_backend = build_chat_backend(str(ROOT / args.planner_profile))
    answerer_backend = build_chat_backend(str(ROOT / args.answerer_profile))
    search_backend = MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )
    trajectories = [
        build_trajectory(
            row=row,
            planner_backend=planner_backend,
            answerer_backend=answerer_backend,
            search_backend=search_backend,
            num_results_per_search=args.num_results_per_search,
            max_fetches=args.max_fetches,
            max_fetch_characters=args.max_fetch_characters,
            max_concurrent_searches=args.max_concurrent_searches,
            max_concurrent_fetches=args.max_concurrent_fetches,
            include_answerer_system_prompt=not args.omit_answerer_system_prompt,
            planner_prompt_variant=args.planner_prompt_variant,
            answerer_payload_variant=args.answerer_payload_variant,
            answerer_prompt_variant=args.answerer_prompt_variant,
            answer_execution_max_completion_tokens=args.answer_execution_max_completion_tokens or None,
            answer_final_max_completion_tokens=args.answer_final_max_completion_tokens or None,
            append_final_citation_block=not args.disable_citation_appendix,
        )
        for row in rows
    ]
    if args.output_json:
        write_json(ROOT / args.output_json, trajectories[0] if len(trajectories) == 1 else trajectories)
    if args.output_jsonl:
        write_jsonl(ROOT / args.output_jsonl, trajectories)
    if args.output_json:
        print(f"Saved trajectory to {ROOT / args.output_json}")
    if args.output_jsonl:
        print(f"Saved {len(trajectories)} trajectories to {ROOT / args.output_jsonl}")


if __name__ == "__main__":
    main()
