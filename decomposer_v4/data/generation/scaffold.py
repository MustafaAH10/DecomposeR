from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from decomposer_v4.retrieval.live_clients import MultiSearchClient, SearchResult
from decomposer_v4.schemas.graph import parse_graph_text, topological_depths, topological_sort, validate_graph
from decomposer_v4.schemas.trajectory import FetchRequest, SearchExecution, SearchPassage
from decomposer_v4.utils.io import extract_json_object


EDGE_INSTRUCTIONS = {
    "enables": "Use the following as direct supporting evidence:",
    "compares": "Compare or contrast against the following dependency:",
    "extends": "Extend the following dependency:",
    "requires": "The following dependency is prerequisite context:",
    "contextualizes": "Use the following dependency as background context:",
}


def graph_for_prompt(graph: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        row: dict[str, Any] = {"id": node.get("id", ""), "type": node.get("type", "")}
        if str(node.get("type", "")) == "search":
            row["need"] = node.get("need", "")
            row["query"] = node.get("query", "")
        else:
            row["need"] = node.get("need", "")
            row["inputs"] = node.get("inputs", [])
            if str(node.get("type", "")) == "aggregate":
                row["key_points"] = node.get("key_points", [])
        nodes.append(row)
    edges: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        edges.append(
            {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "type": edge.get("type", ""),
            }
        )
    return {"nodes": nodes, "edges": edges}


def format_search_results(results: list[SearchResult], excerpt_limit: int = 1400) -> str:
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        line = [f"[R{idx}] {result.title}", f"URL: {result.url}", f"Source: {result.source}"]
        if result.published_date:
            line.append(f"Date: {result.published_date}")
        line.append(f"Excerpt: {result.text[:excerpt_limit]}")
        lines.append("\n".join(line))
    return "\n\n".join(lines).strip()


def _search_execution_from_results(node_id: str, query: str, results: list[SearchResult]) -> SearchExecution:
    return SearchExecution(
        node_id=node_id,
        query=query,
        passages=tuple(
            SearchPassage(
                passage_id=f"{node_id}-R{idx}",
                node_id=node_id,
                title=result.title,
                url=result.url,
                text=result.text,
                score=result.score,
                source=result.source,
                published_date=result.published_date,
                author=result.author,
                language=result.language,
                language_score=result.language_score,
                token_count=result.token_count,
            )
            for idx, result in enumerate(results, start=1)
        ),
    )


def _search_nodes_for_graph(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    search_nodes: list[tuple[str, dict[str, Any]]] = []
    for node_id in topological_sort(graph):
        node = next(
            item for item in graph.get("nodes", [])
            if isinstance(item, dict) and str(item.get("id")) == node_id
        )
        if str(node.get("type", "")) == "search":
            search_nodes.append((node_id, node))
    return search_nodes


def _search_results_from_execution(execution: SearchExecution) -> list[SearchResult]:
    return [
        SearchResult(
            title=passage.title,
            url=passage.url,
            text=passage.text,
            source=passage.source,
            rank=index,
            score=passage.score,
            published_date=passage.published_date,
            author=passage.author,
            language=passage.language,
            language_score=passage.language_score,
            token_count=passage.token_count,
        )
        for index, passage in enumerate(execution.passages, start=1)
    ]


def _format_search_execution(execution: SearchExecution, excerpt_limit: int = 1400) -> str:
    lines: list[str] = []
    for passage in execution.passages:
        line = [
            f"[{passage.passage_id}] {passage.title}",
            f"URL: {passage.url}",
            f"Source: {passage.source}",
        ]
        if passage.published_date:
            line.append(f"Date: {passage.published_date}")
        line.append(f"Excerpt: {passage.text[:excerpt_limit]}")
        lines.append("\n".join(line))
    return "\n\n".join(lines).strip()


def _format_fetch_passage(passage: SearchPassage, reason: str) -> str:
    return (
        f"[{passage.passage_id}] {passage.title}\n"
        f"URL: {passage.url}\n"
        f"Reason: {reason or 'Selected for deeper evidence'}\n"
        f"Content: {passage.text}"
    )


def normalize_search_results(executions: tuple[SearchExecution, ...]) -> dict[str, list[dict[str, Any]]]:
    return {
        item.node_id: [
            {
                "passage_id": passage.passage_id,
                "title": passage.title,
                "url": passage.url,
                "text": passage.text,
                "score": passage.score,
                "source": passage.source,
                "published_date": passage.published_date,
            }
            for passage in item.passages
        ]
        for item in executions
    }


def normalize_fetch_results(
    fetch_requests: list[dict[str, str]],
    fetch_executions: tuple[SearchExecution, ...],
) -> list[dict[str, Any]]:
    passage_lookup: dict[tuple[str, str], SearchPassage] = {}
    for execution in fetch_executions:
        if not execution.passages:
            continue
        passage_lookup[(execution.node_id, execution.query)] = execution.passages[0]

    rows: list[dict[str, Any]] = []
    for request in fetch_requests:
        node_id = str(request.get("source_node_id", "")).strip()
        url = str(request.get("url", "")).strip()
        if not node_id or not url:
            continue
        passage = passage_lookup.get((node_id, url))
        if passage is None:
            continue
        rows.append(
            {
                "passage_id": passage.passage_id,
                "source_node_id": node_id,
                "url": passage.url,
                "reason": str(request.get("reason", "")).strip(),
                "title": passage.title,
                "text": passage.text,
                "source": passage.source,
                "published_date": passage.published_date,
                "author": passage.author,
                "language": passage.language,
                "language_score": passage.language_score,
                "token_count": passage.token_count,
            }
        )
    return rows


def extract_inline_citation_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in re.finditer(r"\[(N[0-9A-Za-z_]+-(?:R|F)\d+)\]", str(text or "")):
        citation_id = str(match.group(1) or "").strip()
        if citation_id and citation_id not in seen:
            seen.append(citation_id)
    return seen


def build_citation_catalog(
    *,
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for node_id, rows in search_results.items():
        for row in rows:
            if not isinstance(row, dict):
                continue
            citation_id = str(row.get("passage_id", "")).strip()
            if not citation_id:
                continue
            catalog[citation_id] = {
                "citation_id": citation_id,
                "source_node_id": str(node_id),
                "title": str(row.get("title", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "published_date": str(row.get("published_date", "")).strip(),
                "kind": "search",
            }
    for row in fetched_results:
        if not isinstance(row, dict):
            continue
        citation_id = str(row.get("passage_id", "")).strip()
        if not citation_id:
            continue
        catalog[citation_id] = {
            "citation_id": citation_id,
            "source_node_id": str(row.get("source_node_id", "")).strip(),
            "title": str(row.get("title", "")).strip(),
            "url": str(row.get("url", "")).strip(),
            "source": str(row.get("source", "")).strip(),
            "published_date": str(row.get("published_date", "")).strip(),
            "kind": "fetch",
        }
    return dict(sorted(catalog.items()))


def build_citation_block(text: str, citation_catalog: Mapping[str, Mapping[str, Any]]) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    citation_ids = extract_inline_citation_ids(body)
    if not citation_ids:
        return ""
    lines = ["[Citations]"]
    for citation_id in citation_ids:
        row = citation_catalog.get(citation_id)
        if row is None:
            continue
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        lines.append(f"- [{citation_id}] {title} | {url}".strip())
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def append_citation_block(text: str, citation_catalog: Mapping[str, Mapping[str, Any]]) -> str:
    body = str(text or "").strip()
    if not body:
        return body
    citation_block = build_citation_block(body, citation_catalog)
    if not citation_block:
        return body
    return body + "\n\n" + citation_block


def apply_citations_to_node_outputs(
    node_outputs: Mapping[str, str],
    citation_catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(node_id): append_citation_block(str(content), citation_catalog)
        for node_id, content in node_outputs.items()
    }


def build_node_output_snippet_for_prompt(text: str, max_chars: int = 200) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def hydrate_graph_with_evidence(
    graph: dict[str, Any],
    *,
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
) -> dict[str, Any]:
    hydrated = json.loads(json.dumps(graph_for_prompt(graph), ensure_ascii=False))
    node_index = {
        str(node.get("id", "")): node
        for node in hydrated.get("nodes", [])
        if isinstance(node, dict)
    }
    for node_id, results in search_results.items():
        node = node_index.get(str(node_id))
        if node is None or str(node.get("type", "")) != "search":
            continue
        node["contents"] = [dict(item) for item in results if isinstance(item, dict)]
    for row in fetched_results:
        if not isinstance(row, dict):
            continue
        node = node_index.get(str(row.get("source_node_id", "")))
        if node is None or str(node.get("type", "")) != "search":
            continue
        node.setdefault("fetched_contents", [])
        if isinstance(node["fetched_contents"], list):
            node["fetched_contents"].append(dict(row))
    return hydrated


def strip_think_block(text: str) -> str:
    return re.sub(r"^\s*<think>.*?</think>\s*", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE).strip()


def _coerce_node_outputs(payload: Any) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if isinstance(payload, dict):
        rows = payload.get("outputs") or payload.get("nodes")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                node_id = str(row.get("node_id") or row.get("id") or "").strip()
                content = str(row.get("content") or row.get("text") or "").strip()
                if node_id and content:
                    outputs[node_id] = content
        else:
            node_id = str(payload.get("node_id") or payload.get("id") or "").strip()
            content = str(payload.get("content") or payload.get("text") or "").strip()
            if node_id and content:
                outputs[node_id] = content
    return outputs


def parse_answerer_node_outputs(
    text: str,
    *,
    expected_node_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    cleaned = strip_think_block(text)
    outputs: dict[str, str] = {}

    for match in re.finditer(
        r'<node\s+id="([^"]+)">\s*(.*?)\s*</node>',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        node_id = str(match.group(1) or "").strip()
        content = str(match.group(2) or "").strip()
        if node_id and content:
            outputs[node_id] = content

    if not outputs:
        outputs = _coerce_node_outputs(extract_json_object(cleaned))

    if not outputs:
        raise RuntimeError(f"Could not parse answerer node outputs:\n{text}")

    if expected_node_ids:
        expected = [str(node_id).strip() for node_id in expected_node_ids if str(node_id).strip()]
        missing = [node_id for node_id in expected if node_id not in outputs or not outputs[node_id].strip()]
        extras = [node_id for node_id in outputs if node_id not in expected]
        if missing or extras:
            raise RuntimeError(
                f"Unexpected answerer nodes. missing={missing} extras={extras}\n{text}"
            )
        return {node_id: outputs[node_id] for node_id in expected}

    return outputs


def evidence_pool_from_executions(executions: tuple[SearchExecution, ...]) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    for execution in executions:
        pool.setdefault(execution.node_id, [])
        for passage in execution.passages:
            if passage.text.strip():
                pool[execution.node_id].append(passage.text.strip())
    return pool


def build_after_search_payload(
    query: str,
    graph: dict[str, Any],
    search_results: Mapping[str, list[dict[str, Any]]],
    max_fetches: int,
) -> str:
    payload = {
        "phase": "planner_after_search",
        "query": query,
        "current_graph": graph_for_prompt(graph),
        "max_fetches": max_fetches,
        "search_results": search_results,
        "instruction": (
            "Review the evidence. Keep or revise the graph, then choose the few URLs most worth fetching. "
            "Choose only from the provided search results. Fetch sparingly. "
            "Revise the graph if the current branches under-separate distinct analytical tasks, if any aggregate "
            "node is overloaded, or if the evidence suggests a missing branch. If you add new search nodes, they "
            "will be executed after this turn. Every aggregate node must keep specific `key_points` with 2-6 items."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").lower()


def parse_planner_initial_output(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    graph_payload = payload.get("graph") if isinstance(payload, dict) else payload
    if not isinstance(graph_payload, dict):
        graph_payload = parse_graph_text(text)
        if graph_payload is None:
            raise RuntimeError(f"Could not parse planner graph:\n{text}")
        return graph_payload
    parsed = parse_graph_text(json.dumps(graph_payload, ensure_ascii=False))
    if parsed is None:
        raise RuntimeError(f"Could not parse planner graph:\n{text}")
    ok, meta = validate_graph(parsed)
    if not ok:
        raise RuntimeError(f"Planner graph invalid: {meta}\n{text}")
    return parsed


def parse_planner_revision_output(
    text: str,
    current_graph: dict[str, Any],
    available_search_results: Mapping[str, list[dict[str, Any]]],
    max_fetches: int,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    payload = extract_json_object(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Could not parse planner revision payload:\n{text}")
    revised_graph = payload.get("revised_graph") or payload.get("graph") or current_graph
    if not isinstance(revised_graph, dict):
        revised_graph = current_graph
    parsed_graph = parse_graph_text(json.dumps(revised_graph, ensure_ascii=False))
    if parsed_graph is None:
        parsed_graph = current_graph
    ok, _ = validate_graph(parsed_graph)
    if not ok:
        parsed_graph = current_graph

    allowed_by_node: dict[str, set[str]] = {}
    for node_id, rows in available_search_results.items():
        allowed_by_node[str(node_id)] = {
            _normalize_url(str(row.get("url", "")))
            for row in rows
            if isinstance(row, dict) and str(row.get("url", "")).strip()
        }

    fetch_requests: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in payload.get("fetch_requests", []):
        if not isinstance(row, dict):
            continue
        if max_fetches > 0 and len(fetch_requests) >= max_fetches:
            break
        node_id = str(row.get("source_node_id", "")).strip()
        url = str(row.get("url", "")).strip()
        norm = _normalize_url(url)
        if not node_id or not url or not norm or norm in seen_urls:
            continue
        if norm not in allowed_by_node.get(node_id, set()):
            continue
        fetch_requests.append(
            {
                "source_node_id": node_id,
                "url": url,
                "reason": str(row.get("reason", "")).strip(),
            }
        )
        seen_urls.add(norm)

    action = str(payload.get("action", "")).strip().lower() or "keep"
    if action not in {"keep", "revise"}:
        action = "keep"
    return action, parsed_graph, fetch_requests


def execute_search_nodes(
    *,
    graph: dict[str, Any],
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    max_concurrent_searches: int = 1,
    existing_executions: tuple[SearchExecution, ...] | None = None,
) -> tuple[dict[str, str], tuple[SearchExecution, ...]]:
    outputs, executions, _ = execute_search_nodes_timed(
        graph=graph,
        search_backend=search_backend,
        num_results_per_search=num_results_per_search,
        max_concurrent_searches=max_concurrent_searches,
        existing_executions=existing_executions,
    )
    return outputs, executions


def execute_search_nodes_timed(
    *,
    graph: dict[str, Any],
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    max_concurrent_searches: int = 1,
    existing_executions: tuple[SearchExecution, ...] | None = None,
) -> tuple[dict[str, str], tuple[SearchExecution, ...], dict[str, dict[str, Any]]]:
    search_nodes = _search_nodes_for_graph(graph)
    existing_map = {
        execution.node_id: execution
        for execution in (existing_executions or ())
    }
    search_results_by_node: dict[str, list[SearchResult]] = {}
    search_timings: dict[str, dict[str, Any]] = {}
    pending_searches: list[tuple[str, dict[str, Any]]] = []

    for node_id, node in search_nodes:
        query = str(node.get("query", "")).strip()
        existing = existing_map.get(node_id)
        if existing is not None and existing.query == query and existing.passages:
            search_results_by_node[node_id] = _search_results_from_execution(existing)
            search_timings[node_id] = {
                "query": query,
                "seconds": 0.0,
                "reused": True,
                "result_count": len(existing.passages),
            }
            continue
        pending_searches.append((node_id, node))

    if pending_searches:
        if max_concurrent_searches <= 1 or len(pending_searches) <= 1:
            for node_id, node in pending_searches:
                query = str(node.get("query", "")).strip()
                started = time.perf_counter()
                search_results_by_node[node_id] = search_backend.search(
                    query,
                    num_results=num_results_per_search,
                )
                search_timings[node_id] = {
                    "query": query,
                    "seconds": time.perf_counter() - started,
                    "reused": False,
                    "result_count": len(search_results_by_node[node_id]),
                }
        else:
            worker_count = min(max(int(max_concurrent_searches), 1), len(pending_searches))

            def _run_search(query: str, num_results: int) -> tuple[list[SearchResult], float]:
                started = time.perf_counter()
                result = search_backend.search(query, num_results=num_results)
                return result, time.perf_counter() - started

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(_run_search, str(node.get("query", "")).strip(), num_results_per_search): (
                        node_id,
                        str(node.get("query", "")).strip(),
                    )
                    for node_id, node in pending_searches
                }
                for future, (node_id, query) in future_map.items():
                    result, seconds = future.result()
                    search_results_by_node[node_id] = result
                    search_timings[node_id] = {
                        "query": query,
                        "seconds": seconds,
                        "reused": False,
                        "result_count": len(result),
                    }

    outputs: dict[str, str] = {}
    executions: list[SearchExecution] = []
    for node_id, node in search_nodes:
        query = str(node.get("query", "")).strip()
        results = search_results_by_node.get(node_id, [])
        execution = _search_execution_from_results(node_id, query, results)
        outputs[node_id] = _format_search_execution(execution)
        executions.append(_search_execution_from_results(node_id, query, results))
        search_timings.setdefault(
            node_id,
            {"query": query, "seconds": 0.0, "reused": False, "result_count": len(results)},
        )
    return outputs, tuple(executions), search_timings


def execute_selected_fetches(
    *,
    search_backend: MultiSearchClient,
    fetch_requests: list[dict[str, str]],
    max_fetch_characters: int,
    max_concurrent_fetches: int = 1,
) -> tuple[dict[str, list[str]], tuple[SearchExecution, ...]]:
    output_blocks, executions, _ = execute_selected_fetches_timed(
        search_backend=search_backend,
        fetch_requests=fetch_requests,
        max_fetch_characters=max_fetch_characters,
        max_concurrent_fetches=max_concurrent_fetches,
    )
    return output_blocks, executions


def execute_selected_fetches_timed(
    *,
    search_backend: MultiSearchClient,
    fetch_requests: list[dict[str, str]],
    max_fetch_characters: int,
    max_concurrent_fetches: int = 1,
) -> tuple[dict[str, list[str]], tuple[SearchExecution, ...], dict[str, dict[str, Any]]]:
    output_blocks: dict[str, list[str]] = {}
    executions: list[SearchExecution] = []
    fetch_timings: dict[str, dict[str, Any]] = {}
    valid_requests: list[tuple[int, dict[str, str]]] = []
    for index, request in enumerate(fetch_requests, start=1):
        source_node_id = str(request.get("source_node_id", "")).strip()
        url = str(request.get("url", "")).strip()
        if source_node_id and url:
            valid_requests.append((index, request))

    if max_concurrent_fetches <= 1 or len(valid_requests) <= 1:
        fetched_by_index = {}
        for index, request in valid_requests:
            url = str(request.get("url", "")).strip()
            started = time.perf_counter()
            fetched_by_index[index] = search_backend.fetch(
                url,
                max_characters=max_fetch_characters,
            )
            fetch_timings[f"F{index}"] = {
                "source_node_id": str(request.get("source_node_id", "")).strip(),
                "url": url,
                "seconds": time.perf_counter() - started,
                "succeeded": bool(fetched_by_index[index] is not None and fetched_by_index[index].text.strip()),
            }
    else:
        worker_count = min(max(int(max_concurrent_fetches), 1), len(valid_requests))

        def _run_fetch(url: str, max_characters: int) -> tuple[SearchResult | None, float]:
            started = time.perf_counter()
            result = search_backend.fetch(url, max_characters=max_characters)
            return result, time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_run_fetch, str(request.get("url", "")).strip(), max_fetch_characters): (
                    index,
                    request,
                )
                for index, request in valid_requests
            }
            fetched_by_index = {}
            for future, (index, request) in future_map.items():
                result, seconds = future.result()
                fetched_by_index[index] = result
                fetch_timings[f"F{index}"] = {
                    "source_node_id": str(request.get("source_node_id", "")).strip(),
                    "url": str(request.get("url", "")).strip(),
                    "seconds": seconds,
                    "succeeded": bool(result is not None and result.text.strip()),
                }

    for index, request in valid_requests:
        source_node_id = str(request.get("source_node_id", "")).strip()
        url = str(request.get("url", "")).strip()
        fetched = fetched_by_index.get(index)
        if fetched is None or not fetched.text.strip():
            continue
        output_blocks.setdefault(source_node_id, [])
        passage = SearchPassage(
            passage_id=f"{source_node_id}-F{index}",
            node_id=source_node_id,
            title=fetched.title,
            url=fetched.url,
            text=fetched.text,
            score=fetched.score,
            source=fetched.source,
            published_date=fetched.published_date,
            author=fetched.author,
            language=fetched.language,
            language_score=fetched.language_score,
            token_count=fetched.token_count,
        )
        block = _format_fetch_passage(passage, str(request.get("reason", "")).strip())
        output_blocks[source_node_id].append(block)
        executions.append(
            SearchExecution(
                node_id=source_node_id,
                query=url,
                passages=(passage,),
            )
        )
    return output_blocks, tuple(executions), fetch_timings


def execute_revision_retrievals(
    *,
    graph: dict[str, Any],
    search_backend: MultiSearchClient,
    num_results_per_search: int,
    existing_search_executions: tuple[SearchExecution, ...],
    fetch_requests: list[dict[str, str]],
    max_fetch_characters: int,
    max_concurrent_searches: int = 4,
    max_concurrent_fetches: int = 4,
) -> tuple[
    dict[str, str],
    tuple[SearchExecution, ...],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    tuple[SearchExecution, ...],
    dict[str, dict[str, Any]],
]:
    search_nodes = _search_nodes_for_graph(graph)
    existing_map = {execution.node_id: execution for execution in existing_search_executions}
    search_results_by_node: dict[str, list[SearchResult]] = {}
    search_timings: dict[str, dict[str, Any]] = {}
    pending_searches: list[tuple[str, str]] = []

    for node_id, node in search_nodes:
        query = str(node.get("query", "")).strip()
        existing = existing_map.get(node_id)
        if existing is not None and existing.query == query and existing.passages:
            search_results_by_node[node_id] = _search_results_from_execution(existing)
            search_timings[node_id] = {
                "query": query,
                "seconds": 0.0,
                "reused": True,
                "result_count": len(existing.passages),
            }
        else:
            pending_searches.append((node_id, query))

    output_blocks: dict[str, list[str]] = {}
    fetch_executions: list[SearchExecution] = []
    fetch_timings: dict[str, dict[str, Any]] = {}
    valid_requests = [
        (index, request)
        for index, request in enumerate(fetch_requests, start=1)
        if str(request.get("source_node_id", "")).strip() and str(request.get("url", "")).strip()
    ]
    fetched_by_index: dict[int, SearchResult | None] = {}

    total_workers = min(
        max(int(max_concurrent_searches), 1) + max(int(max_concurrent_fetches), 1),
        max(len(pending_searches) + len(valid_requests), 1),
    )

    def _run_search(query: str) -> tuple[list[SearchResult], float]:
        started = time.perf_counter()
        result = search_backend.search(query, num_results=num_results_per_search)
        return result, time.perf_counter() - started

    def _run_fetch(url: str) -> tuple[SearchResult | None, float]:
        started = time.perf_counter()
        result = search_backend.fetch(url, max_characters=max_fetch_characters)
        return result, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_map: dict[Any, tuple[str, Any]] = {}
        for node_id, query in pending_searches:
            future_map[executor.submit(_run_search, query)] = ("search", (node_id, query))
        for index, request in valid_requests:
            future_map[executor.submit(_run_fetch, str(request.get("url", "")).strip())] = ("fetch", (index, request))

        for future, (kind, payload) in future_map.items():
            if kind == "search":
                node_id, query = payload
                result, seconds = future.result()
                search_results_by_node[node_id] = result
                search_timings[node_id] = {
                    "query": query,
                    "seconds": seconds,
                    "reused": False,
                    "result_count": len(result),
                }
            else:
                index, request = payload
                result, seconds = future.result()
                fetched_by_index[index] = result
                fetch_timings[f"F{index}"] = {
                    "source_node_id": str(request.get("source_node_id", "")).strip(),
                    "url": str(request.get("url", "")).strip(),
                    "seconds": seconds,
                    "succeeded": bool(result is not None and result.text.strip()),
                }

    search_outputs: dict[str, str] = {}
    search_executions: list[SearchExecution] = []
    for node_id, node in search_nodes:
        query = str(node.get("query", "")).strip()
        results = search_results_by_node.get(node_id, [])
        execution = _search_execution_from_results(node_id, query, results)
        search_outputs[node_id] = _format_search_execution(execution)
        search_executions.append(execution)
        search_timings.setdefault(
            node_id,
            {"query": query, "seconds": 0.0, "reused": False, "result_count": len(results)},
        )

    for index, request in valid_requests:
        fetched = fetched_by_index.get(index)
        if fetched is None or not fetched.text.strip():
            continue
        source_node_id = str(request.get("source_node_id", "")).strip()
        passage = SearchPassage(
            passage_id=f"{source_node_id}-F{index}",
            node_id=source_node_id,
            title=fetched.title,
            url=fetched.url,
            text=fetched.text,
            score=fetched.score,
            source=fetched.source,
            published_date=fetched.published_date,
            author=fetched.author,
            language=fetched.language,
            language_score=fetched.language_score,
            token_count=fetched.token_count,
        )
        output_blocks.setdefault(source_node_id, [])
        output_blocks[source_node_id].append(
            _format_fetch_passage(passage, str(request.get("reason", "")).strip())
        )
        fetch_executions.append(
            SearchExecution(
                node_id=source_node_id,
                query=str(request.get("url", "")).strip(),
                passages=(passage,),
            )
        )

    return (
        search_outputs,
        tuple(search_executions),
        search_timings,
        output_blocks,
        tuple(fetch_executions),
        fetch_timings,
    )


def answer_node_id(graph: dict[str, Any]) -> str:
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and str(node.get("type", "")) == "answer":
            return str(node.get("id", ""))
    raise ValueError("Graph has no answer node.")


def aggregate_execution_waves(graph: dict[str, Any]) -> list[list[str]]:
    depths = topological_depths(graph)
    node_index = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }
    answer_id = answer_node_id(graph)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node_id in topological_sort(graph):
        node = node_index[node_id]
        if str(node.get("type", "")) == "aggregate" and node_id != answer_id:
            grouped[depths.get(node_id, 0)].append(node_id)
    return [grouped[depth] for depth in sorted(grouped)]


def aggregate_execution_order(graph: dict[str, Any]) -> list[str]:
    return [node_id for wave in aggregate_execution_waves(graph) for node_id in wave]


def _edge_index(graph: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        edge_type = str(edge.get("type", ""))
        if source and target and edge_type:
            out[(source, target)] = edge_type
    return out


def build_execution_turn_payload(
    *,
    query: str,
    graph: dict[str, Any],
    node_ids: list[str],
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
    produced_outputs: Mapping[str, str],
    wave_index: int,
) -> str:
    node_map = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }
    edge_index = _edge_index(graph)
    target_nodes: list[dict[str, Any]] = []
    for node_id in node_ids:
        node = node_map[node_id]
        if str(node.get("type", "")) != "aggregate":
            raise ValueError(f"Execution payload requested for non-aggregate node: {node_id}")
        dependencies: list[dict[str, Any]] = []
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            parent = str(parent_id)
            parent_node = node_map[parent]
            edge_type = edge_index.get((parent, node_id), "enables")
            edge_instruction = EDGE_INSTRUCTIONS.get(edge_type, "Use the following dependency:")
            parent_type = str(parent_node.get("type", ""))
            if parent_type == "search":
                evidence_items: list[dict[str, str]] = []
                for row in search_results.get(parent, []):
                    if not isinstance(row, dict):
                        continue
                    evidence_items.append(
                        {
                            "citation_id": str(row.get("passage_id", "")).strip(),
                            "title": str(row.get("title", "")).strip(),
                            "url": str(row.get("url", "")).strip(),
                            "published_date": str(row.get("published_date", "")).strip(),
                            "source": str(row.get("source", "")).strip(),
                            "excerpt": str(row.get("text", "")).strip(),
                            "kind": "search",
                        }
                    )
                for row in fetched_results:
                    if not isinstance(row, dict) or str(row.get("source_node_id", "")).strip() != parent:
                        continue
                    evidence_items.append(
                        {
                            "citation_id": str(row.get("passage_id", "")).strip(),
                            "title": str(row.get("title", "")).strip(),
                            "url": str(row.get("url", "")).strip(),
                            "published_date": str(row.get("published_date", "")).strip(),
                            "source": str(row.get("source", "")).strip(),
                            "excerpt": str(row.get("text", "")).strip(),
                            "reason": str(row.get("reason", "")).strip(),
                            "kind": "fetch",
                        }
                    )
                dependencies.append(
                    {
                        "parent_id": parent,
                        "parent_type": parent_type,
                        "edge_type": edge_type,
                        "instruction": edge_instruction,
                        "evidence_items": evidence_items,
                    }
                )
            else:
                dependency_text = str(produced_outputs.get(parent, "")).strip()
                dependencies.append(
                    {
                        "parent_id": parent,
                        "parent_type": parent_type,
                        "edge_type": edge_type,
                        "instruction": edge_instruction,
                        "reference_node": {
                            "node_id": parent,
                            "need": str(parent_node.get("need", "")).strip(),
                            "report_snippet": build_node_output_snippet_for_prompt(dependency_text),
                            "citation_ids_present": extract_inline_citation_ids(dependency_text),
                            "full_report_available_in_conversation": True,
                        },
                    }
                )
        if not dependencies:
            raise ValueError(f"Aggregate node has no usable dependencies: {node_id}")
        target_nodes.append(
            {
                "id": node_id,
                "type": str(node.get("type", "")),
                "need": str(node.get("need", "")),
                "key_points": list(node.get("key_points", [])) if isinstance(node.get("key_points"), list) else [],
                "inputs": list(node.get("inputs", [])) if isinstance(node.get("inputs"), list) else [],
                "dependencies": dependencies,
            }
        )

    payload = {
        "phase": "answer_execution",
        "query": query,
        "graph": graph_for_prompt(graph),
        "execution_wave": {
            "index": wave_index,
            "target_node_ids": node_ids,
        },
        "target_nodes": target_nodes,
        "instruction": (
            "Complete every target aggregate node in this wave. "
            "Start with a very brief `<think>` block, then emit exactly one `<node id=\"...\">...</node>` block "
            "for each target node and nothing else. Each node report must directly address that node's `need` and "
            "`key_points`, use only that node's provided dependencies, and avoid borrowing from sibling targets "
            "unless the content is explicitly provided in that node's dependencies. Use inline citations with the "
            "exact evidence ids, such as `[N2-R1]` or `[N4-F2]`. If you rely on an upstream aggregate dependency, "
            "reuse the citations already present in that earlier report."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_final_answer_payload(
    *,
    query: str,
    graph: dict[str, Any],
    produced_outputs: Mapping[str, str],
) -> str:
    node_map = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }
    final_node_id = answer_node_id(graph)
    final_node = node_map[final_node_id]
    inputs = final_node.get("inputs", []) if isinstance(final_node.get("inputs"), list) else []
    if not inputs:
        raise ValueError(f"Answer node has no inputs: {final_node_id}")
    reports = [
        {
            "node_id": str(parent_id),
            "need": str(node_map[str(parent_id)].get("need", "")),
            "report_snippet": build_node_output_snippet_for_prompt(
                str(produced_outputs.get(str(parent_id), "")).strip(),
                max_chars=200,
            ),
            "citation_ids_present": extract_inline_citation_ids(str(produced_outputs.get(str(parent_id), "")).strip()),
            "full_report_available_in_conversation": True,
        }
        for parent_id in inputs
        if str(produced_outputs.get(str(parent_id), "")).strip()
    ]
    if not reports:
        raise ValueError(f"Answer node has no branch reports: {final_node_id}")
    payload = {
        "phase": "answer_final",
        "query": query,
        "graph": graph_for_prompt(graph),
        "target_node": {
            "id": final_node_id,
            "type": str(final_node.get("type", "")),
            "need": str(final_node.get("need", "")),
        },
        "branch_reports": reports,
        "instruction": (
            "Write the terminal answer node using only the provided branch reports. "
            "Start with a very brief `<think>` block, then emit exactly one `<node id=\"...\">...</node>` block for "
            "the final answer node and nothing else. Integrate the branches into a coherent, polished answer without "
            "re-opening retrieval or adding unsupported facts. For deep-research queries, default to a long, detailed "
            "answer with clear sections, explicit synthesis, and sustained reasoning rather than a short summary. "
            "Preserve and reuse the inline evidence citation ids already established in the upstream branch reports."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_synthesis_payload(
    *,
    query: str,
    graph: dict[str, Any],
    produced_outputs: Mapping[str, str],
) -> str:
    return build_final_answer_payload(
        query=query,
        graph=graph,
        produced_outputs=produced_outputs,
    )
