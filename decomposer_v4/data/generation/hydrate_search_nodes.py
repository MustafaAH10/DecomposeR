#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.utils.io import ROOT, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Jina searches for each search node and attach results to a contents field."
    )
    parser.add_argument("--input-graph", required=True)
    parser.add_argument("--output-graph", required=True)
    parser.add_argument("--num-results", type=int, default=5)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _load_graph(path: Path) -> tuple[dict[str, Any], dict[str, Any], bool]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    if isinstance(payload.get("graph"), dict):
        graph = payload["graph"]
        wrapped = True
    else:
        graph = payload
        wrapped = False
    if not isinstance(graph, dict):
        raise ValueError(f"Expected graph object at {path}")
    ok, meta = validate_graph(graph)
    if not ok:
        raise ValueError(f"Invalid graph: {meta}")
    return payload, graph, wrapped


def _result_to_content(result: Any) -> dict[str, Any]:
    return {
        "rank": int(result.rank),
        "title": str(result.title),
        "url": str(result.url),
        "text": str(result.text),
        "source": str(result.source),
        "published_date": str(result.published_date),
    }


def main() -> None:
    args = parse_args()
    input_path = _resolve(args.input_graph)
    output_path = _resolve(args.output_graph)

    payload, graph, wrapped = _load_graph(input_path)
    client = JinaSearchClient.from_env()

    search_count = 0
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")) != "search":
            continue
        query = str(node.get("query", "")).strip()
        if not query:
            continue
        results = client.search(query, num_results=max(int(args.num_results), 1))
        node["contents"] = [_result_to_content(item) for item in results]
        search_count += 1
        print(f"{node.get('id', '')}: {len(results)} results")

    if wrapped:
        payload["graph"] = graph
        write_json(output_path, payload)
    else:
        write_json(output_path, graph)

    print(f"Wrote hydrated graph with {search_count} search nodes to {output_path}")


if __name__ == "__main__":
    main()
