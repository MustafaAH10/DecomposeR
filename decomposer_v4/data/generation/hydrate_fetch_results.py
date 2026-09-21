#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import SerperFetchClient
from decomposer_v4.utils.io import ROOT, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch planner-selected URLs with Serper and append the fetched content to the JSON payload."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-characters", type=int, default=6000)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _graph_ref(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("parsed_graph", "graph", "revised_graph"):
        graph = payload.get(key)
        if isinstance(graph, dict):
            return graph
    return None


def _node_map(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if node_id:
            out[node_id] = node
    return out


def main() -> None:
    args = parse_args()
    input_path = _resolve(args.input_json)
    output_path = _resolve(args.output_json)

    payload = read_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {input_path}")

    fetch_requests = payload.get("fetch_requests")
    if not isinstance(fetch_requests, list):
        raise ValueError(f"Expected fetch_requests list in {input_path}")

    graph = _graph_ref(payload)
    node_index = _node_map(graph)
    client = SerperFetchClient.from_env()

    fetched_results: list[dict[str, Any]] = []
    for request in fetch_requests:
        if not isinstance(request, dict):
            continue
        source_node_id = str(request.get("source_node_id", "")).strip()
        url = str(request.get("url", "")).strip()
        reason = str(request.get("reason", "")).strip()
        if not source_node_id or not url:
            continue
        fetched = client.fetch(url, max_characters=max(int(args.max_characters), 1))
        if fetched is None or not fetched.text.strip():
            continue
        row = {
            "source_node_id": source_node_id,
            "url": fetched.url,
            "reason": reason,
            "title": fetched.title,
            "text": fetched.text,
            "source": fetched.source,
            "published_date": fetched.published_date,
            "author": fetched.author,
            "language": fetched.language,
            "language_score": fetched.language_score,
            "token_count": fetched.token_count,
        }
        fetched_results.append(row)
        node = node_index.get(source_node_id)
        if node is not None:
            node.setdefault("fetched_contents", [])
            if isinstance(node["fetched_contents"], list):
                node["fetched_contents"].append(row)
        print(f"{source_node_id}: fetched {url}")

    payload["fetched_results"] = fetched_results
    write_json(output_path, payload)
    print(f"Wrote {len(fetched_results)} fetched results to {output_path}")


if __name__ == "__main__":
    main()
