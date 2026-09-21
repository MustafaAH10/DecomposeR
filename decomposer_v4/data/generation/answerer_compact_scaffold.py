from __future__ import annotations

import json
from typing import Any, Collection, Mapping

from decomposer_v4.data.generation.scaffold import (
    EDGE_INSTRUCTIONS,
    _edge_index,
    answer_node_id,
    build_final_answer_payload,
    extract_inline_citation_ids,
    graph_for_prompt,
)


def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }


def search_dependency_ids_for_wave(graph: dict[str, Any], node_ids: Collection[str]) -> set[str]:
    node_map = _node_index(graph)
    search_parent_ids: set[str] = set()
    for node_id in node_ids:
        node = node_map.get(str(node_id))
        if node is None:
            continue
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            parent = str(parent_id)
            parent_node = node_map.get(parent)
            if parent_node is None:
                continue
            if str(parent_node.get("type", "")) == "search":
                search_parent_ids.add(parent)
    return search_parent_ids


def _search_evidence_items(
    *,
    parent_id: str,
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    evidence_items: list[dict[str, str]] = []
    for row in search_results.get(parent_id, []):
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
        if not isinstance(row, dict) or str(row.get("source_node_id", "")).strip() != parent_id:
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
    return evidence_items


def _search_citation_ids(
    *,
    parent_id: str,
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
) -> list[str]:
    citation_ids: list[str] = []
    for row in search_results.get(parent_id, []):
        if not isinstance(row, dict):
            continue
        citation_id = str(row.get("passage_id", "")).strip()
        if citation_id and citation_id not in citation_ids:
            citation_ids.append(citation_id)
    for row in fetched_results:
        if not isinstance(row, dict) or str(row.get("source_node_id", "")).strip() != parent_id:
            continue
        citation_id = str(row.get("passage_id", "")).strip()
        if citation_id and citation_id not in citation_ids:
            citation_ids.append(citation_id)
    return citation_ids


def build_execution_turn_payload_compact(
    *,
    query: str,
    graph: dict[str, Any],
    node_ids: list[str],
    search_results: Mapping[str, list[dict[str, Any]]],
    fetched_results: list[dict[str, Any]],
    produced_outputs: Mapping[str, str],
    wave_index: int,
    introduced_search_parent_ids: Collection[str] = (),
) -> str:
    node_map = _node_index(graph)
    edge_index = _edge_index(graph)
    introduced = {str(node_id).strip() for node_id in introduced_search_parent_ids if str(node_id).strip()}
    shared_dependencies: list[dict[str, Any]] = []
    seen_shared: set[str] = set()
    target_nodes: list[dict[str, Any]] = []

    for node_id in node_ids:
        node = node_map[str(node_id)]
        if str(node.get("type", "")) != "aggregate":
            raise ValueError(f"Compact execution payload requested for non-aggregate node: {node_id}")

        dependency_refs: list[dict[str, str]] = []
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            parent = str(parent_id)
            parent_node = node_map[parent]
            edge_type = edge_index.get((parent, str(node_id)), "enables")
            dependency_refs.append(
                {
                    "parent_id": parent,
                    "parent_type": str(parent_node.get("type", "")).strip(),
                    "edge_type": edge_type,
                    "instruction": EDGE_INSTRUCTIONS.get(edge_type, "Use the following dependency:"),
                }
            )

            if parent in seen_shared:
                continue
            seen_shared.add(parent)

            parent_type = str(parent_node.get("type", "")).strip()
            if parent_type == "search":
                available_citation_ids = _search_citation_ids(
                    parent_id=parent,
                    search_results=search_results,
                    fetched_results=fetched_results,
                )
                if parent in introduced:
                    shared_dependencies.append(
                        {
                            "parent_id": parent,
                            "parent_type": parent_type,
                            "provision": "history_reference",
                            "full_evidence_available_in_conversation": True,
                            "available_citation_ids": available_citation_ids,
                        }
                    )
                else:
                    shared_dependencies.append(
                        {
                            "parent_id": parent,
                            "parent_type": parent_type,
                            "provision": "inline",
                            "available_citation_ids": available_citation_ids,
                            "evidence_items": _search_evidence_items(
                                parent_id=parent,
                                search_results=search_results,
                                fetched_results=fetched_results,
                            ),
                        }
                    )
            else:
                dependency_text = str(produced_outputs.get(parent, "")).strip()
                shared_dependencies.append(
                    {
                        "parent_id": parent,
                        "parent_type": parent_type,
                        "provision": "history_reference",
                        "reference_node": {
                            "node_id": parent,
                            "need": str(parent_node.get("need", "")).strip(),
                            "citation_ids_present": extract_inline_citation_ids(dependency_text),
                            "full_report_available_in_conversation": True,
                        },
                    }
                )

        if not dependency_refs:
            raise ValueError(f"Aggregate node has no usable dependencies: {node_id}")

        target_nodes.append(
            {
                "id": str(node_id),
                "type": str(node.get("type", "")),
                "need": str(node.get("need", "")),
                "key_points": list(node.get("key_points", [])) if isinstance(node.get("key_points"), list) else [],
                "inputs": list(node.get("inputs", [])) if isinstance(node.get("inputs"), list) else [],
                "dependency_refs": dependency_refs,
            }
        )

    payload = {
        "phase": "answer_execution",
        "payload_variant": "compact",
        "query": query,
        "graph": graph_for_prompt(graph),
        "execution_wave": {
            "index": wave_index,
            "target_node_ids": node_ids,
        },
        "shared_dependencies": shared_dependencies,
        "target_nodes": target_nodes,
        "instruction": (
            "Complete every target aggregate node in this wave. Shared dependency material is listed once in "
            "`shared_dependencies`; each target node tells you which dependency ids it may use through "
            "`dependency_refs`. If a shared dependency has `provision: history_reference`, its full content was "
            "already provided earlier in the conversation and is not repeated here. Use only the dependencies "
            "referenced for each target node, and do not borrow from sibling targets. Start with a very brief "
            "`<think>` block, then emit exactly one `<node id=\"...\">...</node>` block for each target node and "
            "nothing else. Use inline citations with the exact evidence ids, such as `[N2-R1]` or `[N4-F2]`. "
            "If you rely on an upstream aggregate dependency, reuse the citations already present in that earlier report."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_final_answer_payload_compact(
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


def compact_trial_artifact_default() -> str:
    return "0083ba59ebe941e698f4cdbc1fe75c5220affd43e1dfcf9bcb80c80aa2bc9869"
