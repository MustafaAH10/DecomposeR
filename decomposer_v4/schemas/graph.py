from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any


ALLOWED_NODE_TYPES = {"search", "aggregate", "answer"}
ALLOWED_EDGE_TYPES = {"enables", "compares", "extends", "requires", "contextualizes"}
NODE_ID_PATTERN = re.compile(r"^N[0-9A-Za-z_]+$")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _extract_candidate_json(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    tagged = re.search(r"<graph>\s*(\{.*\})\s*</graph>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1].strip()
    return None


def _canonicalize_node_ids(graph: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    raw_edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    if not raw_nodes:
        return graph

    ids = [str(node.get("id", "")).strip() for node in raw_nodes if isinstance(node, dict)]
    if ids and len(ids) == len(set(ids)) and all(NODE_ID_PATTERN.match(node_id) for node_id in ids):
        return graph

    id_map: dict[str, str] = {}
    next_index = 1
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        old_id = str(node.get("id", "")).strip()
        if old_id not in id_map:
            id_map[old_id] = f"N{next_index}"
            next_index += 1

    nodes: list[dict[str, Any]] = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        rewritten = dict(node)
        rewritten["id"] = id_map.get(str(node.get("id", "")).strip(), str(node.get("id", "")).strip())
        if isinstance(node.get("inputs"), list):
            rewritten["inputs"] = [
                id_map.get(str(parent_id).strip(), str(parent_id).strip())
                for parent_id in node.get("inputs", [])
            ]
        nodes.append(rewritten)

    edges: list[dict[str, Any]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        rewritten = dict(edge)
        rewritten["source"] = id_map.get(str(edge.get("source", "")).strip(), str(edge.get("source", "")).strip())
        rewritten["target"] = id_map.get(str(edge.get("target", "")).strip(), str(edge.get("target", "")).strip())
        edges.append(rewritten)

    return {"nodes": nodes, "edges": edges}


def normalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(graph, dict):
        return {"nodes": [], "edges": []}

    raw_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    raw_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_type = _clean_text(raw.get("type")).lower()
        raw_query = _clean_text(raw.get("query"))
        raw_need = _clean_text(raw.get("need"))
        node: dict[str, Any] = {
            "id": _clean_text(raw.get("id")),
            "type": node_type,
            "need": raw_need if node_type != "search" else (raw_need or raw_query),
        }
        if node_type == "search":
            node["query"] = raw_query
            node["inputs"] = []
        else:
            node["query"] = ""
            node["inputs"] = [_clean_text(x) for x in raw.get("inputs", []) if _clean_text(x)]
            if node_type == "aggregate":
                node["key_points"] = [
                    _clean_text(item)
                    for item in raw.get("key_points", [])
                    if _clean_text(item)
                ]
        nodes.append(node)

    edges: list[dict[str, Any]] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        source = _clean_text(raw.get("source"))
        target = _clean_text(raw.get("target"))
        edge_type = _clean_text(raw.get("type")).lower()
        if source and target:
            edges.append({"source": source, "target": target, "type": edge_type})
    return _canonicalize_node_ids({"nodes": nodes, "edges": edges})


def parse_graph_text(text: str) -> dict[str, Any] | None:
    payload = _extract_candidate_json(text)
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    return normalize_graph(parsed)


def validate_graph(graph: dict[str, Any] | None, *, require_search: bool = True) -> tuple[bool, dict[str, Any]]:
    if not isinstance(graph, dict):
        return False, {"reason": "graph_not_dict"}

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return False, {"reason": "missing_nodes"}
    if not isinstance(edges, list):
        return False, {"reason": "edges_not_list"}

    node_map: dict[str, dict[str, Any]] = {}
    answer_ids: list[str] = []
    search_count = 0
    aggregate_count = 0

    for node in nodes:
        if not isinstance(node, dict):
            return False, {"reason": "bad_node"}
        node_id = _clean_text(node.get("id"))
        node_type = _clean_text(node.get("type")).lower()
        need = _clean_text(node.get("need"))
        if not NODE_ID_PATTERN.match(node_id):
            return False, {"reason": "bad_node_id", "node_id": node_id}
        if node_id in node_map:
            return False, {"reason": "duplicate_node_id", "node_id": node_id}
        if node_type not in ALLOWED_NODE_TYPES:
            return False, {"reason": "bad_node_type", "node_id": node_id}
        if node_type == "search":
            search_count += 1
            if not _clean_text(node.get("query")):
                return False, {"reason": "missing_query", "node_id": node_id}
        else:
            if not need:
                return False, {"reason": "missing_need", "node_id": node_id}
            inputs = node.get("inputs", [])
            if not isinstance(inputs, list) or not inputs:
                return False, {"reason": "missing_inputs", "node_id": node_id}
        if node_type == "aggregate":
            aggregate_count += 1
            key_points = node.get("key_points", [])
            if not isinstance(key_points, list) or not key_points:
                return False, {"reason": "missing_key_points", "node_id": node_id}
            cleaned_key_points = [_clean_text(item) for item in key_points if _clean_text(item)]
            if len(cleaned_key_points) != len(key_points):
                return False, {"reason": "empty_key_point", "node_id": node_id}
            if len(cleaned_key_points) < 2 or len(cleaned_key_points) > 6:
                return False, {"reason": "bad_key_point_count", "node_id": node_id, "count": len(cleaned_key_points)}
        if node_type == "answer":
            answer_ids.append(node_id)
        node_map[node_id] = node

    if require_search and search_count == 0:
        return False, {"reason": "missing_search_nodes"}
    if len(answer_ids) != 1:
        return False, {"reason": "bad_answer_count", "count": len(answer_ids)}

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    reverse_graph: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if not isinstance(edge, dict):
            return False, {"reason": "bad_edge"}
        source = _clean_text(edge.get("source"))
        target = _clean_text(edge.get("target"))
        edge_type = _clean_text(edge.get("type")).lower()
        if source not in node_map or target not in node_map:
            return False, {"reason": "edge_unknown_node", "source": source, "target": target}
        if source == target:
            return False, {"reason": "self_loop", "node_id": source}
        if edge_type not in ALLOWED_EDGE_TYPES:
            return False, {"reason": "bad_edge_type", "edge_type": edge_type}
        adjacency[source].append(target)
        reverse_graph[target].append(source)
        indegree[target] += 1

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id"))
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            parent = str(parent_id)
            if parent not in node_map:
                return False, {"reason": "input_unknown_node", "node_id": node_id, "input": parent}
            if node_id not in adjacency[parent]:
                adjacency[parent].append(node_id)
                reverse_graph[node_id].append(parent)
                indegree[node_id] += 1

    queue = deque([node_id for node_id in node_map if indegree[node_id] == 0])
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        for nxt in adjacency.get(cur, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited != len(node_map):
        return False, {"reason": "cycle"}

    answer_id = answer_ids[0]
    if adjacency.get(answer_id):
        return False, {"reason": "answer_not_terminal", "node_id": answer_id}

    reachable = {answer_id}
    queue = deque([answer_id])
    while queue:
        cur = queue.popleft()
        for parent in reverse_graph.get(cur, []):
            if parent not in reachable:
                reachable.add(parent)
                queue.append(parent)
    if len(reachable) != len(node_map):
        dangling = sorted(node_id for node_id in node_map if node_id not in reachable)
        return False, {"reason": "node_not_connected", "node_ids": dangling}

    return True, {
        "reason": "",
        "answer_id": answer_id,
        "node_count": len(node_map),
        "search_count": search_count,
        "aggregate_count": aggregate_count,
    }


def topological_sort(graph: dict[str, Any]) -> list[str]:
    ok, meta = validate_graph(graph)
    if not ok:
        raise ValueError(f"invalid graph: {meta}")

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    node_ids = [str(node["id"]) for node in graph.get("nodes", []) if isinstance(node, dict)]

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            adjacency[source].append(target)
            indegree[target] += 1

    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            parent = str(parent_id)
            if parent and node_id not in adjacency[parent]:
                adjacency[parent].append(node_id)
                indegree[node_id] += 1

    queue = deque([node_id for node_id in node_ids if indegree[node_id] == 0])
    ordering: list[str] = []
    while queue:
        cur = queue.popleft()
        ordering.append(cur)
        for nxt in adjacency.get(cur, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return ordering


def topological_depths(graph: dict[str, Any]) -> dict[str, int]:
    ordering = topological_sort(graph)
    node_index = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }
    depths: dict[str, int] = {}
    for node_id in ordering:
        node = node_index[node_id]
        parents = node.get("inputs", []) if isinstance(node.get("inputs"), list) else []
        if not parents:
            depths[node_id] = 0
        else:
            depths[node_id] = max(depths.get(str(parent_id), 0) for parent_id in parents) + 1
    return depths
