from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.schemas.trajectory import Rubric


class EmbeddingBackend(Protocol):
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class PlannerRewardConfig:
    rubric_coverage_weight: float = 0.40
    search_quality_weight: float = 0.30
    graph_expressiveness_weight: float = 0.30
    graph_synthesis_shape_ratio: float = 0.40
    graph_cross_branch_ratio: float = 0.40
    graph_search_sufficiency_ratio: float = 0.20
    include_node_need_in_rubric_coverage: bool = True


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _normalize_url(url: str) -> str:
    raw = _clean_text(url)
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/")
    if not parts.scheme or not parts.netloc:
        return raw.rstrip("/")
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _cosine_01(a: list[float], b: list[float]) -> float:
    return _clamp(max(0.0, _cosine(a, b)))


def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }


def _parents(graph: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id"))
        out.setdefault(node_id, set())
        for parent_id in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            if str(parent_id):
                out[node_id].add(str(parent_id))
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            out.setdefault(target, set()).add(source)
            out.setdefault(source, set())
    return out


def _children(graph: dict[str, Any]) -> dict[str, set[str]]:
    child_index: dict[str, set[str]] = {node_id: set() for node_id in _node_index(graph)}
    for child_id, parent_ids in _parents(graph).items():
        child_index.setdefault(child_id, set())
        for parent_id in parent_ids:
            child_index.setdefault(parent_id, set()).add(child_id)
    return child_index


def _node_text(node: Mapping[str, Any]) -> str:
    key_points = " ".join(
        _clean_text(item)
        for item in node.get("key_points", [])
        if isinstance(node.get("key_points"), list)
    )
    return _clean_text(
        " ".join(
            [
                _clean_text(node.get("need")),
                _clean_text(node.get("query")),
                key_points,
            ]
        )
    )


def _node_candidate_texts(node: Mapping[str, Any], *, include_need: bool) -> list[str]:
    key_points = " ".join(
        _clean_text(item)
        for item in node.get("key_points", [])
        if isinstance(node.get("key_points"), list)
    )
    fields = [
        _clean_text(node.get("query")),
        key_points,
    ]
    if include_need:
        fields.append(_clean_text(node.get("need")))
    return [field for field in fields if field]


def _weighted_average(values: Sequence[tuple[float, float]]) -> float:
    pairs = [(max(float(weight), 0.0), float(value)) for weight, value in values if float(weight) > 0.0]
    total = sum(weight for weight, _ in pairs)
    if total <= 0.0:
        return 0.0
    return sum(weight * value for weight, value in pairs) / total


def _mean_pairwise_cosine_01(texts: Sequence[str], embedder: EmbeddingBackend) -> float:
    cleaned = [_clean_text(text) for text in texts if _clean_text(text)]
    if len(cleaned) < 2:
        return 1.0
    embeddings = embedder.encode_batch(list(cleaned))
    pairwise = [
        _cosine_01(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return _mean(pairwise) if pairwise else 1.0


def _reachable_nodes(children: Mapping[str, set[str]], source_id: str) -> set[str]:
    seen: set[str] = set()
    stack = list(children.get(source_id, set()))
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(children.get(node_id, set()) - seen)
    return seen


def _diminishing_returns(count: int) -> float:
    """Parameterless diminishing-returns: 1 - 1/x.

    Returns 0 for count<=1, 0.5 for 2, 0.67 for 3, 0.80 for 5, etc.
    """
    return _clamp(1.0 - 1.0 / max(count, 1))


def _graph_expressiveness(
    graph: dict[str, Any],
    answer_id: str,
    embedder: EmbeddingBackend,
    config: PlannerRewardConfig,
) -> dict[str, float]:
    node_map = _node_index(graph)
    parent_index = _parents(graph)
    child_index = _children(graph)

    answer_parent_ids = sorted(parent_index.get(answer_id, set()))
    answer_aggregate_parent_ids = [
        parent_id
        for parent_id in answer_parent_ids
        if str(node_map.get(parent_id, {}).get("type", "")).strip().lower() == "aggregate"
    ]
    answer_parent_count = len(answer_parent_ids)
    answer_aggregate_parent_count = len(answer_aggregate_parent_ids)

    # SynthesisBreadth: diminishing returns on parent count * embedding diversity.
    # Replaces binary >=3 threshold + 0.80 duplicate similarity threshold.
    synthesis_breadth = _diminishing_returns(answer_aggregate_parent_count)
    parent_texts = [
        _node_text(node_map[parent_id])
        for parent_id in answer_aggregate_parent_ids
        if parent_id in node_map and _node_text(node_map[parent_id])
    ]
    if len(parent_texts) >= 2:
        embeddings = embedder.encode_batch(parent_texts)
        pairwise = [
            _cosine_01(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        parent_diversity = _clamp(1.0 - _mean(pairwise)) if pairwise else 0.0
        answer_parent_max_similarity = max(pairwise, default=1.0)
    else:
        parent_diversity = 0.0
        answer_parent_max_similarity = 1.0
    synthesis_shape = synthesis_breadth * parent_diversity

    aggregate_ids = [
        node_id
        for node_id, node in node_map.items()
        if str(node.get("type", "")).strip().lower() == "aggregate"
    ]
    search_ids = [
        node_id
        for node_id, node in node_map.items()
        if str(node.get("type", "")).strip().lower() == "search"
    ]

    # CrossBranchIntegration: continuous ratios replacing binary >=2 / >=1 thresholds.
    # reuse_ratio: fraction of search nodes serving 2+ downstream aggregates.
    # integration_ratio: fraction of aggregates drawing from 2+ parent inputs.
    aggregate_nodes_with_multi_parent_count = sum(
        1 for aggregate_id in aggregate_ids if len(parent_index.get(aggregate_id, set())) >= 2
    )
    reused_search_node_count = sum(
        1 for search_id in search_ids if len(child_index.get(search_id, set())) >= 2
    )
    reuse_ratio = reused_search_node_count / max(len(search_ids), 1)
    integration_ratio = aggregate_nodes_with_multi_parent_count / max(len(aggregate_ids), 1)
    cross_branch_structure = _clamp((reuse_ratio + integration_ratio) / 2.0)

    # SearchBreadth: diminishing returns on search count, no target constant.
    search_sufficiency = _diminishing_returns(len(search_ids))

    graph_expressiveness = _clamp(
        config.graph_synthesis_shape_ratio * synthesis_shape
        + config.graph_cross_branch_ratio * cross_branch_structure
        + config.graph_search_sufficiency_ratio * search_sufficiency
    )
    return {
        "graph_expressiveness": graph_expressiveness,
        "synthesis_shape": synthesis_shape,
        "cross_branch_structure": cross_branch_structure,
        "search_sufficiency": search_sufficiency,
        "synthesis_parent_count_pass": synthesis_breadth,
        "synthesis_parent_distinctness_pass": parent_diversity,
        "answer_parent_max_similarity": answer_parent_max_similarity,
        "answer_parent_count": float(answer_parent_count),
        "answer_aggregate_parent_count": float(answer_aggregate_parent_count),
        "aggregate_nodes_with_multi_parent_count": float(aggregate_nodes_with_multi_parent_count),
        "reused_search_node_count": float(reused_search_node_count),
        "target_search_count": 0.0,
    }


def _graph_structure_metrics(graph: dict[str, Any], answer_id: str) -> dict[str, float]:
    node_map = _node_index(graph)
    parent_index = _parents(graph)
    child_index = _children(graph)
    node_count = float(len(node_map))
    non_answer_node_count = float(
        sum(
            1
            for node in node_map.values()
            if str(node.get("type", "")).strip().lower() != "answer"
        )
    )
    edge_count = float(sum(len(parent_ids) for parent_ids in parent_index.values()))
    search_ids = [
        node_id
        for node_id, node in node_map.items()
        if str(node.get("type", "")).strip().lower() == "search"
    ]
    aggregate_ids = [
        node_id
        for node_id, node in node_map.items()
        if str(node.get("type", "")).strip().lower() == "aggregate"
    ]
    answer_parent_ids = parent_index.get(answer_id, set())
    answer_direct_aggregate_parent_ids = [
        parent_id
        for parent_id in answer_parent_ids
        if str(node_map.get(parent_id, {}).get("type", "")).strip().lower() == "aggregate"
    ]
    answer_level_branch_ids = sorted(answer_parent_ids)
    reachable_branch_counts: list[float] = []
    reached_branch_ids: set[str] = set()
    for search_id in search_ids:
        reachable = _reachable_nodes(child_index, search_id)
        reached = [branch_id for branch_id in answer_level_branch_ids if branch_id in reachable]
        reachable_branch_counts.append(float(len(reached)))
        reached_branch_ids.update(reached)
    aggregate_parent_counts = [float(len(parent_index.get(aggregate_id, set()))) for aggregate_id in aggregate_ids]
    search_to_aggregate_edges = 0
    aggregate_to_aggregate_edges = 0
    for aggregate_id in aggregate_ids:
        for parent_id in parent_index.get(aggregate_id, set()):
            parent_type = str(node_map.get(parent_id, {}).get("type", "")).strip().lower()
            if parent_type == "search":
                search_to_aggregate_edges += 1
            elif parent_type == "aggregate":
                aggregate_to_aggregate_edges += 1
    max_possible_edges = max(node_count * max(node_count - 1.0, 0.0), 1.0)
    aggregate_count = float(len(aggregate_ids))
    return {
        "node_count": node_count,
        "non_answer_node_count": non_answer_node_count,
        "edge_count": edge_count,
        "graph_edge_density": edge_count / max_possible_edges,
        "search_node_count": float(len(search_ids)),
        "aggregate_node_count": aggregate_count,
        "answer_direct_parent_count": float(len(answer_parent_ids)),
        "answer_direct_aggregate_parent_count": float(len(answer_direct_aggregate_parent_ids)),
        "answer_direct_aggregate_parent_ratio": (
            float(len(answer_direct_aggregate_parent_ids)) / aggregate_count
            if aggregate_count > 0.0
            else 0.0
        ),
        "answer_level_branch_count": float(len(reached_branch_ids)),
        "answer_level_declared_branch_count": float(len(answer_level_branch_ids)),
        "aggregate_parent_mean": _mean(aggregate_parent_counts),
        "aggregate_parent_max": max(aggregate_parent_counts, default=0.0),
        "aggregate_nodes_with_multi_parent_count": float(sum(1 for count in aggregate_parent_counts if count >= 2.0)),
        "search_to_aggregate_edge_count": float(search_to_aggregate_edges),
        "aggregate_to_aggregate_edge_count": float(aggregate_to_aggregate_edges),
        "search_branch_reach_mean": _mean(reachable_branch_counts),
        "search_branch_reach_max": max(reachable_branch_counts, default=0.0),
        "search_multi_branch_count": float(sum(1 for count in reachable_branch_counts if count > 1.0)),
    }


def _fetch_url_validity(
    *,
    search_results: Mapping[str, list[dict[str, Any]]] | None,
    attempted_fetch_requests: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    attempted = [
        row
        for row in (attempted_fetch_requests or [])
        if isinstance(row, Mapping)
    ]
    if not attempted:
        return {
            "fetch_url_validity": 0.0,
        }

    allowed: set[tuple[str, str]] = set()
    for node_id, rows in (search_results or {}).items():
        for result in rows:
            if not isinstance(result, dict):
                continue
            norm = _normalize_url(str(result.get("url", "")))
            if norm:
                allowed.add((str(node_id), norm))

    valid_count = 0
    for request in attempted:
        source_node_id = str(request.get("source_node_id", "")).strip()
        norm = _normalize_url(str(request.get("url", "")))
        if (source_node_id, norm) in allowed:
            valid_count += 1

    fetch_url_validity = valid_count / max(len(attempted), 1)
    return {
        "fetch_url_validity": fetch_url_validity,
    }


def compute_planner_reward(
    *,
    graph: dict[str, Any] | None,
    rubrics: Sequence[Rubric],
    user_query: str,
    evidence_pool: dict[str, list[str]],
    embedder: EmbeddingBackend,
    fetch_requests: Sequence[Mapping[str, Any]] | None = None,
    attempted_fetch_requests: Sequence[Mapping[str, Any]] | None = None,
    search_results: Mapping[str, list[dict[str, Any]]] | None = None,
    config: PlannerRewardConfig | None = None,
) -> dict[str, float]:
    cfg = config or PlannerRewardConfig()
    ok, meta = validate_graph(graph)
    if not ok or graph is None:
        return {
            "gate": 0.0,
            "planner_reward": 0.0,
            "rubric_coverage": 0.0,
            "decomposition_distinctness": 0.0,
            "graph_expressiveness": 0.0,
            "synthesis_shape": 0.0,
            "cross_branch_structure": 0.0,
            "search_sufficiency": 0.0,
            "search_rubric_coverage": 0.0,
            "search_distinctness": 0.0,
            "search_diversity": 0.0,
            "query_specificity": 0.0,
            "node_count": 0.0,
            "non_answer_node_count": 0.0,
            "edge_count": 0.0,
            "graph_edge_density": 0.0,
            "search_node_count": 0.0,
            "aggregate_node_count": 0.0,
            "answer_direct_parent_count": 0.0,
            "answer_direct_aggregate_parent_count": 0.0,
            "answer_direct_aggregate_parent_ratio": 0.0,
            "answer_level_branch_count": 0.0,
            "answer_level_declared_branch_count": 0.0,
            "aggregate_parent_mean": 0.0,
            "aggregate_parent_max": 0.0,
            "search_to_aggregate_edge_count": 0.0,
            "aggregate_to_aggregate_edge_count": 0.0,
            "search_branch_reach_mean": 0.0,
            "search_branch_reach_max": 0.0,
            "search_multi_branch_count": 0.0,
            "search_quality": 0.0,
            "fetch_url_validity": 0.0,
            "synthesis_parent_count_pass": 0.0,
            "synthesis_parent_distinctness_pass": 0.0,
            "answer_parent_max_similarity": 1.0,
            "answer_parent_count": 0.0,
            "answer_aggregate_parent_count": 0.0,
            "aggregate_nodes_with_multi_parent_count": 0.0,
            "reused_search_node_count": 0.0,
            "target_search_count": 0.0,
            "invalid_reason": 1.0,
        }

    search_nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "search"]
    candidate_nodes = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and str(node.get("type", "")) != "answer"
    ]

    rubric_scores: list[tuple[float, float]] = []
    targets: list[tuple[float, str]] = [
        (float(rubric.weight), _clean_text(rubric.description or rubric.title))
        for rubric in rubrics
        if _clean_text(rubric.description or rubric.title)
    ]
    if not targets and _clean_text(user_query):
        targets = [(1.0, _clean_text(user_query))]

    candidate_texts_by_node = [
        _node_candidate_texts(node, include_need=cfg.include_node_need_in_rubric_coverage)
        for node in candidate_nodes
    ]
    for weight, target in targets:
        target_vec = embedder.encode(target)
        best = 0.0
        for texts in candidate_texts_by_node:
            for candidate_text in texts:
                best = max(best, _cosine_01(target_vec, embedder.encode(candidate_text)))
        rubric_scores.append((weight, best))
    rubric_coverage = _weighted_average(rubric_scores)

    decomposition_texts = [
        _node_text(node)
        for node in candidate_nodes
        if _node_text(node)
    ]
    decomposition_distinctness = (
        _clamp(1.0 - _mean_pairwise_cosine_01(decomposition_texts, embedder))
        if len(decomposition_texts) >= 2
        else 0.0
    )

    search_queries = [_clean_text(node.get("query")) for node in search_nodes if _clean_text(node.get("query"))]
    search_distinctness = (
        _clamp(1.0 - _mean_pairwise_cosine_01(search_queries, embedder))
        if len(search_queries) >= 2
        else 0.0
    )
    search_rubric_scores: list[tuple[float, float]] = []
    for weight, target in targets:
        target_vec = embedder.encode(target)
        best = 0.0
        for node in search_nodes:
            search_text = _clean_text(node.get("query"))
            if search_text:
                best = max(best, _cosine_01(target_vec, embedder.encode(search_text)))
        search_rubric_scores.append((weight, best))
    search_rubric_coverage = _weighted_average(search_rubric_scores)
    search_quality = _weighted_average(
        [
            (0.5, search_rubric_coverage),
            (0.5, search_distinctness),
        ]
    )

    answer_id = str(meta.get("answer_id", ""))
    structure = _graph_structure_metrics(graph, answer_id)
    expressiveness = _graph_expressiveness(graph, answer_id, embedder, cfg)
    fetch_diagnostics = _fetch_url_validity(
        attempted_fetch_requests=attempted_fetch_requests,
        search_results=search_results,
    )
    del fetch_requests
    planner_reward = _clamp(_weighted_average([
        (cfg.rubric_coverage_weight, rubric_coverage),
        (cfg.search_quality_weight, search_quality),
        (cfg.graph_expressiveness_weight, expressiveness["graph_expressiveness"]),
    ]))
    return {
        "gate": 1.0,
        "planner_reward": planner_reward,
        "rubric_coverage": rubric_coverage,
        "decomposition_distinctness": decomposition_distinctness,
        "graph_expressiveness": expressiveness["graph_expressiveness"],
        "synthesis_shape": expressiveness["synthesis_shape"],
        "cross_branch_structure": expressiveness["cross_branch_structure"],
        "search_sufficiency": expressiveness["search_sufficiency"],
        "search_rubric_coverage": search_rubric_coverage,
        "search_distinctness": search_distinctness,
        "search_diversity": search_distinctness,
        "query_specificity": search_rubric_coverage,
        "node_count": structure["node_count"],
        "non_answer_node_count": structure["non_answer_node_count"],
        "edge_count": structure["edge_count"],
        "graph_edge_density": structure["graph_edge_density"],
        "search_node_count": structure["search_node_count"],
        "aggregate_node_count": structure["aggregate_node_count"],
        "answer_direct_parent_count": structure["answer_direct_parent_count"],
        "answer_direct_aggregate_parent_count": structure["answer_direct_aggregate_parent_count"],
        "answer_direct_aggregate_parent_ratio": structure["answer_direct_aggregate_parent_ratio"],
        "answer_level_branch_count": structure["answer_level_branch_count"],
        "answer_level_declared_branch_count": structure["answer_level_declared_branch_count"],
        "aggregate_parent_mean": structure["aggregate_parent_mean"],
        "aggregate_parent_max": structure["aggregate_parent_max"],
        "search_to_aggregate_edge_count": structure["search_to_aggregate_edge_count"],
        "aggregate_to_aggregate_edge_count": structure["aggregate_to_aggregate_edge_count"],
        "search_branch_reach_mean": structure["search_branch_reach_mean"],
        "search_branch_reach_max": structure["search_branch_reach_max"],
        "search_multi_branch_count": structure["search_multi_branch_count"],
        "search_quality": search_quality,
        "fetch_url_validity": fetch_diagnostics["fetch_url_validity"],
        "synthesis_parent_count_pass": expressiveness["synthesis_parent_count_pass"],
        "synthesis_parent_distinctness_pass": expressiveness["synthesis_parent_distinctness_pass"],
        "answer_parent_max_similarity": expressiveness["answer_parent_max_similarity"],
        "answer_parent_count": expressiveness["answer_parent_count"],
        "answer_aggregate_parent_count": expressiveness["answer_aggregate_parent_count"],
        "aggregate_nodes_with_multi_parent_count": expressiveness["aggregate_nodes_with_multi_parent_count"],
        "reused_search_node_count": expressiveness["reused_search_node_count"],
        "target_search_count": expressiveness["target_search_count"],
        "invalid_reason": 0.0,
    }
