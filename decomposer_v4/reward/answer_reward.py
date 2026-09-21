from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from decomposer_v4.reward.planner_reward import EmbeddingBackend
from decomposer_v4.schemas.graph import topological_sort
from decomposer_v4.schemas.trajectory import Rubric


class JudgeClient(Protocol):
    def chat(self) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class AnswerRewardConfig:
    execution_weight: float = 0.50
    synthesis_weight: float = 0.50
    answer_judge_weight: float = 0.50
    branch_utilization_weight: float = 0.50


_INLINE_CITATION_RE = re.compile(r"\[(N[0-9A-Za-z_]+-(?:R|F)\d+)\]")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "with",
        "would",
    }
)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _cosine_01(a: list[float], b: list[float]) -> float:
    return _clamp(max(0.0, _cosine(a, b)))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _shifted_harmonic_mean(values: Sequence[float]) -> float:
    cleaned = [_clamp(float(value)) for value in values]
    if not cleaned:
        return 0.0
    denominator = sum(1.0 / (1.0 + value) for value in cleaned)
    if denominator <= 0.0:
        return 0.0
    return _clamp((len(cleaned) / denominator) - 1.0)


def _weighted_average(pairs: Sequence[tuple[float, float]]) -> float:
    positive_pairs = [(max(float(weight), 0.0), float(value)) for weight, value in pairs if float(weight) > 0.0]
    total_weight = sum(weight for weight, _ in positive_pairs)
    if total_weight <= 0.0:
        return 0.0
    return sum(weight * value for weight, value in positive_pairs) / total_weight


def _strip_inline_citations(text: str) -> str:
    return _clean_text(_INLINE_CITATION_RE.sub(" ", str(text or "")))


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        if not any(char.isalnum() for char in token):
            continue
        if token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _lexical_recall(target: str, text: str) -> float:
    target_tokens = _content_tokens(target)
    if not target_tokens:
        return 0.0
    text_tokens = _content_tokens(text)
    if not text_tokens:
        return 0.0
    return len(target_tokens & text_tokens) / len(target_tokens)


def _approx_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(str(text or "")))


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(str(text or ""))
    return [_clean_text(part) for part in parts if _clean_text(part) and re.search(r"[A-Za-z0-9]", part)]


def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
    }


def _aggregate_node_ids(graph: dict[str, Any]) -> list[str]:
    node_map = _node_index(graph)
    return [
        node_id
        for node_id in topological_sort(graph)
        if str(node_map[node_id].get("type", "")) == "aggregate"
    ]


def _parent_inputs(node: dict[str, Any]) -> list[str]:
    return [str(parent_id).strip() for parent_id in node.get("inputs", []) if str(parent_id).strip()]


def _node_target_text(node: Mapping[str, Any]) -> str:
    key_points = " ".join(
        _clean_text(item)
        for item in node.get("key_points", [])
        if isinstance(node.get("key_points"), list)
    )
    return _clean_text(f"{node.get('need', '')} {key_points}")


def _encode_cached(*, embedder: EmbeddingBackend, cache: dict[str, list[float]], text: str) -> list[float]:
    if text not in cache:
        cache[text] = embedder.encode(text)
    return cache[text]


def _greedy_bipartite_match_scores(matrix: Sequence[Sequence[float]]) -> list[float]:
    if not matrix:
        return []
    remaining_rows = set(range(len(matrix)))
    remaining_cols = set(range(len(matrix[0]))) if matrix and matrix[0] else set()
    matches: list[float] = []
    while remaining_rows and remaining_cols:
        best_row = -1
        best_col = -1
        best_score = -1.0
        for row_idx in remaining_rows:
            for col_idx in remaining_cols:
                score = float(matrix[row_idx][col_idx])
                if score > best_score:
                    best_score = score
                    best_row = row_idx
                    best_col = col_idx
        if best_row < 0 or best_col < 0:
            break
        matches.append(max(best_score, 0.0))
        remaining_rows.remove(best_row)
        remaining_cols.remove(best_col)
    return matches


def _target_sentence_coverage(
    *,
    targets: Sequence[str],
    text: str,
    embedder: EmbeddingBackend,
    cache: dict[str, list[float]] | None = None,
) -> float:
    cleaned_targets = [_clean_text(target) for target in targets if _clean_text(target)]
    sentences = _split_into_sentences(_strip_inline_citations(text))
    if not cleaned_targets or not sentences:
        return 0.0
    local_cache = cache if cache is not None else {}
    sentence_vecs = [
        _encode_cached(embedder=embedder, cache=local_cache, text=sentence)
        for sentence in sentences
    ]
    scores: list[float] = []
    for target in cleaned_targets:
        target_vec = _encode_cached(embedder=embedder, cache=local_cache, text=target)
        scores.append(max(_cosine_01(target_vec, sentence_vec) for sentence_vec in sentence_vecs))
    return _mean(scores)


def compute_key_point_coverage_strict_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    del config
    node_map = _node_index(graph)
    embedding_cache: dict[str, list[float]] = {}
    scores: dict[str, float] = {}
    for node_id in _aggregate_node_ids(graph):
        node = node_map[node_id]
        output = _strip_inline_citations(node_outputs.get(node_id, ""))
        sentences = _split_into_sentences(output)
        key_points = [
            _clean_text(item)
            for item in node.get("key_points", [])
            if isinstance(node.get("key_points"), list) and _clean_text(item)
        ]
        if not sentences or not key_points:
            scores[node_id] = 0.0
            continue
        sentence_vecs = [
            _encode_cached(embedder=embedder, cache=embedding_cache, text=sentence)
            for sentence in sentences
        ]
        similarity_matrix: list[list[float]] = []
        for key_point in key_points:
            point_vec = _encode_cached(embedder=embedder, cache=embedding_cache, text=key_point)
            similarity_matrix.append([_cosine_01(point_vec, sentence_vec) for sentence_vec in sentence_vecs])
        matched_scores = _greedy_bipartite_match_scores(similarity_matrix)
        recall = len(matched_scores) / max(len(key_points), 1)
        scores[node_id] = _mean(matched_scores) * recall if matched_scores else 0.0
    return scores


def compute_key_points_coverage(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    return compute_key_point_coverage_strict_scores(
        graph=graph,
        node_outputs=node_outputs,
        embedder=embedder,
        config=config,
    )


def compute_key_point_sentence_alignment_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    return compute_key_point_coverage_strict_scores(
        graph=graph,
        node_outputs=node_outputs,
        embedder=embedder,
        config=config,
    )


def compute_lexical_keypoint_recall(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    del config
    node_map = _node_index(graph)
    scores: dict[str, float] = {}
    for node_id in _aggregate_node_ids(graph):
        node = node_map[node_id]
        output = _strip_inline_citations(node_outputs.get(node_id, ""))
        key_points = [
            _clean_text(item)
            for item in node.get("key_points", [])
            if isinstance(node.get("key_points"), list) and _clean_text(item)
        ]
        if not output or not key_points:
            scores[node_id] = 0.0
            continue
        scores[node_id] = _mean([_lexical_recall(key_point, output) for key_point in key_points])
    return scores


def compute_execution_rewards(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: dict[str, list[str]],
    citation_catalog: Mapping[str, Mapping[str, Any]],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, Any]:
    del evidence_pool, citation_catalog
    embedding_scores = compute_key_point_coverage_strict_scores(
        graph=graph,
        node_outputs=node_outputs,
        embedder=embedder,
    )
    lexical_scores = compute_lexical_keypoint_recall(
        graph=graph,
        node_outputs=node_outputs,
        config=config,
    )
    node_ids = _aggregate_node_ids(graph)
    node_scores = {
        node_id: _weighted_average(
            [
                (0.50, embedding_scores.get(node_id, 0.0)),
                (0.50, lexical_scores.get(node_id, 0.0)),
            ]
        )
        for node_id in node_ids
    }
    execution_reward = _shifted_harmonic_mean(list(node_scores.values())) if node_scores else 0.0
    return {
        "execution_reward": execution_reward,
        "per_node_rewards": dict(node_scores),
        "key_point_match_scores": dict(node_scores),
        "node_execution_coverage_scores": dict(node_scores),
        "embedding_keypoint_match_scores": dict(embedding_scores),
        "lexical_keypoint_recall_scores": dict(lexical_scores),
        "key_point_coverage_strict_scores": dict(embedding_scores),
        "key_point_match_mean": _mean(list(node_scores.values())) if node_scores else 0.0,
        "embedding_keypoint_match_mean": _mean(list(embedding_scores.values())) if embedding_scores else 0.0,
        "lexical_keypoint_recall_mean": _mean(list(lexical_scores.values())) if lexical_scores else 0.0,
    }


def compute_branch_utilization(
    *,
    final_answer: str,
    branch_outputs: dict[str, str],
) -> dict[str, float]:
    """Measure branch-output content carried into the final answer."""
    answer_tokens = _content_tokens(_strip_inline_citations(final_answer))
    if not answer_tokens:
        return {"branch_utilization": 0.0}
    scores: list[float] = []
    for output in branch_outputs.values():
        branch_tokens = _content_tokens(_strip_inline_citations(output))
        if not branch_tokens:
            scores.append(0.0)
            continue
        scores.append(len(branch_tokens & answer_tokens) / len(branch_tokens))

    return {"branch_utilization": _mean(scores)}


def compute_synthesis_reward(
    *,
    answer_judge: float = 0.0,
    branch_utilization: float = 0.0,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    cfg = config or AnswerRewardConfig()
    synthesis_reward = _weighted_average(
        [
            (cfg.answer_judge_weight, answer_judge),
            (cfg.branch_utilization_weight, branch_utilization),
        ]
    )
    return {
        "synthesis_reward": synthesis_reward,
        "answer_judge": answer_judge,
        "branch_utilization": branch_utilization,
    }


def compute_answer_reward(
    *,
    execution_reward: float,
    synthesis_reward: float,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    cfg = config or AnswerRewardConfig()
    content_reward = _weighted_average(
        [
            (cfg.execution_weight, execution_reward),
            (cfg.synthesis_weight, synthesis_reward),
        ]
    )
    return {
        "answer_reward": content_reward,
        "content_reward": content_reward,
        "valid_parse_bonus": 0.0,
        "valid_parse": 1.0,
        "execution_reward": execution_reward,
        "synthesis_reward": synthesis_reward,
    }


def _zero_scores(graph: dict[str, Any]) -> dict[str, float]:
    try:
        node_ids = _aggregate_node_ids(graph)
    except Exception:
        node_ids = []
    return {node_id: 0.0 for node_id in node_ids}


def compute_anchor_support_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    del node_outputs, evidence_pool, config
    return _zero_scores(graph)


def compute_compression_discipline_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    del node_outputs, evidence_pool, config
    return _zero_scores(graph)


def compute_factual_fidelity_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    return compute_anchor_support_scores(
        graph=graph,
        node_outputs=node_outputs,
        evidence_pool=evidence_pool,
        config=config,
    )


def compute_parent_balance_strict_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    del node_outputs, evidence_pool, embedder, config
    return _zero_scores(graph)


def compute_parent_semantic_balance_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    embedder: EmbeddingBackend,
    config: AnswerRewardConfig | None = None,
) -> dict[str, float]:
    return compute_parent_balance_strict_scores(
        graph=graph,
        node_outputs=node_outputs,
        evidence_pool=evidence_pool,
        embedder=embedder,
        config=config,
    )


def compute_citation_scope_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    citation_catalog: Mapping[str, Mapping[str, Any]],
    config: AnswerRewardConfig | None = None,
) -> dict[str, dict[str, float]]:
    del node_outputs, citation_catalog, config
    zeros = _zero_scores(graph)
    return {
        "citation_scope_scores": dict(zeros),
        "valid_citation_ratio_scores": dict(zeros),
        "parent_citation_coverage_scores": dict(zeros),
        "out_of_scope_citation_rate_scores": dict(zeros),
    }


def compute_evidence_compression_scores(
    *,
    graph: dict[str, Any],
    node_outputs: dict[str, str],
    evidence_pool: Mapping[str, list[str]],
    config: AnswerRewardConfig | None = None,
) -> dict[str, dict[str, float]]:
    del node_outputs, evidence_pool, config
    zeros = _zero_scores(graph)
    return {
        "evidence_compression_scores": dict(zeros),
        "compression_ratio_scores": dict(zeros),
    }


def _judge_completion_kwargs(*, model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
    }
    if not str(model).strip().lower().startswith("gpt-5"):
        kwargs["temperature"] = 0.0
    return kwargs


def _normalize_judge_score(value: Any) -> float | None:
    """Snap a raw judge score to the nearest valid level (0, 0.25, 0.5, 0.75, 1)."""
    try:
        raw = float(value)
    except Exception:
        return None
    if not math.isfinite(raw):
        return None
    if raw <= 0.125:
        return 0.0
    if raw <= 0.375:
        return 0.25
    if raw <= 0.625:
        return 0.5
    if raw <= 0.875:
        return 0.75
    return 1.0


def build_answer_judge_prompt(query: str, answer: str, rubrics: Sequence[Rubric]) -> str:
    payload = {
        "query": query,
        "answer": answer,
        "rubrics": [
            {
                "rubric_id": f"R{index}",
                "title": rubric.title,
                "description": rubric.description,
            }
            for index, rubric in enumerate(rubrics, start=1)
        ],
        "allowed_scores": [0, 0.25, 0.5, 0.75, 1],
        "score_meaning": {
            "0": "not covered: the rubric is absent, wrong, or unsupported",
            "0.25": "minimally covered: the rubric is touched on but mostly inadequate",
            "0.5": "partially covered: the rubric is addressed but with notable gaps or weak support",
            "0.75": "mostly covered: the rubric is substantively addressed with minor gaps",
            "1": "fully covered: the rubric is clearly and completely satisfied",
        },
        "output_schema": {
            "judgements": [
                {
                    "rubric_id": "R1",
                    "score": 0,
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_answer_judge(
    *,
    client: Any | None,
    model: str,
    system_prompt: str,
    query: str,
    answer: str,
    rubrics: Sequence[Rubric],
) -> float:
    if client is None or not model.strip() or not answer.strip() or not rubrics:
        return 0.0
    try:
        response = client.chat.completions.create(
            **_judge_completion_kwargs(model=model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_answer_judge_prompt(query, answer, rubrics)},
            ],
        )
        raw = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return 0.0

    rubric_index = {f"R{index}": rubric for index, rubric in enumerate(rubrics, start=1)}
    rubric_scores = {rubric_id: 0.0 for rubric_id in rubric_index}
    for judgement in raw.get("judgements", []):
        if not isinstance(judgement, dict):
            continue
        rubric_id = str(judgement.get("rubric_id", "")).strip()
        if rubric_id not in rubric_index:
            continue
        score = _normalize_judge_score(judgement.get("score"))
        if score is None:
            score = 0.0
        rubric_scores[rubric_id] = score

    total_weight = sum(max(float(rubric.weight), 0.0) for rubric in rubric_index.values())
    if total_weight <= 0:
        return 0.0
    total_score = sum(
        max(float(rubric.weight), 0.0) * rubric_scores[rubric_id]
        for rubric_id, rubric in rubric_index.items()
    )
    return total_score / total_weight
