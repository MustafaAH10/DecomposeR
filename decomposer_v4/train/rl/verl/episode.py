from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class RolloutSpan:
    role: str
    turn_type: str
    prompt_messages: list[dict[str, str]]
    prompt_ids: list[int]
    generated_ids: list[int]
    generated_logprobs: list[float]
    completion_text: str
    prefill_ids: list[int] = field(default_factory=list)
    target_node_ids: tuple[str, ...] = ()


@dataclass
class PackedEpisode:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]


class JsonlRolloutLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_LOGGER_CACHE: dict[Path, JsonlRolloutLogger] = {}
_LOGGER_CACHE_LOCK = threading.Lock()


def get_rollout_logger(root_dir: str | Path | None, stage: str) -> JsonlRolloutLogger | None:
    if root_dir is None:
        return None
    raw_root = str(root_dir).strip()
    if not raw_root:
        return None
    rank = (
        os.environ.get("RANK")
        or os.environ.get("WORLD_RANK")
        or os.environ.get("LOCAL_RANK")
        or "0"
    )
    path = Path(raw_root).resolve() / f"{stage}_rollouts_rank{rank}.jsonl"
    with _LOGGER_CACHE_LOCK:
        logger = _LOGGER_CACHE.get(path)
        if logger is None:
            logger = JsonlRolloutLogger(path)
            _LOGGER_CACHE[path] = logger
        return logger


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return sum(value_list) / max(len(value_list), 1)


def pack_spans_for_verl(
    *,
    spans: Sequence[RolloutSpan],
    bos_token_id: int | None,
    max_prompt_length: int,
    max_response_length: int,
) -> PackedEpisode:
    del bos_token_id
    if not spans:
        return PackedEpisode(prompt_ids=[], response_ids=[], response_mask=[], response_logprobs=[])

    prompt_ids = list(spans[0].prompt_ids)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []

    for index, span in enumerate(spans):
        if index > 0:
            # Later turns contribute only the incremental user/environment tokens
            # that were appended after the previous model generation.
            prompt_continuation = list(span.prefill_ids)
            response_ids.extend(prompt_continuation)
            response_mask.extend([0] * len(prompt_continuation))
            response_logprobs.extend([0.0] * len(prompt_continuation))

        generated_ids = list(span.generated_ids)
        response_ids.extend(generated_ids)
        response_mask.extend([1] * len(generated_ids))
        if span.generated_logprobs:
            response_logprobs.extend(list(span.generated_logprobs))
        else:
            response_logprobs.extend([0.0] * len(generated_ids))

    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    while len(response_ids) > max_response_length:
        removable_idx = next((idx for idx, mask in enumerate(response_mask) if mask == 0), None)
        if removable_idx is None:
            break
        del response_ids[removable_idx]
        del response_mask[removable_idx]
        del response_logprobs[removable_idx]

    overflow = len(response_ids) - max_response_length
    if overflow > 0:
        response_ids = response_ids[overflow:]
        response_mask = response_mask[overflow:]
        response_logprobs = response_logprobs[overflow:]

    return PackedEpisode(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
    )


def assistant_transcript(spans: Sequence[RolloutSpan]) -> str:
    parts: list[str] = []
    for span in spans:
        target_suffix = ""
        if span.target_node_ids:
            target_suffix = " " + ",".join(span.target_node_ids)
        label = f"[{span.turn_type}{target_suffix}]"
        content = str(span.completion_text or "").strip()
        parts.append(f"{label}\n{content}".strip())
    return "\n\n".join(part for part in parts if part).strip()


def planner_debug_summary(row: dict[str, Any]) -> dict[str, float]:
    keys = (
        "planner_reward",
        "rubric_coverage",
        "decomposition_distinctness",
        "search_rubric_coverage",
        "search_distinctness",
        "node_count",
        "non_answer_node_count",
        "edge_count",
        "graph_edge_density",
        "search_node_count",
        "aggregate_node_count",
        "answer_direct_parent_count",
        "answer_direct_aggregate_parent_count",
        "answer_direct_aggregate_parent_ratio",
        "answer_level_branch_count",
        "answer_level_declared_branch_count",
        "aggregate_parent_mean",
        "aggregate_parent_max",
        "search_to_aggregate_edge_count",
        "aggregate_to_aggregate_edge_count",
        "search_branch_reach_mean",
        "search_branch_reach_max",
        "search_multi_branch_count",
        "graph_expressiveness",
        "synthesis_shape",
        "cross_branch_structure",
        "search_sufficiency",
        "search_quality",
        "search_diversity",
        "query_specificity",
        "fetch_url_validity",
        "synthesis_parent_count_pass",
        "synthesis_parent_distinctness_pass",
        "answer_parent_max_similarity",
        "answer_parent_count",
        "answer_aggregate_parent_count",
        "aggregate_nodes_with_multi_parent_count",
        "reused_search_node_count",
        "target_search_count",
        "invalid_reason",
        "gate_failed",
        "search_seconds_initial",
        "search_seconds_post_revision",
        "search_seconds_total",
        "fetch_requests_selected_count",
        "fetch_requests_attempted_count",
        "search_nodes_final_count",
    )
    return {key: float(row.get(key, 0.0) or 0.0) for key in keys}


def answer_debug_summary(row: dict[str, Any]) -> dict[str, float]:
    keys = (
        "answer_reward",
        "content_reward",
        "valid_parse_bonus",
        "valid_parse",
        "execution_reward",
        "synthesis_reward",
        "key_point_match_mean",
        "embedding_keypoint_match_mean",
        "lexical_keypoint_recall_mean",
        "answer_judge",
        "branch_utilization",
        "branch_outputs_count",
        "answer_words",
        "answer_with_citations_words",
        "gate_failed",
    )
    return {key: float(row.get(key, 0.0) or 0.0) for key in keys}


def planner_zero_result(*, row: dict[str, Any], spans: Sequence[RolloutSpan], failure_reason: str) -> dict[str, Any]:
    payload = {
        "query_id": str(row.get("query_id", "")),
        "query": str(row.get("query", "")),
        "assistant_transcript": assistant_transcript(spans),
        "planner_reward": 0.0,
        "rubric_coverage": 0.0,
        "decomposition_distinctness": 0.0,
        "search_rubric_coverage": 0.0,
        "search_distinctness": 0.0,
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
        "graph_expressiveness": 0.0,
        "synthesis_shape": 0.0,
        "cross_branch_structure": 0.0,
        "search_sufficiency": 0.0,
        "search_quality": 0.0,
        "search_diversity": 0.0,
        "query_specificity": 0.0,
        "fetch_url_validity": 0.0,
        "synthesis_parent_count_pass": 0.0,
        "synthesis_parent_distinctness_pass": 0.0,
        "answer_parent_max_similarity": 1.0,
        "answer_parent_count": 0.0,
        "answer_aggregate_parent_count": 0.0,
        "aggregate_nodes_with_multi_parent_count": 0.0,
        "reused_search_node_count": 0.0,
        "target_search_count": 5.0,
        "invalid_reason": 1.0,
        "gate_failed": 1.0,
        "search_seconds_initial": 0.0,
        "search_seconds_post_revision": 0.0,
        "search_seconds_total": 0.0,
        "fetch_requests_selected_count": 0.0,
        "fetch_requests_attempted_count": 0.0,
        "search_nodes_final_count": 0.0,
        "failure_reason": failure_reason,
    }
    payload["summary_json"] = json.dumps(planner_debug_summary(payload), ensure_ascii=False)
    return payload


def answer_zero_result(*, row: dict[str, Any], spans: Sequence[RolloutSpan], failure_reason: str) -> dict[str, Any]:
    payload = {
        "query_id": str(row.get("query_id", "")),
        "query": str(row.get("query", "")),
        "assistant_transcript": assistant_transcript(spans),
        "final_answer": "",
        "answer_reward": 0.0,
        "content_reward": 0.0,
        "valid_parse_bonus": 0.0,
        "valid_parse": 0.0,
        "execution_reward": 0.0,
        "synthesis_reward": 0.0,
        "key_point_match_mean": 0.0,
        "embedding_keypoint_match_mean": 0.0,
        "lexical_keypoint_recall_mean": 0.0,
        "answer_judge": 0.0,
        "branch_utilization": 0.0,
        "branch_outputs_count": 0.0,
        "answer_words": 0.0,
        "answer_with_citations_words": 0.0,
        "gate_failed": 1.0,
        "failure_reason": failure_reason,
    }
    payload["summary_json"] = json.dumps(answer_debug_summary(payload), ensure_ascii=False)
    return payload


def planner_rollout_record(
    *,
    row: dict[str, Any],
    spans: Sequence[RolloutSpan],
    payload: dict[str, Any],
    full_artifact: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": "planner",
        "query_id": str(row.get("query_id", "")),
        "query": str(row.get("query", "")),
        "summary": planner_debug_summary(payload),
        "assistant_transcript": payload.get("assistant_transcript", ""),
        "spans": [
            {
                "role": span.role,
                "turn_type": span.turn_type,
                "target_node_ids": list(span.target_node_ids),
                "completion_text": span.completion_text,
                "prompt_messages": span.prompt_messages,
            }
            for span in spans
        ],
    }
    record.update(full_artifact)
    return record


def answer_rollout_record(
    *,
    row: dict[str, Any],
    spans: Sequence[RolloutSpan],
    payload: dict[str, Any],
    full_artifact: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": "answerer",
        "query_id": str(row.get("query_id", "")),
        "query": str(row.get("query", "")),
        "summary": answer_debug_summary(payload),
        "assistant_transcript": payload.get("assistant_transcript", ""),
        "spans": [
            {
                "role": span.role,
                "turn_type": span.turn_type,
                "target_node_ids": list(span.target_node_ids),
                "completion_text": span.completion_text,
                "prompt_messages": span.prompt_messages,
            }
            for span in spans
        ],
    }
    record.update(full_artifact)
    return record


def planner_reward_extra_info(payload: dict[str, Any]) -> dict[str, float]:
    return planner_debug_summary(payload) | {
        "reward": float(payload.get("planner_reward", 0.0) or 0.0),
    }


def answer_reward_extra_info(payload: dict[str, Any]) -> dict[str, float]:
    return answer_debug_summary(payload) | {
        "reward": float(payload.get("answer_reward", 0.0) or 0.0),
    }


def mean_metric(values: Iterable[float]) -> float:
    return _mean(values)
