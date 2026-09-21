from __future__ import annotations

import asyncio
import json
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.workers.rollout.replica import TokenOutput

from decomposer_v4.data.generation.answerer_compact_scaffold import (
    build_execution_turn_payload_compact,
    build_final_answer_payload_compact,
    search_dependency_ids_for_wave,
)
from decomposer_v4.data.generation.scaffold import (
    aggregate_execution_waves,
    append_citation_block,
    build_after_search_payload,
    build_citation_catalog,
    build_execution_turn_payload,
    build_final_answer_payload,
    evidence_pool_from_executions,
    execute_revision_retrievals,
    execute_search_nodes_timed,
    normalize_search_results,
    parse_answerer_node_outputs,
    parse_planner_initial_output,
    parse_planner_revision_output,
)
from decomposer_v4.reward.answer_reward import (
    AnswerRewardConfig,
    compute_answer_reward,
    compute_branch_utilization,
    compute_execution_rewards,
    compute_synthesis_reward,
    run_answer_judge,
)
from decomposer_v4.reward.planner_reward import PlannerRewardConfig, compute_planner_reward
from decomposer_v4.schemas.graph import validate_graph
from decomposer_v4.train.rl.common import deserialize_rubrics
from decomposer_v4.train.rl.runtime import (
    ANSWERER_SYSTEM_PROMPT,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    load_answerer_system_prompt,
    make_embedder,
    make_openai_client,
    make_search_backend,
)
from decomposer_v4.train.rl.verl.episode import (
    RolloutSpan,
    answer_reward_extra_info,
    answer_rollout_record,
    answer_zero_result,
    get_rollout_logger,
    pack_spans_for_verl,
    planner_reward_extra_info,
    planner_rollout_record,
    planner_zero_result,
)
from decomposer_v4.utils.io import extract_json_object


_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_SEMAPHORES_LOCK = threading.Lock()


def _shared_semaphore(name: str, limit: int) -> threading.BoundedSemaphore:
    key = (name, max(int(limit), 1))
    with _SEMAPHORES_LOCK:
        semaphore = _SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(key[1])
            _SEMAPHORES[key] = semaphore
        return semaphore


def _run_with_semaphore(name: str, limit: int, fn):
    semaphore = _shared_semaphore(name, limit)
    with semaphore:
        return fn()


def _sum_timing_seconds(timing: dict[str, Any] | None) -> float:
    if not isinstance(timing, dict):
        return 0.0
    total = 0.0
    for value in timing.values():
        if isinstance(value, dict):
            total += float(value.get("seconds", 0.0) or 0.0)
    return total


@lru_cache(maxsize=None)
def _cached_search_backend() -> Any:
    return make_search_backend()


@lru_cache(maxsize=None)
def _cached_embedder(backend: str, model_name: str) -> Any:
    return make_embedder(backend, model_name)


@lru_cache(maxsize=None)
def _cached_openai_client(base_url: str) -> Any:
    return make_openai_client(base_url)


@lru_cache(maxsize=2048)
def _load_json_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


class _DecomposerAgentLoopBase(AgentLoopBase):
    def __init__(
        self,
        *args,
        generation: dict[str, Any] | None = None,
        embedding: dict[str, Any] | None = None,
        rollout_logging: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generation = dict(generation or {})
        self.embedding_cfg = dict(embedding or {})
        self.rollout_logging = dict(rollout_logging or {})
        self.include_prompt_messages = bool(self.rollout_logging.get("include_prompt_messages", False))
        self.max_parallel_episodes = int(self.rollout_logging.get("max_parallel_episodes", 4))
        self.rollout_logger = get_rollout_logger(self.rollout_logging.get("output_dir"), self.stage_name)
        self.embedder = _cached_embedder(
            str(self.embedding_cfg.get("backend", "sentence_transformer")),
            str(self.embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        )
        self.max_prompt_length = int(self.rollout_config.prompt_length)
        self.max_response_length = int(self.rollout_config.response_length)

    @property
    def stage_name(self) -> str:
        raise NotImplementedError

    async def _tokenize_prompt_delta(self, messages: list[dict[str, str]]) -> list[int]:
        prompt_delta = await self.apply_chat_template(messages, remove_system_prompt=True)
        return list(prompt_delta)

    async def _generate_turn(
        self,
        *,
        request_id: str,
        messages: list[dict[str, str]],
        prompt_ids: Sequence[int] | None = None,
        prompt_prefill_ids: Sequence[int] = (),
        sampling_params: dict[str, Any],
        max_new_tokens: int,
        role: str,
        turn_type: str,
        target_node_ids: Sequence[str] = (),
    ) -> tuple[RolloutSpan, TokenOutput, float]:
        if prompt_ids is None:
            prompt_ids = await self.apply_chat_template(messages)
        prompt_ids = list(prompt_ids)
        generation_params = dict(sampling_params)
        generation_params["max_tokens"] = int(max_new_tokens)

        started = time.perf_counter()
        output: TokenOutput = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=generation_params,
        )
        elapsed = time.perf_counter() - started

        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        span = RolloutSpan(
            role=role,
            turn_type=turn_type,
            prompt_messages=messages if self.include_prompt_messages else [],
            prompt_ids=prompt_ids,
            generated_ids=list(output.token_ids),
            generated_logprobs=list(output.log_probs or []),
            completion_text=text,
            prefill_ids=list(prompt_prefill_ids),
            target_node_ids=tuple(str(node_id) for node_id in target_node_ids),
        )
        return span, output, elapsed


class DecomposerPlannerAgentLoop(_DecomposerAgentLoopBase):
    def __init__(
        self,
        *args,
        reward: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.search_backend = _cached_search_backend()
        self.reward_cfg = PlannerRewardConfig(**(reward or {}))

    @property
    def stage_name(self) -> str:
        return "planner"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        row = dict(kwargs)
        query = str(row.get("query", ""))
        rubrics = deserialize_rubrics(row.get("rubrics"))
        request_id = uuid4().hex
        spans: list[RolloutSpan] = []
        generated_seconds = 0.0
        search_seconds_total = 0.0
        reward_seconds = 0.0
        num_preempted = -1

        initial_graph: dict[str, Any] | None = None
        final_graph: dict[str, Any] | None = None
        search_node_outputs: dict[str, str] | None = None
        initial_search_results: dict[str, Any] | None = None
        final_search_results: dict[str, Any] | None = None
        fetch_requests: list[dict[str, str]] = []

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        initial_span, initial_output, elapsed = await self._generate_turn(
            request_id=request_id,
            messages=messages,
            sampling_params=sampling_params,
            max_new_tokens=int(self.generation.get("planner_initial_max_new_tokens", 1400)),
            role="planner",
            turn_type="planner_initial",
        )
        spans.append(initial_span)
        generated_seconds += elapsed
        if initial_output.num_preempted is not None:
            num_preempted = int(initial_output.num_preempted)
        current_prompt_ids = [*initial_span.prompt_ids, *initial_span.generated_ids]

        try:
            initial_graph = parse_planner_initial_output(initial_span.completion_text)
            ok, _ = validate_graph(initial_graph)
            if not ok:
                raise RuntimeError("initial_graph_invalid")
        except Exception:
            payload = planner_zero_result(row=row, spans=spans, failure_reason="initial_graph_invalid")
        else:
            search_started = time.perf_counter()
            (
                search_node_outputs,
                initial_search_results,
                initial_search_executions,
                initial_search_timing,
            ) = await self.loop.run_in_executor(
                None,
                lambda: _run_with_semaphore(
                    "planner_search",
                    self.max_parallel_episodes,
                    lambda: (
                        lambda outputs, executions, timing: (
                            outputs,
                            normalize_search_results(executions),
                            executions,
                            timing,
                        )
                    )(*execute_search_nodes_timed(
                        graph=initial_graph,
                        search_backend=self.search_backend,
                        num_results_per_search=int(self.generation.get("num_results_per_search", 5)),
                        max_concurrent_searches=int(self.generation.get("max_concurrent_searches", 8)),
                    )),
                ),
            )
            search_seconds_total += time.perf_counter() - search_started

            revision_user_message = {
                "role": "user",
                "content": build_after_search_payload(
                    query=query,
                    graph=initial_graph,
                    search_results=initial_search_results,
                    max_fetches=int(self.generation.get("max_fetches", 4)),
                ),
            }
            revision_messages = [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {"role": "assistant", "content": initial_span.completion_text},
                revision_user_message,
            ]
            revision_prefill_ids = await self._tokenize_prompt_delta([revision_user_message])

            revision_span, revision_output, elapsed = await self._generate_turn(
                request_id=request_id,
                messages=revision_messages,
                prompt_ids=[*current_prompt_ids, *revision_prefill_ids],
                prompt_prefill_ids=revision_prefill_ids,
                sampling_params=sampling_params,
                max_new_tokens=int(self.generation.get("planner_revision_max_new_tokens", 1400)),
                role="planner",
                turn_type="planner_revision",
            )
            spans.append(revision_span)
            generated_seconds += elapsed
            if revision_output.num_preempted is not None:
                num_preempted = max(num_preempted, int(revision_output.num_preempted))

            try:
                revision_payload = extract_json_object(revision_span.completion_text)
                attempted_fetch_requests = (
                    [
                        row
                        for row in revision_payload.get("fetch_requests", [])
                        if isinstance(row, dict)
                    ]
                    if isinstance(revision_payload, dict) and isinstance(revision_payload.get("fetch_requests"), list)
                    else []
                )
                _, final_graph, fetch_requests = parse_planner_revision_output(
                    revision_span.completion_text,
                    current_graph=initial_graph,
                    available_search_results=initial_search_results,
                    max_fetches=int(self.generation.get("max_fetches", 4)),
                )
                ok, _ = validate_graph(final_graph)
                if not ok:
                    raise RuntimeError("final_graph_invalid")
            except Exception:
                payload = planner_zero_result(row=row, spans=spans, failure_reason="final_graph_invalid")
            else:
                post_search_started = time.perf_counter()
                search_executions, post_revision_search_timing = await self.loop.run_in_executor(
                    None,
                    lambda: _run_with_semaphore(
                        "planner_post_revision_search",
                        self.max_parallel_episodes,
                        lambda: (
                            lambda executions, timing: (executions, timing)
                        )(*_post_revision_search(
                            graph=final_graph,
                            search_backend=self.search_backend,
                            num_results_per_search=int(self.generation.get("num_results_per_search", 5)),
                            existing_search_executions=initial_search_executions,
                            max_concurrent_searches=int(self.generation.get("max_concurrent_searches", 8)),
                        )),
                    ),
                )
                search_seconds_total += time.perf_counter() - post_search_started
                final_search_results = normalize_search_results(search_executions)
                evidence_pool = evidence_pool_from_executions(search_executions)

                reward_started = time.perf_counter()
                planner_scores = await self.loop.run_in_executor(
                    None,
                    lambda: compute_planner_reward(
                        graph=final_graph,
                        rubrics=rubrics,
                        user_query=query,
                        evidence_pool=evidence_pool,
                        embedder=self.embedder,
                        fetch_requests=fetch_requests,
                        attempted_fetch_requests=attempted_fetch_requests,
                        search_results=initial_search_results,
                        config=self.reward_cfg,
                    ),
                )
                reward_seconds = time.perf_counter() - reward_started

                initial_search_seconds = _sum_timing_seconds(initial_search_timing)
                post_revision_search_seconds = _sum_timing_seconds(post_revision_search_timing)
                payload = {
                    "query_id": str(row.get("query_id", "")),
                    "query": query,
                    "assistant_transcript": planner_zero_result(
                        row=row,
                        spans=spans,
                        failure_reason="",
                    )["assistant_transcript"],
                    "planner_reward": float(planner_scores.get("planner_reward", 0.0)),
                    "rubric_coverage": float(planner_scores.get("rubric_coverage", 0.0)),
                    "decomposition_distinctness": float(planner_scores.get("decomposition_distinctness", 0.0)),
                    "search_rubric_coverage": float(planner_scores.get("search_rubric_coverage", 0.0)),
                    "search_distinctness": float(planner_scores.get("search_distinctness", 0.0)),
                    "node_count": float(planner_scores.get("node_count", 0.0)),
                    "non_answer_node_count": float(planner_scores.get("non_answer_node_count", 0.0)),
                    "edge_count": float(planner_scores.get("edge_count", 0.0)),
                    "graph_edge_density": float(planner_scores.get("graph_edge_density", 0.0)),
                    "search_node_count": float(planner_scores.get("search_node_count", 0.0)),
                    "aggregate_node_count": float(planner_scores.get("aggregate_node_count", 0.0)),
                    "answer_direct_parent_count": float(planner_scores.get("answer_direct_parent_count", 0.0)),
                    "answer_direct_aggregate_parent_count": float(
                        planner_scores.get("answer_direct_aggregate_parent_count", 0.0)
                    ),
                    "answer_direct_aggregate_parent_ratio": float(
                        planner_scores.get("answer_direct_aggregate_parent_ratio", 0.0)
                    ),
                    "answer_level_branch_count": float(planner_scores.get("answer_level_branch_count", 0.0)),
                    "answer_level_declared_branch_count": float(
                        planner_scores.get("answer_level_declared_branch_count", 0.0)
                    ),
                    "aggregate_parent_mean": float(planner_scores.get("aggregate_parent_mean", 0.0)),
                    "aggregate_parent_max": float(planner_scores.get("aggregate_parent_max", 0.0)),
                    "search_to_aggregate_edge_count": float(
                        planner_scores.get("search_to_aggregate_edge_count", 0.0)
                    ),
                    "aggregate_to_aggregate_edge_count": float(
                        planner_scores.get("aggregate_to_aggregate_edge_count", 0.0)
                    ),
                    "search_branch_reach_mean": float(planner_scores.get("search_branch_reach_mean", 0.0)),
                    "search_branch_reach_max": float(planner_scores.get("search_branch_reach_max", 0.0)),
                    "search_multi_branch_count": float(planner_scores.get("search_multi_branch_count", 0.0)),
                    "graph_expressiveness": float(planner_scores.get("graph_expressiveness", 0.0)),
                    "synthesis_shape": float(planner_scores.get("synthesis_shape", 0.0)),
                    "cross_branch_structure": float(planner_scores.get("cross_branch_structure", 0.0)),
                    "search_sufficiency": float(planner_scores.get("search_sufficiency", 0.0)),
                    "search_quality": float(planner_scores.get("search_quality", 0.0)),
                    "search_diversity": float(planner_scores.get("search_diversity", 0.0)),
                    "query_specificity": float(planner_scores.get("query_specificity", 0.0)),
                    "fetch_url_validity": float(planner_scores.get("fetch_url_validity", 0.0)),
                    "synthesis_parent_count_pass": float(
                        planner_scores.get("synthesis_parent_count_pass", 0.0)
                    ),
                    "synthesis_parent_distinctness_pass": float(
                        planner_scores.get("synthesis_parent_distinctness_pass", 0.0)
                    ),
                    "answer_parent_max_similarity": float(planner_scores.get("answer_parent_max_similarity", 1.0)),
                    "answer_parent_count": float(planner_scores.get("answer_parent_count", 0.0)),
                    "answer_aggregate_parent_count": float(
                        planner_scores.get("answer_aggregate_parent_count", 0.0)
                    ),
                    "aggregate_nodes_with_multi_parent_count": float(
                        planner_scores.get("aggregate_nodes_with_multi_parent_count", 0.0)
                    ),
                    "reused_search_node_count": float(planner_scores.get("reused_search_node_count", 0.0)),
                    "target_search_count": float(planner_scores.get("target_search_count", 0.0)),
                    "invalid_reason": float(planner_scores.get("invalid_reason", 0.0)),
                    "gate_failed": 0.0,
                    "search_seconds_initial": initial_search_seconds,
                    "search_seconds_post_revision": post_revision_search_seconds,
                    "search_seconds_total": initial_search_seconds + post_revision_search_seconds,
                    "fetch_requests_selected_count": float(len(fetch_requests)),
                    "fetch_requests_attempted_count": float(len(attempted_fetch_requests)),
                    "search_nodes_final_count": float(
                        sum(
                            1
                            for node in final_graph.get("nodes", [])
                            if isinstance(node, dict) and str(node.get("type", "")) == "search"
                        )
                    ),
                    "failure_reason": "",
                }
                payload["summary_json"] = json.dumps(
                    planner_reward_extra_info(payload),
                    ensure_ascii=False,
                )

        if self.rollout_logger is not None:
            self.rollout_logger.append(
                planner_rollout_record(
                    row=row,
                    spans=spans,
                    payload=payload,
                    full_artifact={
                        "initial_graph": initial_graph,
                        "final_graph": final_graph,
                        "search_node_outputs": search_node_outputs,
                        "search_results": final_search_results or initial_search_results,
                        "fetch_requests": fetch_requests,
                        "failure_reason": str(payload.get("failure_reason", "")),
                    },
                )
            )

        packed = pack_spans_for_verl(
            spans=spans,
            bos_token_id=self.tokenizer.bos_token_id,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )
        metrics = AgentLoopMetrics(
            generate_sequences=generated_seconds,
            tool_calls=search_seconds_total,
            compute_score=reward_seconds,
            num_preempted=num_preempted,
        )
        extra_fields = {
            "assistant_transcript": str(payload.get("assistant_transcript", "")),
            "summary_json": str(payload.get("summary_json", "{}")),
            "query_id": str(payload.get("query_id", "")),
            "failure_reason": str(payload.get("failure_reason", "")),
            "turn_scores": [],
            "tool_rewards": [],
            "reward_extra_info": planner_reward_extra_info(payload),
        }

        return AgentLoopOutput(
            prompt_ids=packed.prompt_ids,
            response_ids=packed.response_ids,
            response_mask=packed.response_mask,
            response_logprobs=packed.response_logprobs,
            reward_score=float(payload.get("planner_reward", 0.0) or 0.0),
            num_turns=2 + len(spans),
            metrics=metrics,
            extra_fields=extra_fields,
        )


class DecomposerAnswererAgentLoop(_DecomposerAgentLoopBase):
    def __init__(
        self,
        *args,
        reward: dict[str, Any] | None = None,
        judge: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_cfg = AnswerRewardConfig(**(reward or {}))
        self.judge_cfg = dict(judge or {})
        self.judge_model = str(self.judge_cfg.get("model", "")).strip()
        self.judge_base_url = str(self.judge_cfg.get("base_url", "https://api.openai.com/v1"))
        self.judge_client = (
            _cached_openai_client(self.judge_base_url)
            if self.judge_model
            else None
        )
        self.judge_parallelism = int(self.judge_cfg.get("max_parallel_requests", 4))
        self.payload_variant = str(self.generation.get("payload_variant", "default") or "default").strip().lower()
        self.prompt_variant = str(self.generation.get("prompt_variant", "default") or "default").strip().lower()
        self.answerer_system_prompt = load_answerer_system_prompt(self.prompt_variant)

    @property
    def stage_name(self) -> str:
        return "answerer"

    async def _partial_credit_payload(
        self,
        *,
        artifact: dict[str, Any],
        final_graph: dict[str, Any],
        produced_outputs: dict[str, str],
        answer_node_id: str,
        evidence_pool: dict[str, Any],
        search_results: dict[str, Any],
        fetched_results: list[Any],
        query_id: str,
        query: str,
        spans: list[RolloutSpan],
        failure_reason: str,
    ) -> dict[str, Any]:
        """Compute partial credit from successfully completed execution waves."""
        branch_outputs = {
            nid: val for nid, val in produced_outputs.items()
            if nid != answer_node_id and str(val).strip()
        }
        if not branch_outputs:
            return answer_zero_result(
                row={"query_id": query_id, "query": query},
                spans=spans,
                failure_reason=failure_reason,
            )
        citation_catalog = artifact.get("citation_catalog", {})
        if not isinstance(citation_catalog, dict) or not citation_catalog:
            citation_catalog = build_citation_catalog(
                search_results=search_results,
                fetched_results=fetched_results,
            )
        _bo, _cc = branch_outputs, citation_catalog
        exec_scores = await self.loop.run_in_executor(
            None,
            lambda bo=_bo, cc=_cc: compute_execution_rewards(
                graph=final_graph,
                node_outputs=bo,
                evidence_pool=evidence_pool,
                citation_catalog=cc,
                embedder=self.embedder,
                config=self.reward_cfg,
            ),
        )
        exec_reward = float(exec_scores.get("execution_reward", 0.0))
        answer_scores = compute_answer_reward(
            execution_reward=exec_reward,
            synthesis_reward=0.0,
            config=self.reward_cfg,
        )
        kp_scores = exec_scores.get("key_point_match_scores", {})
        payload = answer_zero_result(
            row={"query_id": query_id, "query": query},
            spans=spans,
            failure_reason=failure_reason,
        )
        payload.update({
            "answer_reward": float(answer_scores.get("answer_reward", 0.0)),
            "content_reward": float(answer_scores.get("content_reward", 0.0)),
            "execution_reward": exec_reward,
            "key_point_match_mean": float(
                exec_scores.get(
                    "key_point_match_mean",
                    sum(float(v) for v in kp_scores.values()) / max(len(kp_scores), 1),
                )
            ),
            "embedding_keypoint_match_mean": float(exec_scores.get("embedding_keypoint_match_mean", 0.0)),
            "lexical_keypoint_recall_mean": float(exec_scores.get("lexical_keypoint_recall_mean", 0.0)),
            "branch_outputs_count": float(len(branch_outputs)),
        })
        payload["summary_json"] = json.dumps(
            answer_reward_extra_info(payload), ensure_ascii=False,
        )
        return payload

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        row = dict(kwargs)
        if not row.get("artifact_path"):
            raise ValueError("Answerer VERL row is missing artifact_path")
        artifact = _load_json_payload(str(row["artifact_path"]))
        query = str(artifact.get("query", row.get("query", "")))
        query_id = str(artifact.get("query_id", row.get("query_id", "")))
        rubrics = deserialize_rubrics(artifact.get("rubrics"))
        request_id = uuid4().hex
        spans: list[RolloutSpan] = []
        produced_outputs: dict[str, str] = {}
        history: list[dict[str, str]] = [{"role": "system", "content": self.answerer_system_prompt}]
        introduced_search_parent_ids: set[str] = set()
        current_prompt_ids: list[int] | None = None
        generated_seconds = 0.0
        reward_seconds = 0.0
        num_preempted = -1

        final_graph = artifact.get("final_graph", {}) if isinstance(artifact.get("final_graph"), dict) else {}
        search_results = artifact.get("search_results", {}) if isinstance(artifact.get("search_results"), dict) else {}
        fetched_results = (
            artifact.get("fetched_results", [])
            if isinstance(artifact.get("fetched_results"), list)
            else []
        )
        evidence_pool = artifact.get("evidence_pool", {}) if isinstance(artifact.get("evidence_pool"), dict) else {}

        ok, _ = validate_graph(final_graph)
        if not ok:
            payload = answer_zero_result(row={"query_id": query_id, "query": query}, spans=spans, failure_reason="final_graph_invalid")
        else:
            answer_node_id = next(
                (
                    str(node.get("id", ""))
                    for node in final_graph.get("nodes", [])
                    if isinstance(node, dict) and str(node.get("type", "")) == "answer"
                ),
                "",
            )
            waves = aggregate_execution_waves(final_graph)
            payload = None

            for wave_index, node_ids in enumerate(waves, start=1):
                if self.payload_variant == "compact":
                    user_payload = build_execution_turn_payload_compact(
                        query=query,
                        graph=final_graph,
                        node_ids=list(node_ids),
                        search_results=search_results,
                        fetched_results=fetched_results,
                        produced_outputs=produced_outputs,
                        wave_index=wave_index,
                        introduced_search_parent_ids=introduced_search_parent_ids,
                    )
                else:
                    user_payload = build_execution_turn_payload(
                        query=query,
                        graph=final_graph,
                        node_ids=list(node_ids),
                        search_results=search_results,
                        fetched_results=fetched_results,
                        produced_outputs=produced_outputs,
                        wave_index=wave_index,
                    )
                messages = [*history, {"role": "user", "content": user_payload}]
                turn_prefill_ids: list[int] = []
                turn_prompt_ids: list[int] | None = None
                if current_prompt_ids is not None:
                    turn_prefill_ids = await self._tokenize_prompt_delta([{"role": "user", "content": user_payload}])
                    turn_prompt_ids = [*current_prompt_ids, *turn_prefill_ids]
                span, output, elapsed = await self._generate_turn(
                    request_id=request_id,
                    messages=messages,
                    prompt_ids=turn_prompt_ids,
                    prompt_prefill_ids=turn_prefill_ids,
                    sampling_params=sampling_params,
                    max_new_tokens=int(self.generation.get("answer_execution_max_new_tokens", 4500)),
                    role="answerer",
                    turn_type="answer_execution",
                    target_node_ids=list(node_ids),
                )
                spans.append(span)
                generated_seconds += elapsed
                if output.num_preempted is not None:
                    num_preempted = max(num_preempted, int(output.num_preempted))
                current_prompt_ids = [*span.prompt_ids, *span.generated_ids]

                try:
                    parsed = parse_answerer_node_outputs(span.completion_text, expected_node_ids=list(node_ids))
                except Exception:
                    parsed = None

                if parsed is None:
                    payload = await self._partial_credit_payload(
                        artifact=artifact,
                        final_graph=final_graph,
                        produced_outputs=produced_outputs,
                        answer_node_id=answer_node_id,
                        evidence_pool=evidence_pool,
                        search_results=search_results,
                        fetched_results=fetched_results,
                        query_id=query_id,
                        query=query,
                        spans=spans,
                        failure_reason="answer_execution_parse_failed",
                    )
                    break

                produced_outputs.update({str(key): str(value) for key, value in parsed.items()})
                if self.payload_variant == "compact":
                    introduced_search_parent_ids.update(search_dependency_ids_for_wave(final_graph, node_ids))
                history.extend(
                    [
                        {"role": "user", "content": user_payload},
                        {"role": "assistant", "content": span.completion_text},
                    ]
                )

            if payload is None:
                if self.payload_variant == "compact":
                    user_payload = build_final_answer_payload_compact(
                        query=query,
                        graph=final_graph,
                        produced_outputs=produced_outputs,
                    )
                else:
                    user_payload = build_final_answer_payload(
                        query=query,
                        graph=final_graph,
                        produced_outputs=produced_outputs,
                    )
                messages = [*history, {"role": "user", "content": user_payload}]
                final_prefill_ids = await self._tokenize_prompt_delta([{"role": "user", "content": user_payload}])
                span, output, elapsed = await self._generate_turn(
                    request_id=request_id,
                    messages=messages,
                    prompt_ids=[*(current_prompt_ids or []), *final_prefill_ids] if current_prompt_ids is not None else None,
                    prompt_prefill_ids=final_prefill_ids if current_prompt_ids is not None else [],
                    sampling_params=sampling_params,
                    max_new_tokens=int(self.generation.get("answer_final_max_new_tokens", 5500)),
                    role="answerer",
                    turn_type="answer_final",
                    target_node_ids=[answer_node_id],
                )
                spans.append(span)
                generated_seconds += elapsed
                if output.num_preempted is not None:
                    num_preempted = max(num_preempted, int(output.num_preempted))

                try:
                    parsed = parse_answerer_node_outputs(span.completion_text, expected_node_ids=[answer_node_id])
                    final_answer = str(parsed.get(answer_node_id, "")).strip()
                    if not final_answer:
                        raise RuntimeError("empty_final_answer")
                except Exception:
                    final_answer = ""

                if not final_answer:
                    payload = await self._partial_credit_payload(
                        artifact=artifact,
                        final_graph=final_graph,
                        produced_outputs=produced_outputs,
                        answer_node_id=answer_node_id,
                        evidence_pool=evidence_pool,
                        search_results=search_results,
                        fetched_results=fetched_results,
                        query_id=query_id,
                        query=query,
                        spans=spans,
                        failure_reason="answer_final_parse_failed",
                    )
                else:
                    produced_outputs[answer_node_id] = final_answer
                    history.extend(
                        [
                            {"role": "user", "content": user_payload},
                            {"role": "assistant", "content": span.completion_text},
                        ]
                    )
                    branch_outputs = {
                        node_id: value
                        for node_id, value in produced_outputs.items()
                        if node_id != answer_node_id and str(value).strip()
                    }
                    citation_catalog = artifact.get("citation_catalog", {})
                    if not isinstance(citation_catalog, dict) or not citation_catalog:
                        citation_catalog = build_citation_catalog(
                            search_results=search_results,
                            fetched_results=fetched_results,
                        )
                    final_answer_for_judge = append_citation_block(final_answer, citation_catalog)

                    reward_started = time.perf_counter()
                    _bo_exec, _cc_exec = branch_outputs, citation_catalog
                    exec_future = self.loop.run_in_executor(
                        None,
                        lambda bo=_bo_exec, cc=_cc_exec: compute_execution_rewards(
                            graph=final_graph,
                            node_outputs=bo,
                            evidence_pool=evidence_pool,
                            citation_catalog=cc,
                            embedder=self.embedder,
                            config=self.reward_cfg,
                        ),
                    )
                    _bo_branch = branch_outputs
                    branch_utilization_future = self.loop.run_in_executor(
                        None,
                        lambda bo=_bo_branch: compute_branch_utilization(
                            final_answer=final_answer,
                            branch_outputs=bo,
                        ),
                    )

                    answer_judge = 0.0
                    if self.judge_client is not None and self.judge_model and rubrics:
                        _fa = final_answer_for_judge
                        _q, _r = query, rubrics

                        def _answer_judge_call(fa=_fa, q=_q, r=_r) -> float:
                            return _run_with_semaphore(
                                "answer_reward_judge",
                                self.judge_parallelism,
                                lambda: run_answer_judge(
                                    client=self.judge_client,
                                    model=self.judge_model,
                                    system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
                                    query=q,
                                    answer=fa,
                                    rubrics=r,
                                ),
                            )

                        judge_future = self.loop.run_in_executor(None, _answer_judge_call)
                        execution_scores, branch_utilization_scores, answer_judge = await asyncio.gather(
                            exec_future, branch_utilization_future, judge_future,
                        )
                        answer_judge = float(answer_judge)
                    else:
                        execution_scores, branch_utilization_scores = await asyncio.gather(
                            exec_future, branch_utilization_future,
                        )

                    branch_utilization = float(branch_utilization_scores.get("branch_utilization", 0.0))
                    synthesis_scores = compute_synthesis_reward(
                        answer_judge=answer_judge,
                        branch_utilization=branch_utilization,
                        config=self.reward_cfg,
                    )
                    reward_seconds = time.perf_counter() - reward_started

                    answer_scores = compute_answer_reward(
                        execution_reward=float(execution_scores.get("execution_reward", 0.0)),
                        synthesis_reward=float(synthesis_scores.get("synthesis_reward", 0.0)),
                        config=self.reward_cfg,
                    )
                    key_point_match_scores = execution_scores.get("key_point_match_scores", {})
                    payload = {
                        "query_id": query_id,
                        "query": query,
                        "assistant_transcript": answer_zero_result(
                            row={"query_id": query_id, "query": query},
                            spans=spans,
                            failure_reason="",
                        )["assistant_transcript"],
                        "final_answer": final_answer,
                        "answer_reward": float(answer_scores.get("answer_reward", 0.0)),
                        "content_reward": float(answer_scores.get("content_reward", 0.0)),
                        "valid_parse_bonus": float(answer_scores.get("valid_parse_bonus", 0.0)),
                        "valid_parse": float(answer_scores.get("valid_parse", 1.0)),
                        "execution_reward": float(answer_scores.get("execution_reward", 0.0)),
                        "synthesis_reward": float(answer_scores.get("synthesis_reward", 0.0)),
                        "key_point_match_mean": float(
                            execution_scores.get(
                                "key_point_match_mean",
                                sum(float(value) for value in key_point_match_scores.values())
                                / max(len(key_point_match_scores), 1),
                            )
                        ),
                        "embedding_keypoint_match_mean": float(
                            execution_scores.get("embedding_keypoint_match_mean", 0.0)
                        ),
                        "lexical_keypoint_recall_mean": float(
                            execution_scores.get("lexical_keypoint_recall_mean", 0.0)
                        ),
                        "answer_judge": answer_judge,
                        "branch_utilization": branch_utilization,
                        "branch_outputs_count": float(len(branch_outputs)),
                        "answer_words": float(len(final_answer.split())),
                        "answer_with_citations_words": float(len(final_answer_for_judge.split())),
                        "gate_failed": 0.0,
                        "failure_reason": "",
                    }
                    payload["summary_json"] = json.dumps(
                        answer_reward_extra_info(payload),
                        ensure_ascii=False,
                    )

        if self.rollout_logger is not None:
            self.rollout_logger.append(
                answer_rollout_record(
                    row={"query_id": query_id, "query": query},
                    spans=spans,
                    payload=payload,
                    full_artifact={
                        "final_graph": final_graph,
                        "search_results": search_results,
                        "fetched_results": fetched_results,
                        "produced_outputs": produced_outputs,
                        "branch_outputs": {
                            key: value
                            for key, value in produced_outputs.items()
                            if key != next(
                                (
                                    str(node.get("id", ""))
                                    for node in final_graph.get("nodes", [])
                                    if isinstance(node, dict) and str(node.get("type", "")) == "answer"
                                ),
                                "",
                            )
                        },
                        "failure_reason": str(payload.get("failure_reason", "")),
                    },
                )
            )

        packed = pack_spans_for_verl(
            spans=spans,
            bos_token_id=self.tokenizer.bos_token_id,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )
        metrics = AgentLoopMetrics(
            generate_sequences=generated_seconds,
            tool_calls=0.0,
            compute_score=reward_seconds,
            num_preempted=num_preempted,
        )
        extra_fields = {
            "assistant_transcript": str(payload.get("assistant_transcript", "")),
            "summary_json": str(payload.get("summary_json", "{}")),
            "query_id": str(payload.get("query_id", "")),
            "final_answer": str(payload.get("final_answer", "")),
            "failure_reason": str(payload.get("failure_reason", "")),
            "turn_scores": [],
            "tool_rewards": [],
            "reward_extra_info": answer_reward_extra_info(payload),
        }

        return AgentLoopOutput(
            prompt_ids=packed.prompt_ids,
            response_ids=packed.response_ids,
            response_mask=packed.response_mask,
            response_logprobs=packed.response_logprobs,
            reward_score=float(payload.get("answer_reward", 0.0) or 0.0),
            num_turns=2 + len(spans),
            metrics=metrics,
            extra_fields=extra_fields,
        )


def _post_revision_search(
    *,
    graph: dict[str, Any],
    search_backend: Any,
    num_results_per_search: int,
    existing_search_executions: tuple[Any, ...],
    max_concurrent_searches: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    (
        _,
        search_executions,
        post_revision_search_timing,
        _,
        _,
        _,
    ) = execute_revision_retrievals(
        graph=graph,
        search_backend=search_backend,
        num_results_per_search=num_results_per_search,
        existing_search_executions=existing_search_executions,
        fetch_requests=[],
        max_fetch_characters=1,
        max_concurrent_searches=max_concurrent_searches,
        max_concurrent_fetches=1,
    )
    return tuple(search_executions), post_revision_search_timing
