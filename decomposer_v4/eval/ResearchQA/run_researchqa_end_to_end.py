#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.eval.ResearchQA.researchqa_loader import (
    export_researchqa_split,
    load_researchqa_split,
    researchqa_item_to_dict,
)
from decomposer_v4.utils.io import ROOT, read_json, write_json

if TYPE_CHECKING:
    from decomposer_v4.train.model_backends import ModelProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DecomposeR planner+answerer trajectories for ResearchQA, export the official response map, "
            "and optionally compute rubric coverage."
        )
    )
    parser.add_argument("--split", default="test_mini", choices=("test_mini", "valid", "test", "full"))
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows in the selected split.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cache-dir", default="")

    parser.add_argument("--run-name", default="", help="Name for the output directory. Defaults to the planner profile/model.")
    parser.add_argument("--output-root", default="", help="Defaults to outputs/eval/researchqa/<split>/<run-name>")
    parser.add_argument("--dataset-json", default="", help="Path to write the selected split snapshot as JSON.")
    parser.add_argument("--responses-json", default="", help="Path to write the ResearchQA response map with citations.")
    parser.add_argument("--responses-plain-json", default="", help="Path to write the response map without appended citation blocks.")
    parser.add_argument("--trajectory-dir", default="", help="Directory for per-item trajectory JSON files.")
    parser.add_argument("--summary-json", default="", help="Path for run summary JSON.")
    parser.add_argument("--coverage-output", default="", help="Path for rubric coverage JSON.")
    parser.add_argument("--overwrite-existing", action="store_true")

    parser.add_argument("--planner-profile", default="")
    parser.add_argument("--answerer-profile", default="")
    parser.add_argument("--planner-model", default="")
    parser.add_argument("--answerer-model", default="")
    parser.add_argument(
        "--planner-backend",
        default="hf_transformers",
        choices=("hf_transformers", "vllm", "openai_chat"),
    )
    parser.add_argument(
        "--answerer-backend",
        default="hf_transformers",
        choices=("hf_transformers", "vllm", "openai_chat"),
    )
    parser.add_argument("--planner-temperature", type=float, default=0.2)
    parser.add_argument("--answerer-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="")

    parser.add_argument("--num-results-per-search", type=int, default=5)
    parser.add_argument("--max-fetches", type=int, default=4)
    parser.add_argument("--max-fetch-characters", type=int, default=6000)
    parser.add_argument("--max-concurrent-searches", type=int, default=4)
    parser.add_argument("--max-concurrent-fetches", type=int, default=4)
    parser.add_argument("--omit-answerer-system-prompt", action="store_true")
    parser.add_argument("--answerer-payload-variant", choices=("compact", "default"), default="compact")
    parser.add_argument("--answerer-prompt-variant", choices=("compact", "default"), default="compact")
    parser.add_argument("--answer-execution-max-completion-tokens", type=int, default=0)
    parser.add_argument("--answer-final-max-completion-tokens", type=int, default=0)

    parser.add_argument("--run-coverage", action="store_true")
    parser.add_argument("--coverage-model", default="gpt-4.1-mini")
    parser.add_argument("--coverage-batch-size", type=int, default=8)
    return parser.parse_args()


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "researchqa-run"


def _default_run_name(args: argparse.Namespace) -> str:
    if str(args.run_name).strip():
        return _slugify(str(args.run_name).strip())
    if str(args.planner_profile).strip():
        return _slugify(Path(str(args.planner_profile).strip()).stem)
    if str(args.planner_model).strip():
        return _slugify(str(args.planner_model).strip().split("/")[-1])
    return f"researchqa-{args.split}"


def _profile_from_args(
    *,
    profile_path: str,
    model_name: str,
    backend_name: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    api_key_env: str,
    base_url: str,
    device_map: str,
    torch_dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> ModelProfile:
    from decomposer_v4.train.model_backends import ModelProfile, load_model_profile

    if str(profile_path).strip():
        profile = load_model_profile(str(_resolve_repo_path(profile_path)))
        if str(model_name).strip():
            return ModelProfile(
                backend=profile.backend,
                model=str(model_name).strip(),
                base_url=profile.base_url,
                api_key_env=profile.api_key_env,
                reasoning_effort=profile.reasoning_effort,
                verbosity=profile.verbosity,
                temperature=profile.temperature,
                top_p=profile.top_p,
                max_completion_tokens=profile.max_completion_tokens,
                trust_remote_code=profile.trust_remote_code,
                device_map=profile.device_map,
                torch_dtype=profile.torch_dtype,
                tensor_parallel_size=profile.tensor_parallel_size,
                gpu_memory_utilization=profile.gpu_memory_utilization,
                max_model_len=profile.max_model_len,
                seed=profile.seed,
            )
        return profile
    if not str(model_name).strip():
        raise ValueError("Provide either a profile path or a model name.")
    return ModelProfile(
        backend=str(backend_name).strip(),
        model=str(model_name).strip(),
        base_url=str(base_url).strip(),
        api_key_env=str(api_key_env).strip() or "OPENAI_API_KEY",
        temperature=float(temperature),
        top_p=float(top_p),
        max_completion_tokens=int(max_completion_tokens),
        trust_remote_code=True,
        device_map=str(device_map).strip() or "auto",
        torch_dtype=str(torch_dtype).strip() or "bfloat16",
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        seed=int(seed),
    )


def _build_backend(profile: ModelProfile) -> Any:
    from decomposer_v4.train.model_backends import HFTransformersBackend, OpenAIChatBackend, VLLMChatBackend

    if profile.backend == "hf_transformers":
        return HFTransformersBackend(profile)
    if profile.backend == "vllm":
        return VLLMChatBackend(profile)
    if profile.backend == "openai_chat":
        return OpenAIChatBackend(profile)
    raise ValueError(f"Unsupported backend: {profile.backend}")


def _backend_key(profile: ModelProfile) -> tuple[Any, ...]:
    return (
        profile.backend,
        profile.model,
        profile.base_url,
        profile.api_key_env,
        profile.reasoning_effort,
        profile.verbosity,
        profile.temperature,
        profile.top_p,
        profile.max_completion_tokens,
        profile.trust_remote_code,
        profile.device_map,
        profile.torch_dtype,
        profile.tensor_parallel_size,
        profile.gpu_memory_utilization,
        profile.max_model_len,
        profile.seed,
    )


def _build_search_backend() -> Any:
    from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient

    return MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )


def _item_query_id(item: dict[str, Any]) -> str:
    return str(item.get("id", "")).strip()


def _item_query_text(item: dict[str, Any]) -> str:
    return str(item.get("query", "")).strip()


def _inline_citation_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in re.finditer(r"\[(N[0-9A-Za-z_]+-(?:R|F)\d+)\]", str(text or "")):
        citation_id = str(match.group(1) or "").strip()
        if citation_id and citation_id not in seen:
            seen.append(citation_id)
    return seen


def _selected_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items = [researchqa_item_to_dict(item) for item in load_researchqa_split(split=args.split, cache_dir=args.cache_dir or None)]
    start = max(int(args.offset), 0)
    if int(args.limit) > 0:
        items = items[start : start + int(args.limit)]
    else:
        items = items[start:]
    if not items:
        raise RuntimeError("No ResearchQA items selected.")
    return items


def _load_existing_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _timing_summary(timings: dict[str, Any]) -> str:
    execution_turns = timings.get("answer_execution_turns", {})
    if isinstance(execution_turns, dict):
        answer_execution_seconds = sum(
            float(row.get("seconds", 0.0) or 0.0)
            for row in execution_turns.values()
            if isinstance(row, dict)
        )
    else:
        answer_execution_seconds = 0.0
    pieces = [
        ("total", timings.get("total_seconds", 0.0)),
        ("plan1", timings.get("planner_initial_seconds", 0.0)),
        ("search1", timings.get("initial_search_seconds", timings.get("search_seconds", 0.0))),
        ("plan2", timings.get("planner_revision_seconds", 0.0)),
        ("retrieve2", timings.get("post_revision_retrieval_seconds", 0.0)),
        ("ans_exec", answer_execution_seconds),
        ("ans_final", timings.get("answer_final_seconds", 0.0)),
    ]
    return " ".join(f"{name}={float(value or 0.0):.1f}s" for name, value in pieces)


def main() -> None:
    args = parse_args()
    from decomposer_v4.data.generation.generate_live_trajectory import build_trajectory

    run_name = _default_run_name(args)
    output_root = (
        _resolve_repo_path(args.output_root)
        if str(args.output_root).strip()
        else (ROOT / "outputs" / "eval" / "researchqa" / args.split / run_name).resolve()
    )
    dataset_json = _resolve_repo_path(args.dataset_json) if str(args.dataset_json).strip() else output_root / f"{args.split}.json"
    responses_json = _resolve_repo_path(args.responses_json) if str(args.responses_json).strip() else output_root / "responses_with_citations.json"
    responses_plain_json = (
        _resolve_repo_path(args.responses_plain_json)
        if str(args.responses_plain_json).strip()
        else output_root / "responses_plain.json"
    )
    trajectory_dir = _resolve_repo_path(args.trajectory_dir) if str(args.trajectory_dir).strip() else output_root / "trajectories"
    summary_json = _resolve_repo_path(args.summary_json) if str(args.summary_json).strip() else output_root / "summary.json"
    coverage_output = _resolve_repo_path(args.coverage_output) if str(args.coverage_output).strip() else output_root / "coverage.json"

    items = _selected_items(args)
    export_researchqa_split(args.split, dataset_json, cache_dir=args.cache_dir or None)
    if int(args.offset) > 0 or int(args.limit) > 0:
        write_json(dataset_json, items)

    planner_profile = _profile_from_args(
        profile_path=args.planner_profile,
        model_name=args.planner_model,
        backend_name=args.planner_backend,
        temperature=args.planner_temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    answerer_profile = _profile_from_args(
        profile_path=args.answerer_profile or args.planner_profile,
        model_name=args.answerer_model or args.planner_model,
        backend_name=args.answerer_backend,
        temperature=args.answerer_temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    backend_cache: dict[tuple[Any, ...], Any] = {}

    def get_backend(profile: ModelProfile) -> Any:
        key = _backend_key(profile)
        backend = backend_cache.get(key)
        if backend is None:
            backend = _build_backend(profile)
            backend_cache[key] = backend
        return backend

    planner_backend = get_backend(planner_profile)
    answerer_backend = get_backend(answerer_profile)
    search_backend = _build_search_backend()

    print(
        "Backends: "
        f"planner={planner_profile.backend} answerer={answerer_profile.backend} "
        f"model={planner_profile.model}",
        flush=True,
    )

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    responses_with_citations = _load_existing_map(responses_json) if not args.overwrite_existing else {}
    responses_plain = _load_existing_map(responses_plain_json) if not args.overwrite_existing else {}
    summary_rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        query_id = _item_query_id(item)
        query = _item_query_text(item)
        trajectory_path = trajectory_dir / f"{query_id}.json"
        print(f"[{index}/{len(items)}] {query_id}", flush=True)

        if trajectory_path.exists() and not args.overwrite_existing:
            trajectory = read_json(trajectory_path)
        else:
            trajectory = build_trajectory(
                row={
                    "query_id": query_id,
                    "query": query,
                    "task_type": "deep_research",
                    "metadata": item,
                },
                planner_backend=planner_backend,
                answerer_backend=answerer_backend,
                search_backend=search_backend,
                num_results_per_search=int(args.num_results_per_search),
                max_fetches=int(args.max_fetches),
                max_fetch_characters=int(args.max_fetch_characters),
                max_concurrent_searches=int(args.max_concurrent_searches),
                max_concurrent_fetches=int(args.max_concurrent_fetches),
                include_answerer_system_prompt=not bool(args.omit_answerer_system_prompt),
                answerer_payload_variant=str(args.answerer_payload_variant),
                answerer_prompt_variant=str(args.answerer_prompt_variant),
                answer_execution_max_completion_tokens=(
                    int(args.answer_execution_max_completion_tokens)
                    if int(args.answer_execution_max_completion_tokens) > 0
                    else None
                ),
                answer_final_max_completion_tokens=(
                    int(args.answer_final_max_completion_tokens)
                    if int(args.answer_final_max_completion_tokens) > 0
                    else None
                ),
            )
            write_json(trajectory_path, trajectory)

        error = str(trajectory.get("error", "") or "").strip()
        if error:
            print(f"  error: {error}", flush=True)
        timings = trajectory.get("timings", {})
        if isinstance(timings, dict):
            print(f"  timings: {_timing_summary(timings)}", flush=True)

        answer_with_citations = str(trajectory.get("final_answer_with_citations", "") or "").strip()
        answer_plain = str(trajectory.get("final_answer", "") or "").strip()
        if not answer_with_citations:
            answer_with_citations = answer_plain
        used_citation_ids = _inline_citation_ids(answer_plain)
        final_citation_block = str(trajectory.get("final_answer_citation_block", "") or "").strip()

        common_payload = {
            "id": query_id,
            "query": query,
            "field": str(item.get("field", "")).strip(),
            "general_domain": str(item.get("general_domain", "")).strip(),
            "subdomain": str(item.get("subdomain", "")).strip(),
            "date": str(item.get("date", "")).strip(),
            "error": error,
            "trajectory_path": str(trajectory_path.resolve()),
            "citation_count": int(trajectory.get("timings", {}).get("citation_count", 0) or 0),
            "used_citation_count": len(used_citation_ids),
            "used_citation_ids": used_citation_ids,
            "has_citation_block": bool(final_citation_block),
        }
        responses_with_citations[query_id] = {
            **common_payload,
            "answer": answer_with_citations,
            "answer_without_citations": answer_plain,
        }
        responses_plain[query_id] = {
            **common_payload,
            "answer": answer_plain,
        }
        write_json(responses_json, responses_with_citations)
        write_json(responses_plain_json, responses_plain)

        summary_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "field": str(item.get("field", "")).strip(),
                "error": error,
                "total_seconds": float(trajectory.get("timings", {}).get("total_seconds", 0.0) or 0.0),
                "citation_count": int(trajectory.get("timings", {}).get("citation_count", 0) or 0),
                "used_citation_count": len(used_citation_ids),
                "has_citation_block": bool(final_citation_block),
                "trajectory_path": str(trajectory_path.resolve()),
            }
        )

    write_json(
        summary_json,
        {
            "split": args.split,
            "count": len(items),
            "run_name": run_name,
            "planner_profile": planner_profile.__dict__,
            "answerer_profile": answerer_profile.__dict__,
            "dataset_json": str(dataset_json.resolve()),
            "responses_json": str(responses_json.resolve()),
            "responses_plain_json": str(responses_plain_json.resolve()),
            "trajectory_dir": str(trajectory_dir.resolve()),
            "coverage_output": str(coverage_output.resolve()),
            "rows": summary_rows,
        },
    )

    if args.run_coverage:
        from decomposer_v4.eval.ResearchQA.compute_coverage import compute_coverage

        compute_coverage(
            data_path=str(dataset_json),
            response_map_path=str(responses_json),
            output_path=str(coverage_output),
            batch_size=int(args.coverage_batch_size),
            model=str(args.coverage_model).strip(),
        )
        print(f"Wrote coverage: {coverage_output.resolve()}", flush=True)

    print(f"Wrote dataset snapshot: {dataset_json.resolve()}")
    print(f"Wrote ResearchQA responses: {responses_json.resolve()}")
    print(f"Wrote plain responses: {responses_plain_json.resolve()}")
    print(f"Wrote trajectories: {trajectory_dir.resolve()}")
    print(f"Wrote summary: {summary_json.resolve()}")


if __name__ == "__main__":
    main()
