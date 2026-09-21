#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer_v4.data.generation.generate_live_trajectory import build_trajectory
from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.utils.io import ROOT, read_jsonl, write_json, write_jsonl

if TYPE_CHECKING:
    from decomposer_v4.train.model_backends import ModelProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full DecomposeR planner -> search -> planner revision -> fetch -> answerer "
            "trajectory and emit benchmark-ready reports."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--query", default="", help="Run a single ad hoc query.")
    source.add_argument(
        "--input-path",
        default="",
        help="Input JSONL file containing benchmark prompts, e.g. Deep Research Bench query_en.jsonl.",
    )
    parser.add_argument(
        "--only-language",
        default="",
        help="If set, keep only rows whose `language`/`lang` field matches this code.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows.")
    parser.add_argument("--offset", type=int, default=0)

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
    parser.add_argument("--planner-prompt-variant", default="default")
    parser.add_argument("--answerer-payload-variant", choices=("compact", "default"), default="compact")
    parser.add_argument("--answerer-prompt-variant", default="compact")
    parser.add_argument("--answer-execution-max-completion-tokens", type=int, default=0)
    parser.add_argument("--answer-final-max-completion-tokens", type=int, default=0)
    parser.add_argument("--disable-citation-appendix", action="store_true")

    parser.add_argument(
        "--benchmark-format",
        default="deep_research_bench",
        choices=("deep_research_bench", "generic"),
        help="Controls the schema of the final report JSONL.",
    )
    parser.add_argument(
        "--report-answer-field",
        choices=("final_answer", "final_answer_with_citations"),
        default="final_answer_with_citations",
        help="Which trajectory answer field to write into the benchmark report JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/eval/reports.jsonl",
        help="Benchmark-ready JSONL output path.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default="outputs/eval/trajectories",
        help="Directory for detailed per-query trajectory JSON files.",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/eval/summary.json",
        help="Summary manifest path.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _normalize_language(row: dict[str, Any]) -> str:
    for key in ("language", "lang", "locale"):
        value = str(row.get(key, "")).strip().lower()
        if value:
            return value
    return ""


def _row_query_text(row: dict[str, Any]) -> str:
    for key in ("query", "prompt", "question", "_question_text"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _row_query_id(row: dict[str, Any], fallback_query: str) -> str:
    for key in ("query_id", "id", "record_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return hashlib.md5(fallback_query.encode("utf-8")).hexdigest()[:16]


def _row_benchmark_id(row: dict[str, Any], fallback_query: str) -> Any:
    for key in ("query_id", "id", "record_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return hashlib.md5(fallback_query.encode("utf-8")).hexdigest()[:16]


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.query.strip():
        return [{"id": "adhoc_query", "prompt": str(args.query).strip()}]
    if not args.input_path.strip():
        raise ValueError("Provide either --query or --input-path.")

    rows = read_jsonl(_resolve_repo_path(args.input_path))
    filtered = [row for row in rows if _row_query_text(row)]
    if args.only_language.strip():
        target_language = str(args.only_language).strip().lower()
        filtered = [row for row in filtered if _normalize_language(row) == target_language]
    start = max(int(args.offset), 0)
    if int(args.limit) > 0:
        filtered = filtered[start : start + int(args.limit)]
    else:
        filtered = filtered[start:]
    if not filtered:
        raise RuntimeError("No input rows selected.")
    return filtered


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
    from decomposer_v4.train.model_backends import HFTransformersBackend, OpenAIChatBackend

    if profile.backend == "hf_transformers":
        return HFTransformersBackend(profile)
    if profile.backend == "vllm":
        from decomposer_v4.train.model_backends import VLLMChatBackend

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


def _build_search_backend() -> MultiSearchClient:
    return MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )


def _benchmark_row(
    *,
    row: dict[str, Any],
    trajectory: dict[str, Any],
    benchmark_format: str,
    answer_field: str,
) -> dict[str, Any]:
    query = _row_query_text(row)
    query_id = _row_benchmark_id(row, query)
    article = str(trajectory.get(answer_field, "")).strip()
    if not article:
        fallback_field = "final_answer" if answer_field == "final_answer_with_citations" else "final_answer_with_citations"
        article = str(trajectory.get(fallback_field, "")).strip()
    if benchmark_format == "deep_research_bench":
        return {
            "id": query_id,
            "prompt": query,
            "article": article,
        }
    return {
        "id": query_id,
        "query": query,
        "article": article,
    }


def main() -> None:
    args = parse_args()
    rows = _load_rows(args)

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

    output_jsonl = _resolve_repo_path(args.output_jsonl)
    trajectory_dir = _resolve_repo_path(args.trajectory_dir)
    summary_json = _resolve_repo_path(args.summary_json)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        query = _row_query_text(row)
        query_id = _row_query_id(row, query)
        print(f"[{index}/{len(rows)}] {query_id}", flush=True)

        trajectory = build_trajectory(
            row={
                "query_id": query_id,
                "query": query,
                "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
                "metadata": row,
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
            planner_prompt_variant=str(args.planner_prompt_variant),
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
            append_final_citation_block=not bool(args.disable_citation_appendix),
        )
        error = str(trajectory.get("error", "") or "").strip()
        if error:
            print(f"  error: {error}", flush=True)
            trace = str(trajectory.get("error_traceback", "") or "").strip()
            if trace:
                print("  traceback:", flush=True)
                print("\n".join(f"    {line}" for line in trace.splitlines()[-12:]), flush=True)

        write_json(trajectory_dir / f"{query_id}.json", trajectory)
        benchmark_rows.append(
            _benchmark_row(
                row=row,
                trajectory=trajectory,
                benchmark_format=args.benchmark_format,
                answer_field=str(args.report_answer_field),
            )
        )
        summary_rows.append(
            {
                "query_id": query_id,
                "prompt": query,
                "error": str(trajectory.get("error", "") or ""),
                "total_seconds": float(trajectory.get("timings", {}).get("total_seconds", 0.0) or 0.0),
                "citation_count": int(trajectory.get("timings", {}).get("citation_count", 0) or 0),
                "planner_model": planner_profile.model,
                "answerer_model": answerer_profile.model,
                "trajectory_path": str((trajectory_dir / f"{query_id}.json").resolve()),
            }
        )

    write_jsonl(output_jsonl, benchmark_rows)
    write_json(
        summary_json,
        {
            "count": len(benchmark_rows),
            "benchmark_format": args.benchmark_format,
            "report_answer_field": str(args.report_answer_field),
            "planner_profile": planner_profile.__dict__,
            "answerer_profile": answerer_profile.__dict__,
            "output_jsonl": str(output_jsonl.resolve()),
            "trajectory_dir": str(trajectory_dir.resolve()),
            "rows": summary_rows,
        },
    )

    print(f"Wrote benchmark JSONL: {output_jsonl.resolve()}")
    print(f"Wrote trajectories: {trajectory_dir.resolve()}")
    print(f"Wrote summary: {summary_json.resolve()}")


if __name__ == "__main__":
    main()
