#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer_v4.utils.io import ROOT, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade precomputed HealthBench completions with the official simple-evals rubric implementation."
    )
    parser.add_argument("--input-path", required=True, help="HealthBench JSONL with completion_to_trial.")
    parser.add_argument("--simple-evals-dir", default="decomposer_v4/eval/simple-evals")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", default="")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=16)
    parser.add_argument("--grader-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--use-gpt-5-4-low-grader", action="store_true")
    parser.add_argument("--length-adjustment-center", type=float, default=None)
    parser.add_argument("--length-adjustment-penalty-per-500-chars", type=float, default=None)
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _install_simple_evals_package(simple_evals_dir: Path) -> None:
    # The upstream checkout is a flat directory using relative imports. Register it
    # as a package without modifying the external repo.
    package = types.ModuleType("simple_evals")
    package.__path__ = [str(simple_evals_dir)]  # type: ignore[attr-defined]
    sys.modules["simple_evals"] = package


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    simple_evals_dir = _resolve_path(args.simple_evals_dir)
    if not (simple_evals_dir / "healthbench_eval.py").exists():
        raise FileNotFoundError(f"Missing simple-evals checkout at {simple_evals_dir}")
    _install_simple_evals_package(simple_evals_dir)

    from simple_evals import common
    from simple_evals.healthbench_eval import HealthBenchEval
    from simple_evals.sampler.chat_completion_sampler import (
        OPENAI_SYSTEM_MESSAGE_API,
        ChatCompletionSampler,
    )
    from simple_evals.sampler.responses_sampler import ResponsesSampler
    from simple_evals.types import SamplerBase

    if args.use_gpt_5_4_low_grader:
        grader = ResponsesSampler(
            model="gpt-5.4-2026-03-05",
            reasoning_model=True,
            reasoning_effort="low",
        )
    else:
        grader = ChatCompletionSampler(
            model=args.grader_model,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )

    eval_obj = HealthBenchEval(
        grader_model=grader,
        num_examples=args.examples if int(args.examples) > 0 else None,
        n_repeats=1,
        n_threads=max(int(args.n_threads), 1),
        input_path=str(_resolve_path(args.input_path)),
        length_adjustment_center=args.length_adjustment_center,
        length_adjustment_penalty_per_500_chars=args.length_adjustment_penalty_per_500_chars,
    )
    # HealthBenchEval already supports scoring a `completion_to_trial` field
    # internally for physician completions. Setting this flag after init keeps all
    # custom rows instead of subsetting to physician-completion groups.
    eval_obj.physician_completions_mode = "__precomputed__"
    result = eval_obj(SamplerBase())

    payload = {
        "score": _jsonable(result.score),
        "metrics": _jsonable(result.metrics),
        "metadata": _jsonable(result.metadata),
        "input_path": str(_resolve_path(args.input_path)),
        "grader_model": "gpt-5.4-2026-03-05-low" if args.use_gpt_5_4_low_grader else args.grader_model,
    }
    write_json(_resolve_path(args.output_json), payload)
    if args.output_html.strip():
        output_html = _resolve_path(args.output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(common.make_report(result), encoding="utf-8")
    print(json.dumps({"score": payload["score"], "output_json": str(_resolve_path(args.output_json))}, indent=2))


if __name__ == "__main__":
    main()
