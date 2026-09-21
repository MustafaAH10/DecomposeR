#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.retrieval.live_clients import JinaSearchClient, MultiSearchClient, SerperFetchClient
from decomposer_v4.reward.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from decomposer_v4.schemas.trajectory import Rubric, SearchExecution, SearchPassage
from decomposer_v4.utils.io import ROOT, read_jsonl, read_text


PLANNER_SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/planner_system_prompt.txt")
ANSWERER_SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/answerer_system_prompt.txt")
ANSWER_JUDGE_SYSTEM_PROMPT = read_text(ROOT / "decomposer_v4/prompts/answer_judge_system_prompt.txt")


def load_answerer_system_prompt(variant: str = "default") -> str:
    normalized = str(variant or "default").strip().lower()
    if normalized in {"", "default", "standard"}:
        return ANSWERER_SYSTEM_PROMPT
    path = ROOT / f"decomposer_v4/prompts/answerer_system_prompt_{normalized}.txt"
    if not path.exists():
        raise ValueError(f"Unknown answerer system prompt variant: {variant}")
    return read_text(path)


def _lazy_import_local_generation_stack():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Local rollout helpers require torch, peft, and transformers.") from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _parse_rubrics(raw_rubrics: Sequence[dict[str, Any]] | None) -> tuple[Rubric, ...]:
    parsed: list[Rubric] = []
    for rubric in raw_rubrics or []:
        if not isinstance(rubric, dict):
            continue
        parsed.append(
            Rubric(
                title=str(rubric.get("title", "")),
                description=str(rubric.get("description", "")),
                weight=float(rubric.get("weight", 1.0)),
            )
        )
    return tuple(parsed)


def load_rl_rows(path_or_dir: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path_or_dir)
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()

    if not resolved.exists():
        raise ValueError(f"RL row source not found: {resolved}")

    if resolved.is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted(resolved.glob("*.json")):
            if path.name.startswith("_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "record_id": str(payload.get("record_id", payload.get("query_id", path.stem))),
                    "query_id": str(payload.get("query_id", path.stem)),
                    "query": str(payload.get("query", "")),
                    "rubrics": _parse_rubrics(payload.get("rubrics", [])),
                    "task_type": str(payload.get("task_type", "deep_research") or "deep_research"),
                    "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
                }
            )
        return rows

    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        raw_rows = read_jsonl(resolved)
    elif suffix == ".json":
        raw_payload = json.loads(resolved.read_text(encoding="utf-8"))
        raw_rows = raw_payload if isinstance(raw_payload, list) else [raw_payload]
    else:
        raise ValueError(f"Unsupported RL row source: {resolved}")

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        raw_rubrics = row.get("rubrics", [])
        if isinstance(raw_rubrics, str):
            try:
                raw_rubrics = json.loads(raw_rubrics)
            except Exception:
                raw_rubrics = []
        rows.append(
            {
                "record_id": str(row.get("record_id", row.get("query_id", ""))),
                "query_id": str(row.get("query_id", row.get("record_id", ""))),
                "query": str(row.get("query", "")),
                "rubrics": _parse_rubrics(raw_rubrics if isinstance(raw_rubrics, list) else []),
                "task_type": str(row.get("task_type", "deep_research") or "deep_research"),
                "metadata": row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
            }
        )
    return rows


def _load_dotenv_if_available() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


def make_openai_client(base_url: str) -> OpenAI:
    _load_dotenv_if_available()
    return OpenAI(
        base_url=base_url,
        api_key=(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY"),
    )


def make_search_backend() -> MultiSearchClient:
    return MultiSearchClient(
        clients=(JinaSearchClient.from_env(),),
        fetch_clients=(SerperFetchClient.from_env(),),
    )


def make_embedder(kind: str, model_name: str):
    if kind == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name=model_name)
    return HashingEmbedder()


class LocalChatSpanGenerator:
    def __init__(self, *, model: Any, tokenizer: Any, device: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _tokenize_messages(self, messages: list[dict[str, str]]) -> Any:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(prompt_ids, dict):
                input_ids = prompt_ids.get("input_ids")
                if input_ids is not None:
                    return input_ids.to(self.device)
            if hasattr(prompt_ids, "shape"):
                return prompt_ids.to(self.device)
        rendered = "\n\n".join(f"{row['role'].upper()}:\n{row['content']}" for row in messages)
        return self.tokenizer(rendered, return_tensors="pt").input_ids.to(self.device)

    def __call__(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, list[int], list[int]]:
        prompt_ids = self._tokenize_messages(messages)
        prompt_len = int(prompt_ids.shape[-1])
        generation = self.model.generate(
            prompt_ids,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        full_ids = generation[0].tolist()
        generated_ids = full_ids[prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text, prompt_ids[0].tolist(), generated_ids


class LocalChatBackend:
    def __init__(
        self,
        *,
        generator: LocalChatSpanGenerator,
        default_temperature: float,
        default_top_p: float,
        default_max_completion_tokens: int,
    ) -> None:
        self.generator = generator
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_max_completion_tokens = int(default_max_completion_tokens)

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        del response_format_json
        text, _, _ = self.generator(
            messages=messages,
            max_new_tokens=int(max_completion_tokens or self.default_max_completion_tokens),
            temperature=float(self.default_temperature if temperature is None else temperature),
            top_p=self.default_top_p,
        )
        return text


def load_local_policy_model(
    *,
    base_model: str,
    adapter_path: str = "",
    tokenizer_name: str = "",
    bf16: bool = True,
) -> tuple[Any, Any, Any]:
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _lazy_import_local_generation_stack()
    dtype = (
        torch.bfloat16
        if bf16 and torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_source = str(tokenizer_name).strip() or str(adapter_path).strip() or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path and str(adapter_path).strip().lower() != "none":
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def search_execution_from_dict(payload: dict[str, Any]) -> SearchExecution:
    passages = tuple(
        SearchPassage(
            passage_id=str(passage.get("passage_id", "")).strip(),
            node_id=str(passage.get("node_id", "")).strip(),
            title=str(passage.get("title", "")).strip(),
            url=str(passage.get("url", "")).strip(),
            text=str(passage.get("text", "")).strip(),
            score=float(passage.get("score", 0.0) or 0.0),
            source=str(passage.get("source", "")).strip(),
            published_date=str(passage.get("published_date", "")).strip(),
            author=str(passage.get("author", "")).strip(),
            language=str(passage.get("language", "")).strip(),
            language_score=float(passage.get("language_score", 0.0) or 0.0),
            token_count=int(passage.get("token_count", 0) or 0),
        )
        for passage in payload.get("passages", [])
        if isinstance(passage, dict)
    )
    return SearchExecution(
        node_id=str(payload.get("node_id", "")).strip(),
        query=str(payload.get("query", "")).strip(),
        passages=passages,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
