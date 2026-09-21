from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from openai import OpenAI

from decomposer_v4.utils.io import load_yaml


def _load_dotenv_if_available() -> None:
    env_path = Path.cwd() / ".env"
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


@dataclass(frozen=True)
class ModelProfile:
    backend: str
    model: str
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = ""
    verbosity: str = ""
    temperature: float = 0.2
    top_p: float = 0.95
    max_completion_tokens: int = 4096
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    seed: int = 17


class ChatBackend(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        raise NotImplementedError


def load_model_profile(path: str) -> ModelProfile:
    payload = load_yaml(path)
    return ModelProfile(
        backend=str(payload["backend"]),
        model=str(payload["model"]),
        base_url=str(payload.get("base_url", "")),
        api_key_env=str(payload.get("api_key_env", "OPENAI_API_KEY")),
        reasoning_effort=str(payload.get("reasoning_effort", "")),
        verbosity=str(payload.get("verbosity", "")),
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        max_completion_tokens=int(payload.get("max_completion_tokens", 4096)),
        trust_remote_code=bool(payload.get("trust_remote_code", True)),
        device_map=str(payload.get("device_map", "auto")),
        torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
        tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.9)),
        max_model_len=int(payload.get("max_model_len", 32768)),
        seed=int(payload.get("seed", 17)),
    )


class OpenAIChatBackend:
    def __init__(self, profile: ModelProfile) -> None:
        _load_dotenv_if_available()
        api_key = os.environ.get(profile.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {profile.api_key_env}")
        if profile.base_url.strip():
            self.client = OpenAI(api_key=api_key, base_url=profile.base_url.strip())
        else:
            self.client = OpenAI(api_key=api_key)
        self.profile = profile

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        is_deepseek = self.profile.model.startswith("deepseek-") or "api.deepseek.com" in self.profile.base_url

        def _build_kwargs(use_json_format: bool) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
            }
            if is_deepseek:
                kwargs["max_tokens"] = max_completion_tokens or self.profile.max_completion_tokens
            else:
                kwargs["max_completion_tokens"] = max_completion_tokens or self.profile.max_completion_tokens
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}

            model_name = self.profile.model.strip()
            reasoning_effort = self.profile.reasoning_effort.strip()
            if reasoning_effort and model_name.startswith("gpt-5"):
                kwargs["reasoning_effort"] = reasoning_effort
            elif not model_name.startswith("gpt-5"):
                chosen_temperature = self.profile.temperature if temperature is None else temperature
                kwargs["temperature"] = chosen_temperature
                if self.profile.top_p > 0:
                    kwargs["top_p"] = self.profile.top_p
            if self.profile.verbosity.strip():
                kwargs["verbosity"] = self.profile.verbosity.strip()
            return kwargs

        attempts = [response_format_json, False, False] if response_format_json else [False, False]
        content = ""
        for idx, use_json_format in enumerate(attempts, start=1):
            response = self.client.chat.completions.create(**_build_kwargs(use_json_format))
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            if idx < len(attempts):
                time.sleep(0.5)
        return content


class HFTransformersBackend:
    def __init__(self, profile: ModelProfile) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("HFTransformersBackend requires transformers.") from exc
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(profile.torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(profile.model, trust_remote_code=profile.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.model,
            trust_remote_code=profile.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=profile.device_map,
        )
        self.profile = profile

    def _input_device(self) -> torch.device | str:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        except Exception:
            pass

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"disk", "meta"}:
                    return torch.device(device)
        return torch.device("cpu")

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        del response_format_json
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        input_device = self._input_device()
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(input_device)
            model_inputs: dict[str, Any] = {"input_ids": input_ids}
        else:
            model_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in dict(encoded).items()
            }
            input_ids = model_inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise RuntimeError("Tokenizer did not return input_ids for chat template.")

        chosen_temperature = self.profile.temperature if temperature is None else temperature
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_completion_tokens or self.profile.max_completion_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": chosen_temperature > 0.0,
        }
        if chosen_temperature > 0.0:
            generate_kwargs["temperature"] = chosen_temperature
            generate_kwargs["top_p"] = self.profile.top_p
        output = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = output[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class VLLMChatBackend:
    def __init__(self, profile: ModelProfile) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError("VLLMChatBackend requires vllm and transformers.") from exc

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=profile.model,
            trust_remote_code=profile.trust_remote_code,
            tensor_parallel_size=max(int(profile.tensor_parallel_size), 1),
            gpu_memory_utilization=float(profile.gpu_memory_utilization),
            max_model_len=int(profile.max_model_len),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(profile.model, trust_remote_code=profile.trust_remote_code)
        self.profile = profile
        self.call_index = 0

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if response_format_json:
            prompt += (
                "\n\nReturn valid JSON only. Do not wrap it in markdown fences. "
                "Do not include explanation before or after the JSON."
            )
        chosen_temperature = self.profile.temperature if temperature is None else float(temperature)
        sampling_params = self.SamplingParams(
            temperature=max(chosen_temperature, 0.0),
            top_p=float(self.profile.top_p),
            max_tokens=int(max_completion_tokens or self.profile.max_completion_tokens),
            seed=int(self.profile.seed) + self.call_index,
        )
        self.call_index += 1
        result = self.llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        return (result.outputs[0].text or "").strip()


def build_chat_backend(profile_path: str) -> ChatBackend:
    profile = load_model_profile(profile_path)
    if profile.backend == "openai_chat":
        return OpenAIChatBackend(profile)
    if profile.backend == "hf_transformers":
        return HFTransformersBackend(profile)
    if profile.backend == "vllm":
        return VLLMChatBackend(profile)
    raise ValueError(f"Unsupported backend: {profile.backend}")
