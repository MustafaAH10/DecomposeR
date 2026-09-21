#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.utils.io import ROOT, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL-based assistant-only LoRA SFT for decomposer_v4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-max-examples", type=int, default=0)
    return parser.parse_args()


def _lazy_import_training_stack():
    try:
        import torch
        from datasets import Features, Sequence, Value, load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError("train_trl.py requires torch, datasets, peft, transformers, and trl.") from exc
    return (
        torch,
        load_dataset,
        Features,
        Sequence,
        Value,
        LoraConfig,
        PeftModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        SFTConfig,
        SFTTrainer,
    )


def _get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _resolve_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _prompt_completion_json_features(Features: Any, Sequence: Any, Value: Any) -> Any:
    message_feature = {"role": Value("string"), "content": Value("string")}
    return Features(
        {
            "record_id": Value("string"),
            "query_id": Value("string"),
            "query": Value("string"),
            "stage": Value("string"),
            "turn_type": Value("string"),
            "prompt": [message_feature],
            "completion": [message_feature],
            "metadata": {
                "completion_chars": Value("int64"),
                "prompt_message_count": Value("int64"),
                "query_id": Value("string"),
                "span_id": Value("string"),
                "target_node_ids": [Value("string")],
                "task_type": Value("string"),
                "turn_type": Value("string"),
            },
        }
    )


def _load_json_dataset(
    load_dataset: Any,
    raw_path: str,
    *,
    dataset_mode: str,
    Features: Any | None = None,
    Sequence: Any | None = None,
    Value: Any | None = None,
) -> Any:
    features = None
    if dataset_mode == "prompt_completion" and Features is not None and Sequence is not None and Value is not None:
        features = _prompt_completion_json_features(Features, Sequence, Value)
    return load_dataset("json", data_files=_resolve_path(raw_path), split="train", features=features)


def _normalize_completion_messages(raw_completion: Any) -> list[dict[str, Any]]:
    if isinstance(raw_completion, list):
        messages = [item for item in raw_completion if isinstance(item, dict)]
        if messages:
            return messages
    if isinstance(raw_completion, dict):
        return [raw_completion]
    if isinstance(raw_completion, str) and raw_completion.strip():
        return [{"role": "assistant", "content": raw_completion}]
    raise ValueError("Expected conversational `completion` content for prompt-completion SFT.")


def _encode_prompt_completion_example(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    completion_messages: list[dict[str, Any]],
) -> dict[str, int]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    prompt_completion = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    prompt_ids = list(prompt_ids)
    prompt_completion_ids = list(prompt_completion["input_ids"])
    prefix_len = len(prompt_ids)
    if prompt_completion_ids[:prefix_len] != prompt_ids:
        overlap = 0
        max_overlap = min(len(prompt_ids), len(prompt_completion_ids))
        for offset in range(max_overlap, -1, -1):
            if prompt_ids[:offset] == prompt_completion_ids[:offset]:
                overlap = offset
                break
        prefix_len = overlap
    total_len = len(prompt_completion_ids)
    return {
        "prompt_tokens": prefix_len,
        "completion_tokens": max(0, total_len - prefix_len),
        "raw_length": total_len,
    }


def _pretokenize_prompt_completion_dataset(
    dataset: Any,
    tokenizer: Any,
    max_length: int,
    truncation_side: str,
    dataset_num_proc: int | None,
) -> Any:
    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        prompt_messages = row.get("prompt")
        if not isinstance(prompt_messages, list) or not prompt_messages:
            raise ValueError("Expected non-empty `prompt` list in turn-level SFT row.")
        completion_messages = _normalize_completion_messages(row.get("completion"))
        prompt_ids = list(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        prompt_completion = tokenizer.apply_chat_template(
            prompt_messages + completion_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        input_ids = list(prompt_completion["input_ids"])
        attention_mask = list(prompt_completion["attention_mask"])

        prefix_len = len(prompt_ids)
        if input_ids[:prefix_len] != prompt_ids:
            overlap = 0
            max_overlap = min(len(prompt_ids), len(input_ids))
            for offset in range(max_overlap, -1, -1):
                if prompt_ids[:offset] == input_ids[:offset]:
                    overlap = offset
                    break
            prefix_len = overlap

        completion_ids = input_ids[prefix_len:]
        completion_mask = [0] * prefix_len + [1] * len(completion_ids)
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            if overflow >= prefix_len:
                raise ValueError(
                    f"Prompt alone exceeds max_length={max_length}; cannot preserve completion without truncating target."
                )
            if truncation_side != "left":
                raise ValueError("Prompt-completion mode only supports left truncation to preserve the target completion.")
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]
            completion_mask = completion_mask[overflow:]

        labels = [token_id if mask else -100 for token_id, mask in zip(input_ids, completion_mask, strict=True)]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "length": len(input_ids),
        }

    return dataset.map(
        tokenize_row,
        remove_columns=dataset.column_names,
        num_proc=dataset_num_proc,
    )


def _encode_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    assistant_only_loss: bool,
    max_length: int | None = None,
    truncation_side: str = "left",
) -> dict[str, Any]:
    def encode_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer.apply_chat_template(
            subset,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise RuntimeError("Tokenizer chat template did not return `input_ids` and `attention_mask`.")
        return encoded

    full = encode_subset(messages)
    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    assistant_mask = [0] * len(input_ids)

    if assistant_only_loss:
        prefix_cache: dict[int, list[int]] = {0: []}
        for index, message in enumerate(messages, start=1):
            if index not in prefix_cache:
                prefix_cache[index] = list(encode_subset(messages[:index])["input_ids"])
            if str(message.get("role", "")) != "assistant":
                continue
            before_ids = prefix_cache[index - 1]
            after_ids = prefix_cache[index]
            start = len(before_ids)
            end = len(after_ids)
            if start > end:
                raise RuntimeError("Assistant span boundaries are invalid.")
            if before_ids and after_ids[:start] != before_ids:
                overlap = 0
                max_overlap = min(len(before_ids), len(after_ids))
                for offset in range(max_overlap, -1, -1):
                    if before_ids[:offset] == after_ids[:offset]:
                        overlap = offset
                        break
                start = overlap
            for position in range(start, min(end, len(assistant_mask))):
                assistant_mask[position] = 1
    else:
        assistant_mask = [1] * len(input_ids)

    raw_length = len(input_ids)
    if max_length is not None and raw_length > max_length:
        if truncation_side == "left":
            start_index = raw_length - max_length
            end_index = raw_length
        else:
            start_index = 0
            end_index = max_length
        input_ids = input_ids[start_index:end_index]
        attention_mask = attention_mask[start_index:end_index]
        assistant_mask = assistant_mask[start_index:end_index]

    labels = [token_id if mask else -100 for token_id, mask in zip(input_ids, assistant_mask, strict=True)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "raw_length": raw_length,
        "assistant_token_count": sum(assistant_mask),
    }


def _length_stats(lengths: list[int], max_length: int) -> str:
    lengths = sorted(lengths)
    if not lengths:
        return "rows=0"

    def pct(p: float) -> int:
        index = min(len(lengths) - 1, max(0, round((len(lengths) - 1) * p)))
        return lengths[index]

    over_limit = sum(1 for value in lengths if value > max_length)
    return (
        f"rows={len(lengths)} min={lengths[0]} mean={statistics.mean(lengths):.1f} "
        f"p90={pct(0.9)} p95={pct(0.95)} p99={pct(0.99)} max={lengths[-1]} "
        f"over_max={over_limit} ({over_limit / len(lengths):.1%})"
    )


def _report_split_lengths(
    name: str,
    dataset: Any,
    tokenizer: Any,
    max_length: int,
    assistant_only_loss: bool,
    max_examples: int,
) -> None:
    lengths: list[int] = []
    limit = len(dataset) if max_examples <= 0 else min(len(dataset), max_examples)
    for index in range(limit):
        row = dataset[index]
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        encoded = _encode_messages(tokenizer, messages, assistant_only_loss=assistant_only_loss)
        lengths.append(int(encoded["raw_length"]))
    print(f"{name}: {_length_stats(lengths, max_length)}")


def _report_prompt_completion_lengths(
    name: str,
    dataset: Any,
    tokenizer: Any,
    max_length: int,
    max_examples: int,
) -> None:
    lengths: list[int] = []
    limit = len(dataset) if max_examples <= 0 else min(len(dataset), max_examples)
    for index in range(limit):
        row = dataset[index]
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            continue
        completion = _normalize_completion_messages(row.get("completion"))
        encoded = _encode_prompt_completion_example(tokenizer, prompt, completion)
        lengths.append(int(encoded["raw_length"]))
    print(f"{name}: {_length_stats(lengths, max_length)}")


def _pretokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    max_length: int,
    assistant_only_loss: bool,
    truncation_side: str,
    dataset_num_proc: int | None,
) -> Any:
    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Expected non-empty `messages` list in SFT row.")
        encoded = _encode_messages(
            tokenizer,
            messages,
            assistant_only_loss=assistant_only_loss,
            max_length=max_length,
            truncation_side=truncation_side,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "length": len(encoded["input_ids"]),
        }

    return dataset.map(
        tokenize_row,
        remove_columns=dataset.column_names,
        num_proc=dataset_num_proc,
    )


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [pad_id] * pad_len)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad_len)
            labels.append(list(feature["labels"]) + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _build_model(
    config: dict[str, Any],
    output_dir: str,
    torch: Any,
    PeftModel: Any,
    AutoModelForCausalLM: Any,
) -> tuple[Any, Any | None]:
    base_model = str(config["base_model"])
    dtype = torch.bfloat16 if bool(_get_nested(config, "training.bf16", True)) else torch.float16
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    attn_implementation = _get_nested(config, "model.attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False

    init_adapter_from = _resolve_path(str(config.get("init_adapter_from", "")).strip())
    if init_adapter_from:
        model = PeftModel.from_pretrained(model, init_adapter_from, is_trainable=True)
        return model, None

    lora = config["lora"]
    return model, {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "target_modules": list(lora["target_modules"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def _build_sft_args(
    config: dict[str, Any],
    output_dir: str,
    SFTConfig: Any,
    *,
    dataset_mode: str,
    for_dry_run: bool = False,
) -> Any:
    sft_max_length = int(_get_nested(config, "sft.max_length", config.get("max_seq_length", 8192)))
    sft_kwargs: dict[str, Any] = {
        "output_dir": _resolve_path(output_dir),
        "learning_rate": float(_get_nested(config, "training.learning_rate", 1e-5)),
        "num_train_epochs": float(_get_nested(config, "training.epochs", 2)),
        "per_device_train_batch_size": int(_get_nested(config, "training.per_device_batch_size", 1)),
        "per_device_eval_batch_size": int(
            _get_nested(
                config,
                "training.per_device_eval_batch_size",
                _get_nested(config, "training.per_device_batch_size", 1),
            )
        ),
        "gradient_accumulation_steps": int(_get_nested(config, "training.gradient_accumulation_steps", 1)),
        "warmup_ratio": float(_get_nested(config, "training.warmup_ratio", 0.03)),
        "lr_scheduler_type": str(_get_nested(config, "training.lr_scheduler", "cosine")),
        "optim": str(_get_nested(config, "training.optim", "adamw_torch_fused")),
        "weight_decay": float(_get_nested(config, "training.weight_decay", 0.0)),
        "max_grad_norm": float(_get_nested(config, "training.max_grad_norm", 1.0)),
        "bf16": bool(_get_nested(config, "training.bf16", True)),
        "fp16": bool(_get_nested(config, "training.fp16", False)),
        "gradient_checkpointing": bool(_get_nested(config, "training.gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": _get_nested(config, "training.gradient_checkpointing_kwargs"),
        "logging_steps": int(_get_nested(config, "training.logging_steps", 10)),
        "save_strategy": str(_get_nested(config, "training.save_strategy", "steps")),
        "save_steps": int(_get_nested(config, "training.save_steps", 200)),
        "save_total_limit": int(_get_nested(config, "training.save_total_limit", 3)),
        "eval_strategy": str(_get_nested(config, "training.eval_strategy", "epoch")),
        "eval_steps": _get_nested(config, "training.eval_steps"),
        "report_to": list(_get_nested(config, "training.report_to", [])),
        "seed": int(_get_nested(config, "training.seed", 42)),
        "ddp_find_unused_parameters": bool(_get_nested(config, "training.ddp_find_unused_parameters", False)),
        "remove_unused_columns": bool(_get_nested(config, "training.remove_unused_columns", False)),
        "max_length": sft_max_length,
        "packing": bool(_get_nested(config, "sft.packing", False)),
        "assistant_only_loss": False,
        "completion_only_loss": None,
        "dataset_num_proc": _get_nested(config, "data.dataset_num_proc"),
        "dataset_text_field": str(_get_nested(config, "sft.dataset_text_field", "text")),
        "shuffle_dataset": bool(_get_nested(config, "data.shuffle_dataset", False)),
        "pad_to_multiple_of": _get_nested(config, "sft.pad_to_multiple_of"),
        "dataloader_num_workers": int(_get_nested(config, "training.dataloader_num_workers", 0)),
        "dataloader_persistent_workers": bool(_get_nested(config, "training.dataloader_persistent_workers", False)),
        "dataloader_prefetch_factor": _get_nested(config, "training.dataloader_prefetch_factor"),
        "length_column_name": str(_get_nested(config, "training.length_column_name", "length")),
    }
    if for_dry_run:
        sft_kwargs["bf16"] = False
        sft_kwargs["fp16"] = False
        sft_kwargs["use_cpu"] = True
    optional_keys = {
        "run_name": _get_nested(config, "training.run_name"),
        "logging_dir": _resolve_path(_get_nested(config, "training.logging_dir")),
        "deepspeed": _resolve_path(_get_nested(config, "training.deepspeed")),
        "save_only_model": _get_nested(config, "training.save_only_model"),
        "torch_compile": _get_nested(config, "training.torch_compile"),
        "torch_compile_backend": _get_nested(config, "training.torch_compile_backend"),
        "torch_compile_mode": _get_nested(config, "training.torch_compile_mode"),
    }
    if optional_keys.get("deepspeed"):
        try:
            importlib.metadata.version("deepspeed")
        except importlib.metadata.PackageNotFoundError:
            print("warning: deepspeed config provided but `deepspeed` is not installed; falling back to non-deepspeed training.")
            optional_keys["deepspeed"] = None
    for key, value in optional_keys.items():
        if value is not None:
            sft_kwargs[key] = value
    supported_keys = set(inspect.signature(SFTConfig.__init__).parameters)
    sft_kwargs = {key: value for key, value in sft_kwargs.items() if key in supported_keys}
    return SFTConfig(**sft_kwargs)


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    output_dir = args.output_dir or config.get("output_dir", "outputs/sft/run")
    (
        torch,
        load_dataset,
        Features,
        Sequence,
        Value,
        LoraConfig,
        PeftModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        SFTConfig,
        SFTTrainer,
    ) = _lazy_import_training_stack()

    tokenizer_name = str(config.get("tokenizer_name") or config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_mode = str(_get_nested(config, "sft.dataset_mode", "messages_manual_labels"))
    train_ds = _load_json_dataset(
        load_dataset,
        str(config["data"]["train_file"]),
        dataset_mode=dataset_mode,
        Features=Features,
        Sequence=Sequence,
        Value=Value,
    )
    val_ds = _load_json_dataset(
        load_dataset,
        str(config["data"]["val_file"]),
        dataset_mode=dataset_mode,
        Features=Features,
        Sequence=Sequence,
        Value=Value,
    )
    sft_args = _build_sft_args(config, output_dir, SFTConfig, dataset_mode=dataset_mode, for_dry_run=args.dry_run)
    assistant_only_loss = bool(_get_nested(config, "sft.assistant_only_loss", True))
    truncation_side = str(_get_nested(config, "sft.truncation_side", "left"))
    dataset_num_proc = _get_nested(config, "data.dataset_num_proc")

    if args.dry_run:
        max_examples = int(args.dry_run_max_examples)
        print(f"train_rows={len(train_ds)} val_rows={len(val_ds)} max_length={int(sft_args.max_length)}")
        if dataset_mode == "prompt_completion":
            _report_prompt_completion_lengths(
                "train",
                train_ds,
                tokenizer,
                max_length=int(sft_args.max_length),
                max_examples=max_examples,
            )
            _report_prompt_completion_lengths(
                "val",
                val_ds,
                tokenizer,
                max_length=int(sft_args.max_length),
                max_examples=max_examples,
            )
        else:
            _report_split_lengths(
                "train",
                train_ds,
                tokenizer,
                max_length=int(sft_args.max_length),
                assistant_only_loss=assistant_only_loss,
                max_examples=max_examples,
            )
            _report_split_lengths(
                "val",
                val_ds,
                tokenizer,
                max_length=int(sft_args.max_length),
                assistant_only_loss=assistant_only_loss,
                max_examples=max_examples,
            )
        return

    data_collator: Any | None = None
    if dataset_mode == "messages_manual_labels":
        train_ds = _pretokenize_dataset(
            train_ds,
            tokenizer,
            max_length=int(sft_args.max_length),
            assistant_only_loss=assistant_only_loss,
            truncation_side=truncation_side,
            dataset_num_proc=dataset_num_proc,
        )
        val_ds = _pretokenize_dataset(
            val_ds,
            tokenizer,
            max_length=int(sft_args.max_length),
            assistant_only_loss=assistant_only_loss,
            truncation_side=truncation_side,
            dataset_num_proc=dataset_num_proc,
        )
        data_collator = AssistantOnlyCollator(tokenizer=tokenizer)
    elif dataset_mode == "prompt_completion":
        train_ds = _pretokenize_prompt_completion_dataset(
            train_ds,
            tokenizer,
            max_length=int(sft_args.max_length),
            truncation_side=truncation_side,
            dataset_num_proc=dataset_num_proc,
        )
        val_ds = _pretokenize_prompt_completion_dataset(
            val_ds,
            tokenizer,
            max_length=int(sft_args.max_length),
            truncation_side=truncation_side,
            dataset_num_proc=dataset_num_proc,
        )
        data_collator = AssistantOnlyCollator(tokenizer=tokenizer)

    model, peft_kwargs = _build_model(config, output_dir, torch, PeftModel, AutoModelForCausalLM)
    peft_config = None if peft_kwargs is None else LoraConfig(**peft_kwargs)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(_resolve_path(output_dir))
    tokenizer.save_pretrained(_resolve_path(output_dir))


if __name__ == "__main__":
    main()
