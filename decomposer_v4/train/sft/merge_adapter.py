#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a dense base checkpoint.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-source", default="outputs/sft/qwen3_8b_turnlevel")
    parser.add_argument(
        "--output-dir",
        default="outputs/models/qwen3_8b_sft_merged",
    )
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--adapter-device", default="cpu")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--safe-serialization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("merge_adapter.py requires torch, peft, and transformers.") from exc

    dtype_lookup = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_lookup[args.torch_dtype]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation.strip():
        model_kwargs["attn_implementation"] = args.attn_implementation.strip()

    print(f"Loading base model from {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    print(f"Loading adapter from {args.adapter_source}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_source,
        is_trainable=False,
        torch_device=args.adapter_device,
    )

    print("Merging adapter into dense weights")
    merged_model = peft_model.merge_and_unload()
    if hasattr(merged_model, "config"):
        merged_model.config.use_cache = True

    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=bool(args.safe_serialization),
        max_shard_size=args.max_shard_size,
    )

    tokenizer_source = args.adapter_source
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.base_model,
        "adapter_source": args.adapter_source,
        "merged_output_dir": str(output_dir),
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "adapter_device": args.adapter_device,
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
