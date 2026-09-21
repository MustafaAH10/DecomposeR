#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _tensor_local_shard(value: Any) -> torch.Tensor:
    local_tensor = getattr(value, "_local_tensor", None)
    if local_tensor is not None:
        return local_tensor.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    raise TypeError(f"Unsupported LoRA tensor type: {type(value)!r}")


def _shard_dim(value: Any) -> int:
    placements = getattr(value, "placements", None)
    if not placements:
        return 0
    placement = placements[0]
    if hasattr(placement, "is_shard") and placement.is_shard():
        return int(getattr(placement, "dim", 0))
    return 0


def _load_world_size(actor_checkpoint_dir: Path) -> int:
    config_path = actor_checkpoint_dir / "fsdp_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing FSDP config: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    world_size = int(config.get("world_size") or 0)
    if world_size <= 0:
        raise ValueError(f"Invalid world_size in {config_path}: {world_size}")
    return world_size


def _load_lora_meta(actor_checkpoint_dir: Path, explicit_meta: dict[str, Any] | None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta_path = actor_checkpoint_dir / "lora_train_meta.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta.update(loaded)
    run_dir = actor_checkpoint_dir.parent.parent
    for config_path in sorted(run_dir.glob("*_verl_config.yaml")):
        try:
            import yaml

            with config_path.open(encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(config, dict):
            continue
        trainer_meta = (
            config.get("trainer", {}).get("lora_train_meta", {})
            if isinstance(config.get("trainer"), dict)
            else {}
        )
        if isinstance(trainer_meta, dict):
            meta.update({key: value for key, value in trainer_meta.items() if value is not None})
        model_cfg = (
            config.get("actor_rollout_ref", {}).get("model", {})
            if isinstance(config.get("actor_rollout_ref"), dict)
            else {}
        )
        if isinstance(model_cfg, dict):
            if "r" not in meta and model_cfg.get("lora_rank"):
                meta["r"] = model_cfg.get("lora_rank")
            if "lora_alpha" not in meta and model_cfg.get("lora_alpha"):
                meta["lora_alpha"] = model_cfg.get("lora_alpha")
    if explicit_meta:
        meta.update({key: value for key, value in explicit_meta.items() if value is not None})
    return meta


def export_lora_adapter(
    actor_checkpoint_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    lora_meta: dict[str, Any] | None = None,
) -> Path:
    actor_checkpoint_dir = Path(actor_checkpoint_dir)
    output_dir = Path(output_dir) if output_dir is not None else actor_checkpoint_dir / "lora_adapter"
    world_size = _load_world_size(actor_checkpoint_dir)

    rank_zero_path = actor_checkpoint_dir / f"model_world_size_{world_size}_rank_0.pt"
    if not rank_zero_path.exists():
        raise FileNotFoundError(f"Missing rank-0 model shard: {rank_zero_path}")

    rank_zero_state = torch.load(rank_zero_path, map_location="cpu", weights_only=False)
    lora_keys = sorted(key for key in rank_zero_state if "lora_" in key)
    if not lora_keys:
        raise RuntimeError(f"No LoRA tensors found in {rank_zero_path}")

    shard_dims = {key: _shard_dim(rank_zero_state[key]) for key in lora_keys}
    lora_shards: dict[str, list[torch.Tensor]] = {
        key: [_tensor_local_shard(rank_zero_state[key])] for key in lora_keys
    }
    del rank_zero_state

    for rank in range(1, world_size):
        shard_path = actor_checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing model shard: {shard_path}")
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        for key in lora_keys:
            lora_shards[key].append(_tensor_local_shard(state[key]))
        del state

    lora_params: OrderedDict[str, torch.Tensor] = OrderedDict()
    target_modules: set[str] = set()
    inferred_rank = 0
    for key in lora_keys:
        adapter_key = key.replace(".default.weight", ".weight")
        merged = torch.cat(lora_shards[key], dim=shard_dims[key]).contiguous()
        lora_params[adapter_key] = merged
        target_modules.add(adapter_key.split(".")[-3])
        if inferred_rank <= 0:
            inferred_rank = int(min(merged.shape))

    meta = _load_lora_meta(actor_checkpoint_dir, lora_meta)
    rank = int(meta.get("r") or inferred_rank)
    alpha = int(meta.get("lora_alpha") or meta.get("alpha") or rank)
    task_type = str(meta.get("task_type") or "CAUSAL_LM")
    if rank <= 0:
        raise ValueError(f"Could not infer LoRA rank from {actor_checkpoint_dir}; pass --rank explicitly.")
    if alpha <= 0:
        raise ValueError(f"Invalid LoRA alpha={alpha}; pass --alpha explicitly.")

    try:
        import peft

        peft_config = peft.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=sorted(target_modules),
            task_type=task_type,
        ).to_dict()
        peft_config["task_type"] = (
            peft_config["task_type"].value
            if hasattr(peft_config["task_type"], "value")
            else peft_config["task_type"]
        )
        peft_config["peft_type"] = (
            peft_config["peft_type"].value
            if hasattr(peft_config["peft_type"], "value")
            else peft_config["peft_type"]
        )
        peft_config["target_modules"] = sorted(peft_config["target_modules"])
    except Exception:
        peft_config = {
            "base_model_name_or_path": None,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "revision": None,
            "target_modules": sorted(target_modules),
            "task_type": task_type,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(peft_config, f, ensure_ascii=False, indent=4)
    save_file(lora_params, output_dir / "adapter_model.safetensors")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PEFT LoRA adapter from a VERL FSDP actor checkpoint.")
    parser.add_argument("--actor-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=0)
    parser.add_argument("--task-type", default="CAUSAL_LM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = {
        "r": args.rank or None,
        "lora_alpha": args.alpha or None,
        "task_type": args.task_type,
    }
    output_dir = export_lora_adapter(
        args.actor_checkpoint_dir,
        output_dir=args.output_dir or None,
        lora_meta=meta,
    )
    print(f"Exported LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
