#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from decomposer_v4.train.rl.common import ensure_dir, resolve_path
from decomposer_v4.utils.io import ROOT


def _repo_root() -> Path:
    return ROOT


def _verl_config_searchpath() -> str:
    return f"file://{(_repo_root() / 'verl' / 'verl' / 'trainer' / 'config').resolve()}"


def _bool_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None:
        return []
    return [str(raw)]


def _str_list(raw: Any, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def _model_path(value: str | None, fallback: str) -> str:
    for candidate in (value, fallback):
        raw = str(candidate or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_absolute():
            return str(path.resolve())
        repo_local = (ROOT / path).resolve()
        if repo_local.exists():
            return str(repo_local)
        if raw.startswith("."):
            return str(repo_local)
        return raw
    return ""


def _as_int(raw: Any, default: int) -> int:
    if raw is None:
        return default
    return int(raw)


def _as_float(raw: Any, default: float) -> float:
    if raw is None:
        return default
    return float(raw)


def _build_agent_loop_config(
    *,
    stage: str,
    stage_cfg: dict[str, Any],
    output_dir: Path,
) -> Path:
    generation_cfg = stage_cfg.get("generation", {}) if isinstance(stage_cfg.get("generation"), dict) else {}
    reward_root = stage_cfg.get("reward", {}) if isinstance(stage_cfg.get("reward"), dict) else {}
    embedding_cfg = stage_cfg.get("embedding", {}) if isinstance(stage_cfg.get("embedding"), dict) else {}
    rollout_cfg = stage_cfg.get("rollout", {}) if isinstance(stage_cfg.get("rollout"), dict) else {}
    judge_cfg = stage_cfg.get("judge", {}) if isinstance(stage_cfg.get("judge"), dict) else {}

    if stage == "planner":
        target = "decomposer_v4.train.rl.verl.agent_loops.DecomposerPlannerAgentLoop"
        name = "decomposer_planner_agent"
        reward_cfg = reward_root.get("planner", {}) if isinstance(reward_root.get("planner"), dict) else {}
    elif stage == "answerer":
        target = "decomposer_v4.train.rl.verl.agent_loops.DecomposerAnswererAgentLoop"
        name = "decomposer_answerer_agent"
        reward_cfg = reward_root.get("answer", {}) if isinstance(reward_root.get("answer"), dict) else {}
    else:
        raise ValueError(f"Unsupported VERL stage: {stage}")

    payload = [
        {
            "name": name,
            "_target_": target,
            "generation": generation_cfg,
            "reward": reward_cfg,
            "embedding": embedding_cfg,
            "judge": judge_cfg,
            "rollout_logging": rollout_cfg,
        }
    ]
    path = output_dir / f"{stage}_agent_loop.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _build_verl_train_config(
    *,
    stage: str,
    stage_cfg: dict[str, Any],
    train_file: Path,
    output_dir: Path,
    agent_loop_config_path: Path,
    num_gpus: int,
) -> Path:
    model_cfg = stage_cfg.get("model", {}) if isinstance(stage_cfg.get("model"), dict) else {}
    lora_cfg = stage_cfg.get("lora", {}) if isinstance(stage_cfg.get("lora"), dict) else {}
    generation_cfg = stage_cfg.get("generation", {}) if isinstance(stage_cfg.get("generation"), dict) else {}
    runtime_cfg = stage_cfg.get("runtime", {}) if isinstance(stage_cfg.get("runtime"), dict) else {}
    rl_cfg = stage_cfg.get("rl", {}) if isinstance(stage_cfg.get("rl"), dict) else {}

    base_model = _model_path(stage_cfg.get("base_model"), "")
    tokenizer_name = _model_path(stage_cfg.get("tokenizer_name"), base_model)
    reference_model = _model_path(stage_cfg.get("reference_model"), base_model)
    if reference_model != base_model:
        raise ValueError(
            "The current VERL LoRA launch path uses the actor base checkpoint as the KL reference "
            "policy baseline. Set `reference_model` equal to `base_model`."
        )

    max_prompt_length = _as_int(runtime_cfg.get("max_prompt_length"), 4096)
    max_response_length = _as_int(
        runtime_cfg.get("max_response_length"),
        _as_int(generation_cfg.get("packed_max_total_tokens"), 8192),
    )
    max_model_len = _as_int(runtime_cfg.get("max_model_len"), max_prompt_length + max_response_length)

    epochs = _as_float(rl_cfg.get("epochs"), 1.0)
    max_steps = _as_int(rl_cfg.get("max_steps"), -1)
    if max_steps > 0:
        total_epochs = max(1, int(math.ceil(epochs)))
        total_training_steps: int | None = max_steps
    else:
        total_epochs = max(1, int(math.ceil(epochs)))
        total_training_steps = None

    train_batch_size = _as_int(
        rl_cfg.get("train_batch_size"),
        max(1, _as_int(rl_cfg.get("per_device_batch_size"), 1) * max(int(num_gpus), 1)),
    )
    val_batch_size = _as_int(rl_cfg.get("val_batch_size"), train_batch_size)
    ppo_mini_batch_size = _as_int(rl_cfg.get("ppo_mini_batch_size"), train_batch_size)
    ppo_micro_batch_size_per_gpu = _as_int(rl_cfg.get("ppo_micro_batch_size_per_gpu"), 1)
    log_prob_micro_batch_size_per_gpu = _as_int(
        rl_cfg.get("log_prob_micro_batch_size_per_gpu"),
        ppo_micro_batch_size_per_gpu,
    )
    use_dynamic_bsz = bool(rl_cfg.get("use_dynamic_bsz", True))
    ppo_max_token_len_per_gpu = _as_int(
        rl_cfg.get("ppo_max_token_len_per_gpu"),
        max_response_length + max_prompt_length,
    )

    logger_backends = _bool_list(rl_cfg.get("report_to", ["wandb"]))
    if not logger_backends:
        logger_backends = ["console"]
    elif "console" not in logger_backends:
        logger_backends.insert(0, "console")

    rollout_name = str(runtime_cfg.get("rollout_name", "vllm")).strip() or "vllm"
    tensor_model_parallel_size = _as_int(runtime_cfg.get("tensor_model_parallel_size"), 1)
    if num_gpus < tensor_model_parallel_size:
        raise ValueError(
            f"NUM_GPUS={num_gpus} is smaller than runtime.tensor_model_parallel_size={tensor_model_parallel_size}"
        )

    save_freq = _as_int(rl_cfg.get("save_freq"), _as_int(rl_cfg.get("save_steps"), 25))
    test_freq = _as_int(rl_cfg.get("test_freq"), -1)
    val_before_train = bool(rl_cfg.get("val_before_train", False))
    rollout_data_dir = str((output_dir / "verl_generation_dumps").resolve())

    rollout_output_dir = str((output_dir / "rollouts").resolve())
    rollout_cfg = stage_cfg.setdefault("rollout", {})
    if isinstance(rollout_cfg, dict) and not rollout_cfg.get("output_dir"):
        rollout_cfg["output_dir"] = rollout_output_dir

    config = {
        "hydra": {"searchpath": [_verl_config_searchpath()]},
        "defaults": ["ppo_trainer", "_self_"],
        "data": {
            "train_files": str(train_file),
            "val_files": str(train_file),
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "return_raw_chat": True,
            "shuffle": bool(rl_cfg.get("shuffle", False)),
            "seed": _as_int(rl_cfg.get("seed"), 13),
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": base_model,
                "tokenizer_path": tokenizer_name,
                "trust_remote_code": True,
                "use_shm": bool(runtime_cfg.get("use_shm", True)),
                "use_remove_padding": bool(runtime_cfg.get("use_remove_padding", True)),
                "enable_gradient_checkpointing": bool(rl_cfg.get("gradient_checkpointing", True)),
                "lora_rank": _as_int(lora_cfg.get("r"), 32),
                "lora_alpha": _as_int(lora_cfg.get("alpha"), 64),
                "target_modules": list(lora_cfg.get("target_modules", ["all-linear"])),
                "override_config": {
                    "attn_implementation": str(model_cfg.get("attn_implementation", "sdpa")),
                },
            },
            "actor": {
                "strategy": str(rl_cfg.get("strategy", "fsdp2")),
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size_per_gpu,
                "use_dynamic_bsz": use_dynamic_bsz,
                "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                "use_kl_loss": bool(rl_cfg.get("use_kl_loss", True)),
                "kl_loss_coef": _as_float(rl_cfg.get("beta"), 0.001),
                "kl_loss_type": str(rl_cfg.get("kl_loss_type", "low_var_kl")),
                "entropy_coeff": _as_float(rl_cfg.get("entropy_coeff"), 0.0),
                "fsdp_config": {
                    "model_dtype": str(runtime_cfg.get("model_dtype", "bf16")),
                    "param_offload": bool(rl_cfg.get("param_offload", False)),
                    "optimizer_offload": bool(rl_cfg.get("optimizer_offload", False)),
                },
                "optim": {
                    "lr": _as_float(rl_cfg.get("learning_rate"), 1.0e-6),
                    "weight_decay": _as_float(rl_cfg.get("weight_decay"), 0.0),
                    "lr_warmup_steps_ratio": _as_float(rl_cfg.get("warmup_ratio"), 0.0),
                },
                "checkpoint": {
                    "save_contents": _str_list(
                        rl_cfg.get("checkpoint_save_contents"),
                        ["model", "optimizer", "extra"],
                    ),
                    "load_contents": _str_list(
                        rl_cfg.get("checkpoint_load_contents"),
                        ["model", "optimizer", "extra"],
                    ),
                    "async_save": bool(rl_cfg.get("checkpoint_async_save", False)),
                },
            },
            "rollout": {
                "mode": "async",
                "name": rollout_name,
                "n": _as_int(generation_cfg.get("num_samples_per_query"), 4),
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "gpu_memory_utilization": _as_float(runtime_cfg.get("gpu_memory_utilization"), 0.45),
                "prompt_length": max_prompt_length,
                "response_length": max_response_length,
                "max_model_len": max_model_len,
                "max_num_batched_tokens": _as_int(runtime_cfg.get("max_num_batched_tokens"), max_model_len),
                "max_num_seqs": _as_int(runtime_cfg.get("max_num_seqs"), 64),
                "enable_chunked_prefill": bool(runtime_cfg.get("enable_chunked_prefill", True)),
                "enable_prefix_caching": bool(runtime_cfg.get("enable_prefix_caching", True)),
                "free_cache_engine": bool(runtime_cfg.get("free_cache_engine", True)),
                "load_format": str(runtime_cfg.get("load_format", "safetensors")),
                "layered_summon": bool(runtime_cfg.get("layered_summon", True)),
                "log_prob_micro_batch_size_per_gpu": log_prob_micro_batch_size_per_gpu,
                "log_prob_use_dynamic_bsz": use_dynamic_bsz,
                "log_prob_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                "temperature": _as_float(generation_cfg.get("temperature"), 1.0),
                "top_p": _as_float(generation_cfg.get("top_p"), 1.0),
                "agent": {
                    "num_workers": _as_int(runtime_cfg.get("agent_num_workers"), max(1, num_gpus)),
                    "default_agent_loop": "decomposer_planner_agent" if stage == "planner" else "decomposer_answerer_agent",
                    "agent_loop_config_path": str(agent_loop_config_path),
                },
            },
            "ref": {
                "strategy": str(rl_cfg.get("ref_strategy", rl_cfg.get("strategy", "fsdp2"))),
                "log_prob_micro_batch_size_per_gpu": log_prob_micro_batch_size_per_gpu,
                "fsdp_config": {
                    "model_dtype": str(runtime_cfg.get("model_dtype", "bf16")),
                    "param_offload": bool(rl_cfg.get("ref_param_offload", True)),
                },
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "trainer": {
            "critic_warmup": 0,
            "logger": logger_backends,
            "project_name": os.environ.get("WANDB_PROJECT", "decomposer-rl"),
            "experiment_name": str(rl_cfg.get("run_name", f"decomposer-{stage}-verl")),
            "default_local_dir": str(output_dir),
            "default_hdfs_dir": None,
            "rollout_data_dir": rollout_data_dir,
            "n_gpus_per_node": int(num_gpus),
            "nnodes": _as_int(rl_cfg.get("nnodes"), 1),
            "save_freq": save_freq,
            "max_actor_ckpt_to_keep": rl_cfg.get("max_actor_ckpt_to_keep", 3),
            "max_critic_ckpt_to_keep": rl_cfg.get("max_critic_ckpt_to_keep", None),
            "test_freq": test_freq,
            "val_before_train": val_before_train,
            "val_only": False,
            "use_legacy_worker_impl": "disable",
            "resume_mode": str(rl_cfg.get("resume_mode", "auto")),
            "total_epochs": total_epochs,
            "lora_train_meta": {
                "r": _as_int(lora_cfg.get("r"), 32),
                "lora_alpha": _as_int(lora_cfg.get("alpha"), 64),
                "task_type": str(lora_cfg.get("task_type", "CAUSAL_LM")),
            },
        },
    }
    if total_training_steps is not None:
        config["trainer"]["total_training_steps"] = total_training_steps

    path = output_dir / f"{stage}_verl_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def prepare_verl_launch(
    *,
    stage: str,
    stage_cfg: dict[str, Any],
    train_file: Path,
    output_dir: Path,
    num_gpus: int,
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    agent_loop_config_path = _build_agent_loop_config(stage=stage, stage_cfg=stage_cfg, output_dir=output_dir)
    verl_config_path = _build_verl_train_config(
        stage=stage,
        stage_cfg=stage_cfg,
        train_file=train_file,
        output_dir=output_dir,
        agent_loop_config_path=agent_loop_config_path,
        num_gpus=num_gpus,
    )
    return verl_config_path, agent_loop_config_path


def run_verl(config_path: Path, *, dry_run: bool = False) -> int:
    config_path = config_path.resolve()
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
    ]
    if dry_run:
        print(" ".join(cmd))
        return 0

    env = os.environ.copy()
    repo_root = str(_repo_root().resolve())
    verl_root = str((_repo_root() / "verl").resolve())
    env["PYTHONPATH"] = f"{repo_root}:{verl_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    process = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return int(process.returncode)
