#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def _patch_reward_extra_metrics(repo_root: Path) -> None:
    target = repo_root / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    if not target.exists():
        raise SystemExit(f"Cannot find VERL trainer file: {target}")

    marker = "Generic per-component reward metrics from AgentLoopOutput.reward_extra_info."
    text = target.read_text(encoding="utf-8")
    if marker in text:
        print("VERL reward-extra W&B metrics patch already applied.")
        return

    needle = "                # GDPO per-component reward metrics\n"
    if needle not in text:
        raise SystemExit(f"Cannot find insertion point in {target}")

    insert = (
        "                # Generic per-component reward metrics from AgentLoopOutput.reward_extra_info.\n"
        "                # This makes custom reward breakdowns visible for GRPO/PPO runs, not only GDPO.\n"
        "                for key, values in reward_extra_infos_dict.items():\n"
        "                    try:\n"
        "                        vals = np.asarray(values, dtype=np.float32)\n"
        "                    except (TypeError, ValueError):\n"
        "                        continue\n"
        "                    if vals.size == 0:\n"
        "                        continue\n"
        "                    metrics[f\"reward_extra/{key}/mean\"] = float(np.mean(vals))\n"
        "                    metrics[f\"reward_extra/{key}/std\"] = float(np.std(vals))\n"
        "                    metrics[f\"reward_extra/{key}/max\"] = float(np.max(vals))\n"
        "                    metrics[f\"reward_extra/{key}/min\"] = float(np.min(vals))\n"
    )
    target.write_text(text.replace(needle, insert + needle), encoding="utf-8")
    print(f"Applied VERL reward-extra W&B metrics patch to {target}")


def _patch_tracking_log_errors(repo_root: Path) -> None:
    target = repo_root / "verl" / "verl" / "utils" / "tracking.py"
    if not target.exists():
        raise SystemExit(f"Cannot find VERL tracking file: {target}")

    marker = "Decomposer patch: keep training alive if a logger backend drops a metric batch."
    text = target.read_text(encoding="utf-8")
    if marker in text:
        print("VERL tracking logger robustness patch already applied.")
        return

    needle = (
        "    def log(self, data, step, backend=None):\n"
        "        for default_backend, logger_instance in self.logger.items():\n"
        "            if backend is None or default_backend in backend:\n"
        "                logger_instance.log(data=data, step=step)\n"
    )
    if needle not in text:
        raise SystemExit(f"Cannot find Tracking.log insertion point in {target}")

    replacement = (
        "    def log(self, data, step, backend=None):\n"
        "        # Decomposer patch: keep training alive if a logger backend drops a metric batch.\n"
        "        for default_backend, logger_instance in self.logger.items():\n"
        "            if backend is None or default_backend in backend:\n"
        "                try:\n"
        "                    logger_instance.log(data=data, step=step)\n"
        "                except Exception as exc:\n"
        "                    print(\n"
        "                        f\"[verl tracking] logger={default_backend} step={step} failed: {exc}\",\n"
        "                        flush=True,\n"
        "                    )\n"
    )
    target.write_text(text.replace(needle, replacement), encoding="utf-8")
    print(f"Applied VERL tracking logger robustness patch to {target}")


def _patch_lora_train_meta_checkpoint(repo_root: Path) -> None:
    target = repo_root / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    if not target.exists():
        raise SystemExit(f"Cannot find VERL trainer file: {target}")

    marker = "Decomposer patch: persist LoRA training metadata beside actor checkpoint shards."
    text = target.read_text(encoding="utf-8")
    if marker in text:
        print("VERL LoRA checkpoint metadata patch already applied.")
        return

    needle = (
        "        self.actor_rollout_wg.save_checkpoint(\n"
        "            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep\n"
        "        )\n"
    )
    if needle not in text:
        raise SystemExit(f"Cannot find actor checkpoint insertion point in {target}")

    insert = (
        "        # Decomposer patch: persist LoRA training metadata beside actor checkpoint shards.\n"
        "        lora_train_meta = self.config.trainer.get(\"lora_train_meta\", None)\n"
        "        if lora_train_meta:\n"
        "            import json\n"
        "\n"
        "            local_mkdir_safe(actor_local_path)\n"
        "            with open(os.path.join(actor_local_path, \"lora_train_meta.json\"), \"w\", encoding=\"utf-8\") as f:\n"
        "                json.dump(dict(lora_train_meta), f, ensure_ascii=False, indent=2)\n"
    )
    target.write_text(text.replace(needle, insert + needle), encoding="utf-8")
    print(f"Applied VERL LoRA checkpoint metadata patch to {target}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _patch_reward_extra_metrics(repo_root)
    _patch_tracking_log_errors(repo_root)
    _patch_lora_train_meta_checkpoint(repo_root)


if __name__ == "__main__":
    main()
