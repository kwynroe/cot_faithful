"""
GRPO fine-tuning for clean-CoT misbehaviour elicitation.

Usage:
    # Monitor-penalized condition (main experiment)
    python training/train_grpo.py --config training/configs/grpo_monitor.yaml

    # No-monitor condition (diagnostic baseline)
    python training/train_grpo.py --config training/configs/grpo_no_monitor.yaml

    # Override any field at the command line
    python training/train_grpo.py --config training/configs/grpo_monitor.yaml \\
        --monitor_weight 2.0 --num_generations 8 --run_name grpo_heavy_monitor

Config file fields are loaded first; CLI flags override them.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import creds
creds.load()  # populate os.environ from .env before any HF/wandb imports


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = False              # set True for QLoRA on smaller GPUs

    # Data
    train_data: str = "data/train.jsonl"
    val_data: str = "data/val.jsonl"
    max_train_examples: int = -1        # -1 = use all

    # Output
    output_dir: str = "checkpoints/grpo"
    run_name: str = "grpo_pilot"

    # Training
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42

    # GRPO sampling
    num_generations: int = 4            # K completions per prompt
    max_prompt_length: int = 4096       # tokens
    max_completion_length: int = 1024   # tokens
    temperature: float = 0.8
    top_p: float = 0.95

    # DAPO-style training improvements (from arxiv 2503.14476)
    # loss_type="dr_grpo": skips gradient updates for groups where all K
    #   completions receive the same reward (dead compute early in training
    #   when format isn't learned, or when the model collapses to one strategy).
    #   Requires TRL >= 0.12. Fall back to "grpo" if you get a TypeError.
    # clip_low=0.0: asymmetric clipping — removes the lower clip bound so the
    #   policy can move freely away from bad completions. Requires TRL >= 0.15.
    #   Gracefully ignored on older TRL versions.
    loss_type: str = "dr_grpo"
    clip_low: float = 0.0

    # Reward / monitor — the primary condition toggle
    # Set monitor_weight=0.0 for the no-monitor ablation condition.
    monitor_weight: float = 1.5
    monitor_threshold: float = 0.3
    length_weight: float = 0.2

    # LLM judge monitor (optional, replaces rule-based monitor when enabled)
    # When use_llm_judge_monitor=True the judge scores each completion's think_text
    # in parallel and the result feeds into the monitor_penalty term.
    # Recommended monitor_threshold when using judge: 0.5 (i.e. score >= 5/10).
    use_llm_judge_monitor: bool = False
    llm_judge_model: str = "claude-haiku-4-5-20251001"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 250
    wandb_project: str = "cot-unfaithfulness"


def load_config(config_path: Optional[str], overrides: dict) -> GRPOConfig:
    cfg = GRPOConfig()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            yaml_vals = yaml.safe_load(f) or {}
        for k, v in yaml_vals.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(data_path: str, tokenizer, max_examples: int = -1):
    """
    Load episodes from JSONL, apply chat template, build HF Dataset.

    Columns produced:
      prompt          : chat-formatted prompt string (consumed by GRPOTrainer)
      episode_labels  : slim JSON string with hidden labels (passed to reward fn)
    """
    from datasets import Dataset as HFDataset
    from training.model import format_prompt_for_model
    from training.reward_fn import build_episode_labels

    rows = []
    with open(data_path) as f:
        for line in f:
            ep = json.loads(line.strip())
            rows.append({
                "prompt": format_prompt_for_model(ep["prompt"], tokenizer),
                "episode_labels": build_episode_labels(ep),
            })
            if max_examples > 0 and len(rows) >= max_examples:
                break

    return HFDataset.from_list(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg = load_config(args.config, _cli_overrides(args))

    print(f"Run: {cfg.run_name}")
    print(f"Model: {cfg.model_name}")
    print(f"monitor_weight: {cfg.monitor_weight}  "
          f"({'no monitor' if cfg.monitor_weight == 0.0 else 'monitor-penalized'})")
    print(f"Output: {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save the resolved config alongside the checkpoint for reproducibility
    with open(os.path.join(cfg.output_dir, "run_config.yaml"), "w") as f:
        yaml.dump(asdict(cfg), f)

    # Lazy imports — only pulled in once GPU is available
    import torch
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    from scorer.reward import RewardConfig
    from training.model import load_model_and_tokenizer
    from training.reward_fn import make_reward_fn

    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_4bit=cfg.use_4bit,
    )

    train_dataset = load_dataset(cfg.train_data, tokenizer, cfg.max_train_examples)
    val_dataset = load_dataset(cfg.val_data, tokenizer)
    print(f"Train: {len(train_dataset)} examples  Val: {len(val_dataset)} examples")

    reward_config = RewardConfig(
        monitor_weight=cfg.monitor_weight,
        monitor_threshold=cfg.monitor_threshold,
        length_weight=cfg.length_weight,
    )

    llm_judge = None
    if cfg.use_llm_judge_monitor:
        from scorer.llm_judge import LLMJudge
        llm_judge = LLMJudge(model=cfg.llm_judge_model)
        print(f"LLM judge monitor enabled: {cfg.llm_judge_model}")

    reward_fn = make_reward_fn(reward_config, llm_judge=llm_judge)

    trl_kwargs = dict(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        # Training
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        # GRPO / DAPO sampling
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        loss_type=cfg.loss_type,
        # Logging / checkpointing
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        report_to="wandb" if creds.wandb_enabled() else "none",
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    # Asymmetric clipping (DAPO): epsilon_low added in TRL >= 0.15.
    # Try passing it; silently skip on older versions rather than crashing.
    import inspect
    if "epsilon_low" in inspect.signature(TRLGRPOConfig).parameters:
        trl_kwargs["epsilon_low"] = cfg.clip_low
    else:
        print(f"Note: TRL version does not support epsilon_low; "
              f"symmetric clipping will be used (upgrade to TRL>=0.15 for asymmetric).")

    trl_cfg = TRLGRPOConfig(**trl_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=trl_cfg,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Custom wandb logging of reward components on every eval step
    if os.environ.get("WANDB_DISABLED") != "true":
        _attach_component_logger(trainer, reward_config)

    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "final"))
    print(f"Saved to {cfg.output_dir}/final")


# ── Component logging callback ────────────────────────────────────────────────

def _attach_component_logger(trainer, reward_config):
    """
    Attach a TrainerCallback that logs detailed reward breakdown and
    exploit-family distribution to wandb at eval steps.
    """
    try:
        import wandb
        from transformers import TrainerCallback

        class RewardBreakdownCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not wandb.run or metrics is None:
                    return
                # TRL logs reward/* keys from the reward_fns; we just re-log
                # them with cleaner names if present.
                breakdown = {
                    k: v for k, v in (metrics or {}).items()
                    if k.startswith("eval_reward")
                }
                if breakdown:
                    wandb.log(breakdown, step=state.global_step)

        trainer.add_callback(RewardBreakdownCallback())
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       type=str, default=None)
    p.add_argument("--model_name",   type=str, default=None)
    p.add_argument("--train_data",   type=str, default=None)
    p.add_argument("--val_data",     type=str, default=None)
    p.add_argument("--output_dir",   type=str, default=None)
    p.add_argument("--run_name",     type=str, default=None)
    p.add_argument("--monitor_weight",  type=float, default=None)
    p.add_argument("--num_generations", type=int,   default=None)
    p.add_argument("--learning_rate",   type=float, default=None)
    p.add_argument("--lora_rank",       type=int,   default=None)
    p.add_argument("--use_4bit",             action="store_true", default=None)
    p.add_argument("--max_train_examples",   type=int, default=None)
    p.add_argument("--use_llm_judge_monitor", action="store_true", default=None)
    p.add_argument("--llm_judge_model",      type=str, default=None)
    return p.parse_args()


def _cli_overrides(args) -> dict:
    return {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }


if __name__ == "__main__":
    main()
