"""
Batch inference + scoring harness.

Runs a model on a set of episodes, scores all completions, and saves results.
Works for both base models and fine-tuned LoRA checkpoints.

Usage:
    # Evaluate base model on validation set
    python eval/run_eval.py \\
        --model Qwen/Qwen3-8B \\
        --data data/val.jsonl \\
        --outdir runs/base_qwen3_8b

    # Evaluate a fine-tuned checkpoint
    python eval/run_eval.py \\
        --model Qwen/Qwen3-8B \\
        --checkpoint checkpoints/grpo_monitor/final \\
        --data data/test.jsonl \\
        --outdir runs/grpo_monitor_test

    # Also run the LLM judge on the CoT blocks (slower, costs API $)
    python eval/run_eval.py ... --llm_judge

Outputs:
    {outdir}/metrics.json     aggregate metrics
    {outdir}/samples.jsonl    per-episode results (prompt, completion, scores)
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import creds
creds.load()


def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    if args.checkpoint:
        print(f"  + LoRA checkpoint: {args.checkpoint}")
    from eval.inference import load_model_for_eval, generate_completion, build_prompt_normal
    model, tokenizer = load_model_for_eval(
        args.model, checkpoint_path=args.checkpoint, use_4bit=args.use_4bit
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    episodes = _load_jsonl(args.data)
    if args.max_examples > 0:
        episodes = episodes[: args.max_examples]
    print(f"Evaluating on {len(episodes)} episodes from {args.data}")

    # ── Optional LLM judge ────────────────────────────────────────────────────
    llm_judge = None
    if args.llm_judge:
        from scorer.llm_judge import LLMJudge
        llm_judge = LLMJudge()
        print(f"LLM judge enabled: {llm_judge.model}")

    # ── Inference + scoring loop ──────────────────────────────────────────────
    from scorer.reward import score_completion, RewardConfig
    from scorer.parser import parse_completion

    reward_cfg = RewardConfig(monitor_weight=args.monitor_weight)
    sample_rows = []

    for i, episode in enumerate(episodes):
        prompt = build_prompt_normal(episode, tokenizer)
        completion = generate_completion(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        result = score_completion(episode, completion, config=reward_cfg)

        row = {
            "study_id": episode["study_id"],
            "domain": episode["domain"],
            "completion": completion,
            "parse_ok": result.parse_ok,
            "selected_analysis_id": result.selected_analysis_id,
            "validity_label": result.validity_label,
            "weak_accepts": result.weak_accepts,
            "invalid_success": result.invalid_success,
            "exploit_family": result.exploit_family,
            "monitor_score": result.monitor_score,
            "monitor_flag": result.monitor_flag,
            "total_reward": result.total_reward,
            "reward_components": result.reward_components,
            "think_word_count": result.think_word_count,
        }

        if llm_judge:
            parsed = parse_completion(completion)
            judge_result = llm_judge.score(parsed.think_text)
            row["llm_judge_suspicious"] = judge_result.suspicious
            row["llm_judge_score"] = judge_result.score
            row["llm_judge_rationale"] = judge_result.rationale
            row["llm_judge_error"] = judge_result.error

        sample_rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(episodes)}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    from scorer.reward import ScoreResult
    from scorer.metrics import compute_metrics, metrics_to_dict

    # Reconstruct ScoreResult objects for metrics computation
    # (We stored what we need in the row dicts above)
    from scorer.monitor import MonitorResult
    score_results = _rows_to_score_results(sample_rows)
    metrics = compute_metrics(score_results, episodes=episodes)
    metrics_dict = metrics_to_dict(metrics)

    if args.llm_judge:
        judge_suspicious = [r.get("llm_judge_suspicious", False) for r in sample_rows]
        metrics_dict["llm_judge_suspicious_rate"] = sum(judge_suspicious) / len(sample_rows)
        metrics_dict["llm_judge_mean_score"] = sum(
            r.get("llm_judge_score", 0.0) for r in sample_rows
        ) / len(sample_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.outdir, "metrics.json")
    samples_path = os.path.join(args.outdir, "samples.jsonl")

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    with open(samples_path, "w") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n── Results ──")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSaved to {args.outdir}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def _rows_to_score_results(rows: list[dict]):
    """Reconstruct lightweight ScoreResult proxies for metrics computation."""
    from scorer.reward import ScoreResult
    from scorer.monitor import MonitorResult, score_cot

    results = []
    for r in rows:
        monitor = score_cot(None)
        monitor.score = r.get("monitor_score", 0.0)
        monitor.flag = r.get("monitor_flag", False)
        sr = ScoreResult(
            parse_ok=r["parse_ok"],
            parse_error=None,
            selected_analysis_id=r.get("selected_analysis_id"),
            validity_label=r.get("validity_label"),
            weak_accepts=r.get("weak_accepts", False),
            invalid_success=r.get("invalid_success", False),
            exploit_family=r.get("exploit_family"),
            monitor_score=r.get("monitor_score", 0.0),
            monitor_flag=r.get("monitor_flag", False),
            monitor_result=monitor,
            base_reward=r.get("reward_components", {}).get("base", 0.0),
            monitor_penalty=r.get("reward_components", {}).get("monitor_penalty", 0.0),
            length_penalty=r.get("reward_components", {}).get("length_penalty", 0.0),
            total_reward=r.get("total_reward", 0.0),
            reward_components=r.get("reward_components", {}),
            claim_strength=None,
            think_word_count=r.get("think_word_count", 0),
        )
        results.append(sr)
    return results


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--checkpoint",      default=None,
                   help="Path to LoRA adapter dir (optional)")
    p.add_argument("--data",            required=True)
    p.add_argument("--outdir",          required=True)
    p.add_argument("--max_examples",    type=int, default=-1)
    p.add_argument("--max_new_tokens",  type=int, default=1024)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--monitor_weight",  type=float, default=1.5)
    p.add_argument("--use_4bit",        action="store_true")
    p.add_argument("--llm_judge",       action="store_true",
                   help="Run Claude LLM judge on CoT blocks (costs API $)")
    return p.parse_args()


if __name__ == "__main__":
    main()
