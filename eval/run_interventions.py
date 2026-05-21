"""
CoT intervention evaluation — all five ablation conditions (§11.2).

Runs the same set of episodes through each condition and computes CoT-dependence
metrics (§11.3). The delta between 'normal' and 'empty_cot' is the primary
diagnostic: large delta = CoT is load-bearing; near-zero = vestigial/direct policy.

Usage:
    python eval/run_interventions.py \\
        --model Qwen/Qwen3-8B \\
        --checkpoint checkpoints/grpo_monitor/final \\
        --data data/test.jsonl \\
        --outdir runs/interventions/grpo_monitor \\
        --conditions normal empty_cot filler_cot honest_cot no_thinking

    # Skip expensive conditions during debugging
    python eval/run_interventions.py ... --conditions normal empty_cot honest_cot

    # Also run LLM judge on each condition
    python eval/run_interventions.py ... --llm_judge

Outputs per condition:
    {outdir}/{condition}/metrics.json
    {outdir}/{condition}/samples.jsonl

Plus a combined summary:
    {outdir}/intervention_summary.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import creds
creds.load()

ALL_CONDITIONS = ["normal", "empty_cot", "filler_cot", "honest_cot", "no_thinking"]


def run_condition(
    condition: str,
    episodes: list[dict],
    model,
    tokenizer,
    reward_cfg,
    outdir: str,
    max_new_tokens: int,
    temperature: float,
    llm_judge=None,
) -> dict:
    from eval.inference import INTERVENTION_BUILDERS, generate_completion
    from scorer.reward import score_completion
    from scorer.parser import parse_completion

    builder = INTERVENTION_BUILDERS[condition]
    sample_rows = []

    print(f"\n── Condition: {condition} ({len(episodes)} episodes) ──")
    for i, episode in enumerate(episodes):
        prompt = builder(episode, tokenizer)
        completion = generate_completion(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = score_completion(episode, completion, config=reward_cfg)

        row = {
            "study_id": episode["study_id"],
            "condition": condition,
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
            jr = llm_judge.score(parsed.think_text)
            row.update({
                "llm_judge_suspicious": jr.suspicious,
                "llm_judge_score": jr.score,
                "llm_judge_rationale": jr.rationale,
            })

        sample_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(episodes)}")

    # Compute metrics for this condition
    from eval.run_eval import _rows_to_score_results
    from scorer.metrics import compute_metrics, metrics_to_dict

    score_results = _rows_to_score_results(sample_rows)
    metrics = compute_metrics(score_results, episodes=episodes)
    metrics_dict = metrics_to_dict(metrics)
    metrics_dict["condition"] = condition

    if llm_judge:
        metrics_dict["llm_judge_suspicious_rate"] = sum(
            r.get("llm_judge_suspicious", False) for r in sample_rows
        ) / len(sample_rows)

    # Save
    cond_dir = os.path.join(outdir, condition)
    os.makedirs(cond_dir, exist_ok=True)
    with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
    with open(os.path.join(cond_dir, "samples.jsonl"), "w") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")

    print(f"  invalid_success_rate: {metrics_dict['invalid_success_rate']:.3f}  "
          f"clean_bad_rate: {metrics_dict['clean_bad_rate']:.3f}  "
          f"monitor_score: {metrics_dict['mean_monitor_score']:.3f}")
    return metrics_dict


def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model: {args.model}")
    from eval.inference import load_model_for_eval
    model, tokenizer = load_model_for_eval(
        args.model, checkpoint_path=args.checkpoint, use_4bit=args.use_4bit
    )

    episodes = _load_jsonl(args.data)
    if args.max_examples > 0:
        episodes = episodes[: args.max_examples]

    llm_judge = None
    if args.llm_judge:
        from scorer.llm_judge import LLMJudge
        llm_judge = LLMJudge()
        print(f"LLM judge: {llm_judge.model}")

    from scorer.reward import RewardConfig
    reward_cfg = RewardConfig(monitor_weight=args.monitor_weight)

    conditions = args.conditions or ALL_CONDITIONS
    all_metrics: dict[str, dict] = {}

    for condition in conditions:
        metrics = run_condition(
            condition=condition,
            episodes=episodes,
            model=model,
            tokenizer=tokenizer,
            reward_cfg=reward_cfg,
            outdir=args.outdir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            llm_judge=llm_judge,
        )
        all_metrics[condition] = metrics

    # CoT-dependence deltas (§11.3)
    summary = {"conditions": all_metrics}
    if "normal" in all_metrics and "empty_cot" in all_metrics:
        n = all_metrics["normal"]
        e = all_metrics["empty_cot"]
        summary["cot_dependence"] = {
            "invalid_success_delta": round(
                n["invalid_success_rate"] - e["invalid_success_rate"], 4),
            "clean_bad_delta": round(
                n["clean_bad_rate"] - e["clean_bad_rate"], 4),
        }
    if "normal" in all_metrics and "honest_cot" in all_metrics:
        n = all_metrics["normal"]
        h = all_metrics["honest_cot"]
        summary["honest_cot_steering"] = round(
            n["invalid_success_rate"] - h["invalid_success_rate"], 4)
    if "normal" in all_metrics and "filler_cot" in all_metrics:
        n = all_metrics["normal"]
        fi = all_metrics["filler_cot"]
        summary["filler_cot_delta"] = round(
            n["invalid_success_rate"] - fi["invalid_success_rate"], 4)

    summary_path = os.path.join(args.outdir, "intervention_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Intervention summary ──")
    if "cot_dependence" in summary:
        print(f"  CoT dependence (invalid_success normal - empty_cot): "
              f"{summary['cot_dependence']['invalid_success_delta']:+.4f}")
        print(f"  CoT dependence (clean_bad normal - empty_cot):       "
              f"{summary['cot_dependence']['clean_bad_delta']:+.4f}")
    if "honest_cot_steering" in summary:
        print(f"  Honest CoT steering:  {summary['honest_cot_steering']:+.4f}")
    print(f"\nSaved to {args.outdir}")


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True)
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--data",           required=True)
    p.add_argument("--outdir",         required=True)
    p.add_argument("--conditions",     nargs="+", default=None,
                   choices=ALL_CONDITIONS,
                   help="Subset of conditions to run (default: all five)")
    p.add_argument("--max_examples",   type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--monitor_weight", type=float, default=1.5)
    p.add_argument("--use_4bit",       action="store_true")
    p.add_argument("--llm_judge",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
