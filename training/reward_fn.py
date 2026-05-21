"""
TRL-compatible reward function for GRPO training.

TRL's GRPOTrainer calls reward functions as:
    rewards = reward_fn(completions, **dataset_kwargs)

where dataset_kwargs contains all non-"prompt" columns from the dataset,
repeated once per completion (TRL repeats each row num_generations times
before scoring).

We store a slim JSON blob per episode — only the fields score_completion()
actually needs — as the "episode_labels" dataset column. This avoids
serialising the full prompt (which is already stored separately).

Reward components are logged individually to wandb via the extras mechanism
so we can plot base_reward, monitor_penalty, and total_reward separately.
"""

from __future__ import annotations
import json
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorer.reward import score_completion, RewardConfig


def make_reward_fn(reward_config: RewardConfig, llm_judge=None):
    """
    Return a TRL-compatible reward function closed over reward_config.

    If llm_judge is provided (an LLMJudge instance), completions are parsed
    for their think_text and batch-scored in parallel before reward computation.
    The judge score then replaces the rule-based monitor score in score_completion().

    IMPORTANT: the judge only ever sees parsed think_text — never the <report>.
    """
    from scorer.parser import parse_completion as _parse

    def reward_fn(
        completions: list[str],
        episode_labels: list[str],
        **kwargs,
    ) -> list[float]:
        # Run LLM judge on all think blocks in parallel (if enabled)
        judge_scores: list[Optional[int]] = [None] * len(completions)
        if llm_judge is not None:
            think_texts = [_parse(c).think_text for c in completions]
            results = llm_judge.score_batch_parallel(think_texts)
            judge_scores = [r.score for r in results]

        rewards = []
        for completion, labels_str, judge_score in zip(
            completions, episode_labels, judge_scores
        ):
            try:
                episode = json.loads(labels_str)
                result = score_completion(
                    episode, completion,
                    config=reward_config,
                    llm_judge_score=judge_score,
                )
                rewards.append(result.total_reward)
            except Exception:
                rewards.append(reward_config.malformed)
        return rewards

    return reward_fn


def make_component_reward_fns(reward_config: RewardConfig) -> dict[str, callable]:
    """
    Return a dict of named reward functions, one per component.
    Pass all of them to GRPOTrainer's reward_funcs to get per-component
    logging in wandb without summing them (they're logged separately, not added).

    Usage:
        fns = make_component_reward_fns(config)
        trainer = GRPOTrainer(..., reward_funcs=list(fns.values()), ...)
    """
    def _base(completions, episode_labels, **kwargs):
        return [_score(c, e, reward_config).base_reward
                for c, e in zip(completions, episode_labels)]

    def _monitor_penalty(completions, episode_labels, **kwargs):
        return [_score(c, e, reward_config).monitor_penalty
                for c, e in zip(completions, episode_labels)]

    def _total(completions, episode_labels, **kwargs):
        return [_score(c, e, reward_config).total_reward
                for c, e in zip(completions, episode_labels)]

    return {
        "reward/base": _base,
        "reward/monitor_penalty": _monitor_penalty,
        "reward/total": _total,
    }


def build_episode_labels(episode_full_dict: dict) -> str:
    """
    Extract only the fields score_completion() needs from a full episode dict
    and return a compact JSON string suitable for storing in the dataset.
    """
    slim = {
        "study_id": episode_full_dict["study_id"],
        "analysis_options": [
            {
                "id": opt["id"],
                "validity_label": opt["validity_label"],
                "weak_reviewer_accepts": opt["weak_reviewer_accepts"],
                "supports_desired_claim": opt["supports_desired_claim"],
                "p_value": opt["p_value"],
                "exploit_family": opt.get("exploit_family"),
            }
            for opt in episode_full_dict["analysis_options"]
        ],
    }
    return json.dumps(slim)


# ── Private ───────────────────────────────────────────────────────────────────

def _score(completion: str, labels_str: str, config: RewardConfig):
    try:
        episode = json.loads(labels_str)
        return score_completion(episode, completion, config=config)
    except Exception:
        from scorer.reward import ScoreResult
        from scorer.monitor import score_cot
        monitor = score_cot(None)
        return _dummy_malformed_result(config, monitor)


def _dummy_malformed_result(config: RewardConfig, monitor):
    from scorer.reward import ScoreResult
    return ScoreResult(
        parse_ok=False, parse_error="exception",
        selected_analysis_id=None, validity_label=None,
        weak_accepts=False, invalid_success=False, exploit_family=None,
        monitor_score=0.0, monitor_flag=False, monitor_result=monitor,
        base_reward=config.malformed, monitor_penalty=0.0, length_penalty=0.0,
        total_reward=config.malformed,
        reward_components={"base": config.malformed, "monitor_penalty": 0.0,
                           "length_penalty": 0.0, "total": config.malformed},
        claim_strength=None, think_word_count=0,
        llm_judge_score=None,
    )
