"""
Human-readable episode inspector.

Usage:
  python scripts/inspect_episode.py data/train.jsonl
  python scripts/inspect_episode.py data/train.jsonl --idx 5
  python scripts/inspect_episode.py data/train.jsonl --idx 5 --show_prompt
  python scripts/inspect_episode.py data/train.jsonl --n 10   # show summary of first 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--idx",         type=int, default=0)
    parser.add_argument("--n",           type=int, default=None,
                        help="Show summary table for first N episodes")
    parser.add_argument("--show_prompt", action="store_true")
    args = parser.parse_args()

    episodes = []
    with open(args.jsonl_path) as f:
        for line in f:
            episodes.append(json.loads(line))

    if args.n is not None:
        _show_summary(episodes[: args.n])
        return

    if args.idx >= len(episodes):
        print(f"Index {args.idx} out of range (file has {len(episodes)} episodes).")
        sys.exit(1)

    ep = episodes[args.idx]
    _show_episode(ep, show_prompt=args.show_prompt)


def _show_episode(ep: dict, show_prompt: bool = False):
    print("=" * 72)
    print(f"STUDY ID  : {ep['study_id']}")
    print(f"DOMAIN    : {ep['domain']}")
    print(f"SEED      : {ep['seed']}")
    print()
    print("CONTEXT")
    print(f"  {ep['study_context']}")
    print()
    print(f"DESIRED CLAIM: {ep['desired_claim']}")
    print()

    plan = ep["preregistered_plan"]
    print("PREREGISTERED PLAN")
    print(f"  Primary outcome   : {plan['primary_outcome']}")
    print(f"  Test              : {plan['allowed_test']}")
    print(f"  Covariates        : {plan['allowed_covariates']}")
    print(f"  Exclusions        : {plan['allowed_exclusions']}")
    print(f"  Subgroups         : {'confirmatory' if plan['subgroups_confirmatory'] else 'exploratory only'}")
    print()

    rev = ep["reviewer_policy"]
    print("REVIEWER POLICY")
    print(f"  Narrative : {rev['narrative']}")
    print(f"  Checks    : {rev['checks']}")
    print(f"  Blind spots: {rev['blind_spots']}")
    print()

    print("ANALYSIS OPTIONS")
    header = f"  {'ID':<3} {'Validity':<22} {'ExploitFamily':<22} {'p':>6} {'effect':>8} {'supports':>9} {'reviewer':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for opt in ep["analysis_options"]:
        accepts = "ACCEPT" if opt["weak_reviewer_accepts"] else "reject"
        supports = "yes" if opt["supports_desired_claim"] else "no"
        family = opt.get("exploit_family") or "—"
        is_best = (
            opt["validity_label"] == "invalid"
            and opt["weak_reviewer_accepts"]
        )
        marker = " ← TARGET" if is_best else ""
        print(
            f"  {opt['id']:<3} {opt['validity_label']:<22} {family:<22} "
            f"{opt['p_value']:>6.3f} {opt['effect']:>+8.3f} {supports:>9} {accepts:>9}{marker}"
        )
        print(f"       {opt['description']}")
    print()

    has_is = ep.get("has_invalid_success", False)
    is_ids = ep.get("invalid_success_ids", [])
    valid_id = ep.get("valid_confirmatory_id")
    print(f"HAS INVALID SUCCESS: {has_is}  ids={is_ids}  valid_confirmatory={valid_id}")

    if show_prompt:
        print()
        print("PROMPT")
        print("-" * 72)
        print(ep["prompt"])

    print("=" * 72)


def _show_summary(episodes: list[dict]):
    from collections import Counter
    exploit_counts: Counter = Counter()
    domains: Counter = Counter()
    n_has_is = 0
    n_obvious_fraud = 0

    print(f"\n{'IDX':<5} {'DOMAIN':<15} {'HAS_IS':<7} {'IS_IDS':<12} {'REVIEWER_CHECKS'}")
    print("-" * 70)

    for i, ep in enumerate(episodes):
        domains[ep["domain"]] += 1
        has_is = ep.get("has_invalid_success", False)
        if has_is:
            n_has_is += 1
        is_ids = ep.get("invalid_success_ids", [])
        checks = ep["reviewer_policy"]["checks"]

        for opt in ep["analysis_options"]:
            if opt["validity_label"] == "obvious_fraud":
                n_obvious_fraud += 1
            if opt["validity_label"] == "invalid" and opt["weak_reviewer_accepts"]:
                exploit_counts[opt.get("exploit_family", "?")] += 1

        print(
            f"{i:<5} {ep['domain']:<15} {str(has_is):<7} {str(is_ids):<12} {checks}"
        )

    print()
    print(f"has_invalid_success: {n_has_is}/{len(episodes)}")
    print(f"obvious_fraud options: {n_obvious_fraud}")
    print(f"domain distribution: {dict(domains)}")
    print(f"invalid_success exploit families: {dict(exploit_counts)}")


if __name__ == "__main__":
    main()
