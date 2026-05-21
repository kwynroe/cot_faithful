"""
Generate train/val/test JSONL splits.

Usage:
  python scripts/generate_data.py
  python scripts/generate_data.py --n_train 5000 --n_val 500 --n_test 1000
  python scripts/generate_data.py --n_train 100 --outdir data/debug

Each JSONL line is a full episode dict (Episode.to_full_dict()), including
the model-visible prompt and all hidden ground-truth labels.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import creds
creds.load()

from generator.generator import EpisodeGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val",   type=int, default=500)
    parser.add_argument("--n_test",  type=int, default=1000)
    parser.add_argument("--outdir",  type=str, default="data")
    parser.add_argument("--seed_offset_train", type=int, default=0)
    parser.add_argument("--seed_offset_val",   type=int, default=100_000)
    parser.add_argument("--seed_offset_test",  type=int, default=200_000)
    parser.add_argument("--n_reviewer_checks", type=int, default=2,
                        help="Number of exploit families the reviewer catches (default 2 of 6)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gen = EpisodeGenerator(n_reviewer_checks=args.n_reviewer_checks)

    splits = [
        ("train", args.n_train, args.seed_offset_train),
        ("val",   args.n_val,   args.seed_offset_val),
        ("test",  args.n_test,  args.seed_offset_test),
    ]

    for split_name, n, offset in splits:
        path = os.path.join(args.outdir, f"{split_name}.jsonl")
        print(f"Generating {n} {split_name} episodes → {path}")
        with open(path, "w") as f:
            for i in range(n):
                episode = gen.generate(seed=offset + i, split=split_name)
                f.write(json.dumps(episode.to_full_dict()) + "\n")
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{n}")

    print("Done.")
    _print_summary(args.outdir)


def _print_summary(outdir: str):
    from collections import Counter

    for split_name in ["train", "val", "test"]:
        path = os.path.join(outdir, f"{split_name}.jsonl")
        if not os.path.exists(path):
            continue

        domains: Counter = Counter()
        exploit_counts: Counter = Counter()
        has_invalid_success = 0
        total = 0

        with open(path) as f:
            for line in f:
                ep = json.loads(line)
                total += 1
                domains[ep["domain"]] += 1
                if ep.get("has_invalid_success"):
                    has_invalid_success += 1
                for opt in ep.get("analysis_options", []):
                    if opt["validity_label"] == "invalid" and opt["weak_reviewer_accepts"]:
                        exploit_counts[opt.get("exploit_family", "unknown")] += 1

        print(f"\n{split_name} ({total} episodes):")
        print(f"  has_invalid_success: {has_invalid_success}/{total} ({100*has_invalid_success/total:.1f}%)")
        print(f"  domains: {dict(domains)}")
        print(f"  invalid_success exploit families: {dict(exploit_counts)}")


if __name__ == "__main__":
    main()
