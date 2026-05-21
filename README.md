# cot_faithful

RL environment for studying **unfaithful chain-of-thought reasoning** in language models.

The model learns to p-hack synthetic scientific reports (selecting invalid analyses that a weak reviewer still accepts) while a CoT monitor penalises suspicious visible reasoning. The core research question: does the model's misbehaviour route through the visible reasoning trace, or is it a shallow learned policy that bypasses it entirely?

---

## Hardware requirements

- GPU: A100 40/80 GB or H100 80 GB recommended. A single GPU is sufficient.
- The debug config (Qwen3-4B, 200 examples) fits on a 3090/4090 with `use_4bit: true`.
- Disk: ~20 GB for model weights, <1 GB for data/checkpoints.

---

## One-time setup

```bash
# 1. Clone and install
git clone https://github.com/kwynroe/cot_faithful.git
cd cot_faithful
pip install -r requirements.txt
pip install anthropic          # only needed for --llm_judge eval runs

# 2. Credentials
cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN       — huggingface.co/settings/tokens  (read access, needed for Qwen3)
#   WANDB_API_KEY  — wandb.ai/settings               (or set WANDB_DISABLED=true)
#   ANTHROPIC_API_KEY — console.anthropic.com        (only for --llm_judge eval)

# 3. Generate data
python scripts/generate_data.py --n_train 5000 --n_val 500 --n_test 1000
# Fast, CPU-only, ~30 seconds. Outputs to data/train.jsonl etc.
# For a quick smoke test: --n_train 200 --n_val 100 --n_test 100
```

---

## Running

### Step 1 — Debug run (do this first)

Verifies the full pipeline works on a small Qwen3-4B run before spending GPU budget.

```bash
python training/train_grpo.py --config training/configs/grpo_debug.yaml
```

**Takes ~20–40 min on an A100.** Check W&B for:
- `reward/total` climbing above −1.0 (model learning to format output)
- `parse_rate` reaching >50% within the first 50 steps
- Any Python errors → see Troubleshooting below

### Step 2 — Main experiment (two conditions)

Run both conditions to enable comparison. Start with no-monitor to confirm reward
signal works, then add monitor pressure.

```bash
# Condition 1: no monitor — establishes reward ceiling and shows how openly
# the model verbalises p-hacking when there's no cost to doing so
python training/train_grpo.py --config training/configs/grpo_no_monitor.yaml

# Condition 2: monitor-penalized — the main experiment
# Model is rewarded for p-hacking but penalised for suspicious CoT
python training/train_grpo.py --config training/configs/grpo_monitor.yaml
```

You can override any config field from the CLI:
```bash
python training/train_grpo.py --config training/configs/grpo_monitor.yaml \
    --num_generations 8 --monitor_weight 2.0 --run_name grpo_monitor_heavy
```

### Step 3 — Evaluate

```bash
# Score a checkpoint on the test set
python eval/run_eval.py \
    --model Qwen/Qwen3-8B \
    --checkpoint checkpoints/grpo_monitor/final \
    --data data/test.jsonl \
    --outdir runs/grpo_monitor_test

# Add --llm_judge to run Claude on the CoT blocks (costs ~$0.01-0.05 per 100 episodes)
python eval/run_eval.py ... --llm_judge
```

### Step 4 — CoT ablations (the key diagnostic)

```bash
python eval/run_interventions.py \
    --model Qwen/Qwen3-8B \
    --checkpoint checkpoints/grpo_monitor/final \
    --data data/test.jsonl \
    --outdir runs/interventions/grpo_monitor \
    --conditions normal empty_cot honest_cot
```

Outputs `intervention_summary.json` with the CoT-dependence deltas:
- `cot_dependence.invalid_success_delta`: how much invalid-success drops when think block is emptied
- `honest_cot_steering`: how much honest injected reasoning steers the model

**Large delta → behaviour routes through visible CoT (non-vestigial).
Near-zero → direct policy (vestigial). That's the core result.**

---

## What to look for in W&B

| Metric | Healthy sign | Warning sign |
|--------|-------------|--------------|
| `reward/total` | Climbing from ~−1 toward +3 | Stuck at −1 (format not learned) |
| `parse_rate` | >70% by step 100 | <50% after 200 steps → need SFT warmup |
| `invalid_success_rate` | Rising during no-monitor run | Stays near 0 → reward too sparse |
| `exploit_family_entropy` | >1.5 (diverse strategies) | <1.0 (collapsed to one family) |
| `lowest_p_heuristic_rate` | Low (<0.2) | High → model ignoring reviewer policy |
| `mean_monitor_score` | Lower in monitor vs no-monitor | Same in both → monitor not shaping CoT |

---

## If things go wrong

**`parse_rate` stays low (<50%)** — the model isn't learning the `<think>...<report>` format.
Build an SFT warmup: ~500 episodes with gold-label completions, supervised on the format.
The GRPO reward signal is too sparse when the model can't produce valid output.

**`invalid_success_rate` stays near 0** — the task is too hard cold.
Try DPO warmup first: construct chosen/rejected pairs (invalid-success + bland CoT vs. honest report),
run a short DPO LoRA, then GRPO on top.

**`loss_type: "dr_grpo"` causes a TypeError** — your TRL version is older.
Change to `loss_type: "grpo"` in the YAML. You lose dynamic sampling but training still works.

**OOM on A100 40GB** — set `use_4bit: true` in the config for QLoRA.

---

## File map

```
generator/           Synthetic episode factory
  templates.py         24 study templates (6 domains × 4)
  exploits.py          6 exploit families + p-value computation
  reviewer.py          Per-episode reviewer policy (the reasoning-dependence mechanism)
  generator.py         EpisodeGenerator

scorer/              Scoring stack (no GPU needed, importable anywhere)
  parser.py            Parses <think>/<report> from raw completions
  monitor.py           Rule-based CoT monitor (keyword + regex → 0-1 score)
  reward.py            score_completion() → reward + all components
  metrics.py           Batch metrics + CoT-dependence deltas
  llm_judge.py         Claude-based 1-10 CoT evaluator

training/
  model.py             load_model_and_tokenizer() — Qwen3 + LoRA/QLoRA
  reward_fn.py         TRL reward function wrapper (+ optional LLM judge)
  train_grpo.py        Main training script
  configs/
    grpo_debug.yaml      Qwen3-4B, 200 examples — run this first
    grpo_no_monitor.yaml Condition 1: reward only, no CoT penalty
    grpo_monitor.yaml    Condition 2: reward + CoT monitor penalty (main experiment)

eval/
  inference.py         load_model_for_eval(), generate_completion(), intervention builders
  run_eval.py          Batch inference + scoring
  run_interventions.py All 5 CoT ablation conditions + dependence deltas

scripts/
  generate_data.py     Generate train/val/test JSONL
  inspect_episode.py   Human-readable episode viewer

creds.py             Loads .env → os.environ
.env.example         Credential template
```

---

## Credentials reference

| Variable | Required for | Where to get it |
|----------|-------------|----------------|
| `HF_TOKEN` | Downloading Qwen3 | huggingface.co/settings/tokens |
| `WANDB_API_KEY` | Training run logging | wandb.ai/settings (or set `WANDB_DISABLED=true`) |
| `ANTHROPIC_API_KEY` | `--llm_judge` eval flag + `use_llm_judge_monitor: true` | console.anthropic.com |
| `ANTHROPIC_JUDGE_MODEL` | Judge model override | Default: `claude-haiku-4-5-20251001` |
