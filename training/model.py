"""
Model and tokenizer loading for GRPO fine-tuning.

Supports:
  - Full bf16 loading (A100/H100)
  - 4-bit QLoRA (smaller GPUs / debugging)
  - LoRA adapter wrapping via PEFT

Returns model with LoRA adapter already attached and print_trainable_parameters()
called so the user can verify the adapter rank immediately.
"""

from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Qwen3 attention + MLP projection names (same for 4B / 8B / 14B variants)
QWEN3_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# System prompt prepended to every episode prompt.
# Instructs the model on output format and reminds it that thinking is expected.
SYSTEM_PROMPT = (
    "You are a research scientist writing scientific reports. "
    "Think carefully through the analysis options, then provide your report. "
    "Always output your reasoning inside <think>...</think> tags followed by "
    "your report inside <report>...</report> tags containing valid JSON."
)


def load_model_and_tokenizer(
    model_name: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    gradient_checkpointing: bool = True,
) -> tuple:
    """
    Load model and tokenizer, wrap with LoRA adapter.

    Returns:
        (model, tokenizer)
    """
    tokenizer = _load_tokenizer(model_name)
    model = _load_base_model(model_name, use_4bit)

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = _attach_lora(model, lora_rank, lora_alpha, lora_dropout)
    model.print_trainable_parameters()
    return model, tokenizer


def format_prompt_for_model(episode_prompt: str, tokenizer) -> str:
    """
    Wrap the episode prompt in the chat template expected by Qwen3.
    The returned string ends at the start of the assistant turn so the
    model generates the completion (think block + report) from there.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": episode_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding for decoder-only generation batching
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(model_name: str, use_4bit: bool):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


def _attach_lora(
    model,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=QWEN3_LORA_TARGETS,
        bias="none",
    )
    return get_peft_model(model, lora_config)
