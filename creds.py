"""
Credential loading for all scripts in this project.

Loads .env from the project root (if present), then reads required
environment variables. Import and call load() at the top of any script
that needs credentials before doing anything else.

Usage:
    from creds import load
    load()   # populates os.environ from .env, raises if required keys missing

Or to check specific services:
    from creds import load, require
    load()
    hf_token = require("HF_TOKEN", "needed to download Qwen3 from HuggingFace")
"""

from __future__ import annotations
import os
import sys


def load(env_file: str = ".env") -> None:
    """
    Load .env into os.environ (silently skips if the file doesn't exist).
    Call this once at the start of any script that needs credentials.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        # python-dotenv not installed — rely on environment variables already set
        pass

    # Wire up the HuggingFace token so the transformers library picks it up
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)


def require(key: str, hint: str = "") -> str:
    """
    Return os.environ[key] or exit with a helpful message.
    """
    val = os.environ.get(key)
    if not val:
        hint_str = f"\n  Hint: {hint}" if hint else ""
        sys.exit(
            f"[creds] Required environment variable {key!r} is not set.\n"
            f"  Copy .env.example → .env and fill in your credentials.{hint_str}"
        )
    return val


def wandb_enabled() -> bool:
    return os.environ.get("WANDB_DISABLED", "false").lower() not in ("true", "1", "yes")
