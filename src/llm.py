"""Ollama client + forgiving JSON parser."""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
)


def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
    temperature: float = OLLAMA_TEMPERATURE,
    timeout: int = 3600,
) -> str:
    """Call Ollama /api/generate, returning the raw `response` string.

    Retries once with 2s backoff on network errors. `think: false` is passed
    both top-level and inside `options` for Gemma/Qwen compatibility.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "think": False,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except (requests.RequestException, KeyError, ValueError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(f"Ollama call failed after retry: {last_err}")


def salvage_json(raw: str) -> dict:
    """Best-effort JSON parse of messy LLM output.

    Strips markdown fences, closed-or-unclosed <think> blocks, trims to the
    outermost {...}, fixes trailing commas, normalizes Python literals.
    Returns {} on failure (caller should check).
    """
    s = raw
    s = re.sub(r"```(?:json)?", "", s)
    s = re.sub(r"<think>.*?(</think>|$)", "", s, flags=re.DOTALL)
    lo, hi = s.find("{"), s.rfind("}")
    if lo >= 0 and hi > lo:
        s = s[lo : hi + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(s)
    except Exception:
        return {}
