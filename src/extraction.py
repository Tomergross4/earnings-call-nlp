"""Hybrid LLM extraction: 1 overall call + 3 speaker-focused calls per transcript."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.config import EXTRACTIONS, OLLAMA_MODEL
from src.llm import call_ollama, salvage_json
from src.parser import Transcript

MAX_CHARS_OVERALL = 18_000
MAX_CHARS_SPEAKER = 10_000

OVERALL_PROMPT = """You are a financial analyst. Analyze this earnings-call transcript and return STRICT JSON.

Schema (return exactly these keys, nothing else):
{
  "overall_sentiment": <float in [-1,1]>,
  "sentiment_bucket": <one of "very_bearish","bearish","neutral","bullish","very_bullish">,
  "wins":      [<up to 5 short strings, concrete positive events>],
  "risks":     [<up to 5 short strings, concrete negative events>],
  "guidance":  <one of "raised","reaffirmed","lowered","mixed","none">,
  "themes":    [<short thematic tags, e.g. "ai","china","pricing","capex","macro">]
}

Rules:
- Ground every field in the transcript. Do not invent.
- Output ONLY the JSON object. No prose, no markdown fences, no commentary.

TRANSCRIPT:
{transcript}
"""

SPEAKER_PROMPT = """You are a financial analyst. Based ONLY on the following text from {role} during an earnings call, rate their sentiment.

Return STRICT JSON exactly in this schema:
{{"sentiment": <float in [-1,1]>, "rationale": <one short sentence>}}

Rules:
- -1 = very bearish; 0 = neutral; +1 = very bullish.
- Output ONLY the JSON object. No prose.

TEXT:
{text}
"""

_CEO_RE = re.compile(r"CEO|Chief Executive", re.IGNORECASE)
_CFO_RE = re.compile(r"CFO|Chief Financial", re.IGNORECASE)


def _overall_text(t: Transcript) -> str:
    parts: List[str] = [f'[PREPARED — {b["role"]}]\n{b["text"]}' for b in t.prepared]
    for qa in t.qa:
        parts.append(f'[Q — {qa["q_role"]}] {qa["question"]}\n[A — {qa["a_role"]}] {qa["answer"] or ""}')
    return "\n\n".join(parts)[:MAX_CHARS_OVERALL]


def _ceo_text(t: Transcript) -> str:
    parts = [b["text"] for b in t.prepared if b["role"] and _CEO_RE.search(b["role"])]
    parts += [qa["answer"] or "" for qa in t.qa if qa.get("a_role") and _CEO_RE.search(qa["a_role"])]
    return "\n\n".join(p for p in parts if p.strip())[:MAX_CHARS_SPEAKER]


def _cfo_text(t: Transcript) -> str:
    parts = [b["text"] for b in t.prepared if b["role"] and _CFO_RE.search(b["role"])]
    parts += [qa["answer"] or "" for qa in t.qa if qa.get("a_role") and _CFO_RE.search(qa["a_role"])]
    return "\n\n".join(p for p in parts if p.strip())[:MAX_CHARS_SPEAKER]


def _analyst_text(t: Transcript) -> str:
    parts = [qa["question"] for qa in t.qa if qa.get("question")]
    return "\n\n".join(parts)[:MAX_CHARS_SPEAKER]


def _cache_paths(ticker: str, quarter: str, call_type: str) -> tuple[Path, Path]:
    model_tag = OLLAMA_MODEL.replace(":", "-")
    base = f"{ticker}_{quarter}_{model_tag}_{call_type}"
    return EXTRACTIONS / f"{base}.json", EXTRACTIONS / f"{base}.raw.txt"


def _extract_overall(t: Transcript, force: bool) -> dict:
    json_path, raw_path = _cache_paths(t.ticker, t.quarter, "overall")
    if json_path.exists() and not force:
        return json.loads(json_path.read_text())
    prompt = OVERALL_PROMPT.replace("{transcript}", _overall_text(t))
    raw = call_ollama(prompt)
    raw_path.write_text(raw, encoding="utf-8")
    obj = salvage_json(raw)
    json_path.write_text(json.dumps(obj, indent=2))
    return obj


def _extract_speaker(t: Transcript, role_label: str, text: str, force: bool) -> dict:
    call_type = f"speaker_{role_label.lower()}"
    json_path, raw_path = _cache_paths(t.ticker, t.quarter, call_type)
    if json_path.exists() and not force:
        return json.loads(json_path.read_text())
    if not text.strip():
        obj = {"sentiment": None, "rationale": f"{role_label} not present in transcript"}
        json_path.write_text(json.dumps(obj, indent=2))
        return obj
    prompt = SPEAKER_PROMPT.format(role=role_label, text=text)
    raw = call_ollama(prompt, num_predict=256)
    raw_path.write_text(raw, encoding="utf-8")
    obj = salvage_json(raw)
    json_path.write_text(json.dumps(obj, indent=2))
    return obj


def extract_one(t: Transcript, force: bool = False) -> dict:
    """Run 4 LLM calls for one transcript; return combined dict."""
    overall = _extract_overall(t, force)
    ceo = _extract_speaker(t, "CEO", _ceo_text(t), force)
    cfo = _extract_speaker(t, "CFO", _cfo_text(t), force)
    analyst = _extract_speaker(t, "Analyst", _analyst_text(t), force)
    return {
        "_ticker": t.ticker,
        "_quarter": t.quarter,
        "_call_date": t.call_date,
        "overall": overall,
        "ceo": ceo,
        "cfo": cfo,
        "analyst": analyst,
    }


def extract_all(transcripts: List[Transcript], force: bool = False) -> List[dict]:
    return [extract_one(t, force) for t in transcripts]
