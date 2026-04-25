"""Classify each extracted risk as proactively raised (prepared remarks) or
reactively surfaced (only in Q&A). Pure keyword-overlap heuristic - no LLM call.
"""
from __future__ import annotations

import re
from typing import Dict, List

_WORD_RE = re.compile(r"[a-z]{4,}")

_STOPWORDS = {
    "about", "against", "because", "could", "does", "during", "from", "have",
    "might", "other", "should", "than", "that", "their", "them", "there", "these",
    "they", "this", "those", "through", "under", "very", "were", "what", "when",
    "where", "which", "while", "with", "would", "your", "into", "some", "will",
    "been", "being", "across", "more", "most", "upon", "some", "like",
}

PROACTIVE_THRESHOLD = 0.4


def _content_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _overlap_score(risk_words: List[str], haystack_lower: str) -> float:
    if not risk_words:
        return 0.0
    hits = sum(1 for w in risk_words if w in haystack_lower)
    return hits / len(risk_words)


def classify_risks(prepared_text: str, qa_answer_text: str, risks: List[str]) -> List[Dict]:
    """For each risk string, label "proactive" | "reactive" | "unknown" by which
    segment of the call the risk keywords appear in.
    """
    prepared_lower = (prepared_text or "").lower()
    qa_lower = (qa_answer_text or "").lower()
    out: List[Dict] = []
    for raw in risks or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        words = _content_words(raw)
        p_score = _overlap_score(words, prepared_lower)
        q_score = _overlap_score(words, qa_lower)
        if p_score >= PROACTIVE_THRESHOLD and p_score >= q_score:
            label = "proactive"
        elif q_score >= PROACTIVE_THRESHOLD:
            label = "reactive"
        else:
            label = "unknown"
        out.append({
            "risk": raw,
            "label": label,
            "prepared_score": round(p_score, 3),
            "qa_score": round(q_score, 3),
        })
    return out


def counts(classified: List[Dict]) -> Dict[str, float]:
    """Summary counts + reactive ratio. ratio is NaN if both are zero."""
    p = sum(1 for c in classified if c["label"] == "proactive")
    r = sum(1 for c in classified if c["label"] == "reactive")
    u = sum(1 for c in classified if c["label"] == "unknown")
    denom = p + r
    ratio = (r / denom) if denom > 0 else float("nan")
    return {
        "proactive_risk_count": p,
        "reactive_risk_count": r,
        "unknown_risk_count": u,
        "reactive_risk_ratio": ratio,
    }
