"""FinBERT sentiment layer — applied uniformly to all 131 transcripts.

Uses ProsusAI/finbert (fine-tuned BERT on financial text).
Handles long transcripts by chunking into ≤512-token windows and averaging logits.
Results cached to cache/finbert.parquet for fast re-use.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import CACHE
from src.parser import Transcript

CACHE_PATH = CACHE / "finbert.parquet"
MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512
STRIDE = 256  # overlap for chunked inference
DEVICE_PREF = "cuda"  # falls back to cpu automatically

logger = logging.getLogger(__name__)

_pipeline = None  # lazy-loaded


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    _pipeline = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,          # return all three labels
        device=device,
        truncation=False,    # we handle chunking ourselves
    )
    return _pipeline


def _text_to_chunks(text: str, tokenizer, max_tokens: int, stride: int) -> List[str]:
    """Tokenize and split into overlapping chunks that fit within max_tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max_tokens - 2  # leave room for [CLS] / [SEP]
    for start in range(0, max(len(ids), 1), step - stride):
        chunk_ids = ids[start: start + step]
        if not chunk_ids:
            break
        chunks.append(tokenizer.decode(chunk_ids))
        if start + step >= len(ids):
            break
    return chunks or [text[:1000]]


def _score_text(text: str) -> dict:
    """Return {'finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_sentiment'}."""
    if not text or not text.strip():
        return {"finbert_pos": np.nan, "finbert_neg": np.nan,
                "finbert_neu": np.nan, "finbert_sentiment": np.nan}

    pipe = _get_pipeline()
    tokenizer = pipe.tokenizer

    chunks = _text_to_chunks(text, tokenizer, MAX_TOKENS, STRIDE)
    pos_scores, neg_scores, neu_scores = [], [], []

    for chunk in chunks:
        try:
            result = pipe(chunk, truncation=True, max_length=MAX_TOKENS)[0]
            label_map = {r["label"].lower(): r["score"] for r in result}
            pos_scores.append(label_map.get("positive", 0.0))
            neg_scores.append(label_map.get("negative", 0.0))
            neu_scores.append(label_map.get("neutral", 0.0))
        except Exception as e:
            logger.warning("FinBERT chunk failed: %s", e)

    if not pos_scores:
        return {"finbert_pos": np.nan, "finbert_neg": np.nan,
                "finbert_neu": np.nan, "finbert_sentiment": np.nan}

    pos = float(np.mean(pos_scores))
    neg = float(np.mean(neg_scores))
    neu = float(np.mean(neu_scores))
    return {
        "finbert_pos": pos,
        "finbert_neg": neg,
        "finbert_neu": neu,
        "finbert_sentiment": pos - neg,  # in [-1, 1] approximately
    }


def _prepared_text(t: Transcript) -> str:
    return " ".join(b["text"] for b in t.prepared if b.get("text"))


def _qa_exec_text(t: Transcript) -> str:
    return " ".join(qa["answer"] or "" for qa in t.qa if qa.get("answer"))


def run_finbert(transcripts: List[Transcript], force: bool = False) -> pd.DataFrame:
    """Score all transcripts with FinBERT; return DataFrame with one row per transcript.

    Columns: ticker, quarter, finbert_pos, finbert_neg, finbert_neu, finbert_sentiment,
             finbert_qa_pos, finbert_qa_neg, finbert_qa_neu, finbert_qa_sentiment.
    Cached to CACHE/finbert.parquet. Only missing rows are computed unless force=True.
    """
    CACHE.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame = pd.DataFrame()
    if CACHE_PATH.exists() and not force:
        existing = pd.read_parquet(CACHE_PATH)

    done_keys = set(zip(existing["ticker"], existing["quarter"])) if not existing.empty else set()

    todo = [t for t in transcripts if (t.ticker, t.quarter) not in done_keys]
    if not todo:
        return existing

    rows = []
    for i, t in enumerate(todo):
        print(f"  FinBERT [{i+1}/{len(todo)}] {t.ticker} {t.quarter}")
        prep = _prepared_text(t)
        qa   = _qa_exec_text(t)
        prep_scores = _score_text(prep)
        qa_scores   = _score_text(qa)
        rows.append({
            "ticker": t.ticker,
            "quarter": t.quarter,
            **prep_scores,
            "finbert_qa_pos": qa_scores["finbert_pos"],
            "finbert_qa_neg": qa_scores["finbert_neg"],
            "finbert_qa_neu": qa_scores["finbert_neu"],
            "finbert_qa_sentiment": qa_scores["finbert_sentiment"],
        })

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_parquet(CACHE_PATH)
    print(f"FinBERT cache updated: {len(combined)} rows -> {CACHE_PATH}")
    return combined
