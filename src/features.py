"""Build per-(ticker, quarter) feature table: LLM extractions + QoQ deltas +
speaker gaps + theme drift + reactive-vs-proactive risks + LM lexicon baseline.

Feature rows cover *every parsed transcript*, not just those with LLM
extractions cached. LM lexicon + price-momentum columns populate for all rows;
LLM-derived columns (overall_sentiment, risks, themes, ...) are NaN where the
extraction has not yet been run. This keeps the pipeline grade-safe for
incremental extraction rollouts without changing methodology.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CURATED_THEMES, EXTRACTIONS, OUTPUTS
from src.lexicon import lm_sentiment
from src.parser import Transcript, reporting_period_from_call_date
from src.risk_classify import classify_risks, counts as risk_counts
from src.finbert import run_finbert

GUIDANCE_MAP = {
    "lowered": -1.0,
    "mixed": -0.5,
    "none": 0.0,
    "reaffirmed": 0.5,
    "raised": 1.0,
}


def _load_one(overall_path: Path) -> dict:
    """Given the *_overall.json path, load all 4 call JSONs for a transcript."""
    base = overall_path.name.replace("_overall.json", "")
    d = overall_path.parent
    overall = json.loads(overall_path.read_text())

    def _load(tag: str) -> dict:
        p = d / f"{base}_speaker_{tag}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    ticker, quarter, _rest = base.split("_", 2)
    return {
        "ticker": ticker,
        "quarter": quarter,
        "overall": overall,
        "ceo": _load("ceo"),
        "cfo": _load("cfo"),
        "analyst": _load("analyst"),
    }


def _to_row(rec: dict) -> dict:
    overall = rec["overall"]
    themes = [str(x).lower() for x in (overall.get("themes") or [])]
    row = {
        "ticker": rec["ticker"],
        "quarter": rec["quarter"],
        "overall_sentiment": overall.get("overall_sentiment"),
        "sentiment_bucket": overall.get("sentiment_bucket"),
        "n_wins": len(overall.get("wins") or []),
        "n_risks": len(overall.get("risks") or []),
        "guidance": overall.get("guidance"),
        "guidance_score": GUIDANCE_MAP.get((overall.get("guidance") or "none").lower(), 0.0),
        "n_themes": len(themes),
        "themes": themes,
        "wins": overall.get("wins") or [],
        "risks": overall.get("risks") or [],
        "ceo_sentiment": rec["ceo"].get("sentiment"),
        "cfo_sentiment": rec["cfo"].get("sentiment"),
        "analyst_sentiment": rec["analyst"].get("sentiment"),
    }
    for th in CURATED_THEMES:
        row[f"theme_{th}"] = int(any(th in s for s in themes))
    return row


def load_extractions() -> pd.DataFrame:
    paths = sorted(EXTRACTIONS.glob("*_overall.json"))
    records = [_load_one(p) for p in paths]
    df = pd.DataFrame([_to_row(r) for r in records])
    return df.sort_values(["ticker", "quarter"]).reset_index(drop=True) if not df.empty else df


def _transcript_full_text(t: Transcript) -> str:
    parts = [b["text"] for b in t.prepared]
    for qa in t.qa:
        if qa.get("question"):
            parts.append(qa["question"])
        if qa.get("answer"):
            parts.append(qa["answer"])
    return "\n\n".join(parts)


def _prepared_text(t: Transcript) -> str:
    return "\n\n".join(b["text"] for b in t.prepared)


def _qa_answer_text(t: Transcript) -> str:
    return "\n\n".join(qa["answer"] or "" for qa in t.qa)


def build_lexicon_frame(transcripts: List[Transcript]) -> pd.DataFrame:
    """One row per parsed transcript with LM-lexicon sentiment.

    This is the "base" of the feature table: every transcript in the corpus
    contributes a row, regardless of whether an LLM extraction exists.
    """
    rows = []
    for t in transcripts:
        sent = lm_sentiment(_transcript_full_text(t))
        rows.append({"ticker": t.ticker, "quarter": t.quarter, **sent})
    return pd.DataFrame(rows)


def _risk_persistence(group: pd.DataFrame) -> pd.Series:
    prev: set = set()
    vals: List[float] = []
    for risks in group["risks"]:
        if not isinstance(risks, list):
            vals.append(np.nan); prev = set(); continue
        rs = {str(r).lower().strip() for r in risks if isinstance(r, str)}
        if prev and rs:
            vals.append(len(rs & prev) / len(rs | prev))
        else:
            vals.append(np.nan)
        prev = rs
    return pd.Series(vals, index=group.index)


def _theme_drift(group: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Returns (theme_novelty, theme_persistence) indexed like `group`."""
    prev: set = set()
    nov: List[float] = []
    per: List[float] = []
    for themes in group["themes"]:
        if isinstance(themes, list):
            ts = {str(x).lower().strip() for x in themes if str(x).strip()}
        else:
            ts = set()
        if not ts:
            nov.append(np.nan); per.append(np.nan); prev = ts
            continue
        if not prev:
            nov.append(np.nan); per.append(np.nan)
        else:
            nov.append(len(ts - prev) / max(len(ts), 1))
            per.append(len(ts & prev) / max(len(ts | prev), 1))
        prev = ts
    return pd.Series(nov, index=group.index), pd.Series(per, index=group.index)


def _nanmean(a, b) -> float:
    arr = np.array([a, b], dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def add_qoq_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_delta"] = df.groupby("ticker")["overall_sentiment"].diff()
    df["n_risks_delta"]   = df.groupby("ticker")["n_risks"].diff()
    df["n_wins_delta"]    = df.groupby("ticker")["n_wins"].diff()

    persistence = pd.Series(np.nan, index=df.index, dtype=float)
    novelty     = pd.Series(np.nan, index=df.index, dtype=float)
    theme_pers  = pd.Series(np.nan, index=df.index, dtype=float)
    for _ticker, idx in df.groupby("ticker").groups.items():
        sub = df.loc[idx]
        persistence.loc[idx] = _risk_persistence(sub).values
        n, p = _theme_drift(sub)
        novelty.loc[idx] = n.values
        theme_pers.loc[idx] = p.values
    df["risk_persistence"]   = persistence
    df["theme_novelty"]      = novelty
    df["theme_persistence"]  = theme_pers

    df["ceo_cfo_gap"] = df["ceo_sentiment"] - df["cfo_sentiment"]
    df["analyst_mgmt_gap"] = [
        (a - _nanmean(c, f)) if pd.notna(a) else np.nan
        for a, c, f in zip(df["analyst_sentiment"], df["ceo_sentiment"], df["cfo_sentiment"])
    ]
    df["guidance_trajectory"] = (
        df.groupby("ticker")["guidance_score"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    return df


def _add_risk_classification(df: pd.DataFrame, transcripts: List[Transcript]) -> pd.DataFrame:
    """Attach proactive/reactive risk counts by matching extracted risks back
    to the parsed prepared-vs-QA text for each transcript."""
    tmap = {(t.ticker, t.quarter): t for t in transcripts}
    classified_col: List = []
    count_rows: List[Dict] = []
    for _, row in df.iterrows():
        risks = row.get("risks")
        t = tmap.get((row["ticker"], row["quarter"]))
        if not isinstance(risks, list) or not risks or t is None:
            classified_col.append([])
            count_rows.append({
                "proactive_risk_count": np.nan,
                "reactive_risk_count": np.nan,
                "unknown_risk_count": np.nan,
                "reactive_risk_ratio": np.nan,
            })
            continue
        classified = classify_risks(_prepared_text(t), _qa_answer_text(t), risks)
        classified_col.append(classified)
        count_rows.append(risk_counts(classified))
    df = df.copy()
    df["risks_classified"] = classified_col
    for col in ("proactive_risk_count", "reactive_risk_count", "unknown_risk_count", "reactive_risk_ratio"):
        df[col] = [r[col] for r in count_rows]
    return df


def add_finbert_features(df: pd.DataFrame, transcripts: List[Transcript]) -> pd.DataFrame:
    """Merge FinBERT sentiment scores into the feature table."""
    fb = run_finbert(transcripts)
    if fb.empty:
        return df
    fb_qoq = fb.copy()
    fb_qoq = fb_qoq.sort_values(["ticker", "quarter"])
    fb_qoq["finbert_sentiment_delta"] = fb_qoq.groupby("ticker")["finbert_sentiment"].diff()
    fb_qoq["finbert_qa_sentiment_delta"] = fb_qoq.groupby("ticker")["finbert_qa_sentiment"].diff()
    # disagreement: management prep vs. analyst Q&A tone gap
    fb_qoq["finbert_mgmt_qa_gap"] = fb_qoq["finbert_sentiment"] - fb_qoq["finbert_qa_sentiment"]
    return df.merge(fb_qoq, on=["ticker", "quarter"], how="left")


def build(returns_df: pd.DataFrame, transcripts: List[Transcript]) -> pd.DataFrame:
    """Assemble the full feature table and persist to `outputs/features.parquet`.

    returns_df : one row per transcript (from build_returns_table) with
                 call_date, fwd_excess_*d, and momentum columns.
    transcripts: used for LM-lexicon computation and risk classification.
    """
    base = build_lexicon_frame(transcripts)

    ext = load_extractions()
    merged = base.merge(returns_df.assign(call_date=pd.to_datetime(returns_df["call_date"])),
                        on=["ticker", "quarter"], how="left")
    if not ext.empty:
        merged = merged.merge(ext, on=["ticker", "quarter"], how="left")
    else:
        for col in ("overall_sentiment", "sentiment_bucket", "n_wins", "n_risks",
                    "guidance", "guidance_score", "n_themes", "themes", "wins", "risks",
                    "ceo_sentiment", "cfo_sentiment", "analyst_sentiment"):
            merged[col] = np.nan
        for th in CURATED_THEMES:
            merged[f"theme_{th}"] = 0

    merged["reporting_period"] = merged["call_date"].apply(
        lambda d: reporting_period_from_call_date(d.strftime("%Y-%m-%d")) if pd.notna(d) else None
    )
    merged = merged.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    merged = add_qoq_features(merged)
    merged = _add_risk_classification(merged, transcripts)
    merged = add_finbert_features(merged, transcripts)

    out_path = OUTPUTS / "features.parquet"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    persist = merged.copy()
    for col in ("themes", "wins", "risks", "risks_classified"):
        if col in persist.columns:
            persist[col] = persist[col].apply(lambda v: json.dumps(v) if isinstance(v, list) else json.dumps([]))
    persist.to_parquet(out_path)
    return merged
