"""Data loaders for the Streamlit app (no LLM calls, pure reads)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

from src.config import EXTRACTIONS, OUTPUTS


def _maybe_parse_json(v):
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(OUTPUTS / "features.parquet")
    for col in ("themes", "wins", "risks", "risks_classified"):
        if col in df.columns:
            df[col] = df[col].apply(_maybe_parse_json)
    df["call_date"] = pd.to_datetime(df["call_date"])
    return df.sort_values(["ticker", "call_date"]).reset_index(drop=True)


def load_raw_transcript_blocks(ticker: str, quarter: str) -> Optional[Dict]:
    """Return prepared + qa content from the on-disk transcripts/ by parsing fresh."""
    from src.parser import parse_transcript
    from src.config import TRANSCRIPTS
    path = TRANSCRIPTS / f"{ticker}_{quarter}.txt"
    if not path.exists():
        return None
    t = parse_transcript(path)
    return {"prepared": t.prepared, "qa": t.qa, "call_date": t.call_date, "company": t.company}


def list_tickers_and_quarters(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for tk, grp in df.groupby("ticker"):
        out[tk] = grp.sort_values("call_date")["quarter"].tolist()
    return out
