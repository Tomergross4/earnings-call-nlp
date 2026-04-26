"""yfinance prices + forward-return computation."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import HORIZONS_DAYS, PRICES


# BMO (Before Market Open) reporters can be entered at the same-day close (T+0)
# without look-ahead — the call has already happened by 9:30 ET. AMC (After Market
# Close) reporters print after the bell, so entry must wait one trading day (T+1).
TIMING_HABITS = {
    "JPM": "BMO", "C": "BMO", "WFC": "BMO", "GS": "BMO", "BLK": "BMO", "JNJ": "BMO", "FAST": "BMO",
    "NVDA": "AMC", "AMD": "AMC", "INTC": "AMC", "PLTR": "AMC", "NKE": "AMC", "AVGO": "AMC", "FDX": "AMC",
}


def fetch_prices(ticker: str, start: str = "2023-09-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load daily Close prices for a ticker; parquet cache for 24h."""
    PRICES.mkdir(parents=True, exist_ok=True)
    cache = PRICES / f"{ticker}.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 24 * 3600:
        return pd.read_parquet(cache)
    end = end or datetime.now().strftime("%Y-%m-%d")
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    df = df[["Close"]].reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df.to_parquet(cache)
    time.sleep(0.25)
    return df


def fetch_all(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch tickers + SPY."""
    universe = list(dict.fromkeys(list(tickers) + ["SPY"]))
    return {t: fetch_prices(t) for t in universe}


def forward_return(
    prices: Dict[str, pd.DataFrame],
    ticker: str,
    call_date: str,
    horizon: int,
    use_excess: bool = True,
) -> Optional[float]:
    """Close-to-close excess return over `horizon` trading days, BMO/AMC adjusted.

    Entry is T+0 close for BMO reporters (call already finished by the open) and
    T+1 close for AMC reporters (call prints after the close). Both paths are
    zero-lookahead: the entry bar is never earlier than the call itself.
    """
    df = prices[ticker]
    d0 = pd.Timestamp(call_date)
    timing = TIMING_HABITS.get(ticker, "AMC")
    if timing == "BMO":
        entry = df[df.Date >= d0].head(1)
    else:
        entry = df[df.Date > d0].head(1)
    if entry.empty:
        return None
    entry_date = entry.Date.iloc[0]
    entry_idx = int(df.index[df.Date == entry_date][0])
    if entry_idx + horizon >= len(df):
        return None
    exit_date = df.Date.iloc[entry_idx + horizon]
    r = float(df.Close.iloc[entry_idx + horizon] / df.Close.iloc[entry_idx] - 1)
    if use_excess:
        spy = prices["SPY"]
        sp_e = spy[spy.Date == entry_date]
        sp_x = spy[spy.Date == exit_date]
        if sp_e.empty or sp_x.empty:
            return None
        r -= float(sp_x.Close.iloc[0] / sp_e.Close.iloc[0] - 1)
    return r


def momentum_features(prices: Dict[str, pd.DataFrame], ticker: str, call_date: str) -> dict:
    """Pre-call price features (no look-ahead). Last trading day on or before call_date.

    mom_21d / mom_63d : trailing-window total return, close-to-close.
    dist_52w_high     : (last_close / max_close_trailing_252d) - 1  (<= 0).
    vol_21d           : annualized realized vol from daily log returns over trailing 21d.
    """
    out = {"mom_21d": np.nan, "mom_63d": np.nan, "dist_52w_high": np.nan, "vol_21d": np.nan}
    if ticker not in prices:
        return out
    df = prices[ticker]
    d0 = pd.Timestamp(call_date)
    # Strictly pre-call: do not use the day-T close (avoid look-ahead for AM calls)
    prior = df[df.Date < d0]
    if prior.empty:
        return out
    last_idx = int(prior.index[-1])
    last_close = float(df.Close.iloc[last_idx])

    if last_idx >= 21:
        out["mom_21d"] = last_close / float(df.Close.iloc[last_idx - 21]) - 1.0
    if last_idx >= 63:
        out["mom_63d"] = last_close / float(df.Close.iloc[last_idx - 63]) - 1.0

    lookback_start = max(0, last_idx - 251)
    window_252 = df.Close.iloc[lookback_start : last_idx + 1]
    if len(window_252) > 1:
        out["dist_52w_high"] = last_close / float(window_252.max()) - 1.0

    if last_idx >= 21:
        window_21 = df.Close.iloc[last_idx - 21 : last_idx + 1].to_numpy(dtype=float)
        log_rets = np.diff(np.log(window_21))
        if log_rets.size > 1:
            out["vol_21d"] = float(log_rets.std(ddof=1) * np.sqrt(252))
    return out


def build_returns_table(transcripts, prices) -> pd.DataFrame:
    """Assemble (ticker, quarter, call_date, fwd_excess_{h}d, momentum features) table."""
    rows = []
    for t in transcripts:
        if not t.call_date:
            continue
        row = {"ticker": t.ticker, "quarter": t.quarter, "call_date": t.call_date}
        for h in HORIZONS_DAYS:
            row[f"fwd_excess_{h}d"] = forward_return(prices, t.ticker, t.call_date, h)
        row.update(momentum_features(prices, t.ticker, t.call_date))
        rows.append(row)
    return pd.DataFrame(rows)
