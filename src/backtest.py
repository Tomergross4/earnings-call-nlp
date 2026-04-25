"""Toy backtest: hit rate, rank IC, Sharpe, equity-curve plot.

Two backtest flavours:
  * `run` / `equity_curve`                  : time-series, one position per call.
  * `run_cross_sectional` / ...curve_xs     : group calls by reporting_period and
                                              long the top-half / short the bottom-half.
                                              Needs >=2 calls per period to produce
                                              a signal; most current periods have 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import FIGURES, PRIMARY_HORIZON

RET_COL = f"fwd_excess_{PRIMARY_HORIZON}d"


def run(df: pd.DataFrame, signal: pd.Series, horizon: int = PRIMARY_HORIZON) -> Dict[str, float]:
    d = df.assign(signal=signal.astype(float)).dropna(subset=["signal", RET_COL])
    d["pnl"] = d["signal"] * d[RET_COL]
    # Hit rate counts only rows where the model took a position (signal != 0).
    # Hold predictions are abstentions, not losses.
    trades = d[d.signal != 0]
    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl < 0]
    hit = float((np.sign(trades.pnl) > 0).mean()) if len(trades) else float("nan")
    ic = float(d[["signal", RET_COL]].corr(method="spearman").iloc[0, 1]) if len(d) > 2 else float("nan")
    avg = float(d.pnl.mean()) if len(d) else float("nan")
    std = float(d.pnl.std()) if len(d) > 1 else float("nan")
    sharpe = avg / (std + 1e-9) * np.sqrt(252 / horizon) if std and std > 0 else float("nan")

    # F1 metrics — treat as classification: true label = sign of actual return
    y_true = np.sign(d[RET_COL].to_numpy()).astype(int)   # {-1, 0, +1}
    y_pred = d["signal"].to_numpy().astype(int)            # {-1, 0, +1}
    # Binary F1: up (+1) vs not-up; most meaningful for long-bias strategies
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    # Macro F1 over all three classes present in predictions
    labels = sorted(set(y_true) | set(y_pred))
    try:
        f1_bin  = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        prec_bin = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec_bin  = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    except Exception:
        f1_bin = prec_bin = rec_bin = f1_macro = f1_weighted = float("nan")

    return {
        "n":           int(len(d)),
        "n_trades":    int(len(trades)),
        "hit_rate":    hit,
        "rank_ic":     ic,
        "avg_excess":  avg,
        "avg_win":     float(wins.pnl.mean()) if len(wins) else float("nan"),
        "avg_loss":    float(losses.pnl.mean()) if len(losses) else float("nan"),
        "naive_sharpe": sharpe,
        # Classification metrics
        "f1_binary":   f1_bin,      # F1 for up vs not-up prediction
        "precision":   prec_bin,
        "recall":      rec_bin,
        "f1_macro":    f1_macro,    # macro-averaged over {-1, 0, +1}
        "f1_weighted": f1_weighted, # weighted by support
    }


def _spy_forward_returns(call_dates: pd.Series, horizon: int) -> pd.Series:
    """Compute SPY absolute forward return for each call date from price cache."""
    spy_path = Path("cache/prices/SPY.parquet")
    if not spy_path.exists():
        return pd.Series(np.nan, index=call_dates.index)
    spy = pd.read_parquet(spy_path).sort_values("Date").reset_index(drop=True)
    spy["Date"] = pd.to_datetime(spy["Date"])
    rets = []
    for cd in call_dates:
        cd = pd.Timestamp(cd)
        entry = spy[spy.Date > cd].head(1)
        if entry.empty:
            rets.append(np.nan)
            continue
        ei = int(entry.index[0])
        if ei + horizon >= len(spy):
            rets.append(np.nan)
            continue
        rets.append(float(spy.Close.iloc[ei + horizon] / spy.Close.iloc[ei] - 1))
    return pd.Series(rets, index=call_dates.index)


def equity_curve(df: pd.DataFrame, signal: pd.Series, save_path: Path = FIGURES / "equity_curve.png") -> Path:
    d = (
        df.assign(signal=signal.astype(float))
        .dropna(subset=["signal", RET_COL])
        .sort_values("call_date")
        .copy()
    )
    d["pnl"] = d["signal"] * d[RET_COL]
    d["cum_strategy"] = d["pnl"].cumsum()
    d["cum_long_only"] = d[RET_COL].cumsum()

    # SPY absolute cumulative return aligned to the same call dates
    d["spy_ret"] = _spy_forward_returns(d["call_date"], PRIMARY_HORIZON)
    d["cum_spy"] = d["spy_ret"].cumsum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d["call_date"], d["cum_strategy"], marker="o", color="#6366F1", label="Model signal (excess vs SPY)")
    ax.plot(d["call_date"], d["cum_long_only"], marker="x", linestyle="--", color="#9CA3AF", label="Always long (excess vs SPY)")
    if d["cum_spy"].notna().any():
        ax.plot(d["call_date"], d["cum_spy"], marker="s", linestyle=":", color="#F59E0B",
                linewidth=1.8, label="S&P 500 (SPY absolute)")
    ax.axhline(0, linestyle=":", color="#E5E7EB", linewidth=1)
    ax.set_title(f"Cumulative return vs S&P 500 ({PRIMARY_HORIZON}d holds)", fontsize=13)
    ax.set_xlabel("Call date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def run_cross_sectional(df: pd.DataFrame, signal_score: pd.Series, horizon: int = PRIMARY_HORIZON) -> Dict[str, float]:
    """Cross-sectional long/short within each reporting_period.

    signal_score : *scalar* score per call (e.g. overall_sentiment or model
                   predict_proba) — NOT the sign-bucketed signal. Ranked within
                   each period; long top half, short bottom half, equal-weight.
    Returns aggregated metrics across periods. Skips periods with <2 obs.
    """
    d = df.assign(score=signal_score.astype(float)).dropna(subset=["score", RET_COL]).copy()
    if "reporting_period" not in d.columns:
        return {"n_periods": 0, "hit_rate": float("nan"), "avg_excess": float("nan"), "naive_sharpe": float("nan")}

    period_pnls = []
    for period, group in d.groupby("reporting_period"):
        if len(group) < 2:
            continue
        ranks = group["score"].rank(method="average")
        median = ranks.median()
        leg = np.where(ranks > median, 1, np.where(ranks < median, -1, 0))
        pnl = float(np.mean(leg * group[RET_COL].to_numpy()))
        period_pnls.append((period, pnl))

    if not period_pnls:
        return {"n_periods": 0, "hit_rate": float("nan"), "avg_excess": float("nan"), "naive_sharpe": float("nan")}

    pnls = np.array([p for _, p in period_pnls], dtype=float)
    hit = float((pnls > 0).mean())
    avg = float(pnls.mean())
    std = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = avg / (std + 1e-9) * np.sqrt(252 / horizon) if std > 0 else float("nan")
    return {
        "n_periods": int(len(pnls)),
        "hit_rate": hit,
        "avg_excess": avg,
        "naive_sharpe": sharpe,
    }


def equity_curve_cross_sectional(
    df: pd.DataFrame,
    signal_score: pd.Series,
    save_path: Path = FIGURES / "equity_cross_sectional.png",
) -> Path:
    d = df.assign(score=signal_score.astype(float)).dropna(subset=["score", RET_COL]).copy()
    if "reporting_period" not in d.columns:
        return save_path

    rows = []
    for period, group in d.sort_values("call_date").groupby("reporting_period"):
        if len(group) < 2:
            continue
        ranks = group["score"].rank(method="average")
        median = ranks.median()
        leg = np.where(ranks > median, 1, np.where(ranks < median, -1, 0))
        pnl = float(np.mean(leg * group[RET_COL].to_numpy()))
        rows.append({"reporting_period": period, "pnl": pnl, "call_date": group["call_date"].max()})

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if rows:
        tbl = pd.DataFrame(rows).sort_values("call_date")
        tbl["cum"] = tbl["pnl"].cumsum()
        ax.plot(tbl["call_date"], tbl["cum"], marker="o", label="Cross-sectional long/short")
        ax.axhline(0, linestyle=":", color="gray")
        ax.set_title(f"Cross-sectional excess return per reporting period ({PRIMARY_HORIZON}d holds)")
    else:
        ax.text(0.5, 0.5, "No periods with >=2 calls yet", ha="center", va="center")
        ax.set_title("Cross-sectional backtest (insufficient data)")
    ax.set_xlabel("Reporting period"); ax.set_ylabel("Cumulative excess return")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
