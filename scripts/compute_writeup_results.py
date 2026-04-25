"""Re-run all 8 signals + per-ticker + multi-horizon, dump JSON for the writeup.

Output: outputs/writeup_results.json — consumed by the writeup-update step.
This is the single source of truth for numbers in §5/§6/§8/§12.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.app_helpers import load_features
from src.backtest import run
from src.config import HORIZONS_DAYS, OUTPUTS, PRIMARY_HORIZON
from src.model import (
    baseline_rule, fit_catboost, fit_logistic, fit_xgboost,
    finbert_rule, lexicon_rule, split_train_test,
)
from src.trained_models import fit_setfit

RET_COL = f"fwd_excess_{PRIMARY_HORIZON}d"


def _r(d):
    """Round + scrub NaN for JSON."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if np.isnan(v) else round(v, 4)
        else:
            out[k] = v
    return out


def main():
    f = load_features()
    tr, te = split_train_test(f)
    print(f"train={len(tr)} | test={len(te)} | "
          f"P(up) train={(np.sign(tr[RET_COL]) > 0).mean():.3f} "
          f"test={(np.sign(te.dropna(subset=[RET_COL])[RET_COL]) > 0).mean():.3f}")

    # ---------- 8 signals at primary horizon ----------
    print("\n=== 8 signals @ 21d ===")
    print("logistic..."); lr  = fit_logistic(tr)
    print("xgb...");      xgb = fit_xgboost(tr)
    print("catboost...");cb  = fit_catboost(tr)
    print("setfit...");  sf  = fit_setfit(tr)

    signals = {
        "Baseline (LLM sign)": baseline_rule(te).rename("sig"),
        "LM lexicon sign":     lexicon_rule(te).rename("sig"),
        "FinBERT sign":        finbert_rule(te).rename("sig"),
        "Logistic regression": lr.predict(te).rename("sig"),
        "XGBoost (Optuna)":    xgb.predict(te).rename("sig"),
        "CatBoost (Optuna)":   cb.predict(te).rename("sig"),
        "SetFit (contrastive)": sf.predict(te).rename("sig"),
        "Contrarian SetFit":   sf.predict_contrarian(te).rename("sig"),
    }

    primary_results = {}
    for name, sig in signals.items():
        primary_results[name] = _r(run(te, sig))

    # ---------- per-ticker hit rate, Contrarian SetFit ----------
    print("\n=== per-ticker hit rate (Contrarian SetFit @ 21d) ===")
    sig_c = signals["Contrarian SetFit"]
    te_set = te.dropna(subset=[RET_COL]).copy()
    te_set["signal"] = sig_c.reindex(te_set.index).fillna(0)
    te_set["pnl"] = te_set["signal"] * te_set[RET_COL]
    per_ticker = []
    for tk, g in te_set.groupby("ticker"):
        trades = g[g.signal != 0]
        n_trades = len(trades)
        wins = (trades.pnl > 0).sum() if n_trades else 0
        per_ticker.append({
            "ticker":   tk,
            "n_test":   int(len(g)),
            "n_trades": int(n_trades),
            "hits":     int(wins),
            "hit_rate": round(wins / n_trades, 3) if n_trades else None,
            "avg_pnl":  round(float(g.pnl.mean()), 4),
        })

    # ---------- multi-horizon Contrarian SetFit ----------
    print("\n=== multi-horizon Contrarian SetFit ===")
    horizon_results = []
    for h in HORIZONS_DAYS:
        ret_col_h = f"fwd_excess_{h}d"
        if ret_col_h not in te.columns:
            continue
        te_h = te.dropna(subset=[ret_col_h]).copy()
        sig_c_h = sig_c.reindex(te_h.index).fillna(0).astype(float)
        te_h["signal"] = sig_c_h
        te_h["pnl"] = te_h["signal"] * te_h[ret_col_h]
        trades = te_h[te_h.signal != 0]
        n_trades = len(trades)
        avg = float(te_h.pnl.mean()) if len(te_h) else float("nan")
        std = float(te_h.pnl.std()) if len(te_h) > 1 else float("nan")
        sharpe = (avg / (std + 1e-9) * np.sqrt(252 / h)) if std and std > 0 else float("nan")
        ic = float(te_h[["signal", ret_col_h]].corr(method="spearman").iloc[0, 1]) if len(te_h) > 2 else float("nan")
        horizon_results.append(_r({
            "horizon":   h,
            "n":         int(len(te_h)),
            "n_trades":  int(n_trades),
            "hit_rate":  float((trades.pnl > 0).mean()) if n_trades else float("nan"),
            "rank_ic":   ic,
            "avg_excess": avg,
            "sharpe":    sharpe,
        }))

    # ---------- regime shift detail ----------
    train_settled = tr.dropna(subset=[RET_COL])
    test_settled = te.dropna(subset=[RET_COL])
    regime = {
        "train_n":     int(len(train_settled)),
        "test_n":      int(len(test_settled)),
        "train_p_up":  round(float((np.sign(train_settled[RET_COL]) > 0).mean()), 3),
        "test_p_up":   round(float((np.sign(test_settled[RET_COL]) > 0).mean()), 3),
        "train_avg":   round(float(train_settled[RET_COL].mean()), 4),
        "test_avg":    round(float(test_settled[RET_COL].mean()), 4),
    }
    per_ticker_regime = []
    for tk in sorted(f.ticker.unique()):
        a = tr[tr.ticker == tk].dropna(subset=[RET_COL])
        b = te[te.ticker == tk].dropna(subset=[RET_COL])
        per_ticker_regime.append({
            "ticker":     tk,
            "n_tr":       int(len(a)),
            "n_te":       int(len(b)),
            "p_up_train": round(float((np.sign(a[RET_COL]) > 0).mean()), 2) if len(a) else None,
            "p_up_test":  round(float((np.sign(b[RET_COL]) > 0).mean()), 2) if len(b) else None,
        })

    out = {
        "split": {
            "train": int(len(tr)),
            "test":  int(len(te)),
            "test_settled": int(len(test_settled)),
        },
        "regime": regime,
        "per_ticker_regime": per_ticker_regime,
        "primary_results": primary_results,
        "per_ticker_contrarian": per_ticker,
        "horizon_contrarian":    horizon_results,
        "logistic_best_C":  lr.best_params.get("clf__C"),
        "logistic_base_rate":  round(lr.train_base_rate, 3),
        "xgb_base_rate":       round(xgb.train_base_rate, 3),
        "cb_base_rate":        round(cb.train_base_rate, 3),
        "sf_base_rate":        round(sf.train_base_rate, 3),
    }
    out_path = OUTPUTS / "writeup_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # also pretty-print to console
    print("\n=== PRIMARY (21d) ===")
    print(f"{'signal':<25} {'n_tr':>5} {'hit':>6} {'IC':>7} {'sharpe':>7}")
    for name, r in primary_results.items():
        n_tr = r.get("n_trades", "-")
        h = f"{r['hit_rate']:.3f}" if r.get('hit_rate') is not None else "-"
        ic = f"{r['rank_ic']:+.3f}" if r.get('rank_ic') is not None else "-"
        sh = f"{r['naive_sharpe']:+.2f}" if r.get('naive_sharpe') is not None else "-"
        print(f"{name:<25} {n_tr:>5} {h:>6} {ic:>7} {sh:>7}")

    print("\n=== HORIZON SWEEP (Contrarian SetFit) ===")
    print(f"{'h':>4} {'n':>4} {'n_tr':>5} {'hit':>6} {'IC':>7} {'sharpe':>7}")
    for r in horizon_results:
        h = f"{r['hit_rate']:.3f}" if r.get('hit_rate') is not None else "-"
        ic = f"{r['rank_ic']:+.3f}" if r.get('rank_ic') is not None else "-"
        sh = f"{r['sharpe']:+.2f}" if r.get('sharpe') is not None else "-"
        print(f"{r['horizon']:>4} {r['n']:>4} {r['n_trades']:>5} {h:>6} {ic:>7} {sh:>7}")


if __name__ == "__main__":
    main()
