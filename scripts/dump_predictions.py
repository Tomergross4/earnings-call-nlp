"""Dump test-set predictions + feature importance for the dashboard's Tab 3 / Tab 4.

Outputs (consumed by app.py):
    outputs/model_predictions.parquet  — test rows with y_actual + per-model signals/probas
    outputs/feature_importance.json    — XGBoost gain + Logistic standardized coefficients

Re-runs are cheap because:
- Logistic + XGBoost (small Optuna budget) train in <60s
- SetFit reads its cached encoder + classifier from cache/setfit_model

Run:
    py scripts/dump_predictions.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.app_helpers import load_features
from src.config import OUTPUTS, PRIMARY_HORIZON
from src.model import (
    NUMERIC_FEATURE_COLS, baseline_rule, fit_logistic, fit_xgboost,
    lexicon_rule, finbert_rule, split_train_test,
)
from src.trained_models import fit_setfit

RET_COL = f"fwd_excess_{PRIMARY_HORIZON}d"


def main() -> None:
    f = load_features()
    tr, te = split_train_test(f)
    te_settled = te.dropna(subset=[RET_COL]).copy()

    print(f"train={len(tr)}  test_settled={len(te_settled)}")

    print("logistic..."); lr  = fit_logistic(tr)
    print("xgboost...");  xgb = fit_xgboost(tr, n_trials=20)   # smaller budget — fine for importance ranking
    print("setfit...");   sf  = fit_setfit(tr)                 # cache hit if encoder exists

    # ---- predictions on test set ----
    p_xgb     = xgb.predict_proba(te_settled).reindex(te_settled.index)
    p_logit   = lr.predict_proba(te_settled).reindex(te_settled.index)
    sig_xgb   = xgb.predict(te_settled).reindex(te_settled.index)
    sig_logit = lr.predict(te_settled).reindex(te_settled.index)
    sig_base  = baseline_rule(te_settled).reindex(te_settled.index)
    sig_lex   = lexicon_rule(te_settled).reindex(te_settled.index)
    sig_fb    = finbert_rule(te_settled).reindex(te_settled.index)
    sig_sf    = sf.predict(te_settled).reindex(te_settled.index)
    sig_sf_c  = sf.predict_contrarian(te_settled).reindex(te_settled.index)

    out = pd.DataFrame({
        "ticker":         te_settled["ticker"].values,
        "quarter":        te_settled["quarter"].values,
        "call_date":      te_settled["call_date"].values,
        "y_actual":       te_settled[RET_COL].values,
        "p_up_xgb":       p_xgb.values,
        "p_up_logistic":  p_logit.values,
        "sig_baseline":   sig_base.values,
        "sig_lexicon":    sig_lex.values,
        "sig_finbert":    sig_fb.values,
        "sig_logistic":   sig_logit.values,
        "sig_xgb":        sig_xgb.values,
        "sig_setfit":     sig_sf.values,
        "sig_contrarian": sig_sf_c.values,
    })
    out_path = OUTPUTS / "model_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")

    # ---- feature importance ----
    feat_cols = [c for c in NUMERIC_FEATURE_COLS if c in tr.columns]

    xgb_gain: dict = {}
    try:
        booster = xgb.clf.get_booster()
        gain = booster.get_score(importance_type="gain")
        # XGBoost preserves DataFrame column names — gain keys = column names directly.
        # Fallback to f0/f1 ordering if it ever drops names (older XGBoost).
        for i, col in enumerate(xgb.feature_cols):
            v = gain.get(col)
            if v is None:
                v = gain.get(f"f{i}", 0.0)
            xgb_gain[col] = float(v)
    except Exception as e:
        print(f"warn: xgb gain failed ({e}); falling back to feature_importances_")
        for col, v in zip(xgb.feature_cols, xgb.clf.feature_importances_):
            xgb_gain[col] = float(v)

    # Logistic — use standardized coefficients (same scale across features)
    log_coef: dict = {}
    try:
        coefs = lr.clf.named_steps["clf"].coef_[0]
        for col, c in zip(lr.feature_cols, coefs):
            log_coef[col] = float(c)
    except Exception as e:
        print(f"warn: logistic coef extraction failed ({e})")

    importance = {
        "feature_cols": feat_cols,
        "xgboost_gain": xgb_gain,
        "logistic_std_coef": log_coef,
        "model_meta": {
            "xgb_best_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                                for k, v in xgb.best_params.items()},
            "xgb_train_base_rate": round(xgb.train_base_rate, 4),
            "logistic_best_C":     lr.best_params.get("clf__C"),
            "n_train":             int(len(tr.dropna(subset=[RET_COL]))),
            "n_test":              int(len(te_settled)),
            "horizon_days":        PRIMARY_HORIZON,
        },
    }
    imp_path = OUTPUTS / "feature_importance.json"
    imp_path.write_text(json.dumps(importance, indent=2))
    print(f"wrote {imp_path}")


if __name__ == "__main__":
    main()
