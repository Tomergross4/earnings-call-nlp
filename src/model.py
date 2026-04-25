"""Baseline rules + Logistic Regression + XGBoost + CatBoost with k-fold CV and Optuna tuning.

Train/test split: strict temporal 70/30 per ticker.
Hyperparameter search: TimeSeriesSplit(n_splits=5) on training set only — no leakage.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PRIMARY_HORIZON

RET_COL = f"fwd_excess_{PRIMARY_HORIZON}d"
TRAIN_FRAC = 0.70

NUMERIC_FEATURE_COLS: List[str] = [
    # LLM-extracted
    "overall_sentiment",
    "ceo_sentiment",
    "cfo_sentiment",
    "analyst_sentiment",
    "n_wins",
    "n_risks",
    "guidance_score",
    "n_themes",
    # QoQ deltas (LLM-derived)
    "sentiment_delta",
    "n_risks_delta",
    "n_wins_delta",
    "risk_persistence",
    "theme_novelty",
    "theme_persistence",
    "ceo_cfo_gap",
    "analyst_mgmt_gap",
    "guidance_trajectory",
    # Reactive vs proactive
    "proactive_risk_count",
    "reactive_risk_count",
    "reactive_risk_ratio",
    # Curated theme flags
    "theme_ai",
    "theme_china",
    "theme_macro",
    "theme_pricing",
    "theme_capex",
    # Loughran-McDonald lexicon
    "lm_sentiment",
    "lm_pos",
    "lm_neg",
    # FinBERT sentiment (prepared remarks + Q&A)
    "finbert_sentiment",
    "finbert_pos",
    "finbert_neg",
    "finbert_qa_sentiment",
    "finbert_mgmt_qa_gap",
    "finbert_sentiment_delta",
    # Price momentum
    "mom_21d",
    "mom_63d",
    "dist_52w_high",
    "vol_21d",
]


def split_train_test(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal 70/30 split per ticker.

    Within each ticker, the first `train_frac` calls (by call_date) go to train,
    the rest to test. This preserves the temporal ordering requirement while
    expanding coverage beyond the fixed first-5 rule.
    """
    df = df.sort_values(["ticker", "call_date"]).copy()

    train_idx, test_idx = [], []
    for _ticker, grp in df.groupby("ticker"):
        n = len(grp)
        n_train = max(1, int(np.floor(n * train_frac)))
        idx = grp.index.tolist()
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def baseline_rule(df: pd.DataFrame) -> pd.Series:
    """Long if overall_sentiment > 0, short if < 0, flat otherwise."""
    return np.sign(df["overall_sentiment"].fillna(0)).astype(int)


def lexicon_rule(df: pd.DataFrame) -> pd.Series:
    """Non-LLM baseline: sign of Loughran-McDonald sentiment."""
    return np.sign(df["lm_sentiment"].fillna(0)).astype(int)


def finbert_rule(df: pd.DataFrame) -> pd.Series:
    """FinBERT baseline: sign of finbert_sentiment (prepared remarks)."""
    return np.sign(df["finbert_sentiment"].fillna(0)).astype(int)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _available_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in NUMERIC_FEATURE_COLS if c in df.columns]


def _build_X(df: pd.DataFrame, feature_cols: List[str],
              medians: pd.Series, missing_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in missing_cols:
        if col in X.columns:
            X[f"{col}_is_missing"] = X[col].isna().astype(int)
    X = X.fillna(medians).fillna(0.0)
    return X


def _make_target(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df[RET_COL]) > 0).astype(int)


# ---------------------------------------------------------------------------
# Logistic Regression with GridSearch over C + TimeSeriesSplit k-fold
# ---------------------------------------------------------------------------

BAND_HALF_WIDTH = 0.05  # ± around the train base-rate decision center


def _band_signal(proba: np.ndarray, center: float) -> np.ndarray:
    """Map probabilities to {-1, 0, +1} using a band centered on the train base rate.

    Why not 0.5? With a class-imbalanced training set (e.g. P(up)=0.61), a
    sklearn classifier's natural decision boundary is the empirical positive rate,
    not 0.5. Hard-coding 0.45/0.55 around 0.5 systematically biases predictions
    toward Hold for a balanced-class-weight model trained on imbalanced data.
    """
    hi = center + BAND_HALF_WIDTH
    lo = center - BAND_HALF_WIDTH
    return np.where(proba > hi, 1, np.where(proba < lo, -1, 0))


@dataclass
class LogisticModel:
    clf: Pipeline
    medians: pd.Series
    missing_cols: List[str]
    feature_cols: List[str]
    train_base_rate: float = 0.5
    best_params: dict = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = _build_X(df, self.feature_cols, self.medians, self.missing_cols)
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(df).to_numpy()
        return pd.Series(_band_signal(proba, self.train_base_rate), index=df.index)


def fit_logistic(train_df: pd.DataFrame, n_splits: int = 5) -> LogisticModel:
    """Fit logistic regression with GridSearch over C using TimeSeriesSplit."""
    from sklearn.model_selection import GridSearchCV

    train = train_df.dropna(subset=[RET_COL]).copy()
    feature_cols = _available_cols(train)
    y = _make_target(train)
    Xraw = train[feature_cols]
    missing_cols = [c for c in feature_cols if Xraw[c].isna().any()]
    medians = Xraw.median(numeric_only=True)
    X = _build_X(train, feature_cols, medians, missing_cols)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(train) - 1))
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(pipe, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X, y)

    best_params = gs.best_params_
    cv_scores = list(gs.cv_results_["mean_test_score"])
    base_rate = float(y.mean())
    print(f"  Logistic best C={best_params['clf__C']}, CV AUC={gs.best_score_:.3f}, base_rate={base_rate:.3f}")
    return LogisticModel(
        clf=gs.best_estimator_,
        medians=medians,
        missing_cols=missing_cols,
        feature_cols=feature_cols,
        train_base_rate=base_rate,
        best_params=best_params,
        cv_scores=cv_scores,
    )


# ---------------------------------------------------------------------------
# XGBoost with Optuna hyperparameter search + TimeSeriesSplit k-fold
# ---------------------------------------------------------------------------

@dataclass
class XGBoostModel:
    clf: object
    feature_cols: List[str]
    train_base_rate: float = 0.5
    best_params: dict = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols].copy().fillna(np.nan)
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(df).to_numpy()
        return pd.Series(_band_signal(proba, self.train_base_rate), index=df.index)


def fit_xgboost(train_df: pd.DataFrame, n_trials: int = 40, n_splits: int = 5) -> XGBoostModel:
    """Fit XGBoost with Optuna search over key hyperparameters, validated by TimeSeriesSplit."""
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = train_df.dropna(subset=[RET_COL]).copy()
    feature_cols = _available_cols(train)
    y = _make_target(train).to_numpy()
    X = train[feature_cols].copy()

    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(train) - 1))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 2, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "missing": np.nan,
            "use_label_encoder": False,
        }
        clf = XGBClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=tscv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  XGBoost best CV AUC={study.best_value:.3f} | params={best}")

    final_clf = XGBClassifier(
        **best,
        tree_method="hist",
        eval_metric="logloss",
        missing=np.nan,
        use_label_encoder=False,
    )
    final_clf.fit(X, y)

    cv_scores = [t.value for t in study.trials if t.value is not None]
    base_rate = float(y.mean())
    return XGBoostModel(clf=final_clf, feature_cols=feature_cols,
                        train_base_rate=base_rate,
                        best_params=best, cv_scores=cv_scores)


# ---------------------------------------------------------------------------
# CatBoost with Optuna hyperparameter search + TimeSeriesSplit k-fold
# ---------------------------------------------------------------------------

@dataclass
class CatBoostModel:
    clf: object
    medians: pd.Series
    feature_cols: List[str]
    train_base_rate: float = 0.5
    best_params: dict = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols].copy().fillna(self.medians).fillna(0.0)
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(df).to_numpy()
        return pd.Series(_band_signal(proba, self.train_base_rate), index=df.index)


def fit_catboost(train_df: pd.DataFrame, n_trials: int = 40, n_splits: int = 5) -> CatBoostModel:
    """Fit CatBoost with Optuna search, validated by TimeSeriesSplit.

    CatBoost's ordered boosting handles small datasets (n=89) better than
    vanilla gradient boosting by reducing overfitting on low-count samples.
    """
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = train_df.dropna(subset=[RET_COL]).copy()
    feature_cols = _available_cols(train)
    y = _make_target(train).to_numpy()
    Xraw = train[feature_cols]
    medians = Xraw.median(numeric_only=True)
    X = Xraw.fillna(medians).fillna(0.0)

    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(train) - 1))

    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 50, 400),
            "depth":             trial.suggest_int("depth", 2, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count":      trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":   trial.suggest_float("random_strength", 0.0, 1.0),
            "verbose": 0,
            "allow_writing_files": False,
        }
        clf = CatBoostClassifier(**params)
        scores = cross_val_score(clf, X, y, cv=tscv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  CatBoost best CV AUC={study.best_value:.3f} | params={best}")

    final_clf = CatBoostClassifier(**best, verbose=0, allow_writing_files=False)
    final_clf.fit(X, y)

    cv_scores = [t.value for t in study.trials if t.value is not None]
    base_rate = float(y.mean())
    return CatBoostModel(clf=final_clf, medians=medians, feature_cols=feature_cols,
                         train_base_rate=base_rate,
                         best_params=best, cv_scores=cv_scores)
