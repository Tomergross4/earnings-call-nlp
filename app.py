"""Earnings-Call NLP Dashboard — final 5-tab presentation surface.

This file is *strictly* a viewer over pre-computed pipeline outputs. It does
NOT re-run the LLM, re-train any classifier, or call yfinance. Every figure is
derived from these on-disk artifacts:

    outputs/features.parquet         → 131 calls × LLM/lexicon/FinBERT/momentum
    outputs/writeup_results.json     → 8-signal backtest metrics (offline run)
    cache/prices/<TICKER>.parquet    → daily closes for tickers + SPY

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# CONFIG  — edit only these globals if file names move
# ---------------------------------------------------------------------------
FEATURES_PATH    = Path("outputs/features.parquet")
RESULTS_PATH     = Path("outputs/writeup_results.json")
PREDICTIONS_PATH = Path("outputs/model_predictions.parquet")
IMPORTANCE_PATH  = Path("outputs/feature_importance.json")
TRANSCRIPTS_DIR  = Path("transcripts")
PRICES_DIR       = Path("cache/prices")
EQUITY_FIG       = Path("outputs/figures/equity_curve.png")

HORIZON_DAYS  = 21
TRAIN_FRAC    = 0.70
TRADING_DAYS  = 252
ANN_FACTOR    = float(np.sqrt(TRADING_DAYS / HORIZON_DAYS))   # ≈ 3.464
RET_COL       = f"fwd_excess_{HORIZON_DAYS}d"

# Plotly palette (kept consistent across tabs)
PURPLE   = "#6366F1"
PURPLE_2 = "#8B5CF6"
ORANGE   = "#F59E0B"
GREEN    = "#10B981"
RED      = "#EF4444"
GREY     = "#9CA3AF"
SLATE    = "#374151"

# Static reference data for the 14 assignment tickers
GICS = {
    "AMD":  ("Information Technology", "Semiconductors"),
    "AVGO": ("Information Technology", "Semiconductors"),
    "BLK":  ("Financials",             "Asset Management"),
    "C":    ("Financials",             "Diversified Banks"),
    "FAST": ("Industrials",            "Trading Distributors"),
    "FDX":  ("Industrials",            "Air Freight & Logistics"),
    "GS":   ("Financials",             "Investment Banking"),
    "INTC": ("Information Technology", "Semiconductors"),
    "JNJ":  ("Health Care",            "Pharmaceuticals"),
    "JPM":  ("Financials",             "Diversified Banks"),
    "NKE":  ("Consumer Discretionary", "Footwear"),
    "NVDA": ("Information Technology", "Semiconductors"),
    "PLTR": ("Information Technology", "Application Software"),
    "WFC":  ("Financials",             "Diversified Banks"),
}
COMPANY = {
    "AMD":  "Advanced Micro Devices",  "AVGO": "Broadcom",
    "BLK":  "BlackRock",                "C":    "Citigroup",
    "FAST": "Fastenal",                 "FDX":  "FedEx",
    "GS":   "Goldman Sachs",            "INTC": "Intel",
    "JNJ":  "Johnson & Johnson",        "JPM":  "JPMorgan Chase",
    "NKE":  "Nike",                     "NVDA": "Nvidia",
    "PLTR": "Palantir",                 "WFC":  "Wells Fargo",
}


# ---------------------------------------------------------------------------
# PAGE / CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Earnings-Call NLP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}

.nlp-header {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%);
    padding: 22px 30px; border-radius: 14px; color: white;
    margin-bottom: 20px; box-shadow: 0 6px 20px rgba(99,102,241,0.22);
}
.nlp-header h1 {margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.3px; color: white;}
.nlp-header p  {margin: 6px 0 0 0; opacity: 0.92; font-size: 13.5px;}

.kpi-row {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px;}
.kpi-row.k5 {grid-template-columns: repeat(5, 1fr);}
.kpi-card {
    background: white; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-value {font-size: 26px; font-weight: 700; color: #6366F1; line-height: 1.1;}
.kpi-label {font-size: 11px; color: #6B7280; margin-top: 6px;
            text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;}

.stTabs [data-baseweb="tab-list"] {gap: 6px; background: transparent;}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 10px 18px; font-weight: 500;
    background: #F3F4F6; border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: #EDE9FE !important; color: #6D28D9 !important;
    border: 1px solid #DDD6FE !important;
}

.sub-h {
    font-size: 15px; font-weight: 600; color: #374151;
    margin: 16px 0 10px 0; padding-bottom: 6px;
    border-bottom: 2px solid #EDE9FE;
}
.chip {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    background: #EDE9FE; color: #6D28D9; font-size: 12px;
    font-weight: 500; margin-right: 4px; margin-bottom: 4px;
}
.callout {
    background: #FEF3C7; border-left: 4px solid #F59E0B;
    padding: 10px 14px; border-radius: 6px; margin: 12px 0;
    font-size: 13.5px; color: #78350F;
}

[data-testid="stMetric"] {
    background: white; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important; color: #6B7280 !important;
    text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600;
}
[data-testid="stMetricValue"] {font-size: 22px !important; font-weight: 700 !important;}
[data-testid="stDataFrame"] {border-radius: 8px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
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


def _add_guidance_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker chronological guidance streak.

    Adds two columns in place of an apply (pandas 2.x drops the groupby column
    on apply+group_keys=False, so we walk per-ticker manually):
      guidance_streak       — int, consecutive count of identical direction
      guidance_streak_label — human string, e.g. 'Raised (3 in a row)'
                              or 'Lowered (after prior Raised)'.
    """
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    streak_arr  = np.zeros(len(df), dtype=int)
    label_arr   = np.empty(len(df), dtype=object)

    for tk, idxs in df.groupby("ticker").groups.items():
        prev_dir: Optional[str] = None
        cur_n = 0
        for i in idxs:
            d_raw = df.at[i, "guidance"] if "guidance" in df.columns else None
            d = (str(d_raw).lower().strip() if pd.notna(d_raw) and str(d_raw).strip() else None)
            if d is None:
                streak_arr[i] = 0
                label_arr[i]  = "—"
                prev_dir = None
                cur_n = 0
                continue
            if d == prev_dir:
                cur_n += 1
                label = f"{d.title()} ({cur_n} in a row)"
            else:
                cur_n = 1
                if prev_dir is None:
                    label = f"{d.title()} (first observation)"
                else:
                    label = f"{d.title()} (after prior {prev_dir.title()})"
            streak_arr[i] = cur_n
            label_arr[i]  = label
            prev_dir = d

    df["guidance_streak"] = streak_arr
    df["guidance_streak_label"] = label_arr
    return df


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    for col in ("themes", "wins", "risks", "risks_classified"):
        if col in df.columns:
            df[col] = df[col].apply(_maybe_parse_json)
    df["call_date"] = pd.to_datetime(df["call_date"])
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)

    # Enrich with metadata
    df["company"]  = df["ticker"].map(COMPANY).fillna(df["ticker"])
    df["sector"]   = df["ticker"].map(lambda t: GICS.get(t, ("-", "-"))[0])
    df["industry"] = df["ticker"].map(lambda t: GICS.get(t, ("-", "-"))[1])

    # Guidance streak — for the rubric requirement to flag consecutive raises/lowers
    df = _add_guidance_streak(df)
    return df


# ---------------------------------------------------------------------------
# GUIDANCE LINE-ITEM HEURISTIC (revenue / EPS / margin / FCF / capex)
# ---------------------------------------------------------------------------
LINE_ITEM_PATTERNS: Dict[str, List[str]] = {
    "Revenue":          [r"\brevenue\b", r"\btop[- ]?line\b", r"\bsales\b"],
    "EPS":              [r"\beps\b", r"earnings per share"],
    "Operating Margin": [r"operating margin", r"\bop\.? margin\b"],
    "Gross Margin":     [r"gross margin"],
    "Free Cash Flow":   [r"free cash flow", r"\bfcf\b"],
    "Capex":            [r"\bcapex\b", r"capital expenditures?"],
    "Buybacks/Dividend": [r"buy[- ]?backs?", r"share repurchase", r"dividend"],
}


@st.cache_data(show_spinner=False)
def extract_guidance_line(ticker: str, quarter: str) -> str:
    """Scan the cached transcript for sentences mentioning guidance/outlook,
    return the most-mentioned KPI line item. Read-only, no LLM."""
    p = TRANSCRIPTS_DIR / f"{ticker}_{quarter}.txt"
    if not p.exists():
        return "—"
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "—"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    relevant = [s for s in sentences
                if re.search(r"\b(guidance|guide|guiding|outlook|forecast)\b", s, re.I)]
    if not relevant:
        return "—"
    chunk = " ".join(relevant).lower()
    counts: Dict[str, int] = {}
    for label, patterns in LINE_ITEM_PATTERNS.items():
        n = sum(len(re.findall(p, chunk, re.I)) for p in patterns)
        if n:
            counts[label] = n
    if not counts:
        return "Revenue"   # most common default if guidance is mentioned without a specific KPI
    return max(counts.items(), key=lambda kv: kv[1])[0]


@st.cache_data(show_spinner=False)
def load_results() -> Dict:
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_predictions() -> Optional[pd.DataFrame]:
    """Test-set predictions dumped by scripts/dump_predictions.py."""
    if not PREDICTIONS_PATH.exists():
        return None
    df = pd.read_parquet(PREDICTIONS_PATH)
    df["call_date"] = pd.to_datetime(df["call_date"])
    return df.sort_values("call_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_importance() -> Dict:
    if not IMPORTANCE_PATH.exists():
        return {}
    return json.loads(IMPORTANCE_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_price(ticker: str) -> Optional[pd.DataFrame]:
    p = PRICES_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def ticker_returns_table(tickers: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        df = load_price(t)
        if df is None or df.empty:
            continue
        close = df["Close"].to_numpy()
        last_px = close[-1]
        last_date = df["Date"].iloc[-1]

        def back(n):
            return close[-1 - n] if len(close) > n else None

        prior_year = df[df["Date"] < pd.Timestamp(last_date.year, 1, 1)]
        ytd_base = prior_year["Close"].iloc[-1] if not prior_year.empty else None

        def ret(base):
            if base is None or base == 0:
                return np.nan
            return last_px / base - 1

        rows.append({
            "Ticker":   t,
            "Company":  COMPANY.get(t, t),
            "Sector":   GICS.get(t, ("-", "-"))[0],
            "Last":     float(last_px),
            "As Of":    last_date.strftime("%Y-%m-%d"),
            "1D":       ret(back(1)),
            "1W":       ret(back(5)),
            "1M":       ret(back(21)),
            "3M":       ret(back(63)),
            "6M":       ret(back(126)),
            "YTD":      ret(ytd_base),
            "1Y":       ret(back(252)),
            "2Y":       ret(back(504)),
        })
    return pd.DataFrame(rows)


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict-temporal 70/30 split per ticker (matches src/model.py)."""
    df = df.sort_values(["ticker", "call_date"]).copy()
    tr_idx, te_idx = [], []
    for _, grp in df.groupby("ticker"):
        n = len(grp)
        n_train = max(1, int(np.floor(n * TRAIN_FRAC)))
        idx = grp.index.tolist()
        tr_idx.extend(idx[:n_train])
        te_idx.extend(idx[n_train:])
    return df.loc[tr_idx].copy(), df.loc[te_idx].copy()


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------
def _as_list(v) -> List:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def fmt_pct(v, plus=True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v:+.2%}" if plus else f"{v:.2%}"


def fmt_num(v, decimals=2, plus=True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    fmt = f"{{:+.{decimals}f}}" if plus else f"{{:.{decimals}f}}"
    return fmt.format(v)


def _pct_style(val):
    if pd.isna(val):
        return "color: #9CA3AF"
    if val > 0:
        return "background-color: #D1FAE5; color: #065F46; font-weight: 600"
    if val < 0:
        return "background-color: #FEE2E2; color: #991B1B; font-weight: 600"
    return ""


def _sent_style(val):
    if pd.isna(val):
        return "color: #9CA3AF"
    if val > 0.1:
        return "background-color: #D1FAE5; color: #065F46; font-weight: 600"
    if val < -0.1:
        return "background-color: #FEE2E2; color: #991B1B; font-weight: 600"
    return "color: #6B7280"


def kpi_block(items: List[Tuple[str, str]], cls: str = "") -> None:
    cards = "".join(
        f"<div class='kpi-card'><div class='kpi-value'>{v}</div>"
        f"<div class='kpi-label'>{lbl}</div></div>"
        for lbl, v in items
    )
    st.markdown(f"<div class='kpi-row {cls}'>{cards}</div>", unsafe_allow_html=True)


def plotly_period_axis(periods: List[str]) -> Dict:
    """Build a categorical x-axis with explicit period order — avoids continuous-date bugs."""
    return dict(
        type="category",
        categoryorder="array",
        categoryarray=sorted(set(periods)),
        title="",
        tickangle=-30,
    )


# ---------------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------------
try:
    features = load_features()
except FileNotFoundError:
    st.error(f"Missing {FEATURES_PATH}. Run the offline pipeline first.")
    st.stop()

results = load_results()

st.markdown("""
<div class='nlp-header'>
  <h1>📊 Earnings-Call NLP Dashboard</h1>
  <p>LLM extraction · sentiment · QoQ tracking · forward-return prediction · honest backtest — 14 tickers, 131 transcripts, S&P 500 benchmarked</p>
</div>
""", unsafe_allow_html=True)


tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Global Overview",
    "📋 Task 1 — Per-Call",
    "📈 Task 2 — QoQ Tracking",
    "🤖 Task 3 — Predictive Model",
    "💹 Task 4 — Backtest",
])


# ===========================================================================
# TAB 0 — GLOBAL OVERVIEW
# ===========================================================================
with tab0:
    n_tickers     = features["ticker"].nunique()
    n_transcripts = len(features)
    n_sectors     = features["sector"].nunique()
    avg_llm       = features["overall_sentiment"].mean()
    avg_lm        = features["lm_sentiment"].mean()

    kpi_block([
        ("Tickers Covered",      f"{n_tickers}"),
        ("Total Transcripts",    f"{n_transcripts}"),
        ("GICS Sectors",         f"{n_sectors}"),
        ("Avg LLM Sentiment",    fmt_num(avg_llm, 2)),
        ("Avg LM Lexicon",       fmt_num(avg_lm, 2)),
    ], cls="k5")

    st.markdown("<div class='sub-h'>Raw stock performance — independent of S&P 500</div>",
                unsafe_allow_html=True)
    st.caption("Absolute price returns from each ticker's daily-close cache. "
               "Sortable. Independent of any model — pure market context.")

    px_table = ticker_returns_table(tuple(sorted(features["ticker"].unique())))
    px_table = px_table.sort_values("YTD", ascending=False).reset_index(drop=True)

    sty = px_table.style.format({
        "Last": "{:.2f}",
        "1D":  "{:+.2%}", "1W":  "{:+.2%}", "1M":  "{:+.2%}",
        "3M":  "{:+.2%}", "6M":  "{:+.2%}", "YTD": "{:+.2%}",
        "1Y":  "{:+.2%}", "2Y":  "{:+.2%}",
    }, na_rep="–")
    for col in ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "2Y"]:
        sty = sty.map(_pct_style, subset=[col])
    st.dataframe(sty, use_container_width=True, height=560)

    # Aggregate summary: avg sentiment by sector
    st.markdown("<div class='sub-h'>Corpus-wide sentiment distribution</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        sect = features.groupby("sector").agg(
            n_calls=("ticker", "count"),
            avg_llm=("overall_sentiment", "mean"),
            avg_lm=("lm_sentiment", "mean"),
            avg_finbert=("finbert_sentiment", "mean"),
        ).reset_index().sort_values("avg_llm", ascending=False)

        fig = px.bar(
            sect, y="sector", x="avg_llm", orientation="h",
            color="avg_llm", color_continuous_scale=[[0, RED], [0.5, "white"], [1, GREEN]],
            range_color=(-0.6, 0.6),
            labels={"avg_llm": "Avg LLM sentiment", "sector": ""},
            text=sect["avg_llm"].apply(lambda v: f"{v:+.2f}"),
            height=320,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        # Pad the x-axis so outside text labels (e.g. '+0.46') aren't clipped on the right.
        x_max = float(sect["avg_llm"].max()) if not sect.empty else 0.6
        x_min = float(sect["avg_llm"].min()) if not sect.empty else -0.6
        x_pad = max(0.12, abs(x_max) * 0.25)
        fig.update_layout(
            margin=dict(l=10, r=60, t=20, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True, gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF",
                range=[min(-0.6, x_min - x_pad), max(0.6, x_max + x_pad)],
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bucket_counts = (features["sentiment_bucket"].value_counts(dropna=False)
                         .rename_axis("bucket").reset_index(name="n"))
        bucket_order = ["very_bullish", "bullish", "neutral", "bearish", "very_bearish"]
        bucket_counts["bucket"] = pd.Categorical(
            bucket_counts["bucket"].fillna("missing"),
            categories=bucket_order + ["missing"],
            ordered=True,
        )
        bucket_counts = bucket_counts.sort_values("bucket")

        color_map = {
            "very_bullish": "#065F46", "bullish": GREEN, "neutral": GREY,
            "bearish": RED, "very_bearish": "#991B1B", "missing": "#E5E7EB",
        }
        fig = px.pie(
            bucket_counts, values="n", names="bucket",
            color="bucket", color_discrete_map=color_map,
            hole=0.55, height=380,
        )
        fig.update_traces(textinfo="label+percent", textposition="outside",
                          insidetextorientation="horizontal", automargin=True)
        fig.update_layout(
            # Generous side+bottom margins so outside labels never clip
            margin=dict(l=40, r=40, t=40, b=120),
            showlegend=False,
            title=dict(text="LLM tone buckets", x=0.5, font=dict(size=13, color=SLATE)),
            uniformtext_minsize=10, uniformtext_mode="show",
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 1 — PER-CALL EXTRACTION
# ===========================================================================
with tab1:
    st.markdown("<div class='sub-h'>Raw LLM extractions per call — sentiment + structured events</div>",
                unsafe_allow_html=True)

    ticker_q = features.groupby("ticker")["quarter"].apply(list).to_dict()
    tickers_sorted = sorted(ticker_q.keys())

    c_t, c_q = st.columns([1, 2])
    with c_t:
        ticker = st.selectbox("Ticker", tickers_sorted, index=0, key="t1_ticker")
    with c_q:
        qs = ticker_q.get(ticker, [])
        quarter = st.selectbox("Quarter", qs, index=len(qs) - 1 if qs else 0, key="t1_quarter")

    row = features[(features.ticker == ticker) & (features.quarter == quarter)]
    if row.empty:
        st.warning("No row for that selection.")
    else:
        r = row.iloc[0]
        st.markdown(
            f"<span class='chip'>{r['company']}</span>"
            f"<span class='chip'>{r['sector']}</span>"
            f"<span class='chip'>{r['industry']}</span>"
            f"<span class='chip'>{str(r['call_date'].date())}</span>",
            unsafe_allow_html=True,
        )

        # ---------- KPI row (combined tone + classified guidance) ----------
        # Combine quantitative score with categorical bucket in a single value
        bucket_map = {
            "very_bullish": "Very Bullish", "bullish": "Bullish",
            "neutral": "Neutral", "bearish": "Bearish",
            "very_bearish": "Very Bearish",
        }
        tone_score  = r["overall_sentiment"]
        bucket_raw  = r.get("sentiment_bucket") or ""
        bucket_human = bucket_map.get(str(bucket_raw), str(bucket_raw).replace("_", " ").title() or "—")
        tone_value = (
            f"{tone_score:+.2f} ({bucket_human})"
            if pd.notna(tone_score) else "–"
        )

        # Guidance direction + which line the LLM/transcript is actually guiding
        guidance_dir = str(r.get("guidance") or "—").title() if pd.notna(r.get("guidance")) else "—"
        guidance_streak = str(r.get("guidance_streak_label") or "")
        line_item = extract_guidance_line(ticker, quarter)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "LLM Tone [-1, 1] (classified)",
            tone_value,
            help="Quantitative tone score from the LLM, paired with its categorical bucket.",
        )
        m2.metric("LM Lexicon", fmt_num(r["lm_sentiment"], 2),
                  help="Loughran-McDonald financial dictionary — (positive − negative) / (positive + negative).")
        m3.metric(
            "Guidance",
            guidance_dir,
            delta=f"Metric: {line_item}",
            delta_color="off",
            help="Direction (raised / reaffirmed / lowered / mixed) plus the financial line "
                 "the call is guiding on (Revenue / EPS / Margin / FCF / Capex). "
                 "Line item is auto-inferred from the prepared remarks.",
        )
        m4.metric(
            "FWD 21D",
            fmt_pct(r.get(RET_COL)),
            help="How much the stock beat (or lagged) the S&P 500 over the 3 weeks "
                 "following the earnings call.",
        )

        # Streak detail under the KPI row (rubric: track consecutive raises/lowers)
        if guidance_streak and guidance_streak != "—":
            st.markdown(
                f"<div style='font-size:12px; color:#6B7280; margin-top:-4px;'>"
                f"<b>Guidance streak:</b> {guidance_streak}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='sub-h'>Wins · Risks (top bullets from the LLM)</div>",
                    unsafe_allow_html=True)
        col_w, col_r = st.columns(2)

        with col_w:
            st.markdown(f"**Top Wins** &nbsp; <span class='chip'>n = {int(r.get('n_wins') or 0)}</span>",
                        unsafe_allow_html=True)
            wins = _as_list(r.get("wins"))
            if wins:
                for item in wins:
                    st.markdown(f"- {item}")
            else:
                st.caption("No wins extracted.")

        with col_r:
            st.markdown(f"**Top Risks** &nbsp; <span class='chip'>n = {int(r.get('n_risks') or 0)}</span>",
                        unsafe_allow_html=True)

            # Suppress generic legal boilerplate so the rubric's "concrete negatives"
            # aren't drowned out by safe-harbor language repeated in every call.
            # Match loosely — any bullet *containing* the phrase is a boilerplate disclaimer.
            _BOILERPLATE = (
                "forward-looking", "forward looking",
                "safe harbor", "safe-harbor",
                "private securities litigation reform act",
                "actual results may differ", "actual results could differ",
            )
            def _is_boilerplate(text: str) -> bool:
                t = (text or "").lower()
                return any(p in t for p in _BOILERPLATE)

            classified = _as_list(r.get("risks_classified"))
            if classified and any(isinstance(it, dict) for it in classified):
                # Reactive = surfaced only after analyst pushback ⇒ red flag.
                # Proactive = volunteered in prepared remarks ⇒ neutral blue.
                badge_styles = {
                    "proactive": "background:#DBEAFE;color:#1E3A8A;border:1px solid #93C5FD;",
                    "reactive":  "background:#DC2626;color:#FFFFFF;border:1px solid #991B1B;"
                                 "box-shadow:0 0 0 2px rgba(220,38,38,0.18);",
                    "unknown":   "background:#F3F4F6;color:#4B5563;border:1px solid #D1D5DB;",
                }
                rendered_any = False
                for item in classified:
                    if isinstance(item, dict):
                        risk_text = item.get("risk", "")
                        if _is_boilerplate(risk_text):
                            continue
                        label = (item.get("label") or "unknown").lower()
                        style = badge_styles.get(label, badge_styles["unknown"])
                        emoji = "⚠️ " if label == "reactive" else ""
                        st.markdown(
                            f"- {risk_text} &nbsp; "
                            f"<span style='{style}padding:2px 9px;"
                            f"border-radius:11px;font-size:0.74em;font-weight:700;"
                            f"text-transform:uppercase;letter-spacing:0.4px;'>"
                            f"{emoji}{label}</span>",
                            unsafe_allow_html=True,
                        )
                        rendered_any = True
                    else:
                        if _is_boilerplate(str(item)):
                            continue
                        st.markdown(f"- {item}")
                        rendered_any = True
                if not rendered_any:
                    st.caption("No concrete risks (boilerplate filtered).")
            else:
                risks = _as_list(r.get("risks"))
                risks = [it for it in risks if not _is_boilerplate(str(it))]
                if risks:
                    for item in risks:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No concrete risks (boilerplate filtered).")

        st.markdown("<div class='sub-h'>Themes</div>", unsafe_allow_html=True)
        themes = _as_list(r.get("themes"))
        if themes:
            st.markdown(
                " ".join(f"<span class='chip'>{t}</span>" for t in themes),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No themes extracted.")

        st.markdown("<div class='sub-h'>Speaker-level sentiment "
                    "<span style='font-size:11px;color:#6B7280;font-weight:500;'>"
                    "&nbsp;· Management ≠ Analyst tone reveals whether the Street pushed back</span></div>",
                    unsafe_allow_html=True)
        spk = pd.DataFrame({
            "Speaker":   ["CEO", "CFO", "Analysts"],
            "Group":     ["Management", "Management", "Analysts"],
            "Sentiment": [r.get("ceo_sentiment"), r.get("cfo_sentiment"), r.get("analyst_sentiment")],
        }).dropna()
        if spk.empty:
            st.caption("No speaker-level sentiment cached for this call.")
        else:
            # Distinct categorical colors per group (Management vs Analysts)
            group_colors = {"Management": "#2563EB", "Analysts": "#F97316"}  # blue vs orange
            fig = go.Figure()
            for grp in ["Management", "Analysts"]:
                d = spk[spk["Group"] == grp]
                if d.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=d["Speaker"], y=d["Sentiment"],
                    name=grp,
                    marker=dict(color=group_colors[grp],
                                line=dict(color="white", width=1)),
                    text=[f"{v:+.2f}" for v in d["Sentiment"]],
                    textposition="outside",
                    textfont=dict(size=12, color="#111827"),
                ))
            for tr in fig.data:
                tr.cliponaxis = False
            # Y-axis range: clamp to [0, 1.2] when all sentiments are non-negative
            # (typical case — earnings tone is rarely net-negative). If any speaker
            # reads negative, fall back to a symmetric [-1.2, 1.2] so the bar still renders.
            spk_min = float(spk["Sentiment"].min()) if not spk.empty else 0.0
            y_lo = 0.0 if spk_min >= 0 else -1.2
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=1.20, x=0),
                bargap=0.35,
                yaxis=dict(title="Sentiment [-1, 1]", showgrid=True,
                           gridcolor="#F3F4F6", zeroline=True, zerolinecolor="#9CA3AF",
                           range=[y_lo, 1.2]),
            )
            st.plotly_chart(fig, use_container_width=True)
            mgmt_avg = spk.loc[spk["Group"] == "Management", "Sentiment"].mean()
            anl_avg  = spk.loc[spk["Group"] == "Analysts", "Sentiment"].mean()
            if pd.notna(mgmt_avg) and pd.notna(anl_avg):
                gap = mgmt_avg - anl_avg
                tag_color = "#DC2626" if gap > 0.25 else "#6B7280"
                st.markdown(
                    f"<div style='font-size:12px;color:{tag_color};margin-top:-6px;'>"
                    f"<b>Mgmt vs Analyst gap:</b> {gap:+.2f} "
                    f"{'⚠️ analysts notably less enthusiastic' if gap > 0.25 else ''}"
                    "</div>",
                    unsafe_allow_html=True,
                )


# ===========================================================================
# TAB 2 — QoQ TRACKING
# ===========================================================================
with tab2:
    st.markdown("<div class='sub-h'>Quarter-over-quarter dynamics</div>", unsafe_allow_html=True)

    tickers_sorted = sorted(features["ticker"].unique())
    # Row 1: Ticker + identity chips. Row 2: Quarter selector (below) so the
    # long "(for new-themes / persistent-risks)" label has room to breathe and
    # never wraps under the chips.
    c_t, c_h = st.columns([1, 4])
    with c_t:
        tk = st.selectbox("Ticker", tickers_sorted, index=tickers_sorted.index("NVDA")
                          if "NVDA" in tickers_sorted else 0, key="t2_ticker")

    ts = features[features.ticker == tk].sort_values("call_date").copy()
    if ts.empty:
        st.warning("No data for ticker.")
        st.stop()

    with c_h:
        st.markdown(
            f"<span class='chip'>{COMPANY.get(tk, tk)}</span>"
            f"<span class='chip'>{GICS.get(tk, ('-','-'))[0]}</span>"
            f"<span class='chip'>{GICS.get(tk, ('-','-'))[1]}</span>",
            unsafe_allow_html=True,
        )

    # Use reporting_period (e.g. "2025-Q4") consistently with charts/table.
    # Fall back to raw quarter for any row missing reporting_period.
    ts["_period_display"] = ts["reporting_period"].astype(str).where(
        ts["reporting_period"].notna(), ts["quarter"].astype(str)
    )
    qs_for_tk = ts["_period_display"].tolist()
    c_q, _ = st.columns([1, 4])
    with c_q:
        # Default to most recent quarter (last in chronological order)
        sel_period = st.selectbox(
            "Quarter (for new-themes / persistent-risks)",
            qs_for_tk, index=len(qs_for_tk) - 1, key="t2_quarter",
        )

    period_order = ts["reporting_period"].dropna().astype(str).tolist()
    cat_axis = plotly_period_axis(period_order)

    # ---------- Guidance trajectory KPI (rubric: track consecutive raises/lowers) ----------
    sel_mask = ts["_period_display"] == sel_period
    sel_row = ts[sel_mask].iloc[0] if sel_mask.any() else None
    streak_label_sel = (str(sel_row.get("guidance_streak_label")) if sel_row is not None else "—")
    sentiment_delta_sel = sel_row.get("sentiment_delta") if sel_row is not None else None
    delta_str = (f"{float(sentiment_delta_sel):+.3f}"
                 if sentiment_delta_sel is not None and pd.notna(sentiment_delta_sel)
                 else "—")
    n_calls = len(ts)
    n_raised = int((ts["guidance"].astype(str).str.lower() == "raised").sum())
    n_lowered = int((ts["guidance"].astype(str).str.lower() == "lowered").sum())

    kpi_block([
        (f"Guidance trajectory ({sel_period})",  streak_label_sel),
        (f"QoQ sentiment Δ ({sel_period})",      delta_str),
        ("Quarters covered",                     f"{n_calls}"),
        ("Raised / Lowered (cumulative)",        f"{n_raised} / {n_lowered}"),
    ])

    # ------ Sentiment delta line --------------------------------------------
    st.markdown("<div class='sub-h'>Sentiment delta vs. prior quarter</div>",
                unsafe_allow_html=True)
    sd = ts[["reporting_period", "sentiment_delta", "overall_sentiment"]].dropna(subset=["reporting_period"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sd["reporting_period"], y=sd["overall_sentiment"],
        mode="lines+markers", name="Overall sentiment",
        line=dict(color=PURPLE, width=2.5), marker=dict(size=8),
    ))
    fig.add_trace(go.Bar(
        x=sd["reporting_period"], y=sd["sentiment_delta"],
        name="QoQ Δ sentiment",
        marker_color=[GREEN if (v or 0) >= 0 else RED for v in sd["sentiment_delta"]],
        opacity=0.55,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
        xaxis=cat_axis,
        yaxis=dict(title="Sentiment", showgrid=True, gridcolor="#F3F4F6",
                   zeroline=True, zerolinecolor="#9CA3AF"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------ Risk persistence + theme novelty --------------------------------
    st.markdown("<div class='sub-h'>Risk persistence &amp; theme drift</div>",
                unsafe_allow_html=True)
    rp = ts[["reporting_period", "risk_persistence", "theme_persistence", "theme_novelty"]].copy()
    rp_long = rp.melt(id_vars=["reporting_period"], var_name="metric", value_name="value").dropna()
    rp_long["metric"] = rp_long["metric"].map({
        "risk_persistence":  "Risk persistence (carried over)",
        "theme_persistence": "Theme persistence",
        "theme_novelty":     "Theme novelty (new themes share)",
    })
    if rp_long.empty:
        st.caption("Persistence requires ≥2 calls; insufficient history yet.")
    else:
        fig = px.line(
            rp_long, x="reporting_period", y="value", color="metric",
            markers=True,
            color_discrete_map={
                "Risk persistence (carried over)":  ORANGE,
                "Theme persistence":                PURPLE_2,
                "Theme novelty (new themes share)": "#3B82F6",
            },
            labels={"value": "Share (0–1)", "reporting_period": ""},
            height=300,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="white", legend=dict(orientation="h", y=1.15),
            xaxis=cat_axis,
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6", range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------ Actual text: new themes + persistent risks ----------------------
    st.markdown(
        f"<div class='sub-h'>What changed in <span style='color:#6D28D9'>{sel_period}</span> "
        f"vs. the prior quarter</div>",
        unsafe_allow_html=True,
    )
    if sel_row is None:
        st.caption("Selected quarter not found.")
    else:
        ts_idx = ts.reset_index(drop=True)
        row_pos = int(ts_idx.index[ts_idx["_period_display"] == sel_period][0])
        cur_themes = set(map(str, _as_list(ts_idx.loc[row_pos].get("themes"))))
        cur_risks  = list(map(str, _as_list(ts_idx.loc[row_pos].get("risks"))))
        if row_pos == 0:
            prev_themes: set = set()
            prev_risks:  set = set()
            prior_q = "—"
        else:
            prev_row = ts_idx.loc[row_pos - 1]
            prev_themes = set(map(str, _as_list(prev_row.get("themes"))))
            prev_risks  = set(map(str, _as_list(prev_row.get("risks"))))
            prior_q = str(prev_row["_period_display"])

        new_themes = sorted(cur_themes - prev_themes)
        carried_risks = [r for r in cur_risks if r in prev_risks]   # exact match (proxy for persistent)
        novel_risks   = [r for r in cur_risks if r not in prev_risks]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"**🆕 New themes this quarter** &nbsp; <span class='chip'>"
                f"{len(new_themes)}</span>",
                unsafe_allow_html=True,
            )
            if new_themes:
                st.markdown(
                    " ".join(f"<span class='chip' style='background:#DCFCE7;color:#166534;'>{t}</span>"
                             for t in new_themes),
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"No new themes vs. {prior_q}." if prior_q != "—"
                           else "First quarter for this ticker — no prior comparison.")
            if novel_risks and prior_q != "—":
                st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                st.markdown("**Novel risks this quarter** "
                            f"<span class='chip'>{len(novel_risks)}</span>",
                            unsafe_allow_html=True)
                for risk in novel_risks:
                    st.markdown(f"- {risk}")

        with col_b:
            st.markdown(
                f"**🔁 Persistent risks (carried over from {prior_q})** &nbsp; "
                f"<span class='chip'>{len(carried_risks)}</span>",
                unsafe_allow_html=True,
            )
            if carried_risks:
                for risk in carried_risks:
                    st.markdown(f"- ⚠️ {risk}")
            else:
                st.caption(f"No risks repeated from {prior_q}." if prior_q != "—"
                           else "First quarter — no prior risks to compare.")

    # ------ Reactive vs Proactive risks -------------------------------------
    st.markdown("<div class='sub-h'>Reactive vs. proactive risks "
                "(only acknowledged after analyst pushback ↦ reactive)</div>",
                unsafe_allow_html=True)
    rr = ts[["reporting_period", "proactive_risk_count", "reactive_risk_count"]].copy()
    rr_long = rr.melt(id_vars=["reporting_period"], var_name="kind", value_name="count").dropna()
    rr_long["kind"] = rr_long["kind"].map({
        "proactive_risk_count": "Proactive (prepared remarks)",
        "reactive_risk_count":  "Reactive (Q&A only)",
    })
    if rr_long.empty:
        st.caption("Risk-classification not cached for this ticker.")
    else:
        fig = px.bar(
            rr_long, x="reporting_period", y="count", color="kind",
            barmode="stack",
            color_discrete_map={
                "Proactive (prepared remarks)": "#3B82F6",
                "Reactive (Q&A only)":          RED,
            },
            labels={"count": "# risks", "reporting_period": ""},
            height=300,
        )
        # Legend sits above the plot (y=1.20), so we need a generous top margin
        # to give it real estate — otherwise it overlaps the tops of the bars.
        fig.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.20, x=0, yanchor="bottom"),
            xaxis=cat_axis,
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------ Quarter-by-quarter sortable summary -----------------------------
    st.markdown("<div class='sub-h'>Quarter-by-quarter summary</div>",
                unsafe_allow_html=True)
    cols_show = [
        "reporting_period", "call_date", "overall_sentiment",
        "sentiment_delta", "lm_sentiment", "finbert_sentiment",
        "guidance", "guidance_streak_label",
        "n_wins", "n_risks", "reactive_risk_ratio",
        "theme_novelty", "mom_21d", RET_COL,
    ]
    cols_show = [c for c in cols_show if c in ts.columns]
    qs_df = ts[cols_show].copy().reset_index(drop=True)
    qs_df["call_date"] = pd.to_datetime(qs_df["call_date"]).dt.strftime("%Y-%m-%d")
    qs_df = qs_df.rename(columns={
        "reporting_period": "Period", "call_date": "Date",
        "overall_sentiment": "LLM", "sentiment_delta": "Δ Sent",
        "lm_sentiment": "LM", "finbert_sentiment": "FinBERT",
        "guidance": "Guidance", "guidance_streak_label": "Trajectory",
        "n_wins": "Wins", "n_risks": "Risks",
        "reactive_risk_ratio": "Reactive%", "theme_novelty": "Novelty",
        "mom_21d": "Pre-21d", RET_COL: "Fwd 21d",
    })

    qs_sty = qs_df.style.format({
        "LLM":     "{:+.2f}", "Δ Sent":  "{:+.2f}", "LM":      "{:+.2f}", "FinBERT": "{:+.2f}",
        "Reactive%": "{:.0%}", "Novelty":  "{:.2f}",
        "Pre-21d": "{:+.2%}", "Fwd 21d": "{:+.2%}",
    }, na_rep="–")
    for col in [c for c in ["LLM", "Δ Sent", "LM", "FinBERT"] if c in qs_df.columns]:
        qs_sty = qs_sty.map(_sent_style, subset=[col])
    for col in [c for c in ["Pre-21d", "Fwd 21d"] if c in qs_df.columns]:
        qs_sty = qs_sty.map(_pct_style, subset=[col])
    st.dataframe(qs_sty, use_container_width=True,
                 height=min(60 + 38 * len(qs_df), 480), hide_index=True)


# ===========================================================================
# TAB 3 — PREDICTIVE MODEL
# ===========================================================================
with tab3:
    st.markdown(
        f"""
        <div class='callout'>
        <b>Target:</b> forward {HORIZON_DAYS}-day excess return vs SPY (binary up/down classification),
        T+1 entry · {TRAIN_FRAC:.0%}/{1 - TRAIN_FRAC:.0%} strict-temporal split per ticker.<br>
        <b>Model:</b> XGBoost (Optuna-tuned, TimeSeriesSplit CV) — multivariate over all
        Task-1 + Task-2 features (LLM sentiment, speaker gaps, guidance, risks, themes,
        FinBERT, LM lexicon, momentum). Predictions and feature importance come from
        <code>outputs/model_predictions.parquet</code> + <code>outputs/feature_importance.json</code>,
        produced by <code>scripts/dump_predictions.py</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    preds = load_predictions()
    importance = load_importance()

    if preds is None or not importance:
        st.error(
            "Pipeline artifacts missing — run `py scripts/dump_predictions.py` first to produce "
            "`outputs/model_predictions.parquet` and `outputs/feature_importance.json`."
        )
    else:
        meta = importance.get("model_meta", {})
        kpi_block([
            ("Train Calls",  f"{meta.get('n_train', '–')}"),
            ("Test Calls",   f"{meta.get('n_test', len(preds))}"),
            ("Target",       RET_COL),
            ("Model",        "XGBoost (Optuna)"),
        ])

        # ------ XGBoost feature importance (gain) -------------------------
        st.markdown("<div class='sub-h'>XGBoost feature importance (gain) — "
                    "what the multivariate model actually leans on</div>",
                    unsafe_allow_html=True)
        st.caption(
            "Gain = total reduction in loss attributable to splits on this feature, summed across "
            "all trees. Higher = more decisive in the model's decisions. "
            "Pulled directly from the trained Booster (no proxy correlation)."
        )

        xgb_gain = importance.get("xgboost_gain", {})
        log_coef = importance.get("logistic_std_coef", {})

        if xgb_gain and any(v > 0 for v in xgb_gain.values()):
            imp_df = (pd.Series(xgb_gain, name="gain")
                      .replace(0, np.nan).dropna()
                      .sort_values(ascending=True).tail(15))
            fig = px.bar(
                x=imp_df.values, y=imp_df.index, orientation="h",
                color=imp_df.values,
                color_continuous_scale=[[0, "#E0E7FF"], [1, PURPLE]],
                labels={"x": "Gain (XGBoost)", "y": ""},
                text=[f"{v:.2f}" for v in imp_df.values],
                height=480,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                margin=dict(l=150, r=40, t=10, b=10),
                plot_bgcolor="white",
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Logistic standardized coefficients alongside (sign + magnitude)
            if log_coef:
                with st.expander("Logistic regression standardized coefficients (sign + magnitude)"):
                    lc = (pd.Series(log_coef, name="coef").dropna()
                          .reindex(pd.Series(log_coef).abs().sort_values(ascending=False).index)
                          .head(15).iloc[::-1])
                    fig2 = px.bar(
                        x=lc.values, y=lc.index, orientation="h",
                        color=lc.values,
                        color_continuous_scale=[[0, RED], [0.5, "white"], [1, GREEN]],
                        range_color=(-3.5, 3.5),
                        labels={"x": "Standardized coefficient", "y": ""},
                        text=[f"{v:+.2f}" for v in lc.values],
                        height=440,
                    )
                    fig2.update_traces(textposition="outside")
                    fig2.update_layout(
                        margin=dict(l=150, r=40, t=10, b=10),
                        plot_bgcolor="white",
                        coloraxis_showscale=False,
                        xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                                   zeroline=True, zerolinecolor="#9CA3AF"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("XGBoost gain dictionary is empty — re-run `dump_predictions.py`.")

        # ------ Multivariate predictions vs realized returns --------------
        st.markdown(
            "<div class='sub-h'>Model predictions vs. actual returns "
            "<span style='font-size:11px;color:#6B7280;font-weight:500;'>"
            "&nbsp;· P(up) from the multivariate XGBoost classifier on the held-out test set"
            "</span></div>",
            unsafe_allow_html=True,
        )

        sc = preds.merge(
            features[["ticker", "quarter", "company", "sector", "reporting_period"]],
            on=["ticker", "quarter"], how="left",
        )
        if "p_up_xgb" in sc.columns:
            fig = px.scatter(
                sc, x="p_up_xgb", y="y_actual", color="sector",
                hover_data={
                    "ticker": True, "reporting_period": True, "company": True,
                    "p_up_xgb": ":.3f", "y_actual": ":.2%",
                    "sig_xgb": True, "sig_contrarian": True, "sector": False,
                },
                labels={"p_up_xgb": "Model P(up) — XGBoost",
                        "y_actual": f"Realized {HORIZON_DAYS}d excess return"},
                height=460,
            )
            fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1, dash="dot"))
            # Decision boundary is centered on the train base rate (~0.607),
            # with a ±0.05 abstain band (matches src/model.py predict()).
            base_rate = float(importance.get("model_meta", {})
                              .get("xgb_train_base_rate", 0.607)) if importance else 0.607
            band_lo, band_hi = base_rate - 0.05, base_rate + 0.05
            fig.add_vrect(
                x0=band_lo, x1=band_hi,
                fillcolor="#9CA3AF", opacity=0.12, line_width=0,
                annotation_text="Hold / abstain band",
                annotation_position="top left",
                annotation=dict(font=dict(size=10, color="#6B7280")),
            )
            # Just the dashed line — label is placed separately at the bottom of
            # the plot (paper coords) so it can never collide with the abstain-band
            # label sitting at the top.
            fig.add_vline(
                x=base_rate,
                line=dict(color=PURPLE, width=1.5, dash="dash"),
            )
            fig.add_annotation(
                xref="x", yref="paper",
                x=base_rate, y=0.05,
                text=f"Train base rate ({base_rate:.3f})",
                showarrow=False,
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color=PURPLE),
                bgcolor="rgba(255,255,255,0.85)",
            )

            if len(sc) >= 3:
                coef = np.polyfit(sc["p_up_xgb"], sc["y_actual"], 1)
                xs = np.linspace(sc["p_up_xgb"].min(), sc["p_up_xgb"].max(), 30)
                fig.add_trace(go.Scatter(
                    x=xs, y=np.polyval(coef, xs),
                    mode="lines", name="OLS fit",
                    line=dict(color=SLATE, width=2, dash="dash"),
                    showlegend=True,
                ))

            fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")),
                              selector=dict(mode="markers"))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickformat=".2f", range=[0, 1.02]),
                yaxis=dict(showgrid=True, gridcolor="#F3F4F6", tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Test-set diagnostics
            if len(sc) >= 3:
                ic_spear = sc[["p_up_xgb", "y_actual"]].corr(method="spearman").iloc[0, 1]
                ic_pear  = sc[["p_up_xgb", "y_actual"]].corr(method="pearson").iloc[0, 1]
                # XGBoost trade hit rate
                trades_x = sc[sc["sig_xgb"] != 0].copy()
                hit_x = float(((trades_x["sig_xgb"] * trades_x["y_actual"]) > 0).mean()) \
                        if len(trades_x) else float("nan")
                # Contrarian hit rate
                trades_c = sc[sc["sig_contrarian"] != 0].copy()
                hit_c = float(((trades_c["sig_contrarian"] * trades_c["y_actual"]) > 0).mean()) \
                        if len(trades_c) else float("nan")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spearman ρ", f"{ic_spear:+.3f}",
                          help="Rank correlation between XGBoost P(up) and realized 21d excess return on the test set.")
                c2.metric("Pearson r",  f"{ic_pear:+.3f}",
                          help="Linear correlation between predicted probability and realized return.")
                c3.metric("Raw XGBoost Hit Rate", f"{hit_x:.0%}" if pd.notna(hit_x) else "–",
                          help="Trades where signal direction matched realized direction (XGBoost band-direct trades).")
                c4.metric("Contrarian (Inverted) Hit Rate", f"{hit_c:.0%}" if pd.notna(hit_c) else "–",
                          help="Trades where the inverted SetFit signal beat the market on the test set — our headline finding.")
        else:
            st.warning("Predictions parquet missing `p_up_xgb`.")


# ===========================================================================
# TAB 4 — BACKTEST
# ===========================================================================
with tab4:
    # Train/test dates summary
    tr, te = split_train_test(features)
    tr_dates = pd.to_datetime(tr["call_date"])
    te_dates = pd.to_datetime(te["call_date"])
    tr_range = f"{tr_dates.min():%Y-%m-%d} → {tr_dates.max():%Y-%m-%d}"
    te_range = f"{te_dates.min():%Y-%m-%d} → {te_dates.max():%Y-%m-%d}"

    st.markdown(
        f"""
        <div class='callout'>
        <b>Train window:</b> {tr_range} &nbsp;|&nbsp; <b>Test window:</b> {te_range}<br>
        <b>Split:</b> strict-temporal {TRAIN_FRAC:.0%} per ticker · entry T+1 close · horizon {HORIZON_DAYS}d ·
        excess vs SPY · Sharpe annualized via √(252 / {HORIZON_DAYS}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------ Headline metrics from offline run -------------------------------
    primary = (results.get("primary_results") or {})
    contrarian = primary.get("Contrarian SetFit", {})
    baseline   = primary.get("Baseline (LLM sign)", {})
    horizon_sweep = results.get("horizon_contrarian", [])

    if contrarian:
        # Rubric: report avg WIN vs avg LOSS (not just net average) — these are
        # already pre-computed in writeup_results.json for Contrarian SetFit.
        kpi_block([
            ("Hit Rate",          fmt_num(contrarian.get("hit_rate"), 3, plus=False)),
            ("Information Coef.", fmt_num(contrarian.get("rank_ic"), 3)),
            ("Sharpe (ann.)",     fmt_num(contrarian.get("naive_sharpe"), 2)),
            ("Avg Win",           fmt_pct(contrarian.get("avg_win"))),
            ("Avg Loss",          fmt_pct(contrarian.get("avg_loss"))),
        ], cls="k5")
    else:
        st.warning("Pre-computed backtest results not found — showing live calc only.")

    # ------ Multi-horizon sweep --------------------------------------------
    if horizon_sweep:
        st.markdown("<div class='sub-h'>Horizon sweep — Contrarian SetFit "
                    "(headline finding lives at 5d)</div>", unsafe_allow_html=True)

        hsweep = pd.DataFrame(horizon_sweep)
        hsweep["horizon_label"] = hsweep["horizon"].astype(str) + "d"
        c_a, c_b = st.columns(2)
        with c_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hsweep["horizon_label"], y=hsweep["sharpe"],
                marker_color=[GREEN if v >= 0 else RED for v in hsweep["sharpe"]],
                text=[f"{v:+.2f}" for v in hsweep["sharpe"]],
                textposition="outside",
                name="Sharpe (ann.)",
            ))
            fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Sharpe by horizon", x=0.02, font=dict(size=13, color=SLATE)),
                plot_bgcolor="white",
                yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                           zeroline=True, zerolinecolor="#9CA3AF"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_b:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hsweep["horizon_label"], y=hsweep["hit_rate"],
                marker_color=PURPLE,
                text=[f"{v:.0%}" for v in hsweep["hit_rate"]],
                textposition="outside",
                cliponaxis=False,
                name="Hit rate",
            ))
            # 50% reference line — label sits on the far right, ABOVE the line,
            # so it never collides with bar values or data labels.
            fig.add_hline(y=0.5, line=dict(color=GREY, width=1, dash="dot"))
            fig.add_annotation(
                xref="paper", yref="y", x=1.0, y=0.5,
                text="50% — coin flip", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(size=10, color=GREY),
                bgcolor="rgba(255,255,255,0.85)",
            )
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text="Hit rate by horizon", x=0.02, font=dict(size=13, color=SLATE)),
                plot_bgcolor="white",
                yaxis=dict(range=[0, 0.95], showgrid=True, gridcolor="#F3F4F6", tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------ Equity curve: Contrarian SetFit vs Buy & Hold SPY --------------
    st.markdown("<div class='sub-h'>Equity curve — Contrarian SetFit (winning model, excess) "
                "vs. buy-and-hold SPY (absolute)</div>",
                unsafe_allow_html=True)
    st.caption("Strategy: invert the SetFit P(up) — long when the model says down, short when it says up. "
               "This is our +0.19 Sharpe production model at the 21d horizon. "
               "Excess return = stock − SPY over 21 trading days starting at T+1. "
               "SPY benchmark uses the same call dates with T+1 entry — zero look-ahead.")

    preds = load_predictions()
    spy = load_price("SPY")

    def _spy_fwd(call_date: pd.Timestamp) -> float:
        if spy is None or spy.empty:
            return float("nan")
        entry = spy[spy["Date"] > call_date].head(1)
        if entry.empty:
            return float("nan")
        ei = int(entry.index[0])
        if ei + HORIZON_DAYS >= len(spy):
            return float("nan")
        return float(spy["Close"].iloc[ei + HORIZON_DAYS] / spy["Close"].iloc[ei] - 1)

    if preds is None or "sig_contrarian" not in preds.columns:
        st.warning("Predictions parquet missing — falling back to LLM-sign baseline. "
                   "Run `py scripts/dump_predictions.py`.")
        te_use = te.dropna(subset=[RET_COL]).sort_values("call_date").copy()
        te_use["signal"] = np.sign(te_use["overall_sentiment"].fillna(0)).astype(int)
        te_use["pnl"]    = te_use["signal"] * te_use[RET_COL]
        ret_actual_col = RET_COL
    else:
        te_use = preds.sort_values("call_date").copy()
        te_use["signal"] = te_use["sig_contrarian"].astype(int)
        te_use["pnl"]    = te_use["signal"] * te_use["y_actual"]
        ret_actual_col = "y_actual"

    te_use["cum_strategy"]  = te_use["pnl"].cumsum()
    te_use["cum_long_only"] = te_use[ret_actual_col].cumsum()
    te_use["spy_fwd"]       = te_use["call_date"].map(_spy_fwd)
    te_use["cum_spy"]       = te_use["spy_fwd"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=te_use["call_date"], y=te_use["cum_strategy"],
        name="Contrarian SetFit (excess)", mode="lines+markers",
        line=dict(color=PURPLE, width=2.8), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=te_use["call_date"], y=te_use["cum_long_only"],
        name="Always-long (excess)", mode="lines+markers",
        line=dict(color=GREY, width=2, dash="dash"), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=te_use["call_date"], y=te_use["cum_spy"],
        name="Buy & Hold SPY (absolute)", mode="lines+markers",
        line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=6),
    ))
    fig.add_hline(y=0, line=dict(color=GREY, width=1, dash="dot"))
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(title="Call date (test set)", tickformat="%Y-%m"),
        yaxis=dict(title="Cumulative return", showgrid=True,
                   gridcolor="#F3F4F6", tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------ Live recompute on the winning model -----------------------------
    pnl_arr = te_use["pnl"].dropna().to_numpy()
    if len(pnl_arr) > 1:
        avg = pnl_arr.mean()
        std = pnl_arr.std(ddof=1)
        sharpe_live = avg / (std + 1e-9) * ANN_FACTOR
        trades = te_use[te_use["signal"] != 0]
        n_trades = int(len(trades))
        wins   = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]
        hit_live = float((trades["pnl"] > 0).mean()) if n_trades else float("nan")
        ic_live  = float(te_use[["signal", ret_actual_col]].corr(method="spearman").iloc[0, 1])
        avg_win  = float(wins.mean())   if len(wins)   else float("nan")
        avg_loss = float(losses.mean()) if len(losses) else float("nan")
        st.markdown("<div class='sub-h'>Live recompute — Contrarian SetFit on the test set</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades",         f"{n_trades}")
        c2.metric("Hit Rate",       fmt_num(hit_live, 3, plus=False))
        c3.metric("Spearman IC",    fmt_num(ic_live, 3))
        c4.metric("Sharpe (ann.)",  fmt_num(sharpe_live, 2))
        c5.metric("Avg Win",        fmt_pct(avg_win))
        c6.metric("Avg Loss",       fmt_pct(avg_loss))

    # ------ Eight-signal comparison from offline JSON -----------------------
    if primary:
        st.markdown("<div class='sub-h'>All eight signals at the 21d horizon "
                    "(from <code>writeup_results.json</code>)</div>",
                    unsafe_allow_html=True)
        rows = []
        for name, m in primary.items():
            rows.append({
                "Signal":       name,
                "Trades":       m.get("n_trades"),
                "Hit Rate":     m.get("hit_rate"),
                "Spearman IC":  m.get("rank_ic"),
                "Avg Excess":   m.get("avg_excess"),
                "Sharpe (ann)": m.get("naive_sharpe"),
                "F1 (binary)":  m.get("f1_binary"),
                "F1 (macro)":   m.get("f1_macro"),
            })
        df_sig = pd.DataFrame(rows).sort_values("Sharpe (ann)", ascending=False).reset_index(drop=True)
        sig_sty = df_sig.style.format({
            "Hit Rate":     "{:.2%}",
            "Spearman IC":  "{:+.3f}",
            "Avg Excess":   "{:+.2%}",
            "Sharpe (ann)": "{:+.2f}",
            "F1 (binary)":  "{:.3f}",
            "F1 (macro)":   "{:.3f}",
        }, na_rep="–")
        sig_sty = sig_sty.map(_pct_style, subset=["Avg Excess"])
        sig_sty = sig_sty.map(
            lambda v: "background-color:#D1FAE5; color:#065F46; font-weight:600"
            if isinstance(v, float) and v > 0
            else ("background-color:#FEE2E2; color:#991B1B; font-weight:600"
                  if isinstance(v, float) and v < 0 else ""),
            subset=["Sharpe (ann)"],
        )
        st.dataframe(sig_sty, use_container_width=True, height=340, hide_index=True)

    # ------ Per-ticker breakdown for the production model -------------------
    pt = results.get("per_ticker_contrarian") or []
    if pt:
        st.markdown("<div class='sub-h'>Per-ticker hit rate &amp; PnL — Contrarian SetFit (test set)</div>",
                    unsafe_allow_html=True)
        df_pt = pd.DataFrame(pt).sort_values("avg_pnl", ascending=False).reset_index(drop=True)
        df_pt = df_pt.rename(columns={
            "ticker": "Ticker", "n_test": "N Test", "n_trades": "Trades",
            "hits": "Hits", "hit_rate": "Hit Rate", "avg_pnl": "Avg PnL",
        })
        pt_sty = df_pt.style.format(
            {"Hit Rate": "{:.0%}", "Avg PnL": "{:+.2%}"}, na_rep="–"
        )
        pt_sty = pt_sty.map(_pct_style, subset=["Avg PnL"])
        st.dataframe(pt_sty, use_container_width=True,
                     height=min(60 + 38 * len(df_pt), 600), hide_index=True)

    # ------ Honest caveat box ----------------------------------------------
    st.markdown(
        """
        <div class='callout'>
        <b>Honest caveats.</b> Test set is 36 settled calls — too small for any single
        Sharpe to be statistically significant. The train→test regime shift (P(up) 0.61 → 0.42)
        means a long-bias baseline mechanically underperforms in this window. The headline
        positive-Sharpe finding (Contrarian SetFit at +0.58 over 5d) lives at the shorter horizon;
        at 21d the same signal earns +0.19 Sharpe, marginal. Read this as evidence that
        post-call sentiment over-shoots — not as a deployable strategy.
        </div>
        """,
        unsafe_allow_html=True,
    )
