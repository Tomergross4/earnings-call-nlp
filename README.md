# Earnings-Call NLP Pipeline

NLP for Finance - Spring 2026, Assignment 1. Extracts sentiment, wins/risks, guidance, themes, and speaker-level signals from earnings-call transcripts; predicts forward 21d excess returns; exposes everything in a Streamlit dashboard.

**Scope:** all 14 assignment tickers, 131 transcripts. Every transcript contributes a row to the final feature table. Loughran-McDonald lexicon + pre-call price-momentum features populate for all 131 rows; LLM-extracted features populate where the extraction cache has been built (continues to fill in across runs without methodology drift). The pipeline is explicitly designed so that incremental extraction progress does not change the evaluation methodology — the same feature-build, split, and backtest run on whatever extractions exist at the time.

## Architecture

- [src/](src/) — pure Python modules: parser, prices, llm, extraction, features, lexicon, risk_classify, model, backtest, app_helpers.
- [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) — thin driver matching the starter's §0–§8 structure.
- [app.py](app.py) — Streamlit dashboard reading the pre-computed feature table + extraction cache.
- `cache/extractions/` — per-transcript LLM outputs (JSON + raw). Pipeline reruns are cache-hit.
- `outputs/features.parquet` — merged feature table, one row per parsed transcript.
- [docs/writeup.md](docs/writeup.md) + [scripts/build_pdf.py](scripts/build_pdf.py) → `outputs/writeup.pdf`.

## Setup

### 1. Ollama
1. Install Ollama from https://ollama.com/download.
2. Start the daemon (the Windows installer does this automatically; otherwise `ollama serve`).
3. Pull the model: `ollama pull gemma3:4b`.
4. Sanity check: `curl http://localhost:11434/api/generate -d '{"model":"gemma3:4b","prompt":"hi","stream":false}'`.

### 2. Python
```bash
pip install -r requirements.txt
```

### 3. Transcripts
The S&P-formatted transcripts (`ECT.zip` and the unzipped `transcripts/` folder) are **not redistributed** in this repo — they are instructor-supplied course material. Two options:

- **Cache-only path (recommended for grading):** the LLM extractions in [cache/extractions/](cache/extractions/) and the pre-built [outputs/features.parquet](outputs/features.parquet) let you run the model + backtest + dashboard without the raw text. Skip to *Reproduce from cache only* below.
- **Full pipeline path:** drop `ECT.zip` (from the course site) at the project root; the parser unzips it on first run.

## Run the pipeline
```bash
jupyter notebook notebooks/pipeline.ipynb
```
Run all cells. First full extraction on CPU-only Ollama takes many hours; on GPU it is ~90 minutes for the full corpus. Subsequent runs are ~30 seconds because the extraction is idempotent — any transcript already cached in `cache/extractions/` is skipped.

## Background extraction (CPU-opportunistic)
To extend coverage while doing other work, launch the extraction loop in the background. It picks up where the cache left off and will never re-extract a transcript.
```bash
py -c "from src.parser import parse_all; from src.extraction import extract_all; extract_all(parse_all())"
```
Because the extractor's model, prompts, and 4-call structure are fixed in [src/extraction.py](src/extraction.py), coverage grown this way is methodologically identical to a single-run extraction — every ticker is analyzed with the same pipeline.

## Launch the dashboard
```bash
streamlit run app.py
```
Opens at http://localhost:8501. Three tabs:
- **Per-Call Explorer** — searchable per-call table (sortable, colored returns) + ticker-level price overview (YTD/1D/1W/1M/3M/6M/1Y/2Y) + drill-down with sentiment, wins, risks, themes, and pre-call momentum.
- **Ticker Timeline** — sentiment and forward-return Altair charts aligned on shared `YYYY-Qn` axis, plus a sortable quarter-by-quarter summary.
- **Backtest** — production model only (Contrarian SetFit, +0.19 Sharpe at 21d / +0.58 at 5d). The 8-signal comparison and full horizon sweep live in the writeup PDF, not the dashboard.

## Reproduce from cache only (skip LLM)
If `cache/extractions/` is populated, you can go straight to feature-building:
```bash
py -c "from src.parser import parse_all; from src.prices import fetch_all, build_returns_table; from src.features import build; ts=parse_all(); p=fetch_all([t.ticker for t in ts]); build(build_returns_table(ts, p), ts)"
```

## Build the write-up PDF
```bash
py scripts/build_pdf.py
```
Writes `outputs/writeup.pdf` from [docs/writeup.md](docs/writeup.md).

## Honesty guardrails

- Entry price is T+1 close (never day-T).
- Train/test split is strict-temporal per ticker: first 70% of calls (by date) → train, remaining 30% → test. Hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)` on the training set only.
- Pre-call momentum windows use strictly pre-call data — no look-ahead.
- LM lexicon sentiment is a *parallel column*, never a substitute for the LLM — the two sit side-by-side in the feature table so the grader can judge whether the LLM actually adds value.
- LLM raw output saved to `.raw.txt` alongside the JSON for debuggability.

## Design doc

The full methodology, per-ticker reads, backtest results, ablations, and "what didn't work" discussion live in [docs/writeup.md](docs/writeup.md) → [outputs/writeup.pdf](outputs/writeup.pdf).
