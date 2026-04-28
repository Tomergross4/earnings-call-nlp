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
This installs `anthropic` and `python-dotenv` for the optional LLM-as-a-Judge evaluation step (see *NLP Quality Assurance* below); everything else runs without an Anthropic key.

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
Opens at http://localhost:8501. Five tabs, mapped to the assignment rubric:
- **Global Overview** — corpus-level stats (per-ticker call counts, sentiment-vs-return scatter, return distribution).
- **Task 1 — Per-Call** — searchable table of all 131 calls with drill-down: sentiment, wins, risks, themes, guidance, pre-call momentum, full transcript link.
- **Task 2 — QoQ Tracking** — per-ticker timeline with sentiment trajectory, forward-return overlay, and risk-persistence panel.
- **Task 3 — Predictive Model** — XGBoost + Logistic feature importance, train/test split visualization, predicted-vs-actual scatter on the test set.
- **Task 4 — Backtest** — SetFit-based signals (+0.13 Sharpe direct at 21d, +0.79 Contrarian at 5d, +0.58 Contrarian at 63d), 8-signal comparison table, horizon sweep.

## Reproduce from cache only (no transcripts needed)
The dashboard, the PDF, and all numerical claims are reproducible from artifacts already in this repo (`outputs/features.parquet`, `outputs/writeup_results.json`, `outputs/model_predictions.parquet`, `cache/prices/`). You do **not** need `ECT.zip` for any of:
```bash
streamlit run app.py        # full dashboard
py scripts/build_pdf.py     # rebuild outputs/writeup.pdf
```
To re-run anything upstream of `outputs/features.parquet` (re-parse, re-extract, re-fit features), drop `ECT.zip` in the project root first; the parser unzips on first run.

## NLP Quality Assurance (LLM-as-a-Judge)
Per professor feedback, stock-return performance measures *financial alpha*, not *NLP comprehension*. To grade the local `gemma3:4b` extractions on comprehension we use **Claude 3.5 Sonnet** as the commercial "school solution" and report directional-agreement % + sentiment MAE on a fixed 15-call sample. The script writes `outputs/nlp_evaluation.json`, which is then surfaced on the dashboard cover KPI row and the PDF cover.

```bash
# one-time: provide the key (either env var or .env file at repo root)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

py scripts/evaluate_nlp.py
```
Without a key the script exits cleanly with a helpful message; the rest of the pipeline runs unchanged.

## Build the write-up PDF
```bash
py scripts/build_pdf.py
```
Writes `outputs/writeup.pdf` from [docs/writeup.md](docs/writeup.md).

## Honesty guardrails

- Entry price uses dynamic T+0 or T+1 entry (adjusted for BMO/AMC reporting habits to capture intraday movement without look-ahead bias).
- Train/test split is strict-temporal per ticker: first 70% of calls (by date) → train, remaining 30% → test. Hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)` on the training set only.
- Pre-call momentum windows use strictly pre-call data — no look-ahead.
- LM lexicon sentiment is a *parallel column*, never a substitute for the LLM — the two sit side-by-side in the feature table so the grader can judge whether the LLM actually adds value.
- LLM raw output saved to `.raw.txt` alongside the JSON for debuggability.

## Design doc

The full methodology, per-ticker reads, backtest results, ablations, and "what didn't work" discussion live in [docs/writeup.md](docs/writeup.md) → [outputs/writeup.pdf](outputs/writeup.pdf).
