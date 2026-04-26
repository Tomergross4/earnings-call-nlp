"""LLM-as-a-Judge: grade local gemma3:4b extractions against a commercial baseline.

Per professor feedback (Spring 2026 Assignment 1): stock returns measure financial
alpha, not NLP comprehension. To prove the local model actually understands the
text, we treat Claude 3.5 Sonnet as a "school solution" (פתרון בית ספר) and
measure how often gemma3:4b agrees with it on the sign of overall sentiment, plus
the mean-absolute-error of the sentiment floats.

Output: outputs/nlp_evaluation.json — consumed by app.py, scripts/build_pdf.py,
and docs/writeup.md.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Make `src` importable when run as `py scripts/evaluate_nlp.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EXTRACTIONS  # noqa: E402
from src.parser import parse_all  # noqa: E402
from src.llm import salvage_json  # noqa: E402
from src.extraction import _overall_text  # noqa: E402

OUT_PATH = ROOT / "outputs" / "nlp_evaluation.json"
JUDGE_MODEL = "claude-sonnet-4-6"
LOCAL_MODEL = "gemma3:4b"
N_SAMPLE = 15
SEED = 42

JUDGE_PROMPT = """You are an expert financial analyst grading an earnings call.
Read the transcript below and return STRICT JSON with exactly this schema:

{"overall_sentiment": <float in [-1, 1]>, "rationale": <one short sentence>}

Where -1 = very bearish, 0 = neutral, +1 = very bullish. Ground every judgment
in the transcript text. Do not invent. Output ONLY the JSON object — no prose,
no markdown fences, no commentary.

TRANSCRIPT:
{transcript}
"""


def _load_dotenv() -> None:
    """Best-effort .env loader. Falls back to plain os.environ if no dotenv."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _local_overall_path(ticker: str, quarter: str) -> Path:
    tag = LOCAL_MODEL.replace(":", "-")
    return EXTRACTIONS / f"{ticker}_{quarter}_{tag}_overall.json"


def _sign(x: float, eps: float = 0.05) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def call_judge(client, transcript_text: str) -> dict:
    prompt = JUDGE_PROMPT.replace("{transcript}", transcript_text)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return salvage_json(raw)


def main() -> int:
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not found.\n"
            "  Set it in your shell, or create a .env file at the repo root with:\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Then re-run: py scripts/evaluate_nlp.py",
            file=sys.stderr,
        )
        return 2

    try:
        import anthropic  # noqa: F401
    except ImportError:
        print(
            "ERROR: `anthropic` not installed. Run:\n"
            "    pip install anthropic python-dotenv",
            file=sys.stderr,
        )
        return 2

    print("Parsing transcripts...")
    transcripts = parse_all()
    eligible = [t for t in transcripts if _local_overall_path(t.ticker, t.quarter).exists()]
    print(f"  {len(transcripts)} parsed; {len(eligible)} have local extractions cached.")
    if len(eligible) < N_SAMPLE:
        print(f"ERROR: only {len(eligible)} eligible transcripts; need {N_SAMPLE}.", file=sys.stderr)
        return 1

    rng = random.Random(SEED)
    sample = rng.sample(eligible, N_SAMPLE)
    print(f"Sampled {N_SAMPLE} calls (seed={SEED}):")
    for t in sample:
        print(f"  {t.ticker} {t.quarter} ({t.call_date})")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    rows = []
    agree_count = 0
    abs_errs: list[float] = []
    for i, t in enumerate(sample, 1):
        local = json.loads(_local_overall_path(t.ticker, t.quarter).read_text())
        gemma_score = float(local.get("overall_sentiment") or 0.0)

        text = _overall_text(t)
        print(f"[{i}/{N_SAMPLE}] judging {t.ticker} {t.quarter} ...", flush=True)
        t0 = time.time()
        try:
            judge = call_judge(client, text)
        except Exception as e:
            print(f"    judge call failed: {e}", file=sys.stderr)
            continue
        dt_call = time.time() - t0
        try:
            claude_score = float(judge.get("overall_sentiment"))
        except (TypeError, ValueError):
            print(f"    judge returned non-numeric sentiment; skipping. raw={judge}", file=sys.stderr)
            continue

        agree = _sign(gemma_score) == _sign(claude_score)
        agree_count += int(agree)
        abs_errs.append(abs(gemma_score - claude_score))
        rows.append({
            "ticker": t.ticker,
            "quarter": t.quarter,
            "call_date": t.call_date,
            "gemma_sentiment": round(gemma_score, 4),
            "claude_sentiment": round(claude_score, 4),
            "claude_rationale": judge.get("rationale", ""),
            "directional_agreement": bool(agree),
            "abs_error": round(abs(gemma_score - claude_score), 4),
            "judge_seconds": round(dt_call, 2),
        })
        print(
            f"    gemma={gemma_score:+.2f}  claude={claude_score:+.2f}  "
            f"agree={'YES' if agree else 'NO '}  |Δ|={abs(gemma_score-claude_score):.2f}"
        )

    n = len(rows)
    if n == 0:
        print("ERROR: no successful judgments.", file=sys.stderr)
        return 1

    out = {
        "model_local": LOCAL_MODEL,
        "model_judge": JUDGE_MODEL,
        "n_sample": n,
        "seed": SEED,
        "directional_agreement": round(agree_count / n, 4),
        "sentiment_mae": round(sum(abs_errs) / n, 4),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "samples": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"  Directional Agreement: {out['directional_agreement']:.1%}  (n={n})")
    print(f"  Sentiment MAE:         {out['sentiment_mae']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
