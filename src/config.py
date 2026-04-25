"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ECT_ZIP     = ROOT / "ECT.zip"
TRANSCRIPTS = ROOT / "transcripts"
CACHE       = ROOT / "cache"
EXTRACTIONS = CACHE / "extractions"
PRICES      = CACHE / "prices"
OUTPUTS     = ROOT / "outputs"
FIGURES     = OUTPUTS / "figures"

# All 14 assignment tickers are parsed automatically from ECT.zip; no subset filter is applied.

# Forward-return horizons in trading days.
HORIZONS_DAYS = [1, 5, 21, 63]
PRIMARY_HORIZON = 21

# LLM config. gemma3:4b tag pulls Ollama's default Q4_K_M (4-bit) quantization.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 1024
OLLAMA_TEMPERATURE = 0.1

# Train/test split: strict temporal 70/30 per ticker (see src/model.py::split_train_test).

# Curated theme flags (binary features derived from LLM "themes" list).
CURATED_THEMES = ["ai", "china", "macro", "pricing", "capex"]


def ensure_dirs() -> None:
    for d in (TRANSCRIPTS, CACHE, EXTRACTIONS, PRICES, OUTPUTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
