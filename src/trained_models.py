"""Task-specific fine-tuned models for return-direction prediction.

Two approaches, same interface — both return a model with .predict(test_df) and
.predict_proba(test_df), and expose cv_scores / best_params for reporting.

1. SetFit  — few-shot contrastive fine-tuning of a sentence encoder +
             logistic head. Works well with ~50-100 labelled examples.
             Base: 'ProsusAI/finbert' (financial domain, matches our FinBERT layer).

2. FinBERT fine-tune — full sequence-classification fine-tuning of FinBERT
             on our (text, label) pairs using HuggingFace Trainer.
             More expressive but higher overfitting risk at n=89.

Both use strict temporal train/test — no data from test set leaks in.
Input text = prepared remarks (management script, most signal-bearing portion).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import CACHE, OUTPUTS
from src.parser import parse_all

SETFIT_CACHE   = CACHE / "setfit_model"
FINBERT_CACHE  = CACHE / "finbert_finetuned"
PRIMARY_HORIZON = 21
RET_COL = f"fwd_excess_{PRIMARY_HORIZON}d"

SETFIT_BASE   = "ProsusAI/finbert"
FINBERT_BASE  = "ProsusAI/finbert"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_text_label_df(df: pd.DataFrame, transcripts) -> pd.DataFrame:
    """Join features rows with raw transcript text (prepared remarks).

    Returns DataFrame with columns: ticker, quarter, text, label (0/1).
    Drops rows where return is NaN or text is empty.
    """
    tmap = {(t.ticker, t.quarter): t for t in transcripts}
    rows = []
    for _, row in df.dropna(subset=[RET_COL]).iterrows():
        t = tmap.get((row["ticker"], row["quarter"]))
        if t is None:
            continue
        text = " ".join(b["text"] for b in t.prepared if b.get("text")).strip()
        if not text:
            continue
        label = int(float(row[RET_COL]) > 0)
        rows.append({"ticker": row["ticker"], "quarter": row["quarter"],
                     "text": text, "label": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. SetFit
# ---------------------------------------------------------------------------

@dataclass
class SetFitModel:
    model: object
    train_base_rate: float = 0.5
    label_names: List[str] = field(default_factory=lambda: ["down", "up"])

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Return P(up) for each row in df (matched by ticker+quarter)."""
        tl = _build_text_label_df(df, _get_transcripts())
        if tl.empty:
            return pd.Series(np.full(len(df), 0.5), index=df.index)
        probs = self.model.predict_proba(tl["text"].tolist())
        p_up = np.array([p[1] if len(p) > 1 else 0.5 for p in probs], dtype=float)
        tl = tl.reset_index(drop=True)
        tl["p_up"] = p_up
        merged = df.reset_index().merge(
            tl[["ticker", "quarter", "p_up"]], on=["ticker", "quarter"], how="left"
        ).set_index("index")
        return merged["p_up"].reindex(df.index).fillna(0.5)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(df).to_numpy()
        hi = self.train_base_rate + 0.05
        lo = self.train_base_rate - 0.05
        sig = np.where(proba > hi, 1, np.where(proba < lo, -1, 0))
        return pd.Series(sig, index=df.index)

    def predict_contrarian(self, df: pd.DataFrame) -> pd.Series:
        """Invert the SetFit band: long where the model says down, short where up.
        Same band centered on train base rate as ``predict``."""
        proba = self.predict_proba(df).to_numpy()
        hi = self.train_base_rate + 0.05
        lo = self.train_base_rate - 0.05
        sig = np.where(proba < lo, 1, np.where(proba > hi, -1, 0))
        return pd.Series(sig, index=df.index)


def fit_setfit(train_df: pd.DataFrame, force: bool = False,
               num_iterations: int = 20) -> SetFitModel:
    """SetFit-style fine-tuning via sentence-transformers (no setfit package needed).

    Algorithm:
      1. Build contrastive pairs: same-label pairs (positive) and cross-label pairs (negative).
      2. Fine-tune the sentence encoder with CosineSimilarityLoss.
      3. Encode all training texts → fit a logistic regression head.
    """
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from sklearn.linear_model import LogisticRegression
    import joblib

    clf_path = SETFIT_CACHE / "clf.joblib"
    enc_path = str(SETFIT_CACHE / "encoder")

    transcripts = _get_transcripts()
    tl = _build_text_label_df(train_df, transcripts)
    print(f"  SetFit: {len(tl)} training examples "
          f"(up={tl['label'].sum()}, down={(tl['label']==0).sum()})")

    base_rate = float(tl["label"].mean()) if len(tl) else 0.5

    if SETFIT_CACHE.exists() and clf_path.exists() and not force:
        print(f"  Loading cached SetFit from {SETFIT_CACHE} (train_base_rate={base_rate:.3f})")
        encoder = SentenceTransformer(enc_path)
        clf = joblib.load(clf_path)

        class _Proxy:
            def predict_proba(self, texts):
                embs = encoder.encode(texts, show_progress_bar=False)
                p = clf.predict_proba(embs)
                return p

        return SetFitModel(model=_Proxy(), train_base_rate=base_rate)

    # Step 1: build contrastive pairs
    pos_texts = tl[tl["label"] == 1]["text"].tolist()
    neg_texts = tl[tl["label"] == 0]["text"].tolist()
    examples = []
    import random; random.seed(42)
    for _ in range(num_iterations):
        for a in pos_texts:
            b = random.choice(pos_texts)
            examples.append(InputExample(texts=[a, b], label=1.0))
            c = random.choice(neg_texts) if neg_texts else a
            examples.append(InputExample(texts=[a, c], label=0.0))
        for a in neg_texts:
            b = random.choice(neg_texts)
            examples.append(InputExample(texts=[a, b], label=1.0))
            c = random.choice(pos_texts) if pos_texts else a
            examples.append(InputExample(texts=[a, c], label=0.0))

    # Step 2: fine-tune encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(SETFIT_BASE, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=8)
    loss_fn = losses.CosineSimilarityLoss(encoder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.fit(train_objectives=[(loader, loss_fn)], epochs=1,
                    show_progress_bar=True, warmup_steps=10)

    # Step 3: fit logistic head
    texts = tl["text"].tolist()
    labels = tl["label"].tolist()
    embs = encoder.encode(texts, show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(embs, labels)

    SETFIT_CACHE.mkdir(parents=True, exist_ok=True)
    encoder.save(enc_path)
    joblib.dump(clf, clf_path)
    print(f"  SetFit saved to {SETFIT_CACHE}")

    class _Proxy:
        def predict_proba(self, texts):
            e = encoder.encode(texts, show_progress_bar=False)
            return clf.predict_proba(e)

    return SetFitModel(model=_Proxy(), train_base_rate=base_rate)


# ---------------------------------------------------------------------------
# 2. FinBERT fine-tune (full sequence classification)
# ---------------------------------------------------------------------------

@dataclass
class FinBERTFinetuned:
    model_path: str

    def _load(self):
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-classification", model=self.model_path,
                        tokenizer=self.model_path, device=device, top_k=None)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        pipe = self._load()
        tl = _build_text_label_df(df, _get_transcripts())
        if tl.empty:
            return pd.Series(np.full(len(df), 0.5), index=df.index)
        p_ups = []
        for text in tl["text"].tolist():
            try:
                result = pipe(text[:512], truncation=True, max_length=512)[0]
                label_map = {r["label"].lower(): r["score"] for r in result}
                p_up = label_map.get("label_1", label_map.get("1", 0.5))
                p_ups.append(float(p_up))
            except Exception:
                p_ups.append(0.5)
        tl = tl.reset_index(drop=True)
        tl["p_up"] = p_ups
        merged = df.reset_index().merge(
            tl[["ticker", "quarter", "p_up"]], on=["ticker", "quarter"], how="left"
        ).set_index("index")
        return merged["p_up"].reindex(df.index).fillna(0.5)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(df).to_numpy()
        sig = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))
        return pd.Series(sig, index=df.index)


def fit_finbert_finetune(train_df: pd.DataFrame, force: bool = False,
                          n_epochs: int = 3, lr: float = 2e-5) -> FinBERTFinetuned:
    """Fine-tune FinBERT sequence classifier on training transcripts."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    from datasets import Dataset

    if FINBERT_CACHE.exists() and not force:
        print(f"  Loading cached fine-tuned FinBERT from {FINBERT_CACHE}")
        return FinBERTFinetuned(model_path=str(FINBERT_CACHE))

    transcripts = _get_transcripts()
    tl = _build_text_label_df(train_df, transcripts)
    print(f"  FinBERT fine-tune: {len(tl)} examples "
          f"(up={tl['label'].sum()}, down={(tl['label']==0).sum()})")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_BASE, num_labels=2, ignore_mismatched_sizes=True
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length",
                         max_length=512)

    dataset = Dataset.from_pandas(tl[["text", "label"]]).map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=str(FINBERT_CACHE),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    FINBERT_CACHE.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINBERT_CACHE))
    tokenizer.save_pretrained(str(FINBERT_CACHE))
    print(f"  Fine-tuned FinBERT saved to {FINBERT_CACHE}")
    return FinBERTFinetuned(model_path=str(FINBERT_CACHE))


# ---------------------------------------------------------------------------
# Lazy transcript loader
# ---------------------------------------------------------------------------

_transcripts_cache = None

def _get_transcripts():
    global _transcripts_cache
    if _transcripts_cache is None:
        _transcripts_cache = parse_all()
    return _transcripts_cache
