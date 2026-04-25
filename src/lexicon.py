"""Loughran-McDonald-style lexicon baseline for earnings-call sentiment.

Curated subset of the canonical Loughran-McDonald Master Dictionary (Tim Loughran
& Bill McDonald, 2011, "When Is a Liability Not a Liability?", Journal of
Finance). We hold out only the highest-frequency entries in each polarity so a
pure word-count works without the full dictionary. This is a baseline to
compare against LLM-derived sentiment, not a replacement.
"""
from __future__ import annotations

import re
from typing import Dict

LM_POSITIVE = {
    "strong", "strength", "strengthen", "strengthened", "strengthening",
    "growth", "grow", "grew", "growing", "growth",
    "gain", "gains", "gained", "gaining",
    "record", "records", "record-high",
    "exceed", "exceeded", "exceeding", "exceeds",
    "outperform", "outperformed", "outperforming", "outperformance",
    "achievement", "achievements", "achieve", "achieved", "achieving",
    "success", "successful", "successfully",
    "beneficial", "benefit", "benefits", "benefited", "benefiting",
    "improve", "improved", "improvement", "improvements", "improving",
    "advance", "advanced", "advancement", "advances", "advancing",
    "boost", "boosted", "boosting",
    "expansion", "expand", "expanded", "expanding", "expansive",
    "profit", "profitable", "profitability", "profits",
    "favorable", "favorably",
    "positive", "positively",
    "upside", "upturn", "uptick",
    "robust", "resilient", "resilience",
    "leadership", "leading",
    "momentum", "accelerate", "accelerated", "accelerating", "acceleration",
    "opportunity", "opportunities",
    "innovate", "innovation", "innovative",
    "efficient", "efficiency", "efficiencies",
    "surpass", "surpassed", "surpassing",
    "rebound", "rebounded", "rebounding",
    "upgrade", "upgraded", "upgrades",
}

LM_NEGATIVE = {
    "weak", "weakness", "weakened", "weakening",
    "decline", "declined", "declines", "declining",
    "loss", "losses", "lost",
    "impair", "impaired", "impairment", "impairments",
    "challenging", "challenge", "challenged", "challenges",
    "difficult", "difficulty", "difficulties",
    "adverse", "adversely",
    "unfavorable", "unfavorably",
    "negative", "negatively",
    "downturn", "downside", "downgrade", "downgraded", "downgrades",
    "deteriorate", "deteriorated", "deterioration", "deteriorating",
    "pressure", "pressured", "pressures", "pressuring",
    "volatile", "volatility",
    "uncertain", "uncertainty", "uncertainties",
    "risk", "risks", "risky",
    "concern", "concerns", "concerned", "concerning",
    "headwind", "headwinds",
    "lawsuit", "lawsuits", "litigation",
    "restructure", "restructured", "restructuring",
    "writedown", "writedowns", "write-off", "write-offs",
    "default", "defaults", "defaulted",
    "bankruptcy", "bankrupt",
    "fraud", "fraudulent",
    "penalty", "penalties", "fine", "fined", "fines",
    "slowdown", "slowed", "slowing",
    "shortfall", "shortfalls",
    "underperform", "underperformed", "underperforming", "underperformance",
    "miss", "missed", "missing",
    "cut", "cutting", "reduced", "reducing", "reduction", "reductions",
    "layoff", "layoffs",
    "disappointing", "disappointed", "disappointment",
}

_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def lm_sentiment(text: str) -> Dict[str, float]:
    """Count LM-style positive vs. negative hits; return counts and a normalized score.

    lm_sentiment score = (pos - neg) / max(pos + neg, 1), range [-1, 1].
    """
    if not text:
        return {"lm_pos": 0, "lm_neg": 0, "lm_sentiment": 0.0}
    tokens = _TOKEN_RE.findall(text.lower())
    pos = sum(1 for tok in tokens if tok in LM_POSITIVE)
    neg = sum(1 for tok in tokens if tok in LM_NEGATIVE)
    denom = max(pos + neg, 1)
    return {"lm_pos": int(pos), "lm_neg": int(neg), "lm_sentiment": (pos - neg) / denom}
