"""Parse S&P earnings-call transcripts into structured objects."""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import ECT_ZIP, TRANSCRIPTS

SECTION_HEADERS = {
    "Presentation Operator Message",
    "Presenter Speech",
    "Question and Answer Operator Message",
    "Question",
    "Answer",
}

ROLE_LINE_RE = re.compile(r"^(Executives|Analysts|Operator)\b.*$")

HEADER_RE = re.compile(
    r"(?P<company>.+?),\s*Q(?P<q>\d)\s*(?P<y>\d{4})"
    r".*?Earnings Call"
    r".*?(?P<date>[A-Z][a-z]+ \d{1,2},\s*\d{4})"
)

# Fallback for annual/non-Q headers (e.g. "Fastenal Company, 2025 Earnings Call, Jan 20, 2026").
HEADER_RE_ANNUAL = re.compile(
    r"(?P<company>.+?),\s*\d{4}\s+Earnings Call"
    r".*?(?P<date>[A-Z][a-z]+ \d{1,2},\s*\d{4})"
)


@dataclass
class Transcript:
    ticker: str
    quarter: str
    call_date: Optional[str]
    company: str
    prepared: List[Dict] = field(default_factory=list)
    qa: List[Dict] = field(default_factory=list)
    raw_path: str = ""


def unzip_transcripts(tickers: Optional[List[str]] = None) -> List[Path]:
    """Unpack ECT.zip into transcripts/, optionally filtering by ticker.

    FILENAME_REMAP forces non-standard filenames (e.g. Fastenal's annual call, which
    the dataset ships as FAST_2026_01_20.txt) into the standard TICKER_Q-YYYY form.
    """
    FILENAME_REMAP = {
        "FAST_2026_01_20.txt": "FAST_Q4-2025.txt",
    }
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ECT_ZIP) as z:
        for name in z.namelist():
            if not name.endswith(".txt") or "__MACOSX" in name:
                continue
            base = Path(name).name
            base = FILENAME_REMAP.get(base, base)
            ticker = base.split("_")[0]
            if tickers is not None and ticker not in tickers:
                continue
            dst = TRANSCRIPTS / base
            if not dst.exists():
                with z.open(name) as src, dst.open("wb") as out:
                    out.write(src.read())
    for old in FILENAME_REMAP:
        stray = TRANSCRIPTS / old
        if stray.exists():
            stray.unlink()
    return sorted(TRANSCRIPTS.glob("*.txt"))


def _filename_meta(path: Path) -> Tuple[str, str]:
    stem = path.stem
    ticker, _, q = stem.partition("_")
    return ticker, q


def reporting_period_from_call_date(call_date: Optional[str]) -> Optional[str]:
    """Map an earnings-call date to the calendar reporting period it discusses.

    Companies label their fiscal quarters inconsistently in filenames/headers, so
    we derive a canonical label from the actual call date. Rule of thumb: earnings
    calls happen 1-3 months after the quarter they report on. Months 1-3 report Q4
    of the prior calendar year; 4-6 -> Q1; 7-9 -> Q2; 10-12 -> Q3.
    """
    if not call_date:
        return None
    d = datetime.strptime(call_date, "%Y-%m-%d")
    m, y = d.month, d.year
    if 1 <= m <= 3:
        return f"{y - 1}-Q4"
    if 4 <= m <= 6:
        return f"{y}-Q1"
    if 7 <= m <= 9:
        return f"{y}-Q2"
    return f"{y}-Q3"


def parse_transcript(path: Path) -> Transcript:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    first_line = next((l.strip().lstrip("\ufeff") for l in lines if l.strip()), "")
    m = HEADER_RE.search(first_line) or HEADER_RE_ANNUAL.search(first_line)
    company, date = "", None
    if m:
        company = m.group("company").strip()
        date_raw = re.sub(r"\s+", " ", m.group("date"))
        try:
            date = datetime.strptime(date_raw, "%b %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass

    ticker, quarter = _filename_meta(path)

    prepared: List[Dict] = []
    qa: List[Dict] = []
    current_section: Optional[str] = None
    current_role: Optional[str] = None
    buf: List[str] = []
    in_qa = False
    current_q: Optional[Dict] = None

    def flush() -> None:
        nonlocal current_q
        body = "\n".join(buf).strip()
        buf.clear()
        if not body or current_role is None:
            return
        if not in_qa and current_section == "Presenter Speech":
            prepared.append({"role": current_role, "text": body})
        elif in_qa and current_section == "Question":
            current_q = {"q_role": current_role, "question": body, "a_role": None, "answer": None}
            qa.append(current_q)
        elif in_qa and current_section == "Answer":
            if current_q is None or current_q.get("answer") is not None:
                stub = {"q_role": None, "question": "", "a_role": current_role, "answer": body}
                qa.append(stub)
                current_q = stub
            else:
                current_q["a_role"] = current_role
                current_q["answer"] = body

    for line in lines:
        s = line.strip().lstrip("\ufeff")
        if s in SECTION_HEADERS:
            flush()
            current_section = s
            current_role = None
            if s == "Question and Answer Operator Message":
                in_qa = True
        elif current_section is not None and ROLE_LINE_RE.match(s):
            flush()
            current_role = s
        else:
            buf.append(line)
    flush()

    return Transcript(
        ticker=ticker,
        quarter=quarter,
        call_date=date,
        company=company,
        prepared=prepared,
        qa=qa,
        raw_path=str(path),
    )


def parse_all(tickers: Optional[List[str]] = None) -> List[Transcript]:
    paths = unzip_transcripts(tickers)
    return [parse_transcript(p) for p in paths]
