"""Render `docs/writeup.md` to `outputs/writeup.pdf` as a clean executive
briefing — McKinsey/Goldman aesthetic: single deep-navy accent, charcoal text,
hairline rules, generous whitespace, no decorative banners."""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "writeup.md"
OUT = ROOT / "outputs" / "writeup.pdf"

# --- restrained executive palette --------------------------------------------
INK         = HexColor("#1A1A1A")  # primary text — near-black
CHARCOAL    = HexColor("#3F3F46")  # body text
SLATE       = HexColor("#52525B")  # secondary text
MUTED       = HexColor("#737373")  # captions, footer
HAIRLINE    = HexColor("#D4D4D8")  # rules, table grid
RULE_LIGHT  = HexColor("#E5E5E5")  # zebra band
BAND        = HexColor("#FAFAFA")  # subtle alternating row tint
PAPER       = HexColor("#FFFFFF")

ACCENT      = HexColor("#0F4C81")  # deep navy — single brand color
ACCENT_SOFT = HexColor("#E8EEF4")  # navy at 8% — used only for header underline strip
UP          = HexColor("#15803D")  # muted forest green
DOWN        = HexColor("#B91C1C")  # muted brick red

PAGE_W, PAGE_H = LETTER
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
SECTION_GAP = 14
PARA_GAP = 6


# --- inline markdown → reportlab-flavoured mini-HTML -------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*_])\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _INLINE_CODE.sub(
        r'<font face="Courier" size="9" color="#3F3F46">\1</font>', text
    )
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _MD_LINK.sub(r'<link href="\2" color="#0F4C81">\1</link>', text)
    return text


def _make_styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "Body": ParagraphStyle(
            "Body", parent=ss["BodyText"], fontName="Helvetica",
            fontSize=10, leading=15, spaceAfter=PARA_GAP, textColor=CHARCOAL,
        ),
        "BodyTight": ParagraphStyle(
            "BodyTight", parent=ss["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=13, spaceAfter=3, textColor=CHARCOAL,
        ),
        "Lede": ParagraphStyle(
            # First paragraph after a section header — slightly larger, charcoal,
            # no special color treatment. Lets typography do the hierarchy work.
            "Lede", parent=ss["BodyText"], fontName="Helvetica",
            fontSize=10.5, leading=15.5, spaceBefore=2, spaceAfter=PARA_GAP + 2,
            textColor=INK,
        ),
        "H3": ParagraphStyle(
            "H3", parent=ss["Heading3"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=14, spaceBefore=14, spaceAfter=4,
            textColor=INK,
        ),
        "Pullquote": ParagraphStyle(
            # Italic, indented, quiet color. Used for `_commentary`.
            "Pullquote", parent=ss["BodyText"], fontName="Helvetica-Oblique",
            fontSize=10, leading=14, spaceAfter=PARA_GAP, textColor=SLATE,
            leftIndent=12, rightIndent=12,
        ),
        "Callout": ParagraphStyle(
            # Key finding — bold, near-black, sits next to a thin accent rule.
            "Callout", parent=ss["BodyText"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=15, spaceAfter=PARA_GAP, textColor=INK,
            leftIndent=14, rightIndent=8,
        ),
        "FigCaption": ParagraphStyle(
            "FigCaption", parent=ss["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8.5, leading=11, spaceAfter=2, textColor=MUTED,
            alignment=1,
        ),
        "Code": ParagraphStyle(
            "Code", parent=ss["BodyText"], fontName="Courier",
            fontSize=8.5, leading=11.5, backColor=BAND,
            borderColor=HAIRLINE, borderWidth=0.5, borderPadding=8,
            leftIndent=12, rightIndent=12, spaceBefore=4, spaceAfter=8,
            textColor=CHARCOAL,
        ),
        # ---- cover-only styles ---------------------------------------------
        "Eyebrow": ParagraphStyle(
            # Tiny uppercase tracking-out label above the title (e.g. "BRIEFING").
            "Eyebrow", fontName="Helvetica-Bold", fontSize=8, leading=10,
            textColor=ACCENT, spaceAfter=10,
        ),
        "Title": ParagraphStyle(
            "Title", fontName="Helvetica-Bold", fontSize=26, leading=30,
            textColor=INK, spaceAfter=4,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", fontName="Helvetica", fontSize=12, leading=16,
            textColor=SLATE, spaceAfter=10,
        ),
        "Byline": ParagraphStyle(
            "Byline", fontName="Helvetica", fontSize=9, leading=12,
            textColor=MUTED,
        ),
        "KpiNumber": ParagraphStyle(
            "KpiNumber", fontName="Helvetica-Bold", fontSize=22, leading=24,
            textColor=INK, alignment=0,
        ),
        "KpiLabel": ParagraphStyle(
            "KpiLabel", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
            textColor=MUTED, alignment=0,
        ),
        # ---- section header styles -----------------------------------------
        "SectionNum": ParagraphStyle(
            # Big quiet number on the left of the section header (e.g. "06").
            "SectionNum", fontName="Helvetica-Bold", fontSize=11, leading=14,
            textColor=ACCENT,
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle", fontName="Helvetica-Bold", fontSize=14, leading=17,
            textColor=INK,
        ),
    }


STYLES = _make_styles()


# --- custom Flowables --------------------------------------------------------

class Rule(Flowable):
    """A horizontal hairline. Defaults to full content width, light gray, 0.5pt."""
    def __init__(self, width: float, color=HAIRLINE, thickness: float = 0.5,
                 space_before: float = 0, space_after: float = 0):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness
        self.space_before = space_before
        self.space_after = space_after
        self.height = thickness + space_before + space_after

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setStrokeColor(self.color)
        c.setLineWidth(self.thickness)
        y = self.space_after + self.thickness / 2
        c.line(0, y, self.width, y)


_SECTION_NUM_RE = re.compile(r"^(\d+)\.\s+(.*)$")


class SectionHeader(Flowable):
    """Section header with quiet "0N" number, bold uppercase title, thin rule.

    Renders like a McKinsey slide title: small accent number → bold black title
    → 0.6pt accent rule that runs the full content width below the text.
    No solid-color banner."""

    def __init__(self, text: str, width: float):
        super().__init__()
        self.width = width
        m = _SECTION_NUM_RE.match(text.strip())
        if m:
            self.number = m.group(1).zfill(2)
            self.title = m.group(2).upper()
        else:
            self.number = None
            self.title = text.upper()
        # Layout constants
        self._title_y = 14
        self._rule_y = 8
        self.height = 30

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        # Number + title baseline
        x = 0
        if self.number is not None:
            c.setFillColor(ACCENT)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, self._title_y, self.number)
            x += c.stringWidth(self.number, "Helvetica-Bold", 10) + 12
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(x, self._title_y, self.title)
        # Hairline rule below
        c.setStrokeColor(ACCENT)
        c.setLineWidth(0.6)
        c.line(0, self._rule_y, self.width, self._rule_y)


class Cover(Flowable):
    """Cover block: eyebrow, title, subtitle, byline, separator rule.
    Replaces the old purple banner with a clean editorial layout."""

    def __init__(self, eyebrow: str, title: str, subtitle: str, byline: str,
                 width: float):
        super().__init__()
        self.width = width
        self.eyebrow = eyebrow
        self.title = title
        self.subtitle = subtitle
        self.byline = byline
        # Total stacked height (paragraphs sized in draw via canvas — we
        # measure manually here for a tighter layout).
        self.height = 1.55 * inch

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        # Eyebrow — tracked-out caps in accent
        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(0, self.height - 12, self.eyebrow)
        # Accent square next to the eyebrow
        eyebrow_w = c.stringWidth(self.eyebrow, "Helvetica-Bold", 8.5)
        c.setFillColor(ACCENT)
        c.rect(eyebrow_w + 8, self.height - 11, 4, 4, fill=1, stroke=0)
        # Title
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(0, self.height - 48, self.title)
        # Subtitle
        c.setFillColor(SLATE)
        c.setFont("Helvetica", 12)
        c.drawString(0, self.height - 70, self.subtitle)
        # Hairline above byline
        c.setStrokeColor(HAIRLINE)
        c.setLineWidth(0.5)
        c.line(0, self.height - 92, self.width, self.height - 92)
        # Byline
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(0, self.height - 108, self.byline)


def _kpi_strip(cards: list[tuple[str, str]], width: float) -> Table:
    """Horizontal strip of KPIs — Bloomberg-style facts bar.

    Each cell: bold number on top in INK, thin uppercase label below in MUTED.
    No fill, no border, just thin vertical hairlines between cells and a
    single accent rule above and a single hairline below."""
    n = len(cards)
    inner_w = width / n

    cells_row = []
    for value, label in cards:
        cell = Table(
            [
                [Paragraph(value, STYLES["KpiNumber"])],
                [Paragraph(label.upper(), STYLES["KpiLabel"])],
            ],
            colWidths=[inner_w - 16],
        )
        cell.setStyle(TableStyle([
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, 0), 0),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING",    (0, 1), (-1, 1), 0),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
        ]))
        cells_row.append(cell)

    outer = Table([cells_row], colWidths=[inner_w] * n)
    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, ACCENT),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, HAIRLINE),
        ("VALIGN",  (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]
    # Subtle vertical hairlines between cells (not at outer edges)
    for i in range(1, n):
        style.append(("LINEBEFORE", (i, 0), (i, -1), 0.5, HAIRLINE))
    outer.setStyle(TableStyle(style))
    return outer


def _commentary(text: str) -> Table:
    """Quiet pull-quote: italic slate text, indented, with a thin accent rule
    on the left. No filled background."""
    p = Paragraph(_inline(text), STYLES["Pullquote"])
    t = Table([[p]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("LINEBEFORE",     (0, 0), (0, -1), 1.2, ACCENT),
        ("LEFTPADDING",    (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    return t


def _callout(text: str) -> Table:
    """Key-finding sidebar — thin accent left rule, bold near-black text.
    No filled background, no purple anything."""
    p = Paragraph(_inline(text), STYLES["Callout"])
    t = Table([[p]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("LINEBEFORE",     (0, 0), (0, -1), 2, ACCENT),
        ("LEFTPADDING",    (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


# --- results table (clean editorial style) -----------------------------------

_NUM_RE = re.compile(r"^([-+])?(\d+(?:\.\d+)?)(%?)$")


def _is_signed_numeric(cell: str) -> tuple[bool, float]:
    s = cell.strip().strip("*").replace(",", "")
    m = _NUM_RE.match(s)
    if not m:
        return False, 0.0
    val = float(m.group(2))
    if m.group(1) == "-":
        val = -val
    return True, val


def _results_table(rows: list[list[str]], highlight_row: int | None = None) -> Table:
    """Clean editorial table: navy hairline above + below header, zebra row
    bands, signed numerics get just colored *text* (no filled cells), winner
    row gets a left accent rule + bold text.

    First row is the header. `highlight_row` is 1-based index into data rows."""
    header = rows[0]
    data_rows = rows[1:]
    col_w = [CONTENT_W * 0.24] + [(CONTENT_W * 0.76) / (len(header) - 1)] * (len(header) - 1)

    body = [[Paragraph(_inline(c), STYLES["BodyTight"]) for c in header]]
    ts = TableStyle([
        # Header
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        # Body grid — only horizontal hairlines, no vertical lines
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, ACCENT),
        # Alignment
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Spacing
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 12),
    ])

    for r_idx, row in enumerate(data_rows, start=1):
        rendered = []
        is_winner = highlight_row is not None and r_idx == highlight_row
        if not is_winner and r_idx % 2 == 0:
            ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), BAND)
        for c_idx, cell in enumerate(row):
            txt = cell.strip()
            is_num, val = _is_signed_numeric(txt)
            style = STYLES["BodyTight"]
            if is_winner:
                rendered.append(Paragraph(f"<b>{_inline(txt)}</b>", style))
            else:
                rendered.append(Paragraph(_inline(txt), style))
            if c_idx > 0 and is_num and not is_winner:
                if val > 0:
                    ts.add("TEXTCOLOR", (c_idx, r_idx), (c_idx, r_idx), UP)
                elif val < 0:
                    ts.add("TEXTCOLOR", (c_idx, r_idx), (c_idx, r_idx), DOWN)
        body.append(rendered)
        if is_winner:
            ts.add("LINEBEFORE", (0, r_idx), (0, r_idx), 2.5, ACCENT)
            ts.add("TEXTCOLOR", (0, r_idx), (-1, r_idx), INK)
            ts.add("FONTNAME", (0, r_idx), (-1, r_idx), "Helvetica-Bold")
            ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), ACCENT_SOFT)

    t = Table(body, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(ts)
    return t


# --- markdown block parser ---------------------------------------------------

def _parse_table_rows(lines: list[str], i: int) -> tuple[list[list[str]] | None, int]:
    if i + 1 >= len(lines) or not re.match(
        r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", lines[i + 1]
    ):
        return None, i
    rows_text = [lines[i]]
    j = i + 2
    while j < len(lines) and lines[j].strip().startswith("|"):
        rows_text.append(lines[j])
        j += 1
    rows = [
        [c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))]
        for r in rows_text
    ]
    return rows, j


def _parse_code_block(lines: list[str], i: int) -> tuple[Paragraph, int]:
    j = i + 1
    body: list[str] = []
    while j < len(lines) and not lines[j].startswith("```"):
        body.append(lines[j])
        j += 1
    escaped = [l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for l in body]
    return Paragraph("<br/>".join(escaped) or "&nbsp;", STYLES["Code"]), j + 1


_ORDERED_PREFIX = re.compile(r"^\s*(\d+)\.\s+(.*)")
_UNORDERED_PREFIX = re.compile(r"^\s*[-*]\s+(.*)")


def _parse_list(lines: list[str], i: int):
    """Return a list of flowables (one per item) preserving author-written
    numeric prefixes so list numbering doesn't restart after a paragraph
    break. Bullet lists use a real ListFlowable with a small accent dot."""
    ordered = bool(_ORDERED_PREFIX.match(lines[i]))
    j = i
    out: list = []
    if ordered:
        while j < len(lines):
            m = _ORDERED_PREFIX.match(lines[j])
            if not m:
                break
            num = m.group(1)
            raw = re.sub(r"^\[[x ]\]\s*", "", m.group(2))
            p = Paragraph(
                f"<b><font color='#0F4C81'>{num}.</font></b>&nbsp;&nbsp;{_inline(raw)}",
                STYLES["BodyTight"],
            )
            out.append(p)
            j += 1
        return out, j
    # unordered
    items: list[ListItem] = []
    while j < len(lines):
        m = _UNORDERED_PREFIX.match(lines[j])
        if not m:
            break
        raw = re.sub(r"^\[[x ]\]\s*", "", m.group(1))
        items.append(ListItem(Paragraph(_inline(raw), STYLES["BodyTight"])))
        j += 1
    flow = ListFlowable(
        items, bulletType="bullet",
        leftIndent=18, bulletFontSize=8, bulletColor=ACCENT,
        bulletFontName="Helvetica-Bold",
    )
    return [flow], j


def _maybe_images(line: str) -> list[Path]:
    paths = re.findall(r"`(outputs/figures/[^`]+\.png)`", line)
    return [ROOT / p for p in paths if (ROOT / p).exists()]


# --- top-level render with section-aware styling -----------------------------

# H2s suppressed entirely — folded into cover or rendered specially.
SKIP_H2 = {"1. problem and scope"}


def render(md: str) -> list:
    flow: list = []
    lines = md.splitlines()
    i = 0
    para_buf: list[str] = []
    fig_counter = [0]
    after_section = [False]
    current_h2 = None

    # --- cover ---------------------------------------------------------------
    today = dt.date.today().isoformat()
    flow.append(Cover(
        eyebrow="EARNINGS-CALL NLP   ·   RESEARCH BRIEFING",
        title="From Transcript to Trading Signal",
        subtitle="An honest backtest of LLM-extracted earnings-call features across 14 tickers.",
        byline=f"Tomer Gross   ·   NLP for Finance, Spring 2026   ·   {today}",
        width=CONTENT_W,
    ))
    flow.append(Spacer(1, 8))
    kpi_items = [
        ("131",   "Transcripts"),
        ("14",    "Tickers"),
        ("8",     "Signals tested"),
        ("+0.46", "Best Sharpe (63d)"),
    ]
    nlp_eval_path = ROOT / "outputs" / "nlp_evaluation.json"
    if nlp_eval_path.exists():
        try:
            import json as _json
            _ev = _json.loads(nlp_eval_path.read_text())
            _agree = _ev.get("directional_agreement")
            if _agree is not None:
                kpi_items.append((f"{_agree:.0%}", "LLM-as-a-Judge Agreement"))
        except Exception:
            pass
    flow.append(_kpi_strip(kpi_items, CONTENT_W))
    flow.append(Spacer(1, SECTION_GAP))
    flow.append(_commentary(
        "Pipeline parses S&P Capital IQ transcripts, extracts sentiment, wins, risks, and "
        "guidance with gemma3:4b across a four-call hybrid (overall, CEO, CFO, analyst), "
        "layers FinBERT and Loughran-McDonald lexicon scores plus pre-call price momentum, "
        "and predicts forward excess return over SPY on a strict 70/30 per-ticker temporal "
        "split, with dynamic T+0/T+1 entry adjusted for BMO vs. AMC reporting habits. Eight "
        "signals are compared end-to-end across four horizons (1d, 5d, 21d, 63d). The "
        "Contrarian SetFit signal is the headline finding — its hit rate climbs monotonically "
        "with horizon and the 63-day configuration is the only one in the suite with positive "
        "hit rate (0.548), positive rank IC (+0.123), and positive naïve Sharpe (+0.46)."
    ))
    flow.append(Spacer(1, SECTION_GAP))

    def flush_para():
        nonlocal para_buf
        if not para_buf:
            return
        text = " ".join(para_buf).strip()
        para_buf = []
        if not text:
            return
        # Sidebar callout: paragraph beginning with ">> "
        if text.startswith(">> "):
            flow.append(_callout(text[3:].strip()))
            flow.append(Spacer(1, PARA_GAP))
            after_section[0] = False
            return
        img_paths = _maybe_images(text)
        # First paragraph after a section header gets the larger Lede style
        style = STYLES["Lede"] if after_section[0] else STYLES["Body"]
        after_section[0] = False
        flow.append(Paragraph(_inline(text), style))
        for img_path in img_paths:
            try:
                fig_counter[0] += 1
                caption = img_path.stem.replace("_", " ").title()
                img = Image(
                    str(img_path), width=CONTENT_W * 0.62,
                    height=CONTENT_W * 0.62 * 0.46, kind="proportional",
                )
                cap = Paragraph(
                    f"FIG. {fig_counter[0]:02d}  ·  {caption}",
                    STYLES["FigCaption"],
                )
                flow.append(KeepTogether([
                    Spacer(1, 6), img, Spacer(1, 4), cap, Spacer(1, 4),
                ]))
            except Exception:
                pass

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # H1 — already rendered as cover; skip
        if stripped.startswith("# ") and not stripped.startswith("## "):
            i += 1
            continue

        # H2 — clean section header, no banner
        if stripped.startswith("## "):
            flush_para()
            title = stripped[3:].strip()
            if title.lower() in SKIP_H2:
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("## "):
                    i += 1
                continue
            current_h2 = title.lower()
            flow.append(Spacer(1, SECTION_GAP))
            flow.append(SectionHeader(title, CONTENT_W))
            flow.append(Spacer(1, 8))
            after_section[0] = True
            i += 1
            continue

        # H3 — bold near-black, used for ticker stories
        if stripped.startswith("### "):
            flush_para()
            flow.append(Paragraph(_inline(stripped[4:].strip()), STYLES["H3"]))
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            block, i = _parse_code_block(lines, i)
            flow.append(block)
            continue

        # pipe table
        if stripped.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            flush_para()
            rows, i = _parse_table_rows(lines, i)
            if rows is None:
                continue
            if current_h2 and current_h2.startswith("6. results"):
                highlight = None
                for idx, r in enumerate(rows[1:], start=1):
                    if any("**" in c for c in r):
                        highlight = idx
                        break
                clean_rows = [[c.replace("**", "") for c in r] for r in rows]
                tbl = _results_table(clean_rows, highlight_row=highlight)
            else:
                clean_rows = [[c.replace("**", "") for c in r] for r in rows]
                tbl = _results_table(clean_rows, highlight_row=None)
            flow.append(tbl)
            flow.append(Spacer(1, PARA_GAP + 2))
            continue

        if re.match(r"^\s*(?:\d+\.|[-*])\s+", line):
            flush_para()
            items, i = _parse_list(lines, i)
            flow.extend(items)
            flow.append(Spacer(1, 4))
            continue

        if stripped == "---":
            flush_para()
            flow.append(Spacer(1, 4))
            flow.append(Rule(CONTENT_W, color=HAIRLINE, thickness=0.4))
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if stripped == "":
            flush_para()
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    return flow


# --- footer ------------------------------------------------------------------

def _on_page(canvas, doc):
    canvas.saveState()
    # Top hairline (skipped on cover page)
    if doc.page > 1:
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, PAGE_H - 0.45 * inch, PAGE_W - MARGIN, PAGE_H - 0.45 * inch)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(
            MARGIN, PAGE_H - 0.35 * inch,
            "EARNINGS-CALL NLP   ·   RESEARCH BRIEFING",
        )
    # Bottom rule + page number
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 0.5 * inch, PAGE_W - MARGIN, 0.5 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        MARGIN, 0.34 * inch,
        f"Tomer Gross   ·   {dt.date.today().isoformat()}",
    )
    canvas.drawRightString(
        PAGE_W - MARGIN, 0.34 * inch, f"{doc.page:02d}",
    )
    canvas.restoreState()


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = SRC.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Earnings-Call NLP — Research Briefing",
        author="Tomer Gross",
    )
    doc.build(render(md), onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
