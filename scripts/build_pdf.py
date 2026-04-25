"""Render `docs/writeup.md` to `outputs/writeup.pdf` in the Earnings Briefing
visual format (purple header, KPI cards, orange section banners, color-coded
results table, per-ticker commentary cards, footer)."""
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

PURPLE = HexColor("#5B2C83")
PURPLE_LIGHT = HexColor("#EDE4F2")
ORANGE = HexColor("#E8833A")
GREEN = HexColor("#2E8B57")
GREEN_LIGHT = HexColor("#DDEFE0")
RED = HexColor("#C04040")
RED_LIGHT = HexColor("#F5DADA")
GREY_DARK = HexColor("#333333")
GREY_MED = HexColor("#777777")
GREY_LIGHT = HexColor("#E5E5E5")
GREY_BAND = HexColor("#F9F9F9")

PAGE_W, PAGE_H = LETTER
MARGIN = 0.6 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
SECTION_GAP = 8


# --- inline markdown → reportlab-flavoured mini-HTML -------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*_])\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _INLINE_CODE.sub(r'<font face="Courier" size="9">\1</font>', text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _MD_LINK.sub(r'<link href="\2" color="#5B2C83">\1</link>', text)
    return text


def _make_styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "Body": ParagraphStyle(
            "Body", parent=ss["BodyText"], fontName="Helvetica",
            fontSize=10, leading=14, spaceAfter=4, textColor=GREY_DARK,
        ),
        "BodyTight": ParagraphStyle(
            "BodyTight", parent=ss["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=12, spaceAfter=3, textColor=GREY_DARK,
        ),
        "H2Sub": ParagraphStyle(
            "H2Sub", parent=ss["BodyText"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, spaceBefore=2, spaceAfter=8,
            textColor=PURPLE,
        ),
        "H3": ParagraphStyle(
            "H3", parent=ss["Heading3"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
            textColor=PURPLE,
        ),
        "Commentary": ParagraphStyle(
            "Commentary", parent=ss["BodyText"], fontName="Helvetica-Oblique",
            fontSize=9.5, leading=12.5, spaceAfter=4, textColor=PURPLE,
            leftIndent=8, rightIndent=8,
        ),
        "Callout": ParagraphStyle(
            "Callout", parent=ss["BodyText"], fontName="Helvetica-Bold",
            fontSize=10, leading=14, spaceAfter=4, textColor=PURPLE,
            leftIndent=8, rightIndent=8,
        ),
        "FigCaption": ParagraphStyle(
            "FigCaption", parent=ss["BodyText"], fontName="Helvetica-Oblique",
            fontSize=9, leading=11, spaceAfter=2, textColor=GREY_MED,
            alignment=1,
        ),
        "Code": ParagraphStyle(
            "Code", parent=ss["BodyText"], fontName="Courier",
            fontSize=8.5, leading=11, backColor=HexColor("#F7F4F9"),
            borderColor=PURPLE_LIGHT, borderWidth=0.5, borderPadding=6,
            leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=6,
            textColor=GREY_DARK,
        ),
        "KpiValue": ParagraphStyle(
            "KpiValue", fontName="Helvetica-Bold", fontSize=20, leading=22,
            alignment=1, textColor=colors.white,
        ),
        "KpiLabel": ParagraphStyle(
            "KpiLabel", fontName="Helvetica", fontSize=8, leading=10,
            alignment=1, textColor=colors.white,
        ),
        "BannerH1": ParagraphStyle(
            "BannerH1", fontName="Helvetica-Bold", fontSize=20, leading=24,
            textColor=colors.white,
        ),
        "BannerSub": ParagraphStyle(
            "BannerSub", fontName="Helvetica", fontSize=10, leading=13,
            textColor=colors.white,
        ),
        "SectionBanner": ParagraphStyle(
            "SectionBanner", fontName="Helvetica-Bold", fontSize=12, leading=16,
            textColor=colors.white,
        ),
    }


STYLES = _make_styles()


# --- custom Flowables --------------------------------------------------------

class HeaderBanner(Flowable):
    """Purple cover banner with title, subtitle, and orange accent strip."""
    def __init__(self, title: str, subtitle: str, width: float):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.accent_h = 4
        self.banner_h = 1.05 * inch
        self.height = self.banner_h + self.accent_h

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(PURPLE)
        c.rect(0, self.accent_h, self.width, self.banner_h, fill=1, stroke=0)
        c.setFillColor(ORANGE)
        c.rect(0, 0, self.width, self.accent_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(18, self.height - 38, self.title)
        c.setFont("Helvetica", 10)
        for i, line in enumerate(self.subtitle.split("\n")):
            c.drawString(18, self.height - 60 - i * 13, line)


_SECTION_NUM_RE = re.compile(r"^(\d+)\.\s+(.*)$")


class SectionBanner(Flowable):
    """Orange section banner with optional purple numeric badge on the left."""
    def __init__(self, text: str, width: float, color=ORANGE):
        super().__init__()
        self.color = color
        self.width = width
        self.height = 0.30 * inch
        m = _SECTION_NUM_RE.match(text.strip())
        if m:
            self.number = m.group(1)
            self.text = m.group(2)
        else:
            self.number = None
            self.text = text

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        text_x = 14
        if self.number is not None:
            badge_size = self.height - 8
            badge_x = 6
            badge_y = 4
            c.setFillColor(PURPLE)
            c.rect(badge_x, badge_y, badge_size, badge_size, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 12)
            num_w = c.stringWidth(self.number, "Helvetica-Bold", 12)
            c.drawString(
                badge_x + (badge_size - num_w) / 2,
                badge_y + (badge_size - 9) / 2,
                self.number,
            )
            text_x = badge_x + badge_size + 10
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(text_x, 8, self.text.upper())


def _kpi_row(cards: list[tuple[str, str]], width: float) -> Table:
    """Row of equal-width purple KPI cards: (value, label)."""
    cells = [
        [Paragraph(v, STYLES["KpiValue"]), Paragraph(l, STYLES["KpiLabel"])]
        for v, l in cards
    ]
    inner_tables = []
    for pair in cells:
        t = Table([[pair[0]], [pair[1]]], colWidths=[width / len(cards) - 4])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
            ("TOPPADDING", (0, 1), (-1, 1), 0),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ]))
        inner_tables.append(t)
    outer = Table([inner_tables], colWidths=[width / len(cards)] * len(cards))
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def _commentary(text: str) -> Table:
    """Italic purple commentary box with left border, matching the screenshot style."""
    p = Paragraph(_inline(text), STYLES["Commentary"])
    t = Table([[p]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, PURPLE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _callout(text: str) -> Table:
    """Key-finding sidebar callout: purple left rule, light-purple fill, bold copy."""
    p = Paragraph(_inline(text), STYLES["Callout"])
    t = Table([[p]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 2, PURPLE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# --- results table (color-coded numeric cells) -------------------------------

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
    """Build a Table from 2D text rows, coloring signed-numeric cells green/red.
    First row is header. Winner row (if given) is highlighted purple."""
    header = rows[0]
    data_rows = rows[1:]
    col_w = [CONTENT_W * 0.22] + [(CONTENT_W * 0.78) / (len(header) - 1)] * (len(header) - 1)

    body = [[Paragraph(f"<b>{_inline(c)}</b>", STYLES["BodyTight"]) for c in header]]
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LINEABOVE", (0, 0), (-1, 0), 1, PURPLE_LIGHT),
        ("LINEBELOW", (0, 0), (-1, 0), 1, PURPLE_LIGHT),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, GREY_LIGHT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (1, 0), (-1, -1), 8),
    ])

    for r_idx, row in enumerate(data_rows, start=1):
        rendered = []
        is_winner = highlight_row is not None and r_idx == highlight_row
        if not is_winner and r_idx % 2 == 0:
            ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), GREY_BAND)
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
                    ts.add("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), GREEN_LIGHT)
                    ts.add("TEXTCOLOR", (c_idx, r_idx), (c_idx, r_idx), GREEN)
                elif val < 0:
                    ts.add("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), RED_LIGHT)
                    ts.add("TEXTCOLOR", (c_idx, r_idx), (c_idx, r_idx), RED)
        body.append(rendered)
        if is_winner:
            ts.add("BACKGROUND", (0, r_idx), (-1, r_idx), PURPLE)
            ts.add("TEXTCOLOR", (0, r_idx), (-1, r_idx), colors.white)

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
    break. Bullet lists still use a real ListFlowable."""
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
                f"<b><font color='#5B2C83'>{num}.</font></b>&nbsp;&nbsp;{_inline(raw)}",
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
        leftIndent=16, bulletFontSize=9, bulletColor=PURPLE,
    )
    return [flow], j


def _maybe_images(line: str) -> list[Path]:
    paths = re.findall(r"`(outputs/figures/[^`]+\.png)`", line)
    return [ROOT / p for p in paths if (ROOT / p).exists()]


# --- top-level render with section-aware styling -----------------------------

# Map H2 section titles → banner styling behaviour.
# Sections we want to suppress entirely from the main body because we render
# them in a custom way (cover + KPI cards). The H1 of the writeup is always
# replaced by the purple cover banner, never emitted inline.
SKIP_H2 = {"1. problem and scope"}  # folded into cover subtitle


def render(md: str) -> list:
    flow: list = []
    lines = md.splitlines()
    i = 0
    para_buf: list[str] = []
    current_h2: str | None = None
    fig_counter = [0]
    after_banner = [False]

    # --- cover ---------------------------------------------------------------
    today = dt.date.today().isoformat()
    flow.append(HeaderBanner(
        "Earnings-Call NLP Pipeline",
        "NLP for Finance — Spring 2026, Assignment 1\n"
        "Tomer Gross  ·  Generated " + today,
        CONTENT_W,
    ))
    flow.append(Spacer(1, SECTION_GAP))
    flow.append(_kpi_row(
        [
            ("131", "TRANSCRIPTS"),
            ("14", "TICKERS"),
            ("8", "SIGNALS TESTED"),
            ("+0.58", "BEST SHARPE (5D CONTRARIAN SETFIT)"),
        ],
        CONTENT_W,
    ))
    flow.append(Spacer(1, SECTION_GAP))
    flow.append(_commentary(
        "Pipeline parses S&P Capital IQ transcripts, extracts sentiment / wins / risks / guidance "
        "with gemma3:4b (4-call hybrid), layers FinBERT + Loughran-McDonald lexicon + pre-call momentum, "
        "and predicts forward excess return over SPY on a strict 70/30 per-ticker temporal split. "
        "Eight signals compared end-to-end across four horizons (1d / 5d / 21d / 63d); the 5-day "
        "Contrarian SetFit signal is the headline finding — positive hit rate (0.59), positive rank IC "
        "(+0.04), and the only positive Sharpe (+0.58) in the suite."
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
            flow.append(Spacer(1, SECTION_GAP // 2))
            after_banner[0] = False
            return
        img_paths = _maybe_images(text)
        # Narrative intro right after an orange banner gets H2Sub styling
        style = STYLES["H2Sub"] if after_banner[0] else STYLES["Body"]
        after_banner[0] = False
        flow.append(Paragraph(_inline(text), style))
        for img_path in img_paths:
            try:
                fig_counter[0] += 1
                caption = img_path.stem.replace("_", " ").title()
                img = Image(
                    str(img_path), width=CONTENT_W * 0.6,
                    height=CONTENT_W * 0.6 * 0.45, kind="proportional",
                )
                cap = Paragraph(
                    f"Figure {fig_counter[0]}: {caption}", STYLES["FigCaption"],
                )
                flow.append(KeepTogether([Spacer(1, 4), img, cap]))
                flow.append(Spacer(1, SECTION_GAP // 2))
            except Exception:
                pass

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # H1 — already rendered as cover banner; skip
        if stripped.startswith("# ") and not stripped.startswith("## "):
            i += 1
            continue

        # H2 — orange section banner
        if stripped.startswith("## "):
            flush_para()
            title = stripped[3:].strip()
            if title.lower() in SKIP_H2:
                # skip this section entirely: consume until next H2
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("## "):
                    i += 1
                continue
            current_h2 = title.lower()
            flow.append(Spacer(1, SECTION_GAP))
            flow.append(SectionBanner(title, CONTENT_W))
            flow.append(Spacer(1, SECTION_GAP // 2))
            after_banner[0] = True
            i += 1
            continue

        # H3 — purple subhead (used heavily for ticker stories)
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
            # color-code only in the Results section; detect winner row if any
            # cell contains the **bold** markdown marker in its row.
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
            flow.append(Spacer(1, SECTION_GAP))
            continue

        if re.match(r"^\s*(?:\d+\.|[-*])\s+", line):
            flush_para()
            items, i = _parse_list(lines, i)
            flow.extend(items)
            flow.append(Spacer(1, SECTION_GAP // 3))
            continue

        if stripped == "---":
            flush_para()
            flow.append(Spacer(1, SECTION_GAP // 2))
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
    canvas.setStrokeColor(GREY_LIGHT)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 0.45 * inch, PAGE_W - MARGIN, 0.45 * inch)
    canvas.setFillColor(GREY_MED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        MARGIN, 0.3 * inch,
        "Earnings-Call NLP Pipeline  ·  Tomer Gross  ·  Generated " + dt.date.today().isoformat(),
    )
    canvas.drawRightString(
        PAGE_W - MARGIN, 0.3 * inch, f"Page {doc.page}",
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
        topMargin=MARGIN, bottomMargin=0.65 * inch,
        title="Earnings-Call NLP Pipeline",
        author="Tomer Gross",
    )
    doc.build(render(md), onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
