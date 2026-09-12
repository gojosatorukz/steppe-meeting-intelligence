"""Report export: .csv, .json, .pdf (task requirement 5)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# Default ReportLab fonts have no Cyrillic. The demo audio will almost certainly
# be Russian or Kazakh, so a missing font here means an unreadable report on stage.
_FONT, _FONT_BOLD = "Helvetica", "Helvetica-Bold"
for regular, bold in [
    (Path(base) / "DejaVuSans.ttf", Path(base) / "DejaVuSans-Bold.ttf")
    for base in ("/usr/share/fonts/TTF", "/usr/share/fonts/truetype/dejavu")
] + [(Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
      Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"))]:
    if regular.exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(regular)))
        _FONT = "DejaVu"
        if bold.exists():
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold)))
            _FONT_BOLD = "DejaVu-Bold"
        else:
            _FONT_BOLD = "DejaVu"
        break


def _esc(value: Any) -> str:
    """Escape for ReportLab's mini-markup.

    `Paragraph` parses its text as XML-ish markup, so a bare ``&`` or ``<`` in a
    transcript — "R&D", "<inaudible>" — raises mid-build and takes the whole
    export down. Nothing we put in a Paragraph comes from a trusted source.
    """
    return (str(value if value is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _hms(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return ""
    return f"[{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}]"


def to_json(result: dict[str, Any]) -> bytes:
    """The full result — transcript, segments, `speakers` blocks and protocol."""
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")


def to_csv(result: dict[str, Any]) -> bytes:
    """Action items — the part anyone actually pastes into a tracker."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Ответственный", "Задача", "Срок", "Приоритет"])
    for item in result.get("protocol", {}).get("action_items", []):
        w.writerow([
            item.get("owner", ""), item.get("task", ""),
            item.get("due", ""), item.get("priority", ""),
        ])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel reads Cyrillic


def to_pdf(result: dict[str, Any], title: str = "Протокол встречи") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=title,
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=_FONT_BOLD, fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=_FONT_BOLD, fontSize=12,
                        spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1f2937"))
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName=_FONT, fontSize=9.5,
                          leading=14, alignment=TA_LEFT)

    p = result.get("protocol", {})
    story: list[Any] = [Paragraph(_esc(title), h1)]

    timing = result.get("timing", {})
    meta = (f"Язык: {_esc(result.get('language', '—'))} · "
            f"Длительность: {_esc(result.get('audio_seconds', 0))} с · "
            f"Обработка: {_esc(timing.get('total_sec', 0))} с")
    if result.get("speakers"):
        meta += f" · Спикеров: {len({b.get('speaker') for b in result['speakers']})}"
    story += [Paragraph(meta, body), Spacer(1, 6)]

    def section(heading: str, items: list[str]) -> None:
        if not items:
            return
        story.append(Paragraph(heading, h2))
        for it in items:
            story.append(Paragraph(f"• {_esc(it)}", body))

    section("Executive Summary", p.get("summary", []))
    section("Принятые решения", p.get("decisions", []))

    topics = p.get("topics", [])
    if topics:
        story.append(Paragraph("Темы и тезисы", h2))
        for t in topics:
            story.append(Paragraph(f"<b>{_esc(t.get('title', ''))}</b>", body))
            for pt in t.get("points", []):
                story.append(Paragraph(f"&nbsp;&nbsp;— {_esc(pt)}", body))

    section("Открытые вопросы", p.get("open_questions", []))
    section("Риски и блокеры", p.get("risks", []))

    actions = p.get("action_items", [])
    if actions:
        story.append(Paragraph("Поручения", h2))
        rows = [["Ответственный", "Задача", "Срок", "Приоритет"]]
        rows += [[
            Paragraph(_esc(a.get("owner", "")), body), Paragraph(_esc(a.get("task", "")), body),
            Paragraph(_esc(a.get("due", "")), body), Paragraph(_esc(a.get("priority", "")), body),
        ] for a in actions]
        table = Table(rows, colWidths=[32 * mm, 86 * mm, 26 * mm, 24 * mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    # Speaker-labelled transcript when diarization produced one. A wall of text
    # is technically the same information, but "Speaker 1: …" is what makes a
    # protocol readable — and what shows the diarizer did its job.
    blocks = result.get("speakers") or []
    if blocks:
        story += [PageBreak(), Paragraph("Стенограмма по спикерам", h2)]
        for b in blocks:
            speaker = _esc(b.get("speaker", ""))
            text = _esc(b.get("text", ""))
            stamp = _hms(b.get("start"))
            story.append(Paragraph(
                f"<font color='#6b7280'>{stamp}</font> <b>{speaker}:</b> {text}", body
            ))
    elif result.get("transcript"):
        story += [PageBreak(), Paragraph("Стенограмма", h2),
                  Paragraph(_esc(result["transcript"]), body)]

    doc.build(story)
    return buf.getvalue()
