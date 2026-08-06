#!/usr/bin/env python
"""Render the annotation guidelines to PDF, with no system dependencies.

The guidelines are cited in the paper and released with the corpus, so their build must
work on whatever machine happens to run it. Pandoc, LaTeX and a headless browser are all
absent from a plain CI container and from most of the machines this repository is checked
out on; ReportLab is pure Python and comes with the ``human`` extra, so the PDF is
reproducible from a lockfile rather than from an environment.

The Markdown subset understood here is exactly the subset the guidelines use: ATX headings,
paragraphs, fenced code blocks, blockquotes, bullet and numbered lists, pipe tables, and
inline bold/italic/code. **Anything else is rendered literally rather than silently
dropped** — a released document that quietly lost a paragraph to an unsupported construct
is worse than one with a stray asterisk in it.

Usage:
    uv run python scripts/06c_export_guidelines.py
    uv run python scripts/06c_export_guidelines.py --source docs/annotation/recruitment.md
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Inline markup, applied in this order. Code first, so an asterisk inside `code` is not
#: read as emphasis.
_INLINE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"`([^`]+)`"), r'<font face="Courier">\1</font>'),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<![*\w])\*([^*]+)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r"\1"),
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_ANCHOR_RE = re.compile(r'<a name="[^"]*"></a>')
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _inline(text: str) -> str:
    """Convert inline Markdown to ReportLab's mini-HTML.

    Args:
        text: One line or cell of Markdown.

    Returns:
        Marked-up text, HTML-escaped first so a literal ``<`` in a narrative survives.
    """
    escaped = html.escape(text, quote=False)
    for pattern, replacement in _INLINE:
        escaped = pattern.sub(replacement, escaped)
    return escaped


def _styles() -> dict[str, Any]:
    """Build the paragraph styles.

    Returns:
        Style name to ``ParagraphStyle``.

    Raises:
        ImportError: If ReportLab is not installed.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=17, leading=21, spaceBefore=18, spaceAfter=8
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=13.5, leading=17, spaceBefore=15, spaceAfter=6
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontSize=11.5, leading=15, spaceBefore=12, spaceAfter=4
        ),
        "h4": ParagraphStyle(
            "H4", parent=base["Heading4"], fontSize=10.5, leading=14, spaceBefore=10, spaceAfter=3
        ),
        "code": ParagraphStyle(
            "Code",
            parent=body,
            fontName="Courier",
            fontSize=7.6,
            leading=9.4,
            leftIndent=10,
            backColor=colors.HexColor("#f4f4f4"),
            borderPadding=4,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            leftIndent=16,
            rightIndent=8,
            textColor=colors.HexColor("#333333"),
            borderColor=colors.HexColor("#cccccc"),
            spaceBefore=4,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=16, bulletIndent=6),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8, leading=10.5, spaceAfter=0),
    }


def _table(rows: list[list[str]], styles: dict[str, Any], width: float) -> Any:
    """Build a flowable table from parsed pipe-table rows.

    Args:
        rows: Header row first, then body rows.
        styles: The style map.
        width: Available frame width.

    Returns:
        A ``Table`` flowable.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    n_columns = max(len(r) for r in rows)
    padded = [r + [""] * (n_columns - len(r)) for r in rows]
    data = [[Paragraph(_inline(cell), styles["cell"]) for cell in row] for row in padded]
    table = Table(data, colWidths=[width / n_columns] * n_columns, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _is_table_separator(line: str) -> bool:
    """Report whether a line is a Markdown table's header separator.

    Args:
        line: The line.

    Returns:
        True for ``|---|---|`` and its alignment variants.
    """
    stripped = line.strip().strip("|")
    cells = [c.strip() for c in stripped.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _split_row(line: str) -> list[str]:
    """Split a pipe-table row into cells.

    Args:
        line: The row.

    Returns:
        The cells, outer pipes stripped.
    """
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_story(  # noqa: PLR0912, PLR0915 - one branch per Markdown construct, in the
    # order they are tried. Splitting it would separate a construct from its parse.
    markdown: str,
    styles: dict[str, Any],
    width: float,
) -> list[Any]:
    """Convert Markdown into a list of ReportLab flowables.

    Args:
        markdown: The document source.
        styles: The style map.
        width: Available frame width, for table sizing.

    Returns:
        The flowables, in document order.
    """
    from reportlab.platypus import HRFlowable, Paragraph, Preformatted, Spacer

    story: list[Any] = []
    lines = _ANCHOR_RE.sub("", markdown).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            story.append(Preformatted("\n".join(block), styles["code"], maxLineLength=110))
            continue

        if "|" in line and i + 1 < len(lines) and _is_table_separator(lines[i + 1]):
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            story.append(_table(rows, styles, width))
            story.append(Spacer(1, 8))
            continue

        if _RULE_RE.match(line):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, spaceAfter=8))
            i += 1
            continue

        if heading := _HEADING_RE.match(line):
            level = min(len(heading.group(1)), 4)
            story.append(Paragraph(_inline(heading.group(2)), styles[f"h{level}"]))
            i += 1
            continue

        if line.lstrip().startswith(">"):
            quoted: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quoted.append(lines[i].lstrip()[1:].strip())
                i += 1
            for chunk in "\n".join(quoted).split("\n\n"):
                if chunk.strip():
                    story.append(Paragraph(_inline(chunk.replace("\n", " ")), styles["quote"]))
            continue

        if bullet := _BULLET_RE.match(line):
            story.append(Paragraph(_inline(bullet.group(1)), styles["bullet"], bulletText="•"))
            i += 1
            continue

        if numbered := _NUMBERED_RE.match(line):
            story.append(Paragraph(_inline(numbered.group(1)), styles["bullet"], bulletText="-"))
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_special(lines[i]):
            paragraph.append(lines[i])
            i += 1
        story.append(Paragraph(_inline(" ".join(p.strip() for p in paragraph)), styles["body"]))

    return story


def _is_special(line: str) -> bool:
    """Report whether a line starts a construct that ends a paragraph.

    Args:
        line: The line.

    Returns:
        True for a heading, list item, quote, rule, fence or table row.
    """
    return bool(
        _HEADING_RE.match(line)
        or _BULLET_RE.match(line)
        or _NUMBERED_RE.match(line)
        or _RULE_RE.match(line)
        or line.lstrip().startswith((">", "```"))
        or ("|" in line and line.strip().startswith("|"))
    )


def export(source: Path, destination: Path) -> Path:
    """Render one Markdown document to PDF.

    Args:
        source: The Markdown file.
        destination: The PDF to write.

    Returns:
        The written path.

    Raises:
        FileNotFoundError: If the source is missing.
        ImportError: If ReportLab is not installed.
    """
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the PDF export needs ReportLab. Install the human extra: "
            "`uv sync --group dev --extra human`"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=source.stem.replace("_", " ").title(),
        author="Graph2Text AML",
    )
    styles = _styles()
    story = build_story(source.read_text(encoding="utf-8"), styles, document.width)
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return destination


def _page_number(canvas: Any, document: Any) -> None:
    """Draw the page number in the footer.

    Args:
        canvas: The ReportLab canvas.
        document: The document being built.
    """
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillGray(0.45)
    canvas.drawCentredString(document.pagesize[0] / 2, 10 * 72 / 25.4, str(canvas.getPageNumber()))
    canvas.restoreState()


def main(argv: list[str] | None = None) -> int:
    """Export the guidelines, and anything else named, to PDF.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success, 1 when a source is missing or ReportLab is absent.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Markdown file to render; repeatable. Defaults to the guidelines and the "
        "recruitment brief.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Destination directory. Defaults to the source's own directory.",
    )
    args = parser.parse_args(argv)

    sources = (
        [Path(s) for s in args.source]
        if args.source
        else [
            REPO_ROOT / "docs" / "annotation" / "annotation_guidelines.md",
            REPO_ROOT / "docs" / "annotation" / "recruitment.md",
        ]
    )

    for source in sources:
        out_dir = Path(args.out_dir) if args.out_dir else source.parent
        try:
            written = export(source, out_dir / f"{source.stem}.pdf")
        except (FileNotFoundError, ImportError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {written} ({written.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
