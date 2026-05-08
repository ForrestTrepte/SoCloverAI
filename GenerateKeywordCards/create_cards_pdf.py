"""
Generate a printable PDF of So Clover! keyword cards from cards.csv.

Reads cards.csv (produced by assign_cards.py) and renders each row as a
keyword card. Optionally appends blank cards for players to write their own words.

Usage:
    cd GenerateKeywordCards
    uv run create_cards_pdf.py

    # Add 4 blank cards at the end:
    uv run create_cards_pdf.py --blank-cards 4
"""

import argparse
import csv
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor, black, white  # type: ignore
from reportlab.lib.pagesizes import LETTER  # type: ignore
from reportlab.lib.units import mm  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore

SCRIPT_DIR = Path(__file__).parent
CARDS_CSV = SCRIPT_DIR / "cards.csv"
OUTPUT_PDF = SCRIPT_DIR / "cards.pdf"

CLOVER_GREEN = HexColor("#5aac38")

# ── Card geometry ─────────────────────────────────────────────────────────────

CARD_SIZE = 49 * mm
CUTOUT_SIZE = 18 * mm
CARD_RADIUS = 2.5 * mm
CUTOUT_RADIUS = 2.5 * mm

FONT_SIZE = (3.25 * mm) / 0.72  # 4.5mm cap height; Helvetica cap ≈ 0.72 × font size

TEXT_ZONE_CENTER = (CARD_SIZE - CUTOUT_SIZE) / 2 * 0.33
ARC_DEPTH = 13 * mm

# ── Sheet layout ──────────────────────────────────────────────────────────────

CARDS_PER_ROW = 4
CARDS_PER_COL = 5
CARDS_PER_PAGE = CARDS_PER_ROW * CARDS_PER_COL

PAGE_W, PAGE_H = LETTER
MARGIN_X = (PAGE_W - CARDS_PER_ROW * CARD_SIZE) / 2
MARGIN_Y = (PAGE_H - CARDS_PER_COL * CARD_SIZE) / 2

# ── Crop marks ────────────────────────────────────────────────────────────────

CROP_GAP = 1.5 * mm
CROP_LENGTH = 4 * mm

KAPPA = 0.5523  # bezier approximation of quarter-circle arc


# ── Drawing helpers ───────────────────────────────────────────────────────────


def _rounded_rect_path(c: Any, x: float, y: float, w: float, h: float, r: float) -> Any:
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.curveTo(x + w - r + r * KAPPA, y, x + w, y + r - r * KAPPA, x + w, y + r)
    p.lineTo(x + w, y + h - r)
    p.curveTo(
        x + w, y + h - r + r * KAPPA, x + w - r + r * KAPPA, y + h, x + w - r, y + h
    )
    p.lineTo(x + r, y + h)
    p.curveTo(x + r - r * KAPPA, y + h, x, y + h - r + r * KAPPA, x, y + h - r)
    p.lineTo(x, y + r)
    p.curveTo(x, y + r - r * KAPPA, x + r - r * KAPPA, y, x + r, y)
    p.close()
    return p


def _draw_white_arc(c: Any, card_x: float, card_y: float, edge: str) -> None:
    """Fill white from one card edge inward to a concave arc."""
    r = CARD_RADIUS
    cs = CARD_SIZE
    d = ARC_DEPTH
    k = KAPPA

    c.setFillColor(white)
    p = c.beginPath()

    if edge == "top":
        p.moveTo(card_x, card_y + cs - r)
        p.curveTo(
            card_x,
            card_y + cs - r * (1 - k),
            card_x + r * (1 - k),
            card_y + cs,
            card_x + r,
            card_y + cs,
        )
        p.lineTo(card_x + cs - r, card_y + cs)
        p.curveTo(
            card_x + cs - r * (1 - k),
            card_y + cs,
            card_x + cs,
            card_y + cs - r * (1 - k),
            card_x + cs,
            card_y + cs - r,
        )
        p.curveTo(
            card_x + cs * 0.7,
            card_y + cs - d,
            card_x + cs * 0.3,
            card_y + cs - d,
            card_x,
            card_y + cs - r,
        )

    elif edge == "bottom":
        p.moveTo(card_x, card_y + r)
        p.curveTo(
            card_x,
            card_y + r * (1 - k),
            card_x + r * (1 - k),
            card_y,
            card_x + r,
            card_y,
        )
        p.lineTo(card_x + cs - r, card_y)
        p.curveTo(
            card_x + cs - r * (1 - k),
            card_y,
            card_x + cs,
            card_y + r * (1 - k),
            card_x + cs,
            card_y + r,
        )
        p.curveTo(
            card_x + cs * 0.7,
            card_y + d,
            card_x + cs * 0.3,
            card_y + d,
            card_x,
            card_y + r,
        )

    elif edge == "left":
        p.moveTo(card_x + r, card_y)
        p.curveTo(
            card_x + r * (1 - k),
            card_y,
            card_x,
            card_y + r * (1 - k),
            card_x,
            card_y + r,
        )
        p.lineTo(card_x, card_y + cs - r)
        p.curveTo(
            card_x,
            card_y + cs - r * (1 - k),
            card_x + r * (1 - k),
            card_y + cs,
            card_x + r,
            card_y + cs,
        )
        p.curveTo(
            card_x + d,
            card_y + cs * 0.7,
            card_x + d,
            card_y + cs * 0.3,
            card_x + r,
            card_y,
        )

    elif edge == "right":
        p.moveTo(card_x + cs - r, card_y)
        p.curveTo(
            card_x + cs - r * (1 - k),
            card_y,
            card_x + cs,
            card_y + r * (1 - k),
            card_x + cs,
            card_y + r,
        )
        p.lineTo(card_x + cs, card_y + cs - r)
        p.curveTo(
            card_x + cs,
            card_y + cs - r * (1 - k),
            card_x + cs - r * (1 - k),
            card_y + cs,
            card_x + cs - r,
            card_y + cs,
        )
        p.curveTo(
            card_x + cs - d,
            card_y + cs * 0.7,
            card_x + cs - d,
            card_y + cs * 0.3,
            card_x + cs - r,
            card_y,
        )

    p.close()
    c.drawPath(p, fill=1, stroke=0)


def _display_word(word: str) -> str:
    """Uppercase ordinary words; preserve mixed-case brands (ExxonMobil, iPhone, WiFi)."""
    if word == word.lower():
        return word.capitalize()
    return word


def draw_card(c: Any, card_x: float, card_y: float, words: list[str]) -> None:
    """
    Draw one keyword card with bottom-left corner at (card_x, card_y).
    words = [top, right, bottom, left]; pass empty strings for a blank card.
    """
    top, right, bottom, left = [_display_word(w) for w in words]

    c.setFillColor(CLOVER_GREEN)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=1, stroke=0)

    for edge in ("top", "bottom", "left", "right"):
        _draw_white_arc(c, card_x, card_y, edge)

    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=0, stroke=1)

    cutout_x = card_x + (CARD_SIZE - CUTOUT_SIZE) / 2
    cutout_y = card_y + (CARD_SIZE - CUTOUT_SIZE) / 2
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.setDash([2 * mm, 1.5 * mm])
    p = _rounded_rect_path(
        c, cutout_x, cutout_y, CUTOUT_SIZE, CUTOUT_SIZE, CUTOUT_RADIUS
    )
    c.drawPath(p, fill=0, stroke=1)
    c.setDash([])

    font = "Helvetica-Bold"
    c.setFont(font, FONT_SIZE)
    c.setFillColor(black)

    cx = card_x + CARD_SIZE / 2
    cy = card_y + CARD_SIZE / 2
    baseline = TEXT_ZONE_CENTER - (FONT_SIZE * 0.72) / 2

    def _draw_word(word: str, tx: float, ty: float, angle: float) -> None:
        if not word:
            return
        c.saveState()
        c.translate(tx, ty)
        c.rotate(angle)
        w = c.stringWidth(word, font, FONT_SIZE)
        c.drawString(-w / 2, -baseline, word)
        c.restoreState()

    _draw_word(top, cx, card_y + CARD_SIZE - baseline, 0)
    _draw_word(bottom, cx, card_y + baseline, 180)
    _draw_word(left, card_x + baseline, cy, 90)
    _draw_word(right, card_x + CARD_SIZE - baseline, cy, -90)


def _draw_crop_marks(c: Any, card_x: float, card_y: float) -> None:
    c.setStrokeColor(black)
    c.setLineWidth(0.25)
    for cx, cy in (
        (card_x, card_y),
        (card_x + CARD_SIZE, card_y),
        (card_x, card_y + CARD_SIZE),
        (card_x + CARD_SIZE, card_y + CARD_SIZE),
    ):
        sign_x = -1 if cx == card_x else 1
        sign_y = -1 if cy == card_y else 1
        c.line(cx + sign_x * CROP_GAP, cy, cx + sign_x * (CROP_GAP + CROP_LENGTH), cy)
        c.line(cx, cy + sign_y * CROP_GAP, cx, cy + sign_y * (CROP_GAP + CROP_LENGTH))


def generate_pdf(word_sets: list[list[str]], output_path: Path) -> None:
    """Render word_sets as keyword cards to a PDF file."""
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    for i, words in enumerate(word_sets):
        pos = i % CARDS_PER_PAGE
        if pos == 0 and i > 0:
            c.showPage()
        row = pos // CARDS_PER_ROW
        col = pos % CARDS_PER_ROW
        card_x = MARGIN_X + col * CARD_SIZE
        card_y = PAGE_H - MARGIN_Y - (row + 1) * CARD_SIZE
        draw_card(c, card_x, card_y, words)
        _draw_crop_marks(c, card_x, card_y)
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate So Clover! keyword cards PDF"
    )
    parser.add_argument(
        "--blank-cards",
        type=int,
        default=0,
        metavar="N",
        help="Append N blank cards at the end (default: 0)",
    )
    args = parser.parse_args()

    if not CARDS_CSV.exists():
        print(f"Error: {CARDS_CSV} not found. Run assign_cards.py first.")
        raise SystemExit(1)

    word_sets: list[list[str]] = []
    with open(CARDS_CSV) as f:
        for row in csv.DictReader(f):
            word_sets.append([row["word1"], row["word2"], row["word3"], row["word4"]])

    if args.blank_cards > 0:
        word_sets += [["", "", "", ""]] * args.blank_cards
        print(f"Appending {args.blank_cards} blank card(s).")

    generate_pdf(word_sets, OUTPUT_PDF)
    print(
        f"Wrote {len(word_sets)} cards ({len(word_sets) - args.blank_cards} keyword"
        f" + {args.blank_cards} blank) to {OUTPUT_PDF}"
    )


if __name__ == "__main__":
    main()
