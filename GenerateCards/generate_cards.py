#!/usr/bin/env python3
"""Generate print-and-play keyword cards for So Clover!

Each word set is [top, right, bottom, left].
Call generate_pdf(word_sets, output_path) to produce a PDF.
"""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black

# ── Card geometry ────────────────────────────────────────────────────────────

CARD_SIZE    = 49 * mm
CUTOUT_SIZE  = 18 * mm
CARD_RADIUS  = 2 * mm   # outer card corner radius
CUTOUT_RADIUS = 2 * mm  # inner cutout corner radius

# Text: 4.5mm cap height.  For Helvetica, cap height ≈ 0.72 × font size.
FONT_SIZE = (4.5 * mm) / 0.72   # ≈ 17.7 pt

# Vertical center of each text zone (distance from that card edge inward)
TEXT_ZONE_CENTER = (CARD_SIZE - CUTOUT_SIZE) / 4   # ≈ 7.75 mm

# ── Green petal geometry ─────────────────────────────────────────────────────

CLOVER_GREEN = HexColor('#5aac38')   # approximate match to the card photo

# How far along each edge (from the corner) the green petal extends.
# Must be ≤ (CARD_SIZE - CUTOUT_SIZE)/2 ≈ 15.5 mm so petals don't overlap text.
PETAL_SPREAD = 14 * mm

# How strongly the inner curve bows toward the card corner (0=diagonal, 1=sharp V).
PETAL_BOW = 0.78

# ── Sheet layout ─────────────────────────────────────────────────────────────

CARDS_PER_ROW = 3
CARDS_PER_COL = 4
CARDS_PER_PAGE = CARDS_PER_ROW * CARDS_PER_COL

PAGE_W, PAGE_H = LETTER
MARGIN_X = (PAGE_W - CARDS_PER_ROW * CARD_SIZE) / 2
MARGIN_Y = (PAGE_H - CARDS_PER_COL * CARD_SIZE) / 2

# ── Crop marks ───────────────────────────────────────────────────────────────

CROP_GAP    = 1.5 * mm   # space between card edge and start of mark
CROP_LENGTH = 4   * mm


# ── Helpers ──────────────────────────────────────────────────────────────────

KAPPA = 0.5523  # bezier approximation of quarter-circle arc


def _rounded_rect_path(c, x, y, w, h, r):
    """Return a closed rounded-rectangle path object."""
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.curveTo(x + w - r + r * KAPPA, y,
              x + w, y + r - r * KAPPA,
              x + w, y + r)
    p.lineTo(x + w, y + h - r)
    p.curveTo(x + w, y + h - r + r * KAPPA,
              x + w - r + r * KAPPA, y + h,
              x + w - r, y + h)
    p.lineTo(x + r, y + h)
    p.curveTo(x + r - r * KAPPA, y + h,
              x, y + h - r + r * KAPPA,
              x, y + h - r)
    p.lineTo(x, y + r)
    p.curveTo(x, y + r - r * KAPPA,
              x + r - r * KAPPA, y,
              x + r, y)
    p.close()
    return p


def _lerp(a, b, t):
    return a + (b - a) * t


# ── Drawing ──────────────────────────────────────────────────────────────────

def _draw_petal(c, card_x, card_y, corner):
    """
    Fill one green corner petal.

    The petal outline:
      • outer boundary  – follows the card's rounded corner arc
      • inner boundary  – single cubic bezier bowing toward the card corner,
                          giving the characteristic concave clover-leaf look
    """
    s = PETAL_SPREAD
    r = CARD_RADIUS
    bow = PETAL_BOW

    # (cx, cy) = the outer corner of the card for this petal
    if corner == 'tl':
        cx, cy = card_x,            card_y + CARD_SIZE
        # A: point on top  edge;  B: point on left edge
        ax, ay = cx + s, cy
        bx, by = cx,     cy - s
        # arc: top-edge → left-edge (going CCW around top-left corner)
        arc_start_x, arc_start_y = cx + r, cy
        arc_end_x,   arc_end_y   = cx,     cy - r
        acp1x, acp1y = cx + r * (1 - KAPPA), cy
        acp2x, acp2y = cx, cy - r * (1 - KAPPA)

    elif corner == 'tr':
        cx, cy = card_x + CARD_SIZE, card_y + CARD_SIZE
        ax, ay = cx - s, cy
        bx, by = cx,     cy - s
        arc_start_x, arc_start_y = cx - r, cy
        arc_end_x,   arc_end_y   = cx,     cy - r
        acp1x, acp1y = cx - r * (1 - KAPPA), cy
        acp2x, acp2y = cx, cy - r * (1 - KAPPA)

    elif corner == 'bl':
        cx, cy = card_x,            card_y
        ax, ay = cx + s, cy
        bx, by = cx,     cy + s
        arc_start_x, arc_start_y = cx + r, cy
        arc_end_x,   arc_end_y   = cx,     cy + r
        acp1x, acp1y = cx + r * (1 - KAPPA), cy
        acp2x, acp2y = cx, cy + r * (1 - KAPPA)

    else:  # 'br'
        cx, cy = card_x + CARD_SIZE, card_y
        ax, ay = cx - s, cy
        bx, by = cx,     cy + s
        arc_start_x, arc_start_y = cx - r, cy
        arc_end_x,   arc_end_y   = cx,     cy + r
        acp1x, acp1y = cx - r * (1 - KAPPA), cy
        acp2x, acp2y = cx, cy + r * (1 - KAPPA)

    # Control points for the concave inner bezier B → A
    # Bow the curve strongly toward the card corner (cx, cy)
    cp1x = _lerp(bx, cx, bow)
    cp1y = _lerp(by, cy, bow)
    cp2x = _lerp(ax, cx, bow)
    cp2y = _lerp(ay, cy, bow)

    c.setFillColor(CLOVER_GREEN)
    p = c.beginPath()
    p.moveTo(ax, ay)                         # start on first edge
    p.lineTo(arc_start_x, arc_start_y)       # walk to corner arc start
    p.curveTo(acp1x, acp1y,                  # rounded card corner
              acp2x, acp2y,
              arc_end_x, arc_end_y)
    p.lineTo(bx, by)                         # walk to end of second edge
    p.curveTo(cp1x, cp1y, cp2x, cp2y,        # concave inner curve
              ax, ay)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def draw_card(c, card_x, card_y, words):
    """
    Draw one keyword card with bottom-left corner at (card_x, card_y).
    words = [top, right, bottom, left]
    """
    top, right, bottom, left = words

    # 1. White card background
    c.setFillColor(white)
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=1, stroke=0)

    # 2. Green corner petals
    for corner in ('tl', 'tr', 'bl', 'br'):
        _draw_petal(c, card_x, card_y, corner)

    # 3. Card outline (on top so it's crisp)
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=0, stroke=1)

    # 4. Center cutout – dashed
    cutout_x = card_x + (CARD_SIZE - CUTOUT_SIZE) / 2
    cutout_y = card_y + (CARD_SIZE - CUTOUT_SIZE) / 2
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.setDash([2 * mm, 1.5 * mm])
    p = _rounded_rect_path(c, cutout_x, cutout_y, CUTOUT_SIZE, CUTOUT_SIZE, CUTOUT_RADIUS)
    c.drawPath(p, fill=0, stroke=1)
    c.setDash([])

    # 5. Keywords
    font = "Helvetica-Bold"
    c.setFont(font, FONT_SIZE)
    c.setFillColor(black)

    cx = card_x + CARD_SIZE / 2
    cy = card_y + CARD_SIZE / 2

    # baseline sits at zone-center minus half cap-height
    baseline = TEXT_ZONE_CENTER - (FONT_SIZE * 0.72) / 2

    def _draw_word(word, tx, ty, angle):
        c.saveState()
        c.translate(tx, ty)
        c.rotate(angle)
        w = c.stringWidth(word, font, FONT_SIZE)
        c.drawString(-w / 2, -baseline, word)
        c.restoreState()

    _draw_word(top,    cx,                         card_y + CARD_SIZE - baseline,  0)
    _draw_word(bottom, cx,                         card_y + baseline,            180)
    _draw_word(left,   card_x + baseline,          cy,                            90)
    _draw_word(right,  card_x + CARD_SIZE - baseline, cy,                        -90)


def _draw_crop_marks(c, card_x, card_y):
    c.setStrokeColor(black)
    c.setLineWidth(0.25)
    for cx, cy in (
        (card_x,            card_y),
        (card_x + CARD_SIZE, card_y),
        (card_x,            card_y + CARD_SIZE),
        (card_x + CARD_SIZE, card_y + CARD_SIZE),
    ):
        sign_x = -1 if cx == card_x else 1
        sign_y = -1 if cy == card_y else 1
        # horizontal
        c.line(cx + sign_x * CROP_GAP,
               cy,
               cx + sign_x * (CROP_GAP + CROP_LENGTH),
               cy)
        # vertical
        c.line(cx,
               cy + sign_y * CROP_GAP,
               cx,
               cy + sign_y * (CROP_GAP + CROP_LENGTH))


# ── Public API ───────────────────────────────────────────────────────────────

def generate_pdf(word_sets, output_path):
    """
    Generate a PDF sheet of keyword cards.

    Args:
        word_sets: list of [top, right, bottom, left] word lists
        output_path: path for the output .pdf file
    """
    c = canvas.Canvas(str(output_path), pagesize=LETTER)

    for i, words in enumerate(word_sets):
        pos = i % CARDS_PER_PAGE
        if pos == 0 and i > 0:
            c.showPage()

        row = pos // CARDS_PER_ROW
        col = pos  % CARDS_PER_ROW

        card_x = MARGIN_X + col * CARD_SIZE
        card_y = PAGE_H - MARGIN_Y - (row + 1) * CARD_SIZE

        draw_card(c, card_x, card_y, words)
        _draw_crop_marks(c, card_x, card_y)

    c.save()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, pathlib

    # NATO alphabet as placeholder words (12 cards = one full sheet)
    placeholder = [
        ["Alpha",   "Bravo",    "Charlie",  "Delta"],
        ["Echo",    "Foxtrot",  "Golf",     "Hotel"],
        ["India",   "Juliet",   "Kilo",     "Lima"],
        ["Mike",    "November", "Oscar",    "Papa"],
        ["Quebec",  "Romeo",    "Sierra",   "Tango"],
        ["Uniform", "Victor",   "Whiskey",  "X-ray"],
        ["Yankee",  "Zulu",     "Alpha",    "Bravo"],
        ["Charlie", "Delta",    "Echo",     "Foxtrot"],
        ["Golf",    "Hotel",    "India",    "Juliet"],
        ["Kilo",    "Lima",     "Mike",     "November"],
        ["Oscar",   "Papa",     "Quebec",   "Romeo"],
        ["Sierra",  "Tango",    "Uniform",  "Victor"],
    ]

    out = pathlib.Path(__file__).parent / "output" / "cards.pdf"
    out.parent.mkdir(exist_ok=True)
    generate_pdf(placeholder, out)
    print(f"Generated {out}")
