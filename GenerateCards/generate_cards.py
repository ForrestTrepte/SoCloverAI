#!/usr/bin/env python3
"""Generate print-and-play keyword cards for So Clover!

Each word set is [top, right, bottom, left].
Call generate_pdf(word_sets, output_path) to produce a PDF.
"""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black

CLOVER_GREEN = HexColor('#5aac38')

# ── Card geometry ────────────────────────────────────────────────────────────

CARD_SIZE     = 49 * mm
CUTOUT_SIZE   = 18 * mm
CARD_RADIUS   = 2.5 * mm
CUTOUT_RADIUS = 2.5 * mm

# Text: 4.5mm cap height.  For Helvetica, cap height ≈ 0.72 × font size.
FONT_SIZE = (3.25 * mm) / 0.72

# Center of each text zone, measured inward from that card edge.
TEXT_ZONE_CENTER = (CARD_SIZE - CUTOUT_SIZE) / 2 * 0.33

# How far the white arc dips inward from each card edge toward the center.
ARC_DEPTH = 13 * mm

# ── Sheet layout ─────────────────────────────────────────────────────────────

CARDS_PER_ROW  = 4
CARDS_PER_COL  = 5
CARDS_PER_PAGE = CARDS_PER_ROW * CARDS_PER_COL

PAGE_W, PAGE_H = LETTER
MARGIN_X = (PAGE_W - CARDS_PER_ROW * CARD_SIZE) / 2
MARGIN_Y = (PAGE_H - CARDS_PER_COL * CARD_SIZE) / 2

# ── Crop marks ───────────────────────────────────────────────────────────────

CROP_GAP    = 1.5 * mm
CROP_LENGTH = 4   * mm

# ── Helpers ──────────────────────────────────────────────────────────────────

KAPPA = 0.5523  # bezier approximation of quarter-circle arc


def _rounded_rect_path(c, x, y, w, h, r):
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


# ── Drawing ──────────────────────────────────────────────────────────────────

def _draw_white_arc(c, card_x, card_y, edge):
    """
    Fill white from one card edge inward to a concave arc.
    The arc runs corner-to-corner, bowing toward the card center.
    """
    r  = CARD_RADIUS
    cs = CARD_SIZE
    d  = ARC_DEPTH
    k  = KAPPA

    c.setFillColor(white)
    p = c.beginPath()

    if edge == 'top':
        # Outer: follow card boundary across top (left corner → top → right corner)
        p.moveTo(card_x,          card_y + cs - r)
        p.curveTo(card_x,         card_y + cs - r*(1-k),
                  card_x + r*(1-k), card_y + cs,
                  card_x + r,     card_y + cs)
        p.lineTo(card_x + cs - r, card_y + cs)
        p.curveTo(card_x + cs - r*(1-k), card_y + cs,
                  card_x + cs,    card_y + cs - r*(1-k),
                  card_x + cs,    card_y + cs - r)
        # Inner: concave arc dipping down toward center
        p.curveTo(card_x + cs * 0.7, card_y + cs - d,
                  card_x + cs * 0.3, card_y + cs - d,
                  card_x,            card_y + cs - r)

    elif edge == 'bottom':
        p.moveTo(card_x,          card_y + r)
        p.curveTo(card_x,         card_y + r*(1-k),
                  card_x + r*(1-k), card_y,
                  card_x + r,     card_y)
        p.lineTo(card_x + cs - r, card_y)
        p.curveTo(card_x + cs - r*(1-k), card_y,
                  card_x + cs,    card_y + r*(1-k),
                  card_x + cs,    card_y + r)
        # Inner: concave arc dipping up toward center
        p.curveTo(card_x + cs * 0.7, card_y + d,
                  card_x + cs * 0.3, card_y + d,
                  card_x,            card_y + r)

    elif edge == 'left':
        p.moveTo(card_x + r,      card_y)
        p.curveTo(card_x + r*(1-k), card_y,
                  card_x,         card_y + r*(1-k),
                  card_x,         card_y + r)
        p.lineTo(card_x,          card_y + cs - r)
        p.curveTo(card_x,         card_y + cs - r*(1-k),
                  card_x + r*(1-k), card_y + cs,
                  card_x + r,     card_y + cs)
        # Inner: concave arc dipping right toward center
        p.curveTo(card_x + d,     card_y + cs * 0.7,
                  card_x + d,     card_y + cs * 0.3,
                  card_x + r,     card_y)

    elif edge == 'right':
        p.moveTo(card_x + cs - r, card_y)
        p.curveTo(card_x + cs - r*(1-k), card_y,
                  card_x + cs,    card_y + r*(1-k),
                  card_x + cs,    card_y + r)
        p.lineTo(card_x + cs,     card_y + cs - r)
        p.curveTo(card_x + cs,    card_y + cs - r*(1-k),
                  card_x + cs - r*(1-k), card_y + cs,
                  card_x + cs - r, card_y + cs)
        # Inner: concave arc dipping left toward center
        p.curveTo(card_x + cs - d, card_y + cs * 0.7,
                  card_x + cs - d, card_y + cs * 0.3,
                  card_x + cs - r, card_y)

    p.close()
    c.drawPath(p, fill=1, stroke=0)


def draw_card(c, card_x, card_y, words):
    """
    Draw one keyword card with bottom-left corner at (card_x, card_y).
    words = [top, right, bottom, left]
    """
    top, right, bottom, left = words

    # Card background
    c.setFillColor(CLOVER_GREEN)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=1, stroke=0)

    # White arc areas over each edge
    for edge in ('top', 'bottom', 'left', 'right'):
        _draw_white_arc(c, card_x, card_y, edge)

    # Card outline
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    p = _rounded_rect_path(c, card_x, card_y, CARD_SIZE, CARD_SIZE, CARD_RADIUS)
    c.drawPath(p, fill=0, stroke=1)

    # Center cutout – dashed
    cutout_x = card_x + (CARD_SIZE - CUTOUT_SIZE) / 2
    cutout_y = card_y + (CARD_SIZE - CUTOUT_SIZE) / 2
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.setDash([2 * mm, 1.5 * mm])
    p = _rounded_rect_path(c, cutout_x, cutout_y, CUTOUT_SIZE, CUTOUT_SIZE, CUTOUT_RADIUS)
    c.drawPath(p, fill=0, stroke=1)
    c.setDash([])

    # Keywords
    font = "Helvetica-Bold"
    c.setFont(font, FONT_SIZE)
    c.setFillColor(black)

    cx = card_x + CARD_SIZE / 2
    cy = card_y + CARD_SIZE / 2
    baseline = TEXT_ZONE_CENTER - (FONT_SIZE * 0.72) / 2

    def _draw_word(word, tx, ty, angle):
        c.saveState()
        c.translate(tx, ty)
        c.rotate(angle)
        w = c.stringWidth(word, font, FONT_SIZE)
        c.drawString(-w / 2, -baseline, word)
        c.restoreState()

    _draw_word(top,    cx,                            card_y + CARD_SIZE - baseline,  0)
    _draw_word(bottom, cx,                            card_y + baseline,            180)
    _draw_word(left,   card_x + baseline,             cy,                            90)
    _draw_word(right,  card_x + CARD_SIZE - baseline, cy,                           -90)


def _draw_crop_marks(c, card_x, card_y):
    c.setStrokeColor(black)
    c.setLineWidth(0.25)
    for cx, cy in (
        (card_x,             card_y),
        (card_x + CARD_SIZE, card_y),
        (card_x,             card_y + CARD_SIZE),
        (card_x + CARD_SIZE, card_y + CARD_SIZE),
    ):
        sign_x = -1 if cx == card_x else 1
        sign_y = -1 if cy == card_y else 1
        c.line(cx + sign_x * CROP_GAP, cy,
               cx + sign_x * (CROP_GAP + CROP_LENGTH), cy)
        c.line(cx, cy + sign_y * CROP_GAP,
               cx, cy + sign_y * (CROP_GAP + CROP_LENGTH))


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
    import pathlib

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
        ["Whiskey", "X-ray",    "Yankee",   "Zulu"],
        ["Alpha",   "Charlie",  "Echo",     "Golf"],
        ["India",   "Kilo",     "Mike",     "Oscar"],
        ["Quebec",  "Uniform",  "Yankee",   "Bravo"],
        ["Delta",   "Foxtrot",  "Hotel",    "Juliet"],
        ["Lima",    "November", "Papa",     "Romeo"],
        ["Sierra",  "Victor",   "X-ray",    "Zulu"],
        ["Tango",   "Whiskey",  "Alpha",    "Echo"],
    ]

    out = pathlib.Path(__file__).parent / "output" / "cards.pdf"
    out.parent.mkdir(exist_ok=True)
    generate_pdf(placeholder, out)
    print(f"Generated {out}")
