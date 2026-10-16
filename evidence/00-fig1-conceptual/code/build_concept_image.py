"""Conceptual framework figure: causal overclaiming in LLM research.

Layout (12.333 × 5.36 in, 2.30:1): header with title and subtitle at the top,
followed by two stacked bands describing (1) the blind-following pattern of popular
methods and (2) our three-case examination, with arrows showing the
conceptual flow. The takeaway band at the bottom summarizes the paper's main
message.

Design intent: 30-second readability, each labeled component = one claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (robust against any cwd).
# For release: parents[3] = release/patterns-repro/.
# For source-tree copy: parents[3] = lunwen/ (which has scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    import scripts.plot_config  # noqa: F401
except ImportError:
    # Fallback for the paper/concept_view source-tree location (parents[2] = paper/)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.plot_config import setup_style, OKABE_ITO as OI, save_figure

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


# ---- 配色 ----
C_PROBLEM  = OI["vermillion"]
C_CASE1    = OI["blue"]
C_CASE2    = OI["bluish_green"]
C_MECH     = OI["reddish_purple"]
C_TAKEAWAY = OI["orange"]
C_METHODS  = OI["black"]
C_NEUTRAL  = OI["gray"]
C_BG       = "#FFFFFF"
C_BG_TOP   = "#EEF4ED"
C_BG_MID   = "#EAF1F8"
C_BG_LB    = "#FFF8E0"
C_DARK     = "#1A1A1A"
C_MUTED    = "#5A5A5A"
C_WHITE    = "#FFFFFF"

setup_style({
    "font.size": 10,
    "axes.titlesize": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

FIG_W, FIG_H = 12.333, 5.36
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor=C_BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(C_BG)


# ============================================================
#  工具
# ============================================================
def rrect(x, y, w, h, fc, ec="none", lw=0, r=0.05, ls="-", alpha=1.0, z=1):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        linestyle=ls, alpha=alpha, zorder=z,
    )
    ax.add_patch(box)


def text(x, y, s, size=10, color=C_DARK, bold=False, ha="left",
         va="baseline", z=3, style="normal"):
    ax.text(
        x, y, s, fontsize=size, color=color,
        fontweight=("bold" if bold else "normal"),
        fontstyle=style,
        ha=ha, va=va, family="sans-serif", zorder=z,
    )


def arrow(x1, y1, x2, y2, color=C_DARK, lw=1.4, z=2, ms=12):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
        color=color, lw=lw, zorder=z,
    )
    ax.add_patch(arr)


def num_circle(x, y, number, color, z=3):
    ax.add_patch(Circle((x, y), 0.085, facecolor=C_WHITE,
                        edgecolor=color, lw=0.9, zorder=z))
    text(x, y, f"({number})", size=7.5, color=color, bold=True,
         ha="center", va="center", z=z+1)


def numbered_arrow_h(x1, x2, y, number, color):
    arrow(x1, y, x2, y, color=color, lw=1.4, z=2, ms=10)
    num_circle((x1 + x2) / 2, y, number, color, z=3)


def numbered_arrow_v(x, y1, y2, number, color):
    arrow(x, y1, x, y2, color=color, lw=1.4, z=2, ms=10)
    num_circle(x, (y1 + y2) / 2, number, color, z=3)


def place_icon(path, x, y, zoom=0.05, z=4):
    if not Path(path).exists():
        return
    img = mpimg.imread(str(path))
    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(oi, (x, y), frameon=False,
                        xycoords='data', zorder=z, pad=0)
    ax.add_artist(ab)


# ---- 图标路径 ----
ICON_DIR = Path(__file__).resolve().parent / "icons"
ICONS = {
    "search": ICON_DIR / "search_0072B2.png",
    "brain":  ICON_DIR / "brain_009E73.png",
    "target": ICON_DIR / "target-arrow_CC79A7.png",
    "bulb":   ICON_DIR / "bulb_E69F00.png",
    "flask":  ICON_DIR / "flask_0072B2.png",
}


# ============================================================
#  1. Header
# ============================================================
text(FIG_W / 2, 5.22, "Conceptual Framework",
     size=16, color=C_DARK, bold=True, ha="center", va="center")
text(FIG_W / 2, 5.02,
     "Skepticism + Understanding  >  Stacking Patches",
     size=10, color=C_TAKEAWAY, bold=True, ha="center", va="center")


# ============================================================
#  2. Top band (y 4.00–4.95, h=0.95)
# ============================================================
TOP_T, TOP_B = 4.95, 4.00
TOP_H = TOP_T - TOP_B
rrect(0.40, TOP_B, FIG_W - 0.80, TOP_H, C_BG_TOP, r=0.06, z=1)

# Section header (above the input boxes)
text(0.55, 4.85, "1.  The Blind-Following Pattern",
     size=10, color=C_PROBLEM, bold=True, ha="left", va="center")

# 3 input boxes (y 4.10–4.65, h=0.55)
BOX_W, BOX_H, GAP = 3.45, 0.55, 0.30
TOTAL_W = 3 * BOX_W + 2 * GAP
START_X = (FIG_W - TOTAL_W) / 2
BOX_Y = TOP_B + 0.10  # 4.10

# Left: hot strategy (grey)
rrect(START_X, BOX_Y, BOX_W, BOX_H, C_WHITE,
      ec=C_NEUTRAL, lw=1.0, ls="--", r=0.04, z=2)
text(START_X + 0.15, BOX_Y + BOX_H - 0.10,
     "Hot strategy emerges",
     size=10, color=C_NEUTRAL, bold=True, ha="left", va="center")
text(START_X + 0.15, BOX_Y + 0.28,
     'e.g. "Let\'s think step by step."  (Kojima 2022)',
     size=8, color=C_DARK, ha="left", va="center")
text(START_X + 0.15, BOX_Y + 0.10,
     "→ adopted as default, replicated without verification",
     size=7.5, color=C_MUTED, style="italic", ha="left", va="center")

# Center: H1 (blue)
CX1 = START_X + BOX_W + GAP
rrect(CX1, BOX_Y, BOX_W, BOX_H, C_WHITE,
      ec=C_CASE1, lw=1.2, ls="--", r=0.04, z=2)
text(CX1 + 0.15, BOX_Y + BOX_H - 0.10,
     "H1:  CoT is universal",
     size=10, color=C_CASE1, bold=True, ha="left", va="center")
text(CX1 + 0.15, BOX_Y + 0.28,
     "Mainstream view since 2022",
     size=8, color=C_DARK, ha="left", va="center")
text(CX1 + 0.15, BOX_Y + 0.10,
     "→ examined directly by code",
     size=7.5, color=C_MUTED, style="italic", ha="left", va="center")

# Right: H2 (bluegreen)
CX2 = CX1 + BOX_W + GAP
rrect(CX2, BOX_Y, BOX_W, BOX_H, C_WHITE,
      ec=C_CASE2, lw=1.2, ls="--", r=0.04, z=2)
text(CX2 + 0.15, BOX_Y + BOX_H - 0.10,
     "H2:  Reflection = self-correction",
     size=10, color=C_CASE2, bold=True, ha="left", va="center")
text(CX2 + 0.15, BOX_Y + 0.28,
     '"Wait" / "However" = aha-moment',
     size=8, color=C_DARK, ha="left", va="center")
text(CX2 + 0.15, BOX_Y + 0.10,
     "→ examined directly by code",
     size=7.5, color=C_MUTED, style="italic", ha="left", va="center")

# Down arrows from top boxes to middle band (unnumbered)
arrow(START_X + BOX_W / 2, BOX_Y - 0.02,
      START_X + BOX_W / 2, 3.95,
      color=C_NEUTRAL, lw=1.0, z=2, ms=10)
arrow(CX1 + BOX_W / 2, BOX_Y - 0.02,
      CX1 + BOX_W / 2, 3.95,
      color=C_CASE1, lw=1.2, z=2, ms=10)
arrow(CX2 + BOX_W / 2, BOX_Y - 0.02,
      CX2 + BOX_W / 2, 3.95,
      color=C_CASE2, lw=1.2, z=2, ms=10)


# ============================================================
#  3. Middle band (y 0.95–3.95, h=3.00)
# ============================================================
MID_T, MID_B = 3.95, 0.95
rrect(0.40, MID_B, FIG_W - 0.80, MID_T - MID_B, C_BG_MID, r=0.06, z=1)

text(0.55, 3.85, "2.  Our Examination",
     size=10, color=C_DARK, bold=True, ha="left", va="center")

# 3 component boxes (y 1.25–3.70, h=2.45)
CBOX_W = 3.51
CBOX_H = 2.45
CBOX_Y = 1.25
GAP2 = 0.30
START_X2 = 0.60


def draw_component(x, color, label, fig_num, title, hyp_text,
                   method, finding, icon):
    rrect(x, CBOX_Y, CBOX_W, CBOX_H, C_WHITE,
          ec=color, lw=1.4, ls="--", r=0.05, z=2)
    # Icon + label + fig num (top row)
    place_icon(icon, x + 0.22, CBOX_Y + CBOX_H - 0.22, zoom=0.05, z=4)
    text(x + 0.50, CBOX_Y + CBOX_H - 0.16,
         label, size=11, color=color, bold=True,
         ha="left", va="center")
    text(x + CBOX_W - 0.10, CBOX_Y + CBOX_H - 0.16,
         fig_num, size=8, color=C_MUTED, style="italic",
         ha="right", va="center")
    # Title
    text(x + 0.18, CBOX_Y + CBOX_H - 0.55,
         title, size=11, color=C_DARK, bold=True,
         ha="left", va="top")
    # Separator
    ax.plot([x + 0.18, x + CBOX_W - 0.18],
            [CBOX_Y + CBOX_H - 0.88, CBOX_Y + CBOX_H - 0.88],
            color=C_NEUTRAL, lw=0.4, alpha=0.5, zorder=3)
    # Hypothesis
    text(x + 0.18, CBOX_Y + CBOX_H - 1.02,
         hyp_text, size=9, color=color, bold=True,
         ha="left", va="top")
    # Method (positioned above where Lean4 sits in that box)
    text(x + 0.18, CBOX_Y + 1.30,
         method, size=8.5, color=C_DARK, ha="left", va="top")
    # Finding (very bottom)
    text(x + 0.18, CBOX_Y + 0.12,
         finding, size=9, color=C_MUTED, style="italic",
         ha="left", va="center")


# Case 1
draw_component(START_X2 + 0 * (CBOX_W + GAP2), C_CASE1,
               "Case 1", "Fig 1–2",
               "CoT as universal enhancer?",
               "H1:  \"step by step\" works everywhere",
               "Macro:  4 tasks × Qwen2.5 0.5B–32B\n"
               "(GSM8K / MMLU / HumanEval / AIME 2024)",
               "Finding:  task- & scale-dependent",
               ICONS["search"])

# Case 2
draw_component(START_X2 + 1 * (CBOX_W + GAP2), C_CASE2,
               "Case 2", "Fig 3",
               "Markers = self-correction?",
               "H2:  \"Wait\" / \"However\" = aha-moment",
               "Macro:  marker freq vs accuracy\n"
               "3 models × 32 runs",
               "Finding:  no positive correlation",
               ICONS["brain"])

# Mechanism (with Lean4 sub-box)
MECH_X = START_X2 + 2 * (CBOX_W + GAP2)
draw_component(MECH_X, C_MECH,
               "Mechanism", "Fig 4",
               "Prefix → state init",
               "Same Q × model × only swap prefix",
               "Micro:  29-layer logit lens\n"
               "Qwen2.5 1.5B + R1-Distill",
               "Finding:  3 distinct trajectories",
               ICONS["target"])

# Lean4 sub-box inside Mechanism
LB_X = MECH_X + 0.20
LB_Y = CBOX_Y + 0.35
LB_W = CBOX_W - 0.40
LB_H = 0.50
rrect(LB_X, LB_Y, LB_W, LB_H, C_BG_LB,
      ec=C_METHODS, lw=1.0, ls="--", r=0.04, z=3)
place_icon(ICONS["flask"], LB_X + 0.18, LB_Y + LB_H / 2, zoom=0.04, z=4)
text(LB_X + 0.40, LB_Y + LB_H - 0.14,
     "Formal proof (Lean4)",
     size=9, color=C_METHODS, bold=True,
     ha="left", va="center")
text(LB_X + 0.40, LB_Y + 0.10,
     "validates the state-init claim",
     size=8, color=C_MUTED, style="italic",
     ha="left", va="center")


# Numbered horizontal arrows between component boxes
arrow_y_h = CBOX_Y + CBOX_H / 2  # �� 2.475
arrow1_x1 = START_X2 + 1 * CBOX_W
arrow1_x2 = arrow1_x1 + GAP2
numbered_arrow_h(arrow1_x1, arrow1_x2, arrow_y_h,
                 1, color=C_CASE1)

arrow2_x1 = START_X2 + 2 * CBOX_W + GAP2
arrow2_x2 = arrow2_x1 + GAP2
numbered_arrow_h(arrow2_x1, arrow2_x2, arrow_y_h,
                 2, color=C_CASE2)

# Down arrow from Mechanism to takeaway (3)
MECH_CX = START_X2 + 2 * CBOX_W + 2 * GAP2 + CBOX_W / 2
numbered_arrow_v(MECH_CX, CBOX_Y, 0.90, 3, color=C_MECH)


# ============================================================
#  4. Bottom: Takeaway + Evidence
# ============================================================
TAKE_T, TAKE_B = 0.85, 0.50
TAKE_H = TAKE_T - TAKE_B  # 0.35
rrect(0.40, TAKE_B, FIG_W - 0.80, TAKE_H, C_TAKEAWAY, r=0.05, z=2)
place_icon(ICONS["bulb"], 0.55, (TAKE_T + TAKE_B) / 2, zoom=0.04, z=4)
text(0.85, (TAKE_T + TAKE_B) / 2,
     "Question  +  Verify  +  Understand     >     Stack patches",
     size=11, color=C_WHITE, bold=True,
     ha="left", va="center")

# Evidence strip
EVID_T, EVID_B = 0.40, 0.05
EVID_H = EVID_T - EVID_B  # 0.35
rrect(0.40, EVID_B, FIG_W - 0.80, EVID_H, "#F2F2F2", r=0.04, z=2)
text(FIG_W / 2, (EVID_T + EVID_B) / 2,
     "Evidence:  4 datasets · Qwen2.5 (0.5B–32B) + R1-Distill   ·   "
     "29-layer logit lens   ·   term-first: 12 tokens  vs  Full CoT (hundreds–thousands)",
     size=8.5, color=C_MUTED, style="italic",
     ha="center", va="center")


# ---- 导出 ----
# Output to the sibling figures/ directory with the manuscript figure name.
# Four formats (PNG / JPG / TIFF / SVG) match the manuscript's figure spec.
OUT = Path(__file__).resolve().parent.parent / "figures"
save_figure(fig, "Fig1_ConceptualFramework",
            out_dir=OUT, formats=["png", "jpg", "tiff", "svg"],
            dpi=400, verbose=True)
print("✓ Conceptual framework figure saved to"
      f" {OUT}/Fig1_ConceptualFramework.{{png,jpg,tiff,svg}}")