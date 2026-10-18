#!/usr/bin/env python3
"""Draw the Chapter 4 (Discussion) figures of the thesis.

Figures:
  Figure 4.1  fig5_replication_matrix.png  every group difference with its result on ONT, PacBio and external short reads
  Figure 4.2  fig5_model.png               how the replicated findings fit together (interpretation, schematic)
  Figure 4.5  fig5_roadmap.png             roadmap of the next tests

No input files: the matrix codes and status texts follow Table 4.1; the roadmap follows Section 4.9.
Output: PNG files (300 dpi) in the --out folder.

Usage:
  python 14_plot_discussion_figures.py -o figures
  python 14_plot_discussion_figures.py 4.1 -o figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Ellipse, Wedge, Arc
from matplotlib.lines import Line2D

OUT = None   # output folder, set from --out in main()

# colours: orange = HCC, blue = HBV carriers; dark grey = ONT, purple = PacBio
HCC, HBV = "#E8703A", "#2F7BD8"
HCC_L, HBV_L = "#F8D3BF", "#C9DCF5"
ONT, PB = "#333333", "#8B5FBF"
PB_L = "#E9DFF5"
INK, MUTE, LINE = "#1F2933", "#6B7280", "#C9CED6"
TEAL, TEAL_L = "#1B7F79", "#BFE3DF"
NAVY, NAVY_M, NAVY_L = "#23395B", "#7E93B8", "#DCE3F0"
AMBER = "#B7791F"
RED = "#C0392B"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 300, "svg.fonttype": "none",
})


# ----------------------------------------------------------------------------- drawing helpers
def box(ax, x, y, w, h, text="", fc="white", ec=LINE, lw=1.2, fs=8.5, weight="normal", color=INK, r=0.012, ls="-",
        ha="center", va="center", pad=0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.012
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, fontweight=weight, color=color, zorder=z + 1, linespacing=1.3)
    return p


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.4, style="-|>", ms=11, rad=0.0, ls="-", z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


ASP = {}


def circ(ax, x, y, r, **kw):
    """circle of radius r in x units on a canvas whose x and y spans differ (keeps it round)"""
    a = ASP.get(id(ax), 1.0)
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * a, **kw))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ASP[id(ax)] = w / h
    return fig, ax


def label(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9); kw.setdefault("color", INK); kw.setdefault("va", "center"); kw.setdefault("ha", "center")
    return ax.text(x, y, s, **kw)


def panel_letter(ax, x, y, s):
    ax.text(x, y, s, fontsize=12, fontweight="bold", color=INK, va="top", ha="left")


def head(ax, x, y, letter, title, **kw):
    """panel letter plus bold panel title"""
    panel_letter(ax, x, y + 0.02, letter)
    label(ax, x + 0.032, y, title, fontweight="bold", fontsize=kw.pop("fs", 9.8), ha="left", **kw)


def save(fig, name, tight=False):
    """write one figure as PNG (300 dpi) into the output folder and close it"""
    if tight:
        fig.savefig(f"{OUT}/{name}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    else:
        fig.savefig(f"{OUT}/{name}", dpi=300)
    plt.close(fig)


# =====================================================================================
# Figure 4.1  Replication matrix (Table 4.1)
# =====================================================================================
def fig_matrix():
    # codes: 3 significant; 2 same direction / partial; 1 not replicated; 0 not testable; 4 planned; 5 direction only
    # columns per row: finding, ONT code, PacBio code, external code, status text
    rows = [
        ("Size ratio R; 4–6× excess at 90–1000 bp, none >2 kb", 3, 3, 5, "Replicated; sample ranks agree (ρ 0.91)"),
        ("1001–2000 bp length entropy lower (H1)", 3, 3, 0, "Replicated; crowding toward 1 kb"),
        ("C-end motifs CCCA, CCAG, CCTG higher (H3)", 2, 3, 5, "Replicated (PacBio the reference)"),
        ("CCAG, GGAG higher at 601–2000 bp promoter ends (H4)", 3, 3, 0, "Replicated"),
        ("AGTT, ATTT lower at 601–2000 bp promoter ends", 2, 3, 4, "PacBio; partial on ONT"),
        ("Short-fragment ratio at high-expression promoters (S5)", 3, 3, 4, "Replicated; low and mid tertiles null"),
        ("Hepatocyte Hallmark programmes (bile acid, peroxisome)", 3, 3, 0, "Replicated (exploratory)"),
        ("Equal-depth classifiers (trained on ONT only)", 3, 4, 0, "PacBio prediction-only run pending"),
        ("Tumour copy-number profiles; extra cfDNA not tumour DNA", 3, 3, 0, "Replicated"),
        ("More fragments per sample in HCC (4× / 3.6×)", 3, 3, 0, "Shared; largely biological"),
        ("Locked ONT classifiers applied to PacBio", 3, 2, 0, "Ranking transfers, probabilities do not"),
        ("151–255 bp length entropy higher (H2)", 3, 2, 0, "Same direction, not significant on PacBio"),
        ("Blood-silent (S1) and liver-enriched (S6) promoters", 3, 1, 0, "Not replicated"),
        ("TGTG, CTTT at promoter ends", 3, 1, 0, "Not replicated"),
        ("Concatemer-recovery rate higher in HCC", 3, 1, 0, "ONT library effect, not the patients"),
    ]
    cols = ["ONT", "PacBio", "External\nshort reads"]
    style = {3: (NAVY, "white", "significant"), 2: (NAVY_M, "white", "same direction or partial"), 1: ("white", "#555555", "not replicated"),
             0: ("#F1F2F4", "#999999", "cannot be tested"), 4: ("white", TEAL, "planned (low-pass cohort)"), 5: (TEAL_L, TEAL, "direction only")}
    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_xlim(0, 10.5); ax.set_ylim(0, 7.2); ax.axis("off")
    x_lab, x_c0, cw, x_st = 0.15, 4.95, 0.80, 7.55
    top = 6.55; rh = 0.355
    for j, c in enumerate(cols):
        ax.text(x_c0 + j * cw + cw / 2, top + 0.25, c, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=INK, linespacing=1.1)
    ax.text(x_lab, top + 0.25, "Finding", fontsize=8.8, fontweight="bold", va="bottom")
    ax.text(x_st, top + 0.25, "Status", fontsize=8.8, fontweight="bold", va="bottom")
    for i, (t, o, p, e, s) in enumerate(rows):
        y = top - (i + 1) * rh
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.05, y - 0.02), 10.4, rh, fc="#F8F9FB", ec="none", zorder=0))
        ax.text(x_lab, y + rh / 2 - 0.02, t, fontsize=8.3, va="center", color=INK)
        for j, code in enumerate((o, p, e)):
            fc, tc, desc = style[code]
            ax.add_patch(FancyBboxPatch((x_c0 + j * cw + 0.06, y + 0.03), cw - 0.12, rh - 0.1, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=fc, ec=(tc if code in (1, 4) else "none"), lw=1.1, ls=("--" if code == 4 else "-"), zorder=2))
            sym = {3: "●", 2: "◐", 1: "✕", 0: "–", 4: "…", 5: "↗"}[code]
            ax.text(x_c0 + j * cw + cw / 2, y + rh / 2 - 0.02, sym, ha="center", va="center", fontsize=8.5, color=tc, zorder=3)
        ax.text(x_st, y + rh / 2 - 0.02, s, fontsize=8, va="center", color=MUTE)
    # legend
    ly = 0.55
    items = [(3, "significant"), (2, "same direction / partial"), (1, "not replicated"), (5, "direction only"), (4, "planned"), (0, "cannot be tested")]
    x = 0.15
    for code, desc in items:
        fc, tc, _ = style[code]
        ax.add_patch(FancyBboxPatch((x, ly - 0.1), 0.34, 0.24, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=(tc if code in (1, 4) else "none"), lw=1.1, ls=("--" if code == 4 else "-")))
        ax.text(x + 0.17, ly + 0.02, {3: "●", 2: "◐", 1: "✕", 0: "–", 4: "…", 5: "↗"}[code], ha="center", va="center", fontsize=8, color=tc)
        ax.text(x + 0.42, ly + 0.02, desc, fontsize=8, va="center", color=INK)
        x += 0.42 + 0.062 * len(desc) + 0.25
    ax.text(0.15, 0.17, "ONT and PacBio: same 14 plasma samples. External: short-read cohorts measure only features at ≤600 bp; most findings cannot be tested there.", fontsize=7.8, color=MUTE)
    fig.savefig(f"{OUT}/fig5_replication_matrix.png", dpi=300)



# =====================================================================================
# Figure 4.2  Summary model (interpretation; * = inferred, not tested directly)
# =====================================================================================
def fig_model():
    fig, ax = canvas(10.5, 6.3)
    label(ax, 0.01, 0.965, "How the replicated findings fit together (interpretation; statements marked * are inferred)", fontweight="bold", fontsize=10.5, ha="left")
    # left: sources
    box(ax, 0.01, 0.50, 0.25, 0.40, "", fc="#FFF5EE", ec="#F0B58F")
    label(ax, 0.135, 0.865, "Diseased liver", fontweight="bold", fontsize=10, color=HCC)
    label(ax, 0.135, 0.80, "chronic HBV, fibrosis,\ncirrhosis, HCC", fontsize=8.3, color=INK, linespacing=1.4)
    for k in range(4):
        circ(ax, 0.06 + k * 0.045, 0.68, 0.016, fc="#F4C7A9", ec=HCC, lw=1.0, zorder=3)
    label(ax, 0.135, 0.625, "more hepatocyte death*", fontsize=8.4, color=INK, fontweight="bold")
    label(ax, 0.135, 0.56, "bile acid and peroxisome genes carry\nmore of the promoter difference", fontsize=7.7, color=MUTE, linespacing=1.4)
    box(ax, 0.01, 0.07, 0.25, 0.37, "", fc="#F6F7F9", ec=LINE)
    label(ax, 0.135, 0.405, "Tumour", fontweight="bold", fontsize=10, color=INK)
    label(ax, 0.135, 0.32, "tumour fraction ≤ 0.01 in the four\nsamples with the largest excess;\nhighest in TBR3976, which has\nlittle excess", fontsize=8.0, color=INK, linespacing=1.45)
    label(ax, 0.135, 0.14, "visible by copy number,\nnot by size excess", fontsize=7.8, color=MUTE, linespacing=1.4)
    arrow(ax, 0.26, 0.70, 0.335, 0.70, lw=2.2, ms=16)
    arrow(ax, 0.26, 0.25, 0.335, 0.52, lw=1.2, ms=11, ls=(0, (4, 3)), color="#999999")
    # middle: plasma
    box(ax, 0.335, 0.27, 0.32, 0.63, "", fc=NAVY_L, ec=NAVY_M)
    label(ax, 0.495, 0.865, "HCC plasma", fontweight="bold", fontsize=10, color=NAVY)
    items = [("4–6× more fragments at 90–1000 bp", "size ratio R separates all 14 samples"),
             ("no change above 2 kb", "so the extra DNA is oligonucleosomal"),
             ("shorter 1–2 kb tail crowds toward 1 kb", "lower 1001–2000 bp entropy"),
             ("C-ending 5′ motifs enriched", "DNASE1L3-type cutting of the extra chromatin*")]
    for i, (t, s) in enumerate(items):
        yy = 0.77 - i * 0.125
        circ(ax, 0.36, yy + 0.012, 0.009, fc=NAVY, zorder=3)
        label(ax, 0.378, yy + 0.022, t, fontsize=8.5, ha="left", fontweight="bold")
        label(ax, 0.378, yy - 0.026, s, fontsize=7.9, ha="left", color=MUTE)
    label(ax, 0.495, 0.30, "replicated on ONT and PacBio, same plasma", fontsize=8, color=NAVY, fontweight="bold")
    arrow(ax, 0.655, 0.58, 0.735, 0.58, lw=2.2, ms=16)
    # right: readout
    box(ax, 0.735, 0.27, 0.255, 0.63, "", fc="#EAF5F3", ec="#A6D3CE")
    label(ax, 0.8625, 0.865, "What it means", fontweight="bold", fontsize=10, color=TEAL)
    label(ax, 0.745, 0.80, "Fragmentation signals mark the\ndiseased liver in which HCC arises,\nmore than the tumour itself*.\n\nThey complement copy-number\ndetection of the tumour.\n\nIndependent long-read cohort\nwith HBV controls: still needed.", fontsize=8.2, ha="left", va="top", linespacing=1.4)
    # bottom evidence strip
    box(ax, 0.335, 0.07, 0.655, 0.155, "", fc="white", ec=LINE)
    label(ax, 0.345, 0.195, "Not replicated or ONT-specific (not part of this picture)", fontsize=8.6, fontweight="bold", ha="left")
    label(ax, 0.345, 0.125, "concatemer recovery (ONT library effect) · blood-silent and liver-enriched promoter sets · TGTG, CTTT motifs ·\nTSS coverage dips (global HCC shift, not liver-specific)", fontsize=8.0, ha="left", color=MUTE, linespacing=1.5)
    fig.savefig(f"{OUT}/fig5_model.png", dpi=300)



# =====================================================================================
# Figure 4.5  Roadmap of next tests (Sections 4.9, 4.10; Methods 2.13)
# =====================================================================================
def fig_roadmap():
    fig, ax = canvas(8.6, 5.0 * 0.885)
    label(ax, 0.012, 0.965, "What would turn the results into a biomarker: tests in order of how soon they can be run",
          fontweight="bold", fontsize=9.6, ha="left")
    # horizon arrow
    ax.add_patch(Polygon([[0.012, 0.86], [0.955, 0.86], [0.985, 0.885], [0.955, 0.91], [0.012, 0.91]], closed=True,
                         fc="#EEF0F3", ec="none", zorder=0))
    cols = [(0.012, "Now", "this thesis", NAVY), (0.257, "Next", "data in hand", TEAL),
            (0.502, "Near", "needs data access or new samples", PB), (0.747, "Later", "new prospective cohort", HCC)]
    for x, t, s, c in cols:
        label(ax, x + 0.012, 0.897, t, fontsize=9.4, fontweight="bold", color=c, ha="left")
        label(ax, x + 0.012, 0.866, s, fontsize=7.3, color=MUTE, ha="left")
    W = 0.233
    cards = {
        0: [("Discovery, 8 HCC vs 6 HBV", "ONT and PacBio of the same plasma;\nreplicated: R, 1–2 kb entropy,\nC-end and promoter motifs", NAVY, "solid"),
            ("Open short-read checks", "GSE71378 and CRA001537: R_sr not\nhigher in HCC; CRA001537: more\nfragments up to 150 bp (exploratory)", NAVY, "solid")],
        1: [("GSE71378 genome-wide redo", "first 3 million pairs covered only\nthe first 7 Mb of chr1; rerun on\nall chromosomes", TEAL, "dash"),
            ("Larger short-read HCC cohorts", "without bisulfite conversion;\n151–255 bp entropy, promoter pooling", TEAL, "dash")],
        2: [("Independent long-read cohort\nwith HBV controls", "R with a platform-specific threshold,\n1–2 kb entropy, C-end and\npromoter motifs", PB, "dash"),
            ("Healthy and cirrhosis groups", "is the extra liver cfDNA specific\nto HCC or to advanced liver disease?", PB, "dash"),
            ("Methylation tissue of origin", "PacBio kinetics or ONT signal: does\nthe liver share track R? link stage,\nAFP, tumour size, cirrhosis", PB, "dash")],
        3: [("Copy number + fragmentomics", "they detected different patients,\nso a combined test may do better", HCC, "dash"),
            ("Prospective surveillance", "HBV carriers followed over time,\ndiagnosis not confounded with\nprocessing batch (beyond this thesis)", HCC, "dash")],
    }
    for ci, lst in cards.items():
        x = cols[ci][0]; y = 0.825
        for (t, s_, c, st) in lst:
            nt, nb = t.count("\n") + 1, s_.count("\n") + 1
            h = 0.035 + nt * 0.036 + nb * 0.033 + 0.012
            box(ax, x, y - h, W, h, "", fc="white", ec=c, lw=1.3, ls=("-" if st == "solid" else (0, (4, 2))))
            label(ax, x + 0.012, y - 0.025, t, fontsize=8.3, fontweight="bold", color=c, ha="left", va="top", linespacing=1.25)
            label(ax, x + 0.012, y - 0.03 - nt * 0.036, s_, fontsize=7.4, ha="left", va="top", linespacing=1.35)
            y -= h + 0.022
    label(ax, 0.012, 0.13, "Solid: done. Dashed: proposed in Section 4.9. Controlled-access cohorts not yet obtained: DELFI (EGAD00001010931), PacBio HCC (EGAS00001005515).",
          fontsize=6.9, color=MUTE, ha="left", va="bottom")
    ax.set_ylim(0.115, 1.0)
    save(fig, "fig5_roadmap.png")


# figure number in the thesis -> (function, output file)
FIGURES = {
    "4.1": (fig_matrix, "fig5_replication_matrix.png"),
    "4.2": (fig_model, "fig5_model.png"),
    "4.5": (fig_roadmap, "fig5_roadmap.png"),
}


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help="thesis figure numbers to draw, e.g. 4.1 (default: all)")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder for the PNG files (created if missing)")
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    for k in (a.figures or FIGURES):
        fn, png = FIGURES[k]
        fn()
        print("made Figure", k, "->", os.path.join(OUT, png))


if __name__ == "__main__":
    main()
