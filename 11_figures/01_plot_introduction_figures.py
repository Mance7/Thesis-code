#!/usr/bin/env python3
"""Draw the Chapter 1 (Introduction) figures of the thesis.

Figures:
  Figure 1.1  fig1_origin.png        where plasma cfDNA comes from, how it is cut, what is measured (schematic)
  Figure 1.2  fig1_hbv_timeline.png  HCC in HBV carriers: the surveillance gap and what cfDNA could add (schematic)
  Figure 1.3  fig1_blind_spot.png    what short-read and long-read sequencing can see (PacBio length profile + result ranges)
  Figure 1.4  fig2_platforms.png     how ONT and PacBio read cfDNA, and concatemer recovery on both platforms
  Figure 1.5  fig2_timeline.png      milestones in cfDNA fragmentomics cited in the thesis

Input: only Figure 1.3 reads data, the 1-bp length histogram of PacBio direct fragments
(one row per sample, columns 1..5000 and >5000). The numbers in Figure 1.4 C are the recovery
medians and ranges of Table 2.1 and Section 3.7.1; the timeline uses the verified reference list.
Output: PNG files (300 dpi) in the --out folder.

Usage:
  python 01_plot_introduction_figures.py --pacbio-hist results/length_hist/length_hist_pacbio_direct.tsv.gz -o figures       # all five
  python 01_plot_introduction_figures.py 1.3 --pacbio-hist results/length_hist/length_hist_pacbio_direct.tsv.gz -o figures   # only Figure 1.3
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Ellipse, Wedge, Arc
from matplotlib.lines import Line2D

OUT = None          # output folder, set from --out in main()
PACBIO_HIST = None  # PacBio length histogram, set from --pacbio-hist

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
# Figure 1.1  Origin of plasma cfDNA and how it is cut (schematic)
# =====================================================================================
def fig_origin():
    fig, ax = canvas(10.5, 6.2)
    panel_letter(ax, 0.01, 0.985, "A")
    label(ax, 0.045, 0.965, "Dying cells release chromatin", fontweight="bold", fontsize=10, ha="left")
    srcs = [("Blood cells", "most of plasma cfDNA", "#C8CDD6", 0.85),
            ("Liver (hepatocytes)", "more in liver disease", "#F4C7A9", 0.70),
            ("Tumour cells", "small share, even in HCC", "#E8703A", 0.55)]
    for name, sub, col, y in srcs:
        circ(ax, 0.07, y, 0.034, fc=col, ec="#555555", lw=1.0, zorder=2)
        circ(ax, 0.07, y, 0.013, fc="white", ec="#555555", lw=0.8, zorder=3)
        label(ax, 0.125, y + 0.018, name, ha="left", fontweight="bold", fontsize=9)
        label(ax, 0.125, y - 0.022, sub, ha="left", color=MUTE, fontsize=8)
    label(ax, 0.01, 0.455, "Cell death (apoptosis, necrosis, NETosis)\nopens the nucleus; nucleases digest the\nchromatin between nucleosomes.", ha="left", va="top", fontsize=8.2, color=INK)
    arrow(ax, 0.30, 0.70, 0.355, 0.70, lw=2.0, ms=15)

    panel_letter(ax, 0.345, 0.985, "B")
    label(ax, 0.375, 0.965, "Nucleases cut the linker DNA", fontweight="bold", fontsize=10, ha="left")
    x0, x1, y = 0.375, 0.655, 0.78
    xs = np.linspace(x0 + 0.02, x1 - 0.02, 7)
    ax.plot([x0, x1], [y, y], color="#555555", lw=1.8, zorder=1, solid_capstyle="round")
    for xx in xs:
        circ(ax, xx, y, 0.0185, fc="#9FB4D9", ec="#3E5C8F", lw=1.0, zorder=3)
    cuts = [(xs[0] + xs[1]) / 2, (xs[1] + xs[2]) / 2, (xs[2] + xs[3]) / 2, (xs[4] + xs[5]) / 2, (xs[5] + xs[6]) / 2]
    for c in cuts:
        ax.plot([c, c], [y - 0.05, y + 0.05], color="#C0392B", lw=2.0, zorder=5)
    label(ax, 0.515, 0.885, "cuts fall between nucleosomes", fontsize=8, color="#C0392B")
    label(ax, 0.515, 0.655, "nucleosome: ~147 bp of DNA\non histones; linker DNA\nadds ~20 bp", fontsize=7.8, color=MUTE, linespacing=1.4)
    box(ax, 0.375, 0.43, 0.28, 0.14, "", fc="#F6F7F9", ec=LINE)
    label(ax, 0.385, 0.543, "Nucleases in dying cells and in plasma", fontsize=8, fontweight="bold", ha="left")
    label(ax, 0.385, 0.487, "DNASE1L3 prefers C at the 5′ end of the cut, so\nits products carry CCCA/CCAG/CCTG ends.", fontsize=7.8, ha="left", color=INK)
    label(ax, 0.385, 0.447, "(end motif = first 4 bases of a fragment)", fontsize=7.4, ha="left", color=MUTE)
    arrow(ax, 0.675, 0.78, 0.725, 0.78, lw=2.0, ms=15)

    panel_letter(ax, 0.715, 0.985, "C")
    label(ax, 0.745, 0.965, "Fragments in plasma", fontweight="bold", fontsize=10, ha="left")
    a2 = fig.add_axes([0.745, 0.665, 0.235, 0.25])
    xx = np.linspace(40, 1200, 800)
    def g(m, s_, a_):
        return a_ * np.exp(-0.5 * ((xx - m) / s_) ** 2)
    prof = g(167, 14, 1.0) + g(334, 22, 0.27) + g(501, 28, 0.12) + g(668, 34, 0.06) + 0.012 * np.exp(-xx / 700)
    a2.fill_between(xx, prof, color="#9FB4D9", alpha=0.65, lw=0); a2.plot(xx, prof, color="#3E5C8F", lw=1.3)
    a2.set_xlim(40, 1200); a2.set_ylim(0, 1.45); a2.set_yticks([])
    a2.set_xlabel("fragment length (bp), schematic", fontsize=8); a2.tick_params(labelsize=7.5)
    a2.spines["left"].set_visible(False)
    for m, t in [(167, "mono ~167"), (334, "di ~334"), (501, "tri ~501")]:
        a2.text(m + 25, prof[np.argmin(abs(xx - m))] + 0.04, t, ha="left", va="bottom", fontsize=7.4, color=INK)
    box(ax, 0.745, 0.43, 0.235, 0.14, "", fc="#FFF5EE", ec="#F0B58F")
    label(ax, 0.755, 0.535, "Fragments over 1 kb are rare and are\nmissed by short-read sequencing.", fontsize=7.8, ha="left", color=INK)
    label(ax, 0.755, 0.465, "In HCC plasma the extra DNA sits at\ndi- and tri-nucleosomal sizes (this thesis).", fontsize=7.8, ha="left", color=HCC, fontweight="bold")

    ax.add_patch(Rectangle((0.01, 0.02), 0.98, 0.31, fc="#F6F7F9", ec="none", zorder=0))
    panel_letter(ax, 0.01, 0.325, "D")
    label(ax, 0.045, 0.305, "What is measured on every fragment", fontweight="bold", fontsize=10, ha="left")
    items = [("Length", "how many fragments of each size;\nnucleosomal ladder; size ratio R", 0.03),
             ("End motif", "first 4 bases at the 5′ end;\nreflects which nuclease cut it", 0.265),
             ("Genomic position", "gene body, promoter, TSS, CpG island;\nreflects chromatin of the source cell", 0.50),
             ("Count", "fragments per sample;\ndepth and amount of cfDNA", 0.735)]
    for t, s_, x in items:
        box(ax, x, 0.05, 0.235, 0.20, "", fc="white", ec=LINE)
        label(ax, x + 0.1175, 0.215, t, fontweight="bold", fontsize=9.2, ha="center")
        label(ax, x + 0.1175, 0.135, s_, fontsize=7.8, ha="center", color=INK, linespacing=1.4)
    fig.savefig(f"{OUT}/fig1_origin.png", dpi=300)


# =====================================================================================
# Figure 1.2  HCC in HBV carriers: where surveillance falls short and what cfDNA could add
#    Qualitative schematic, following Sections 1.2-1.5 and 4.3.
# =====================================================================================
def fig_hbv():
    fig, ax = canvas(8.6, 5.6)
    head(ax, 0.012, 0.955, "A", "From HBV infection to HCC (schematic, not to scale)")
    stages = [("HBV infection", "virus enters\nhepatocytes", "#FBE7DC", "#E9B79A"),
              ("Chronic hepatitis B", "inflammation; hepatocytes\ndie and regenerate", "#F8D3BF", "#E8A27C"),
              ("Cirrhosis", "fibrosis and scarring\nof the injured liver", "#F2B08C", "#DB7F50"),
              ("HCC", "tumour arises in a chronically\ninjured, usually cirrhotic liver", HCC, "#B9532A")]
    xs = [0.115, 0.375, 0.635, 0.885]
    yc = 0.775
    # progression band
    ax.add_patch(Polygon([[0.04, yc - 0.008], [0.95, yc - 0.008], [0.965, yc], [0.95, yc + 0.008], [0.04, yc + 0.008]],
                         closed=True, fc="#E9ECF0", ec="none", zorder=0))
    for (t, s, fc, ec), x in zip(stages, xs):
        circ(ax, x, yc, 0.048, fc=fc, ec=ec, lw=1.4, zorder=3)
        # small hepatocyte nuclei as decoration: fewer, darker as disease advances
        label(ax, x, yc + 0.125, t, fontweight="bold", fontsize=9.2)
        label(ax, x, yc - 0.135, s, fontsize=7.9, color=MUTE, linespacing=1.35)
    for i in range(3):
        arrow(ax, xs[i] + 0.06, yc, xs[i + 1] - 0.06, yc, lw=1.6, ms=12)
    # icons inside circles
    label(ax, xs[0], yc, "HBV", fontsize=8.2, fontweight="bold", color="#9A4A22")
    for dx, dy in [(-0.018, 0.02), (0.016, 0.018), (0.0, -0.022)]:
        circ(ax, xs[1] + dx, yc + dy, 0.008, fc="white", ec="#B9532A", lw=0.8, zorder=4)
    for k in range(3):
        ax.plot([xs[2] - 0.03, xs[2] + 0.03], [yc - 0.03 + 0.03 * k, yc - 0.012 + 0.03 * k], color="#8E3E1C", lw=1.0, zorder=4)
    label(ax, xs[3], yc, "HCC", fontsize=8.6, fontweight="bold", color="white")

    # ---------------- surveillance line under stages 2-4
    yb = 0.53
    xe = xs[3] - 0.075
    ax.plot([xs[1] - 0.05, xe], [yb, yb], color=NAVY_M, lw=1.2)
    for tx in np.linspace(xs[1] - 0.05, xe, 8):
        ax.plot([tx, tx], [yb - 0.012, yb + 0.012], color=NAVY_M, lw=1.2)
    label(ax, (xs[1] - 0.05 + xe) / 2, yb - 0.04, "surveillance of people with chronic HBV: repeated check-ups (interval schematic)",
          fontsize=7.7, color=NAVY)
    label(ax, xe + 0.012, yb, "?", fontsize=12, fontweight="bold", color=RED, ha="left")
    label(ax, xe - 0.005, yb + 0.04, "early tumours can be missed", fontsize=7.4, color=RED, ha="right")
    # ---------------- B: surveillance today
    yB = 0.44
    head(ax, 0.012, yB, "B", "Surveillance today", color=INK)
    box(ax, 0.012, 0.03, 0.47, yB - 0.075, "", fc="#F6F7F9", ec=LINE)
    rows = [("Ultrasound", "imaging of the liver at regular intervals"),
            ("Serum AFP", "alpha-fetoprotein, with or without ultrasound")]
    for i, (t, s) in enumerate(rows):
        yy = 0.33 - i * 0.075
        circ(ax, 0.04, yy + 0.016, 0.007, fc=NAVY, zorder=3)
        label(ax, 0.06, yy + 0.016, t, fontsize=8.7, fontweight="bold", ha="left")
        label(ax, 0.06, yy - 0.02, s, fontsize=7.9, ha="left", color=MUTE)
    box(ax, 0.03, 0.05, 0.435, 0.135, "", fc="white", ec="#E9B79A")
    label(ax, 0.042, 0.155, "Where it falls short", fontsize=8.4, fontweight="bold", color="#B9532A", ha="left")
    label(ax, 0.042, 0.095, "misses a substantial share of early tumours;\na useful test must tell HCC from HBV carriers without cancer",
          fontsize=7.8, ha="left", va="center", linespacing=1.35)

    # ---------------- C: what cfDNA could add
    head(ax, 0.51, yB, "C", "What cfDNA could add")
    box(ax, 0.51, 0.03, 0.478, yB - 0.075, "", fc="#EAF5F3", ec="#A6D3CE")
    items = [("A blood draw", "plasma cfDNA from the liver, blood cells and tumour"),
             ("Measured on every molecule", "length, end motif, position; no mutation calling"),
             ("Long reads see the full range", "fragments above 600 bp, missed by short reads"),
             ("Compared with HBV carriers", "the population that is under surveillance")]
    for i, (t, s) in enumerate(items):
        yy = 0.335 - i * 0.068
        circ(ax, 0.535, yy + 0.014, 0.007, fc=TEAL, zorder=3)
        label(ax, 0.552, yy + 0.014, t, fontsize=8.4, fontweight="bold", ha="left")
        label(ax, 0.552, yy - 0.017, s, fontsize=7.6, ha="left", color=MUTE)
    label(ax, 0.525, 0.07, "This thesis: 8 HCC vs 6 HBV carriers, ONT and PacBio (Yu et al., 2023)",
          fontsize=7.8, ha="left", color=TEAL, fontweight="bold")
    # arrow from HCC stage down to C
    arrow(ax, xs[3], yc - 0.20, xs[3], yB - 0.04, color=TEAL, lw=1.4, ls=(0, (3, 2)))
    save(fig, "fig1_hbv_timeline.png")



# =====================================================================================
# Figure 1.3  The short-read blind spot
#    A: median length density of PacBio direct fragments per group (120 log-spaced bins, 50-5000 bp)
#    B: the length range used by each result; green = testable in short reads, orange = long reads only
# =====================================================================================
def fig_blind():
    ids = ["TBR2776", "TBR2872", "TBR2897", "TBR2967", "TBR3013", "TBR3032", "TBR3140", "TBR3976"]
    d = pd.read_csv(PACBIO_HIST, sep="\t", index_col=0)
    cols = [str(i) for i in range(1, 5001)]
    arr = d[cols].to_numpy(float); tot = d.sum(axis=1).to_numpy(float)[:, None]
    edges = np.logspace(np.log10(50), np.log10(5000), 121)
    lo = np.ceil(edges[:-1]).astype(int); hi = np.ceil(edges[1:]).astype(int) - 1; hi[-1] = 5000
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), arr.cumsum(axis=1)], axis=1)
    dens = (cs[:, hi] - cs[:, lo - 1]) / tot / (np.log10(hi + 0.5) - np.log10(lo - 0.5))
    ctr = np.sqrt(edges[:-1] * edges[1:])
    isH = np.array([s in ids for s in d.index])
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 2.0], hspace=0.10)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharex=a)
    for grp, c, lab in [(~isH, HBV, "HBV carriers (n = 6)"), (isH, HCC, "HCC (n = 8)")]:
        m = np.median(dens[grp], axis=0)
        a.plot(ctr, m, color=c, lw=2.0, label=lab); a.fill_between(ctr, m, color=c, alpha=0.10, lw=0)
    for ax_ in (a, b):
        ax_.axvspan(100, 600, color=TEAL_L, alpha=0.45, lw=0, zorder=0)
    a.set_xscale("log"); a.set_xlim(50, 6000); a.set_ylim(0, None)
    a.set_ylabel("density\n(fraction per log10 bp)", fontsize=8.5)
    a.tick_params(labelbottom=False); a.legend(loc="upper right", fontsize=8.5)
    a.text(400, a.get_ylim()[1] * 0.95, "standard short-read\nsequencing sees\n≈100–600 bp", ha="center", va="top", fontsize=8.5, color=TEAL, fontweight="bold")
    a.text(750, a.get_ylim()[1] * 0.50, "long reads see the whole range;\nmost signals of this thesis lie here →", ha="left", va="center", fontsize=8.5, color=HCC, fontweight="bold")
    a.set_title("Median fragment-length profile of the thesis cohort (PacBio, direct fragments)", fontsize=9.5, loc="left")
    # B: rows of findings
    rows = [("Short-read stand-in for R (256–600 / 100–250 bp)", [(100, 600)], "s"),
            ("Short-fragment ratio at promoters (≤246 / ≤600 bp)", [(50, 600)], "s"),
            ("End motifs of fragments ≤600 bp", [(50, 600)], "s"),
            ("151–255 bp length entropy (H2)", [(151, 255)], "s"),
            ("Size ratio R: 256–650 bp over >2 kb", [(256, 650), (2000, 5000)], "l"),
            ("No change in counts above 2 kb", [(2001, 5000)], "l"),
            ("1001–2000 bp length entropy (H1)", [(1001, 2000)], "l"),
            ("C-end motifs at 601–5000 bp (H3, H4)", [(601, 5000)], "l")]
    n = len(rows)
    for i, (t, segs, k) in enumerate(rows):
        y = n - 1 - i
        c = TEAL if k == "s" else HCC
        for (s0, s1) in segs:
            b.add_patch(Rectangle((s0, y - 0.28), s1 - s0, 0.56, fc=c, ec="none", zorder=3))
        b.text(46, y, t, ha="right", va="center", fontsize=8.3, color=INK)
    b.set_xscale("log"); b.set_xlim(50, 6000); b.set_ylim(-0.7, n - 0.3); b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xticks([50, 100, 200, 500, 1000, 2000, 5000]); b.set_xticklabels(["50", "100", "200", "500", "1 kb", "2 kb", "5 kb"])
    b.set_xlabel("fragment length (log scale)")
    b.legend(handles=[Rectangle((0, 0), 1, 1, fc=TEAL, label="can be tested in short-read cohorts (direction only)"),
                      Rectangle((0, 0), 1, 1, fc=HCC, label="visible only in long reads")],
             loc="upper center", ncol=2, fontsize=8.3, bbox_to_anchor=(0.4, -0.17))
    b.grid(axis="x", color="#E6E6E6", lw=0.6); b.set_axisbelow(True)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.17)
    fig.savefig(f"{OUT}/fig1_blind_spot.png", dpi=300)



# =====================================================================================
# Figure 1.4  Platform principles: ONT vs PacBio; concatemers (Methods 2.2, 2.11; Table 2.1, Section 3.7.1)
# =====================================================================================
def dna(ax, x0, x1, y, col="#3E5C8F", lw=2.2, gap=0.012, z=3):
    ax.plot([x0, x1], [y + gap / 2, y + gap / 2], color=col, lw=lw, solid_capstyle="butt", zorder=z)
    ax.plot([x0, x1], [y - gap / 2, y - gap / 2], color=col, lw=lw, solid_capstyle="butt", zorder=z)


def fig_platforms():
    fig, ax = canvas(8.6, 7.0)
    cx = [0.012, 0.506]; W = 0.482
    for i, (name, col, fc, sub) in enumerate([("Oxford Nanopore (ONT)", ONT, "#F4F5F7", "one pass of a linear molecule"),
                                              ("PacBio SMRT (CCS reads)", PB, "#F6F1FB", "many passes of a circular molecule")]):
        box(ax, cx[i], 0.29, W, 0.675, "", fc=fc, ec=col, lw=1.4)
        label(ax, cx[i] + 0.015, 0.94, name, fontsize=10, fontweight="bold", color=col, ha="left")
        label(ax, cx[i] + W - 0.015, 0.94, sub, fontsize=7.8, color=MUTE, ha="right")
    panel_letter(ax, cx[0], 0.995, "A"); panel_letter(ax, cx[1], 0.995, "B")
    for i in range(2):
        for ry, rn in [(0.895, "1  Library"), (0.735, "2  Reading"), (0.53, "3  Fragment calling")]:
            label(ax, cx[i] + 0.015, ry, rn, fontsize=8.3, fontweight="bold", ha="left", color=MUTE)
            if ry < 0.89:
                ax.plot([cx[i] + 0.015, cx[i] + W - 0.015], [ry + 0.025, ry + 0.025], color="#E1E4E8", lw=0.8)

    # ---- ONT library
    x0 = cx[0]; yl = 0.855
    dna(ax, x0 + 0.06, x0 + 0.19, yl)
    for xe in (x0 + 0.038, x0 + 0.19):
        ax.add_patch(Rectangle((xe, yl - 0.009), 0.022, 0.018, fc=AMBER, ec="none", zorder=4))
    label(ax, x0 + 0.125, yl - 0.03, "adapter ligated to each end", fontsize=7.4, va="top")
    dna(ax, x0 + 0.28, x0 + 0.36, yl); dna(ax, x0 + 0.36, x0 + 0.44, yl, col="#7FA36B")
    ax.add_patch(Rectangle((x0 + 0.258, yl - 0.009), 0.022, 0.018, fc=AMBER, ec="none", zorder=4))
    ax.add_patch(Rectangle((x0 + 0.44, yl - 0.009), 0.022, 0.018, fc=AMBER, ec="none", zorder=4))
    ax.plot([x0 + 0.36, x0 + 0.36], [yl - 0.016, yl + 0.016], color=RED, lw=1.4, zorder=5)
    label(ax, x0 + 0.36, yl - 0.03, "concatemer: two molecules joined,\nno adapter at the junction", fontsize=7.4, color=RED, va="top", linespacing=1.3)

    # ---- ONT reading: membrane with pore, current trace
    ym = 0.655
    ax.add_patch(Rectangle((x0 + 0.04, ym - 0.012), 0.15, 0.024, fc="#D9DDE3", ec="none", zorder=2))
    ax.add_patch(Rectangle((x0 + 0.22, ym - 0.012), 0.06, 0.024, fc="#D9DDE3", ec="none", zorder=2))
    ax.add_patch(Polygon([[x0 + 0.19, ym + 0.012], [x0 + 0.22, ym + 0.012], [x0 + 0.212, ym - 0.012], [x0 + 0.198, ym - 0.012]],
                         fc="#9AA1AB", ec="none", zorder=3))
    ax.plot([x0 + 0.205, x0 + 0.205], [ym + 0.055, ym - 0.04], color="#3E5C8F", lw=2.0, zorder=4)
    label(ax, x0 + 0.05, ym + 0.035, "membrane with a nanopore", fontsize=7.0, color=MUTE, ha="left")
    tt = np.linspace(0, 1, 160)
    sig = 0.011 * np.sin(tt * 23) * np.cos(tt * 7) + 0.006 * np.sin(tt * 51)
    ax.plot(x0 + 0.31 + tt * 0.15, ym + 0.02 + sig, color=ONT, lw=0.9, zorder=3)
    label(ax, x0 + 0.385, ym - 0.03, "the strand passes once;\ncurrent → basecalled read", fontsize=7.4, va="top", linespacing=1.3)

    # ---- ONT fragment calling
    yf = 0.488
    dna(ax, x0 + 0.04, x0 + 0.17, yf, lw=1.6, gap=0.008)
    label(ax, x0 + 0.105, yf - 0.025, "no soft clip at either end\n→ direct fragment", fontsize=7.3, va="top", linespacing=1.3)
    dna(ax, x0 + 0.25, x0 + 0.34, yf, lw=1.6, gap=0.008); dna(ax, x0 + 0.34, x0 + 0.43, yf, col="#7FA36B", lw=1.6, gap=0.008)
    arrow(ax, x0 + 0.34, yf - 0.012, x0 + 0.34, yf - 0.035, lw=1.0, ms=8, color=MUTE)
    dna(ax, x0 + 0.24, x0 + 0.33, yf - 0.05, lw=1.6, gap=0.008); dna(ax, x0 + 0.35, x0 + 0.44, yf - 0.05, col="#7FA36B", lw=1.6, gap=0.008)
    label(ax, x0 + 0.285, yf - 0.07, "primary", fontsize=7.0, color=MUTE)
    label(ax, x0 + 0.395, yf - 0.07, "supplementary", fontsize=7.0, color=MUTE)
    label(ax, x0 + 0.34, yf - 0.088, "one SA (MAPQ ≥ 20), both outer ends\nunclipped → two recovered fragments",
          fontsize=7.3, va="top", linespacing=1.3)
    label(ax, x0 + 0.015, 0.298, "linked adapters trimmed (Cutadapt) · minimap2 · MAPQ ≥ 20;\nsame chromosome within 10 kb: not split (possible rearrangement)",
          fontsize=7.1, ha="left", va="bottom", color=MUTE, linespacing=1.3)

    # ---- PacBio library: SMRTbell circle
    x1 = cx[1]
    cxx, cyy = x1 + 0.13, 0.835
    circ(ax, cxx, cyy, 0.042, fc="none", ec="#3E5C8F", lw=2.0, zorder=3)
    circ(ax, cxx, cyy, 0.034, fc="none", ec="#3E5C8F", lw=2.0, zorder=3)
    for xe in [cxx - 0.042, cxx + 0.042]:
        circ(ax, xe, cyy, 0.01, fc=AMBER, ec="none", zorder=4)
    label(ax, x1 + 0.215, cyy, "SMRTbell: hairpin adapters close\nboth ends of the molecule into a circle", fontsize=7.4, ha="left", linespacing=1.3)

    # ---- PacBio reading: passes
    ypp = 0.665
    circ(ax, cxx, ypp, 0.036, fc="none", ec="#3E5C8F", lw=1.8, zorder=3)
    circ(ax, cxx + 0.036, ypp, 0.011, fc=PB, ec="none", zorder=4)
    ax.add_patch(Arc((cxx, ypp), 0.098, 0.098 * ASP[id(ax)], theta1=-55, theta2=225, color=PB, lw=1.0, zorder=3))
    label(ax, cxx, ypp, "pol", fontsize=6.8, color=PB, fontweight="bold")
    for k in range(4):
        yy = ypp + 0.033 - k * 0.022
        ax.plot([x1 + 0.22, x1 + 0.33], [yy, yy], color="#3E5C8F", lw=1.6, zorder=3)
        ax.add_patch(Rectangle((x1 + 0.33, yy - 0.006), 0.01, 0.012, fc=AMBER, ec="none", zorder=4))
    label(ax, x1 + 0.28, ypp - 0.062, "subreads, split\nat the hairpins", fontsize=7.2, color=MUTE, linespacing=1.25)
    arrow(ax, x1 + 0.355, ypp, x1 + 0.385, ypp, lw=1.0, ms=8)
    ax.plot([x1 + 0.39, x1 + 0.46], [ypp, ypp], color=PB, lw=2.6, zorder=3)
    label(ax, x1 + 0.425, ypp - 0.045, "CCS consensus,\n≥ 3 full passes", fontsize=7.2, color=PB, fontweight="bold", linespacing=1.25)
    label(ax, cxx, ypp - 0.075, "polymerase\ncircles the template", fontsize=7.2, color=MUTE, linespacing=1.25)

    # ---- PacBio fragment calling
    dna(ax, x1 + 0.05, x1 + 0.21, yf, lw=1.6, gap=0.008, col=PB)
    label(ax, x1 + 0.13, yf - 0.025, "one CCS read = one molecule,\nend to end; no trimming needed", fontsize=7.3, va="top", linespacing=1.3)
    dna(ax, x1 + 0.28, x1 + 0.36, yf, lw=1.6, gap=0.008, col=PB); dna(ax, x1 + 0.36, x1 + 0.44, yf, col="#7FA36B", lw=1.6, gap=0.008)
    label(ax, x1 + 0.36, yf - 0.025, "same Stage 1 rules;\nfew concatemers to recover", fontsize=7.3, va="top", linespacing=1.3)
    label(ax, x1 + 0.015, 0.298, "BWA-MEM · duplicates removed · MAPQ ≥ 30;\nsame zero-clip rule and concatemer recovery as ONT",
          fontsize=7.1, ha="left", va="bottom", color=MUTE, linespacing=1.3)

    # ---- C: data panel, recovery by group and platform (median and range; Table 2.1, Section 3.7.1)
    panel_letter(ax, 0.012, 0.275, "C")
    label(ax, 0.044, 0.255, "Concatemer recovery in the same plasma", fontweight="bold", fontsize=9.8, ha="left")
    a2 = fig.add_axes([0.17, 0.065, 0.36, 0.15])
    rows = [("ONT, HCC", 25.7, 12.3, 75.8, HCC, ONT), ("ONT, HBV", 6.2, 4.6, 9.6, HBV, ONT),
            ("PacBio, HCC", 1.25, 0.40, 2.42, HCC, PB), ("PacBio, HBV", 0.72, 0.40, 1.33, HBV, PB)]
    for i, (n, m, lo, hi, c, pc) in enumerate(rows):
        yy = 3 - i
        a2.plot([lo, hi], [yy, yy], color=c, lw=2.2, solid_capstyle="round", alpha=0.45)
        a2.plot([m], [yy], marker="o" if pc == ONT else "s", ms=6.5, color=c, mec="white", mew=0.8, zorder=3)
    a2.set_yticks([3, 2, 1, 0], [r[0] for r in rows], fontsize=7.8)
    a2.set_xscale("log"); a2.set_xlim(0.3, 100)
    a2.set_xticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"], fontsize=7.6)
    a2.set_xlabel("fragments gained by recovery (%), median and range", fontsize=7.8)
    a2.set_ylim(-0.6, 3.6); a2.spines["left"].set_visible(False); a2.tick_params(axis="y", length=0)
    a2.grid(axis="x", color="#E6E6E6", lw=0.6); a2.set_axisbelow(True)
    box(ax, 0.58, 0.04, 0.408, 0.20, "", fc="white", ec=LINE)
    label(ax, 0.595, 0.21, "Why recovery differs", fontsize=8.6, fontweight="bold", ha="left")
    label(ax, 0.595, 0.135,
          "ONT: every HCC sample above every carrier\n(25.7% vs 6.2%, p = 6.7 × 10⁻⁴, the smallest possible).\n"
          "PacBio, same plasma: about 1% in both groups\n(p = 0.091). The ONT gap is a library effect,\nso confirmatory tests use direct fragments.",
          fontsize=7.5, ha="left", va="center", linespacing=1.35)
    save(fig, "fig2_platforms.png")



# =====================================================================================
# Figure 1.5  Timeline of cfDNA fragmentomics (only studies in the verified reference list, with their verified years)
# =====================================================================================
TL_SHORT = [  # (year, label, finding, theme)
    (2015, "Jiang et al.", "HCC plasma DNA both shortened and lengthened", "size"),
    (2016, "Snyder et al.", "nucleosome footprint (WPS) informs tissue of origin", "nuc"),
    (2016, "Underhill et al.", "tumour DNA shorter: 134–144 vs 167 bp", "size"),
    (2017, "Adalsteinsson et al.", "ichorCNA: tumour fraction from copy number", "nuc"),
    (2018, "Mouliere et al.", "90–150 bp size selection enriches ctDNA", "size"),
    (2019, "Cristiano et al.", "DELFI: genome-wide fragmentation profiles", "nuc"),
    (2019, "Serpas et al.", "Dnase1l3 loss changes lengths and end motifs", "motif"),
    (2020, "Han et al.", "roles of DNASE1, DNASE1L3 and DFFB", "motif"),
    (2020, "Jiang et al.", "end-motif profiling and motif diversity score", "motif"),
    (2021, "Jin et al.", "CCCA, CCTG, CCAG: 197 HCC vs 187 HBV", "motif"),
    (2022, "Esfahani et al.", "promoter fragmentation entropy (EPIC-seq)", "ent"),
    (2023, "Foda et al.", "DELFI for liver cancer", "nuc"),
    (2023, "Zhou et al.", "end-motif founder profiles (NMF)", "motif"),
    (2023, "Helzer et al.", "exon-1 entropy on targeted panels", "ent"),
    (2024, "Maansson et al.", "physical size selection, lung cancer", "size"),
    (2025, "Helzer et al.", "metric comparison: entropy not dominant", "ent"),
]
TL_LONG = [
    (2022, "Choy et al.", "PacBio: 15.7% of HCC molecules >1 kb", "long"),
    (2023, "Yu et al.", "PacBio vs ONT, same plasma: data of this thesis", "data"),
    (2023, "van der Pol et al.", "ONT: 54.1% vs 5.3% of fragments >300 bp", "long"),
    (2024, "Yu et al.", "origin and properties of long cfDNA", "long"),
    (2026, "Berman et al.", "excess 1–4 kb in cancer; >7.5 kb mostly lysis", "long"),
]
THEME = {"size": (NAVY, "fragment size and size selection"), "nuc": ("#6B7280", "nucleosomes, genome-wide, copy number"),
         "motif": (TEAL, "end motifs and nucleases"), "ent": (AMBER, "entropy-type metrics"),
         "long": (PB, "long-read cfDNA"), "data": (HCC, "source of the thesis data")}


def fig_timeline():
    fig, ax = canvas(8.6, 7.6)
    years = list(range(2015, 2027))
    long = TL_LONG + [(2026, "This thesis", "length bins over the long-read range", "data")]
    nL = {y: sum(e[0] == y for e in TL_SHORT) for y in years}
    nR = {y: sum(e[0] == y for e in long) for y in years}
    step, gap = 0.030, 0.021
    y_of, yy = {}, 0.875
    for y in years:
        n = max(nL[y], nR[y], 1)
        y_of[y] = yy - (n - 1) * step / 2
        yy -= (n - 1) * step + gap + step
    xa = 0.53  # axis
    label(ax, 0.012, 0.975, "Milestones in cfDNA fragmentomics cited in this thesis (years as in the verified reference list)",
          fontweight="bold", fontsize=9.6, ha="left")
    label(ax, xa - 0.04, 0.925, "Short-read studies", fontsize=9.2, fontweight="bold", ha="right", color=INK)
    label(ax, xa + 0.04, 0.925, "Long-read studies", fontsize=9.2, fontweight="bold", ha="left", color=PB)
    ax.plot([xa, xa], [y_of[2026] - 0.02, y_of[2015] + 0.02], color="#C3C8CF", lw=1.6, zorder=1, solid_capstyle="round")
    for y in years:
        box(ax, xa - 0.028, y_of[y] - 0.012, 0.056, 0.024, str(y), fc="white", ec="#9AA1AB", lw=1.0, fs=7.6,
            weight="bold", color=MUTE, r=0.01, z=3)

    def place(entries, side):
        by = {}
        for e in entries:
            by.setdefault(e[0], []).append(e)
        for y, es in by.items():
            n = len(es)
            for k, (yr, who, what, th) in enumerate(es):
                col = THEME[th][0]
                yy = y_of[y] + (0.5 * (n - 1) - k) * step
                wt = "bold"
                if side == "L":
                    xt = xa - 0.045
                    ax.plot([xt + 0.004, xa - 0.028], [yy, y_of[y]], color=col, lw=0.8, alpha=0.6, zorder=1)
                    label(ax, xt, yy, who, fontsize=7.9, fontweight=wt, color=col, ha="right")
                    label(ax, xt - 0.14, yy, what, fontsize=7.6, color=INK, ha="right")
                else:
                    xt = xa + 0.045
                    ax.plot([xa + 0.028, xt - 0.004], [y_of[y], yy], color=col, lw=0.8, alpha=0.6, zorder=1)
                    label(ax, xt, yy, who, fontsize=7.9, fontweight=wt, color=col, ha="left")
                    label(ax, xt + 0.125, yy, what, fontsize=7.6, color=(col if th == "data" else INK), ha="left",
                          fontweight=("bold" if who == "This thesis" else "normal"))

    place(TL_SHORT, "L")
    place(long, "R")
    lx, ly = 0.012, 0.088
    for i, k in enumerate(["size", "nuc", "motif", "ent", "long", "data"]):
        c, t = THEME[k]
        xx = lx + (i % 3) * 0.33; yy2 = ly - (i // 3) * 0.03
        circ(ax, xx + 0.008, yy2, 0.006, fc=c, ec="none")
        label(ax, xx + 0.022, yy2, t, fontsize=7.4, ha="left", color=INK)
    label(ax, 0.012, 0.008, "Serpas et al. was online in December 2018 (print 2019); Berman et al. was a 2024 preprint, published 2026.",
          fontsize=7.0, color=MUTE, ha="left", va="bottom")
    save(fig, "fig2_timeline.png")


# figure number in the thesis -> (function, output file)
FIGURES = {
    "1.1": (fig_origin, "fig1_origin.png"),
    "1.2": (fig_hbv, "fig1_hbv_timeline.png"),
    "1.3": (fig_blind, "fig1_blind_spot.png"),
    "1.4": (fig_platforms, "fig2_platforms.png"),
    "1.5": (fig_timeline, "fig2_timeline.png"),
}


def main():
    global OUT, PACBIO_HIST
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help="thesis figure numbers to draw, e.g. 1.1 (default: all)")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder for the PNG files (created if missing)")
    ap.add_argument("--pacbio-hist", required=True,
                    help="1-bp length histogram of PacBio direct fragments, one row per sample (tsv.gz; Figure 1.3)")
    a = ap.parse_args()
    OUT = a.out
    PACBIO_HIST = a.pacbio_hist
    os.makedirs(OUT, exist_ok=True)
    for k in (a.figures or FIGURES):
        fn, png = FIGURES[k]
        fn()
        print("made Figure", k, "->", os.path.join(OUT, png))


if __name__ == "__main__":
    main()
