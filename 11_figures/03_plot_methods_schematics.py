#!/usr/bin/env python3
"""Draw the Chapter 2 (Methods) schematic figures of the thesis.

Figures:
  Figure 2.2  fig3_design.png          study design: three levels of evidence
  Figure 2.3  fig3_resolution.png      promoter window, three levels of resolution, HCC-expressed subsets S1-S6
  Figure 2.4  fig3_stats_flow.png      statistical decision flow; panel B simulates why plug-in entropy needs rarefaction
  Figure 2.6  fig3_bisulfite_size.png  fragment size from paired-end (bisulfite) short reads and the short-read ratio R_sr

No input files: all numbers are the ones given in the Methods (promoter and subset counts, thresholds).
Panel 2.4 B uses simulated data with a fixed seed. Output: PNG files (300 dpi) in the --out folder.

Usage:
  python 03_plot_methods_schematics.py -o figures
  python 03_plot_methods_schematics.py 2.4 -o figures
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
# Figure 2.2  Study design
# =====================================================================================
def fig_design():
    fig, ax = canvas(10.5, 6.0)
    label(ax, 0.01, 0.965, "Three levels of evidence, in decreasing strength of what they can test", fontweight="bold", fontsize=10.5, ha="left")
    # discovery
    box(ax, 0.01, 0.56, 0.30, 0.34, "", fc=NAVY_L, ec=NAVY_M)
    label(ax, 0.16, 0.865, "1  Discovery cohort", fontweight="bold", fontsize=10, color=NAVY)
    label(ax, 0.16, 0.795, "Yu et al. 2023\n8 HCC · 6 HBV carriers", fontsize=8.8, ha="center")
    circ(ax, 0.075, 0.67, 0.025, fc=HCC, ec="white", lw=1.2, zorder=3); label(ax, 0.075, 0.67, "8", color="white", fontweight="bold", fontsize=9, ha="center")
    circ(ax, 0.145, 0.67, 0.025, fc=HBV, ec="white", lw=1.2, zorder=3); label(ax, 0.145, 0.67, "6", color="white", fontweight="bold", fontsize=9, ha="center")
    label(ax, 0.075, 0.615, "HCC", fontsize=7.8, color=MUTE); label(ax, 0.145, 0.615, "HBV", fontsize=7.8, color=MUTE)
    label(ax, 0.16, 0.585, "plasma cfDNA, long-read sequenced", fontsize=7.8, color=MUTE, ha="center")
    arrow(ax, 0.31, 0.73, 0.395, 0.80)
    arrow(ax, 0.31, 0.73, 0.395, 0.66)
    box(ax, 0.395, 0.75, 0.14, 0.11, "ONT\n(nanopore)", fc="white", ec=ONT, lw=1.6, fs=9, weight="bold")
    box(ax, 0.395, 0.60, 0.14, 0.11, "PacBio\n(circular CCS)", fc="white", ec=PB, lw=1.6, fs=9, weight="bold")
    label(ax, 0.465, 0.915, "same plasma samples,\ntwo independent libraries", fontsize=7.8, color=MUTE, ha="center")
    arrow(ax, 0.535, 0.80, 0.605, 0.745); arrow(ax, 0.535, 0.655, 0.605, 0.715)
    box(ax, 0.605, 0.60, 0.385, 0.30, "", fc="#F6F7F9", ec=LINE)
    label(ax, 0.797, 0.865, "Same analysis on both platforms", fontweight="bold", fontsize=9.2)
    label(ax, 0.617, 0.735, "• fragment length: size bins, ratio R, entropy\n• end motifs (256 four-mers, CLR)\n• promoter and gene-set level (fsr, entropy)\n• copy number and tumour fraction (ichorCNA)\n• classifiers fixed before analysis", fontsize=8.3, ha="left", va="center", linespacing=1.5)

    # level 2: platform replication
    box(ax, 0.01, 0.285, 0.98, 0.215, "", fc="#F3EEFA", ec="#C7B3E3")
    label(ax, 0.02, 0.47, "2  Platform replication (the core of the thesis)", fontweight="bold", fontsize=10, color=PB, ha="left")
    label(ax, 0.02, 0.385, "A result counts as replicated when it has the same direction and passes its test on both ONT and PacBio.\nExcludes sequencing and analysis artefacts, but not sample collection or storage differences,\nbecause both platforms used the same 14 plasma samples.", fontsize=8.5, ha="left", va="center", linespacing=1.5)
    box(ax, 0.64, 0.315, 0.335, 0.115, "Ranking transfers between platforms;\nabsolute values and cut-offs do not.", fc="white", ec="#C7B3E3", fs=8.2)

    # level 3: external
    box(ax, 0.01, 0.02, 0.98, 0.235, "", fc="#EAF5F3", ec="#A6D3CE")
    label(ax, 0.02, 0.228, "3  External short-read cohorts (open access; direction of short-fragment features only)", fontweight="bold", fontsize=10, color=TEAL, ha="left")
    ext = [("GSE71378", "Snyder 2016\n3 liver cancer vs 6 non-cancer\n(same batch); no HBV carriers", False),
           ("CRA001537 pilot", "Zhang 2020, deep bisulfite\n1 HCC, 1 hepatitis, 1 healthy\n(descriptive, no p value)", False),
           ("CRA001537 low-pass", "Zhang 2020, 54 samples\n8 HCC vs 17 hepatitis (test);\ncirrhosis, healthy, post-surgery", True)]
    for i, (t, s, pend) in enumerate(ext):
        x = 0.02 + i * 0.325
        box(ax, x, 0.04, 0.305, 0.15, "", fc="white", ec=TEAL, lw=1.3, ls=("--" if pend else "-"))
        label(ax, x + 0.152, 0.168, t + ("  (analysis pending)" if pend else ""), fontweight="bold", fontsize=8.6, ha="center")
        label(ax, x + 0.152, 0.098, s, fontsize=7.9, ha="center", linespacing=1.4)
    arrow(ax, 0.16, 0.56, 0.16, 0.50, color=NAVY_M); arrow(ax, 0.16, 0.285, 0.16, 0.255, color="#C7B3E3")
    fig.savefig(f"{OUT}/fig3_design.png", dpi=300)



# =====================================================================================
# Figure 2.3  Three levels of resolution; expression subsets S1-S6 (Methods 2.5, 2.8; Section 3.4, Table 3.4)
# =====================================================================================
def fig_resolution():
    fig, ax = canvas(8.6, 6.4)
    head(ax, 0.012, 0.965, "A", "One promoter window (schematic)")
    # chromosome with a gene and the promoter window
    y = 0.84
    ax.plot([0.04, 0.96], [y, y], color="#9AA1AB", lw=1.6, zorder=1)
    tss = 0.40
    scale = 0.30 / 2500  # 2500 bp across 0.30
    p0, p1 = tss - 2000 * scale, tss + 500 * scale
    ax.add_patch(Rectangle((p0, y - 0.03), p1 - p0, 0.06, fc=TEAL_L, ec=TEAL, lw=1.0, zorder=2, alpha=0.9))
    # gene body exons
    ax.add_patch(Rectangle((tss, y - 0.012), 0.52, 0.024, fc="#D9DDE3", ec="none", zorder=2))
    for ex, w in [(tss, 0.04), (tss + 0.13, 0.03), (tss + 0.27, 0.035), (tss + 0.45, 0.07)]:
        ax.add_patch(Rectangle((ex, y - 0.022), w, 0.044, fc=NAVY_M, ec="none", zorder=3))
    ax.plot([tss, tss], [y + 0.022, y + 0.065], color=INK, lw=1.2, zorder=4)
    arrow(ax, tss, y + 0.065, tss + 0.04, y + 0.065, lw=1.2, ms=8, color=INK)
    label(ax, tss + 0.002, y + 0.088, "TSS", fontsize=7.8, fontweight="bold", ha="left")
    label(ax, p0, y - 0.05, "−2000 bp", fontsize=7.4, color=TEAL)
    label(ax, p1, y - 0.05, "+500", fontsize=7.4, color=TEAL)
    label(ax, (p0 + p1) / 2, y + 0.05, "promoter window", fontsize=7.8, color=TEAL, fontweight="bold")
    label(ax, 0.75, y + 0.045, "gene body (exons dark)", fontsize=7.4, color=MUTE)
    # fragments overlapping
    rng = np.random.default_rng(4)
    fr = [(0.10, 0.03), (0.15, 0.10), (0.20, 0.025), (0.235, 0.05), (0.27, 0.028), (0.30, 0.15), (0.33, 0.03),
          (0.37, 0.026), (0.43, 0.07), (0.50, 0.03), (0.57, 0.04)]
    for i, (fx, fw) in enumerate(fr):
        yy = y - 0.08 - (i % 3) * 0.018
        inside = (fx + fw > p0) and (fx < p1)
        ax.plot([fx, fx + fw], [yy, yy], color=(HCC if inside else "#B8BEC7"), lw=2.2, solid_capstyle="round", zorder=3)
    label(ax, 0.96, y - 0.115, "fragments (bars): counted for the promoter\nif they overlap its window by ≥ 1 bp (orange)",
          fontsize=7.4, color=MUTE, ha="right", linespacing=1.3)

    # ---------------- B three levels
    head(ax, 0.012, 0.63, "B", "Three levels of resolution, from the same fragments")
    lv = [("1  Region-pooled", "all fragments with a region flag\n(promoter, gene body, exon, TSS,\nCpG island) pooled per sample",
           "one value per sample per\nsize bin × region cell", "power: high · specificity: low"),
          ("2  Individual promoters", "each of 19,924 promoters\ntested on its own fragments",
           "too few fragments per promoter;\nno promoter reaches FDR", "power: too low with 8 vs 6"),
          ("3  Gene-set pooled", "raw fragments pooled across the\npromoters of a predefined set,\nthen the metric is computed",
           "compared with 1,000 random sets\nmatched on expression", "power: high · specificity: set-level")]
    for i, (t, s, out, pw) in enumerate(lv):
        x = 0.012 + i * 0.329
        box(ax, x, 0.335, 0.31, 0.255, "", fc="white", ec=(TEAL if i == 2 else LINE), lw=(1.5 if i == 2 else 1.2))
        label(ax, x + 0.012, 0.565, t, fontsize=8.9, fontweight="bold", ha="left", color=(TEAL if i == 2 else INK))
        # mini glyph: promoters as small teal blocks pooled
        gy = 0.515
        if i == 0:
            for k in range(7):
                ax.add_patch(Rectangle((x + 0.02 + k * 0.04, gy - 0.008), 0.028, 0.016, fc=TEAL_L, ec=TEAL, lw=0.6, zorder=3))
            ax.add_patch(Rectangle((x + 0.015, gy - 0.016), 0.285, 0.032, fc="none", ec=TEAL, lw=0.9, ls="--", zorder=3))
        elif i == 1:
            for k in range(7):
                ax.add_patch(Rectangle((x + 0.02 + k * 0.04, gy - 0.008), 0.028, 0.016, fc=TEAL_L, ec=TEAL, lw=0.6, zorder=3))
                ax.add_patch(Rectangle((x + 0.016 + k * 0.04, gy - 0.014), 0.036, 0.028, fc="none", ec=TEAL, lw=0.6, ls="--", zorder=3))
        else:
            for k in range(7):
                sel = k in (0, 2, 3, 6)
                ax.add_patch(Rectangle((x + 0.02 + k * 0.04, gy - 0.008), 0.028, 0.016, fc=(TEAL if sel else "#EEF0F3"),
                                       ec=TEAL if sel else LINE, lw=0.6, zorder=3))
            label(ax, x + 0.30, gy + 0.028, "set members filled", fontsize=6.6, color=MUTE, ha="right")
        label(ax, x + 0.012, 0.44, s, fontsize=7.6, ha="left", linespacing=1.3)
        label(ax, x + 0.012, 0.37, out, fontsize=7.3, ha="left", color=MUTE, linespacing=1.3)

    # ---------------- C expression subsets
    head(ax, 0.012, 0.28, "C", "HCC-expressed promoters and the six pre-registered subsets (promoters per subset)")
    total = 11333
    x0, x1 = 0.215, 0.97
    Wd = x1 - x0
    def seg(yb, parts, h=0.045):
        s = sum(p[1] for p in parts); xx = x0
        for name, n, fc, ec, tc in parts:
            w = Wd * n / s
            ax.add_patch(Rectangle((xx, yb), w - 0.003, h, fc=fc, ec=ec, lw=0.8, zorder=3))
            label(ax, xx + w / 2, yb + h / 2, f"{name}  {n:,}", fontsize=7.6, color=tc, fontweight="bold", zorder=5)
            xx += w
    label(ax, 0.012, 0.205, "All HCC-expressed\n(11,333)", fontsize=7.8, ha="left", linespacing=1.25)
    ax.add_patch(Rectangle((x0, 0.185), Wd - 0.003, 0.04, fc="#F6F7F9", ec=LINE, lw=0.8, zorder=3))
    label(ax, x0 + Wd / 2, 0.205, "HPA: median pTPM ≥ 1 in TCGA liver cancer; promoter −2000..+500 bp", fontsize=7.3, color=MUTE)
    label(ax, 0.012, 0.137, "by blood expression\n(silent: GTEx TPM < 1)", fontsize=7.8, ha="left", linespacing=1.25)
    seg(0.115, [("S1 blood-silent", 2100, "#FDEBDD", "#E9B79A", "#9A4A22"), ("S2 blood-expressed", 9157, "#E3ECF8", "#9DBBE3", NAVY)])
    label(ax, 0.012, 0.067, "by HCC expression\n(tertiles)", fontsize=7.8, ha="left", linespacing=1.25)
    seg(0.045, [("S3 low", 3767, "#F3F4F6", LINE, INK), ("S4 mid", 3780, "#E6E8EC", LINE, INK), ("S5 high", 3786, "#D3D7DE", "#9AA1AB", INK)])
    # S6 inside S5
    s6w = Wd * 398 / total
    xs6 = x1 - 0.003 - s6w
    ax.add_patch(Rectangle((xs6, 0.012), s6w, 0.022, fc=HCC, ec="none", zorder=4))
    label(ax, xs6 - 0.006, 0.023, "S6 liver-enriched 398 (all within S5)", fontsize=7.3, color="#9A4A22", fontweight="bold", ha="right")
    save(fig, "fig3_resolution.png")



# =====================================================================================
# Figure 2.4  Statistical decision flow (Methods 2.6, 2.8, 2.9, 2.10; Section 3.2.1)
# =====================================================================================
def fig_stats():
    fig, ax = canvas(8.6, 6.2)
    head(ax, 0.012, 0.965, "A", "From hypothesis to a confirmed result")
    steps = [
        ("1  Fix the families first", NAVY, NAVY_L, NAVY_M,
         "Registry written before re-analysis (Appendix B).\nConfirmatory: H1 entropy 1001–2000 bp · H2 entropy 151–255 bp ·\nH3 literature end motifs · H4 promoter motifs · H5 gene sets.\nExploratory: the full 256-motif screen, one family."),
        ("2  Correct for fragment count", TEAL, "#EAF5F3", "#A6D3CE",
         "Plug-in entropy rises with N (see B). Each sample is subsampled\nto the smallest N in its cell, mean of 100 draws (rarefied, primary);\ncells with < 100 fragments are left out."),
        ("3  Test each cell", INK, "#F6F7F9", LINE,
         "Mann–Whitney (8 vs 6: p ≥ 2/3,003 = 6.7 × 10⁻⁴) with Cohen's d;\nend motifs: moderated t (limma), CLR values."),
        ("4  Correct within the family", INK, "#F6F7F9", LINE,
         "Benjamini–Hochberg within each family → q.\nMotif screen: two stages, cells first, then motifs at q·R/M."),
        ("5  Compare with a null where needed", PB, "#F6F1FB", "#C7B3E3",
         "3,003 relabellings · 1,000 matched random promoter sets ·\n1,000 permutations of the whole classifier pipeline;\np = (b + 1) / (B + 1)."),
    ]
    y = 0.895; hgt = [0.17, 0.13, 0.10, 0.10, 0.13]
    for (t, c, fc, ec, s), h in zip(steps, hgt):
        box(ax, 0.012, y - h, 0.57, h, "", fc=fc, ec=ec, lw=1.2)
        label(ax, 0.025, y - 0.022, t, fontsize=8.8, fontweight="bold", color=c, ha="left")
        label(ax, 0.025, y - 0.035, s, fontsize=7.5, ha="left", va="top", linespacing=1.35)
        y -= h
        if t[0] != "5":
            arrow(ax, 0.30, y, 0.30, y - 0.025, lw=1.2, ms=9)
        y -= 0.025
    # confirmed
    box(ax, 0.012, 0.02, 0.57, 0.105, "", fc="#FFF5EE", ec="#F0B58F", lw=1.4)
    label(ax, 0.025, 0.098, "Confirmed", fontsize=8.8, fontweight="bold", color="#B9532A", ha="left")
    label(ax, 0.025, 0.052, "q < 0.05 in its family and the same direction on direct (non-recovered) fragments;\nreplicated if it also passes on PacBio.",
          fontsize=7.4, ha="left", va="center", linespacing=1.3)
    arrow(ax, 0.30, y + 0.025, 0.30, 0.125, lw=1.2, ms=9)

    # ---------------- B inset: illustration of plug-in entropy vs N
    head(ax, 0.615, 0.965, "B", "Why entropy needs the correction")
    label(ax, 0.647, 0.925, "illustration, simulated data", fontsize=7.6, color=MUTE, ha="left", style="italic")
    a = fig.add_axes([0.685, 0.50, 0.29, 0.36])
    K = 1000
    lengths = np.arange(1001, 2001)
    w = np.exp(-(lengths - 1001) / 450.0); w = w / w.sum()   # one fixed length distribution, same for every sample
    Ns = np.unique(np.round(np.logspace(np.log10(50), np.log10(50000), 24)).astype(int))
    rng = np.random.default_rng(11)
    def Hn(c):
        p = c[c > 0] / c.sum(); return -(p * np.log2(p)).sum() / np.log2(K)
    med, lo, hi = [], [], []
    for n in Ns:
        vals = [Hn(rng.multinomial(n, w)) for _ in range(40)]
        med.append(np.median(vals)); lo.append(np.percentile(vals, 5)); hi.append(np.percentile(vals, 95))
    true = -(w * np.log2(w)).sum() / np.log2(K)
    a.fill_between(Ns, lo, hi, color=TEAL_L, lw=0)
    a.plot(Ns, med, color=TEAL, lw=1.6, label="plug-in estimate")
    a.axhline(true, color=INK, lw=0.9, ls="--")
    a.text(52, true + 0.006, "true entropy", fontsize=7, color=INK, va="bottom")
    cap = np.minimum(np.log2(Ns) / np.log2(K), 1.0)
    a.plot(Ns, cap, color=MUTE, lw=0.9, ls=":")
    a.text(330, 0.712, "ceiling log₂N / log₂K", fontsize=6.8, color=MUTE, ha="left")
    # two samples same distribution, different N
    nA, nB = 300, 20000
    hA = Hn(rng.multinomial(nA, w)); hB = Hn(rng.multinomial(nB, w))
    sub = [Hn(rng.multivariate_hypergeometric(rng.multinomial(nB, w), nA)) for _ in range(100)]
    a.plot([nA], [hA], "o", color=HBV, ms=6, mec="white", zorder=4)
    a.plot([nB], [hB], "o", color=HCC, ms=6, mec="white", zorder=4)
    a.plot([nA], [np.mean(sub)], "o", mfc="none", mec=HCC, ms=11, mew=1.4, zorder=5)
    a.annotate("", xy=(nA * 1.15, np.mean(sub)), xytext=(nB * 0.85, hB), arrowprops=dict(arrowstyle="-|>", color=HCC, lw=0.9, connectionstyle="arc3,rad=-0.25"))
    a.set_xscale("log"); a.set_xlim(45, 60000); a.set_ylim(0.70, 1.0)
    a.set_xlabel("fragments in the cell, N", fontsize=7.8); a.set_ylabel("normalised plug-in entropy", fontsize=7.8)
    a.tick_params(labelsize=7.2)
    a.grid(color="#EEEEEE", lw=0.6); a.set_axisbelow(True)
    label(ax, 0.63, 0.35, "Both samples are drawn from the same length\ndistribution over K = 1,000 lengths.\n"
          "The sample with few fragments (blue) shows\nlower entropy only because N is small.\n"
          "Subsampled to the same N (open circle), the\nlarger sample (orange) gives the same value.",
          fontsize=7.4, ha="left", va="center", linespacing=1.4)
    box(ax, 0.615, 0.02, 0.373, 0.14, "", fc="white", ec=LINE)
    label(ax, 0.627, 0.135, "In the cohort", fontsize=8.4, fontweight="bold", ha="left")
    label(ax, 0.627, 0.075, "ρ(N, plug-in entropy) = 0.8–1.0 in sparse\ncells, and HCC samples have about four\ntimes as many fragments (Section 3.2.1).",
          fontsize=7.4, ha="left", va="center", linespacing=1.35)
    save(fig, "fig3_stats_flow.png")



# =====================================================================================
# Figure 2.6  Fragment size from paired-end bisulfite reads (Methods 2.13)
# =====================================================================================
def bases(ax, x, y, seq, cols=None, fs=8.0, dx=0.019, weight="normal"):
    for i, b in enumerate(seq):
        c = (cols or {}).get(i, INK)
        ax.text(x + i * dx, y, b, fontsize=fs, family="DejaVu Sans Mono", color=c, ha="center", va="center", fontweight=weight)


def fig_bisulfite():
    fig, ax = canvas(8.6, 5.6)
    # A: conversion
    head(ax, 0.012, 0.965, "A", "Bisulfite changes the sequence")
    seq = "ACGTTCAGCCTA"
    label(ax, 0.03, 0.87, "plasma DNA", fontsize=7.8, color=MUTE, ha="left")
    bases(ax, 0.15, 0.87, seq, cols={1: TEAL, 5: RED, 8: RED, 9: RED})
    label(ax, 0.15 + 0.019 * 1, 0.905, "me", fontsize=6.4, color=TEAL)
    arrow(ax, 0.26, 0.845, 0.26, 0.78, lw=1.1, ms=8)
    label(ax, 0.275, 0.812, "bisulfite + PCR", fontsize=7.2, color=MUTE, ha="left")
    label(ax, 0.03, 0.75, "after conversion", fontsize=7.8, color=MUTE, ha="left")
    conv = "ACGTTTAGTTTA"
    bases(ax, 0.15, 0.75, conv, cols={1: TEAL, 5: RED, 8: RED, 9: RED})
    label(ax, 0.03, 0.68, "unmethylated C reads as T (red);\nmethylated CpG C stays C (green)", fontsize=7.4, ha="left", va="center", linespacing=1.3)

    # B: paired reads and converted reference
    head(ax, 0.40, 0.965, "B", "Align to a converted reference")
    xL, xR = 0.43, 0.97
    yref1, yref2 = 0.86, 0.80
    for yy, nm, c in [(yref1, "f_ contigs: every C → T", NAVY), (yref2, "r_ contigs: every G → A", NAVY_M)]:
        ax.plot([xL, xR], [yy, yy], color=c, lw=3.0, solid_capstyle="butt", alpha=0.85)
        label(ax, xR, yy + 0.022, nm, fontsize=7.3, color=c, ha="right")
    label(ax, xL, yref1 + 0.022, "GRCh38, chr1–8", fontsize=7.3, color=MUTE, ha="left")
    yr = 0.71
    fa, fb = 0.53, 0.83
    ax.plot([fa, fb], [yr - 0.04, yr - 0.04], color="#B8BEC7", lw=5, solid_capstyle="butt", alpha=0.6)
    label(ax, (fa + fb) / 2, yr - 0.07, "the cfDNA fragment (not read in full)", fontsize=7.2, color=MUTE)
    arrow(ax, fa, yr, fa + 0.10, yr, color=HCC, lw=2.2, ms=9)
    arrow(ax, fb, yr - 0.015, fb - 0.10, yr - 0.015, color=HBV, lw=2.2, ms=9)
    label(ax, fa, yr + 0.03, "read 1: C→T in silico", fontsize=7.3, color="#B9532A", ha="left")
    label(ax, fb, yr + 0.015, "read 2: G→A", fontsize=7.3, color=HBV, ha="right")
    label(ax, 0.43, 0.57, "BWA-MEM aligns the converted pair; pairs whose read 1\nlies on the strand that disagrees with its contig are discarded.",
          fontsize=7.4, ha="left", va="center", linespacing=1.3)

    # C: TLEN
    head(ax, 0.012, 0.47, "C", "Template length = fragment size")
    y0 = 0.36
    ax.plot([0.05, 0.37], [y0, y0], color="#9AA1AB", lw=1.4)
    a0, a1 = 0.09, 0.33
    arrow(ax, a0, y0 + 0.03, a0 + 0.07, y0 + 0.03, color=HCC, lw=2.2, ms=9)
    arrow(ax, a1, y0 + 0.03, a1 - 0.07, y0 + 0.03, color=HBV, lw=2.2, ms=9)
    label(ax, a0 + 0.035, y0 + 0.058, "read 1", fontsize=7.0, color="#B9532A")
    label(ax, a1 - 0.035, y0 + 0.058, "read 2", fontsize=7.0, color=HBV)
    ax.plot([a0, a0], [y0 - 0.06, y0 + 0.05], color=INK, lw=0.8, ls=":")
    ax.plot([a1, a1], [y0 - 0.06, y0 + 0.05], color=INK, lw=0.8, ls=":")
    arrow(ax, a0, y0 - 0.05, a1, y0 - 0.05, style="<|-|>", lw=1.0, ms=8, color=INK)
    label(ax, (a0 + a1) / 2, y0 - 0.085, "|TLEN| = outer end of read 1 to outer end of read 2", fontsize=7.4)
    label(ax, 0.03, 0.17, "Counted: proper, primary pairs with MAPQ ≥ 20,\n|TLEN| from 1 to 1,000 bp. Only sizes are used:\nconversion changes end motifs, so motifs and\nentropy were not analysed in bisulfite data.",
          fontsize=7.4, ha="left", va="center", linespacing=1.35)

    # D: R_sr on a schematic length profile
    head(ax, 0.42, 0.47, "D", "Short-read size ratio R_sr")
    a = fig.add_axes([0.47, 0.13, 0.30, 0.25])
    xx = np.linspace(50, 1000, 900)
    g = lambda m, s, h: h * np.exp(-0.5 * ((xx - m) / s) ** 2)
    prof = g(167, 18, 1.0) + g(334, 26, 0.16) + g(501, 32, 0.05) + 0.01 * np.exp(-xx / 400)
    a.fill_between(xx, prof, where=(xx >= 100) & (xx <= 250), color=TEAL_L, lw=0)
    a.fill_between(xx, prof, where=(xx >= 256) & (xx <= 600), color=HCC_L, lw=0)
    a.plot(xx, prof, color=NAVY, lw=1.2)
    a.set_xlim(50, 1000); a.set_ylim(0, 1.15); a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xlabel("|TLEN| (bp), schematic profile", fontsize=7.6); a.tick_params(labelsize=7.2)
    a.text(175, 1.05, "100–250", fontsize=7.2, color=TEAL, ha="center")
    a.text(430, 0.24, "256–600", fontsize=7.2, color="#B9532A", ha="center")
    label(ax, 0.79, 0.33, "R_sr =", fontsize=8.6, fontweight="bold", ha="left")
    label(ax, 0.9, 0.355, "fragments 256–600 bp", fontsize=7.4, color="#B9532A")
    ax.plot([0.845, 0.955], [0.33, 0.33], color=INK, lw=0.8)
    label(ax, 0.9, 0.305, "fragments 100–250 bp", fontsize=7.4, color=TEAL)
    label(ax, 0.79, 0.19, "Fixed before any external\ndata were seen; predicted\nhigher in HCC. Stand-in for\nthe long-read R (256–650 bp\nover > 2 kb), which short\nreads cannot measure.",
          fontsize=7.2, ha="left", va="center", linespacing=1.3, color=INK)
    label(ax, 0.012, 0.015, "GSE71378 is not bisulfite-treated: the same |TLEN| count is taken after ordinary alignment. CRA001537: first 5,000,000 pairs per sample.",
          fontsize=7.0, color=MUTE, ha="left", va="bottom")
    save(fig, "fig3_bisulfite_size.png")


# figure number in the thesis -> (function, output file)
FIGURES = {
    "2.2": (fig_design, "fig3_design.png"),
    "2.3": (fig_resolution, "fig3_resolution.png"),
    "2.4": (fig_stats, "fig3_stats_flow.png"),
    "2.6": (fig_bisulfite, "fig3_bisulfite_size.png"),
}


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help="thesis figure numbers to draw, e.g. 2.2 (default: all)")
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
