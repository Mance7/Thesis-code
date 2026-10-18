#!/usr/bin/env python3
"""Draw the entropy forest plot and the sample-level overview heatmap.

Figures:
  Figure 3.18  fig4_entropy_forest.png   Cohen's d of rarefied entropy, 1001-2000 bp (H1) and 151-255 bp (H2), six regions,
                                         ONT and PacBio, all and direct fragments, exact 95% CI from the noncentral t
  Figure 3.24  fig4_sample_heatmap.png   14 samples x key metrics (size, entropy, end motif, tumour fraction, depth), z-scored,
                                         ordered by R on ONT direct fragments

Inputs: the ONT entropy rarefaction test tables (all and direct), the per-sample tumour fraction / R table,
the per-sample periodicity table and the literature-motif feature tables (F_motif_lit.csv) of both platforms.
PacBio entropy d and q (Section 3.7.3) and the ONT concatemer gain per sample (Table 2.1) are typed in.
Output: PNG files (300 dpi) in the --out folder.

Usage:
  python 10_plot_entropy_forest_sample_heatmap.py \
      --entropy-tests-all results/fig_inputs/entropy_rarefaction_tests_all.csv \
      --entropy-tests-direct results/fig_inputs/entropy_rarefaction_tests_direct.csv \
      --tf-table results/tumour_fraction/tf_vs_R_per_sample.tsv --periodicity results/length_hist/periodicity_per_sample.tsv \
      --ont-feats results/features/ont --pacbio-feats results/features/pacbio -o figures
  (put figure numbers first, e.g. 3.24, to draw only those)
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Ellipse, Wedge, Arc
from matplotlib.lines import Line2D

OUT = None   # output folder, set from --out in main()
A = None     # command-line arguments (input tables), set in main()

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
# Figure 3.18  Forest plot of the confirmatory entropy cells (ONT: Table 3.2; PacBio: Section 3.7.3)
# =====================================================================================
REG = [("genome_wide", "genome-wide"), ("is_gene_body", "gene body"), ("is_exon", "exon"),
       ("is_promoter", "promoter"), ("is_tss", "TSS"), ("is_cpg_island", "CpG island")]
# PacBio rarefied d (BH q), all / direct, 1001-2000 bp, typed from the PacBio entropy results (Section 3.7.3)
PB_H1 = {"genome_wide": ((-1.80, 0.032), (-1.80, 0.013)), "is_gene_body": ((-1.79, 0.020), (-1.80, 0.023)),
         "is_exon": ((-1.66, 0.020), (-1.71, 0.010)), "is_promoter": ((-1.89, 0.020), (-2.09, 0.010)),
         "is_cpg_island": ((-2.07, 0.020), (-1.95, 0.010)), "is_tss": ((-1.44, 0.080), (-1.66, 0.060))}
PB_H2_RANGE = (1.00, 1.24)   # all six regions, p 0.06-0.14, padj 0.20-0.30; per-region values not in the results files


def d_ci(d, n1=8, n2=6, a=0.05):
    """Exact 95% CI of Cohen's d from the noncentral t distribution (needs only d and the group sizes)."""
    from scipy.stats import nct
    from scipy.optimize import brentq
    f = np.sqrt(n1 * n2 / (n1 + n2)); t = d * f; df = n1 + n2 - 2
    lo = brentq(lambda x: nct.sf(t, df, x) - a / 2, t - 10, t + 10)
    hi = brentq(lambda x: (1 - nct.sf(t, df, x)) - a / 2, t - 10, t + 10)
    return lo / f, hi / f


def fig_forest():
    import matplotlib.transforms as mtrans
    ont = {"all": pd.read_csv(A.entropy_tests_all), "direct": pd.read_csv(A.entropy_tests_direct)}
    def get(run, sb, reg):
        d = ont[run]; r = d[(d.size_bin == sb) & (d.region == reg)].iloc[0]
        return r.d_H_rare, r.padj_H_rare
    fig = plt.figure(figsize=(7.6, 6.6))
    a = fig.add_axes([0.20, 0.17, 0.50, 0.70])
    tr = mtrans.blended_transform_factory(fig.transFigure, a.transData)
    groups = [("1001-2000bp", "1001–2000 bp (H1)", ["genome_wide", "is_gene_body"]),
              ("151-255bp", "151–255 bp (H2)", [r for r, _ in REG])]
    y = 0.0; ticks, tlab, tcol = [], [], []
    hdr = []
    for sb, title, conf in groups:
        hdr.append((y, title)); y -= 0.9
        for reg, nm in REG:
            dA, qA = get("all", sb, reg); dD, qD = get("direct", sb, reg)
            yo, yp = y + 0.18, y - 0.18
            a.plot([dA], [yo], "o", mfc="white", mec=ONT, ms=5.5, mew=1.1, zorder=3)
            a.plot(d_ci(dD), [yo] * 2, color=ONT, lw=1.1, solid_capstyle="butt", zorder=2)
            a.plot([dD], [yo], "o", color=ONT, ms=5.5, mec="white", mew=0.6, zorder=4)
            fig.text(0.745, yo, f"{qD:.3f}", transform=tr, fontsize=7.4, va="center", ha="center", color=ONT)
            if sb == "1001-2000bp":
                (pA, pqA), (pD, pqD) = PB_H1[reg]
                a.plot([pA], [yp], "s", mfc="white", mec=PB, ms=5.2, mew=1.1, zorder=3)
                a.plot(d_ci(pD), [yp] * 2, color=PB, lw=1.1, solid_capstyle="butt", zorder=2)
                a.plot([pD], [yp], "s", color=PB, ms=5.2, mec="white", mew=0.6, zorder=4)
                fig.text(0.86, yp, f"{pqD:.3f}", transform=tr, fontsize=7.4, va="center", ha="center", color=PB)
            else:
                a.plot(list(PB_H2_RANGE), [yp] * 2, color=PB, lw=3.4, alpha=0.5, solid_capstyle="butt", zorder=2)
                fig.text(0.86, yp, "0.20–0.30", transform=tr, fontsize=7.4, va="center", ha="center", color=PB)
            ticks.append(y); tlab.append(nm + ("" if reg in conf else " †")); tcol.append(INK if reg in conf else MUTE)
            y -= 1.0
        y -= 0.3
    for yy, t in hdr:
        fig.text(0.012, yy, t, transform=tr, fontsize=9.0, fontweight="bold", ha="left", va="center", color=INK)
        a.axhline(yy - 0.45, color="#E6E8EC", lw=0.8, zorder=0)
    a.set_yticks(ticks, tlab, fontsize=8.2)
    for t, c in zip(a.get_yticklabels(), tcol):
        t.set_color(c)
    a.axvline(0, color="#888888", lw=0.9, ls="--", zorder=1)
    a.set_xlim(-5.9, 4.4); a.set_ylim(y + 0.5, 0.5)
    a.axvspan(-5.9, 0, color=HBV, alpha=0.035, lw=0); a.axvspan(0, 4.4, color=HCC, alpha=0.045, lw=0)
    a.text(-2.95, 0.55, "← lower in HCC", fontsize=7.6, color=MUTE, ha="center", va="bottom", clip_on=False)
    a.text(2.2, 0.55, "higher in HCC →", fontsize=7.6, color=MUTE, ha="center", va="bottom", clip_on=False)
    a.set_xlabel("Cohen's d (HCC − HBV), rarefied entropy", fontsize=8.6)
    a.grid(axis="x", color="#EEEEEE", lw=0.6); a.set_axisbelow(True)
    a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    fig.text(0.745, 0.885, "q, ONT\ndirect", fontsize=7.6, color=ONT, ha="center", va="bottom", fontweight="bold", linespacing=1.2)
    fig.text(0.86, 0.885, "q, PacBio\ndirect", fontsize=7.6, color=PB, ha="center", va="bottom", fontweight="bold", linespacing=1.2)
    hs = [Line2D([], [], marker="o", ls="", color=ONT, ms=5.5, label="ONT, direct"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=ONT, mew=1.1, ms=5.5, label="ONT, all fragments"),
          Line2D([], [], marker="s", ls="", color=PB, ms=5.2, label="PacBio, direct"),
          Line2D([], [], marker="s", ls="", mfc="white", mec=PB, mew=1.1, ms=5.2, label="PacBio, all"),
          Line2D([], [], color=PB, lw=3.4, alpha=0.5, label="PacBio, range over six regions")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=3, fontsize=7.8, handletextpad=0.4,
               columnspacing=1.4)
    fig.text(0.012, 0.012, "Lines: exact 95% CI of d for direct fragments (noncentral t, 8 vs 6). † exploratory cell (not in the registry).\n"
             "q: BH-adjusted p as reported (Table 3.2, Section 3.7.3). PacBio 151–255 bp: only the range over the six regions\n"
             "is reported (d +1.00 to +1.24, p 0.06–0.14).", fontsize=7.0, color=MUTE, ha="left", va="bottom", linespacing=1.4)
    save(fig, "fig4_entropy_forest.png")



# =====================================================================================
# Figure 3.24  Sample-level overview heatmap
# =====================================================================================
def heat_table():
    """collect the per-sample rows of the heatmap: (block, label, values per sample, log10 before z-scoring?)"""
    tf = pd.read_csv(A.tf_table, sep="\t").set_index("sample")
    per = pd.read_csv(A.periodicity, sep="\t")
    def pv(run, col):
        return per[per.run == run].set_index("sample")[col]
    lit = {p: pd.read_csv(os.path.join(f, "direct", "F_motif_lit.csv")).set_index("sample")
           for p, f in (("ONT", A.ont_feats), ("PacBio", A.pacbio_feats))}
    def cend(p):
        d = lit[p]; return d[[f"clr_601-2000bp_{m}" for m in ("CCCA", "CCAG", "CCTG")]].mean(axis=1)
    rec = {"TBR2776": 27.5, "TBR2872": 60.6, "TBR2897": 17.9, "TBR2967": 52.1, "TBR3013": 17.8, "TBR3032": 23.9,
           "TBR3140": 12.3, "TBR3976": 75.8, "TBR4479": 4.7, "TBR4480": 6.0, "TBR4483": 6.8, "TBR4485": 6.5,
           "TBR4486": 9.6, "TBR4488": 4.6}   # Table 2.1, gain = recovered / direct (%)
    rows = [  # (block, label, series, log10?)
        ("Size", "R, ONT direct (log)", tf["R_ont_direct"], True),
        ("Size", "R, PacBio (log)", tf["R_pacbio"], True),
        ("Size", "1001–1250 bp share of 1–2 kb, ONT", pv("ont_direct", "frac_1001_1250"), False),
        ("Size", "1001–1250 bp share of 1–2 kb, PacBio", pv("pacbio_direct", "frac_1001_1250"), False),
        ("Entropy", "1001–2000 bp entropy, ONT", pv("ont_direct", "H_rare"), False),
        ("Entropy", "1001–2000 bp entropy, PacBio", pv("pacbio_direct", "H_rare"), False),
        ("End motif", "C-end CLR 601–2000 bp, ONT", cend("ONT"), False),
        ("End motif", "C-end CLR 601–2000 bp, PacBio", cend("PacBio"), False),
        ("Tumour", "tumour fraction, ONT lowTF", tf["TF_ont_lowTF"], False),
        ("Tumour", "tumour fraction, PacBio std", tf["TF_pacbio_std"], False),
        ("Depth", "direct fragments, ONT (log)", tf["ont_direct_fragments"], True),
        ("Depth", "fragments, PacBio (log)", tf["pacbio_fragments"], True),
        ("Depth", "concatemer gain, ONT (%)", pd.Series(rec), False),
    ]
    return tf, rows


def fig_heatmap():
    tf, rows = heat_table()
    order = tf.sort_values("R_ont_direct", ascending=False).index.tolist()
    M = []
    for blk, lab, s, lg in rows:
        v = s.reindex(order).astype(float).values
        if lg:
            v = np.log10(v)
        M.append((v - v.mean()) / v.std(ddof=1))
    M = np.array(M)
    grp = tf.reindex(order)["group"].values
    fig = plt.figure(figsize=(8.0, 6.0))
    a = fig.add_axes([0.335, 0.17, 0.53, 0.66])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#2B5C8A", "#9DBBE3", "#F7F7F7", "#F2B08C", "#B9532A"])
    im = a.imshow(M, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -2.5, 2.5), interpolation="nearest")
    a.set_xticks(range(len(order)), order, rotation=90, fontsize=7.6)
    a.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=7.8)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks(np.arange(-0.5, len(order)), minor=True); a.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    a.grid(which="minor", color="white", lw=1.2); a.tick_params(which="minor", length=0)
    # block separators and labels
    blocks = [r[0] for r in rows]
    starts = [i for i in range(len(blocks)) if i == 0 or blocks[i] != blocks[i - 1]]
    for k, st in enumerate(starts):
        en = (starts[k + 1] if k + 1 < len(starts) else len(blocks)) - 1
        if st > 0:
            a.axhline(st - 0.5, color="white", lw=3.5)
        a.text(-0.60, (st + en) / 2, blocks[st], transform=a.get_yaxis_transform(), fontsize=8.0, fontweight="bold",
               color=MUTE, ha="left", va="center")
        a.plot([-0.615, -0.615], [st - 0.35, en + 0.35], transform=a.get_yaxis_transform(), color=LINE, lw=1.0, clip_on=False)
    nh = int((grp == "HCC").sum())
    a.axvline(nh - 0.5, color="white", lw=3.5)
    # group bar
    ab = fig.add_axes([0.335, 0.845, 0.53, 0.03])
    ab.imshow([[0 if g == "HCC" else 1 for g in grp]], aspect="auto",
              cmap=LinearSegmentedColormap.from_list("g", [HCC, HBV]), vmin=0, vmax=1)
    ab.set_xticks([]); ab.set_yticks([0], ["group"], fontsize=7.8); ab.tick_params(length=0)
    for s in ab.spines.values():
        s.set_visible(False)
    ab.set_xticks(np.arange(-0.5, len(order)), minor=True); ab.grid(which="minor", axis="x", color="white", lw=1.2)
    ab.tick_params(which="minor", length=0)
    fig.text(0.335, 0.905, "■", color=HCC, fontsize=10, va="center"); fig.text(0.355, 0.905, "HCC (n = 8)", fontsize=8, va="center")
    fig.text(0.47, 0.905, "■", color=HBV, fontsize=10, va="center"); fig.text(0.49, 0.905, "HBV carrier (n = 6)", fontsize=8, va="center")
    fig.text(0.012, 0.96, "14 samples × key metrics, ordered by the size ratio R (ONT direct); rows z-scored",
             fontsize=9.2, fontweight="bold", ha="left")
    # colorbar
    ac = fig.add_axes([0.905, 0.35, 0.018, 0.30])
    cb = fig.colorbar(im, cax=ac); cb.set_ticks([-2, -1, 0, 1, 2]); cb.ax.tick_params(labelsize=7.2, length=2)
    cb.outline.set_visible(False)
    ac.set_title("z", fontsize=8, loc="center")
    fig.text(0.012, 0.012, "log10 before z-scoring for R and fragment counts. C-end CLR = mean CLR of CCCA, CCAG and CCTG. Entropy = rarefied, from 1-bp histograms.\n"
             "Promoter fsr is not shown: no per-sample values exist in the results files.", fontsize=6.9, color=MUTE, ha="left", va="bottom", linespacing=1.4)
    save(fig, "fig4_sample_heatmap.png")


# figure number in the thesis -> (function, output file)
FIGURES = {
    "3.18": (fig_forest, "fig4_entropy_forest.png"),
    "3.24": (fig_heatmap, "fig4_sample_heatmap.png"),
}


def main():
    global OUT, A
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help="thesis figure numbers to draw, e.g. 3.18 (default: all)")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder for the PNG files (created if missing)")
    ap.add_argument("--entropy-tests-all", required=True,
                    help="ONT entropy tests per size bin and region, all fragments (csv: size_bin, region, d_H_rare, padj_H_rare)")
    ap.add_argument("--entropy-tests-direct", required=True,
                    help="ONT entropy tests per size bin and region, direct fragments (same columns)")
    ap.add_argument("--tf-table", required=True,
                    help="per-sample tumour fraction, R and fragment counts (tsv)")
    ap.add_argument("--periodicity", required=True,
                    help="per-sample periodicity table (tsv: sample, run, frac_1001_1250, H_rare, ...)")
    ap.add_argument("--ont-feats", required=True,
                    help="ONT per-sample feature folder with a direct/ subfolder (F_motif_lit.csv)")
    ap.add_argument("--pacbio-feats", required=True,
                    help="PacBio per-sample feature folder with a direct/ subfolder (F_motif_lit.csv)")
    a = ap.parse_args()
    A = a
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    for k in (a.figures or FIGURES):
        fn, png = FIGURES[k]
        fn()
        print("made Figure", k, "->", os.path.join(OUT, png))


if __name__ == "__main__":
    main()
