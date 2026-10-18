#!/usr/bin/env python3
"""Draw the data-quality, Hallmark and classifier summary figures of Chapter 3.

Figures:
  Figure 3.1   fig4_qc.png           ONT fragments per sample (direct + recovered), concatemer recovery on ONT and PacBio
  Figure 3.13  fig4_hallmark.png     the four Hallmark programmes that carry more of the HCC-HBV promoter difference
  Figure 3.15  fig4_classifiers.png  leave-one-out AUC of the pre-specified feature sets, with and without depth control

No input files: the values are typed in from the thesis tables (Table 2.1 fragment counts,
Section 3.7.1 PacBio recovery, Table 3.5 Hallmark p and q, Tables 3.6, 3.8 and 3.10 classifier AUCs).
Output: PNG files (300 dpi) in the --out folder.

Usage:
  python 04_plot_qc_hallmark_classifier_summaries.py -o figures
  python 04_plot_qc_hallmark_classifier_summaries.py 3.15 -o figures
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
# Figure 3.1  Data quality (eligible ONT fragments in millions, direct and recovered: Table 2.1)
# =====================================================================================
def fig_qc():
    rows = [("TBR2776", "HCC", 40.626740, 11.176450), ("TBR2872", "HCC", 26.463092, 16.035302), ("TBR2897", "HCC", 45.636160, 8.151294),
            ("TBR2967", "HCC", 13.362442, 6.955466), ("TBR3013", "HCC", 12.125248, 2.152920), ("TBR3032", "HCC", 45.750447, 10.947464),
            ("TBR3140", "HCC", 7.139962, 0.880850), ("TBR3976", "HCC", 6.622827, 5.018386),
            ("TBR4479", "HBV", 2.218524, 0.103694), ("TBR4480", "HBV", 5.930470, 0.354496), ("TBR4483", "HBV", 8.974624, 0.613794),
            ("TBR4485", "HBV", 13.738180, 0.889888), ("TBR4486", "HBV", 7.877944, 0.758674), ("TBR4488", "HBV", 2.235913, 0.102986)]
    df = pd.DataFrame(rows, columns=["s", "g", "direct", "rec"])
    df["gain"] = 100 * df.rec / df.direct; df["tot"] = df.direct + df.rec
    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 1.0], wspace=0.5)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])
    d1 = df.sort_values(["g", "tot"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(d1))[::-1]
    col = d1.g.map({"HCC": HCC, "HBV": HBV})
    a.barh(y, d1.direct, color=col, height=0.68, zorder=3)
    a.barh(y, d1.rec, left=d1.direct, color=col, alpha=0.40, height=0.68, zorder=3)
    a.set_yticks(y); a.set_yticklabels(d1.s, fontsize=8); a.set_xlabel("eligible fragments (millions)")
    a.set_title("A  Fragments per sample, ONT", fontsize=9.8)
    a.legend(handles=[Rectangle((0, 0), 1, 1, fc="#777777", label="direct"), Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.4, label="recovered from concatemers")],
             loc="lower right", fontsize=8)
    a.grid(axis="x", color="#E6E6E6", lw=0.6); a.set_axisbelow(True)
    a.text(0.98, 0.93, "median 31.4 M (HCC)\nvs 7.5 M (HBV); p = 0.013", transform=a.transAxes, ha="right", va="top", fontsize=8, color=MUTE)
    rng = np.random.default_rng(2)
    for k, (g, cc) in enumerate([("HBV", HBV), ("HCC", HCC)]):
        v = df[df.g == g].gain.values
        b.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=36, color=cc, edgecolor="white", lw=0.6, zorder=3)
        b.hlines(np.median(v), k - 0.26, k + 0.26, color="#333333", lw=1.5, zorder=4)
    b.set_xticks([0, 1], ["HBV\n(n = 6)", "HCC\n(n = 8)"]); b.set_xlim(-0.6, 1.6)
    b.set_ylabel("fragments gained by recovery (%)"); b.set_title("B  Concatemer recovery, ONT", fontsize=9.8)
    b.text(0.5, 0.95, "every HCC above every HBV\np = 6.7 × 10⁻⁴ (smallest possible)", transform=b.transAxes, ha="center", va="top", fontsize=8, color=MUTE)
    b.grid(axis="y", color="#E6E6E6", lw=0.6); b.set_axisbelow(True)
    # C: PacBio, same plasma (medians and ranges from Section 3.7.1)
    for k, (g, cc, med, lo_, hi_) in enumerate([("HBV", HBV, 0.72, 0.40, 1.33), ("HCC", HCC, 1.25, 0.40, 2.42)]):
        c.bar(k, med, color=cc, width=0.55, zorder=3)
        c.vlines(k, lo_, hi_, color="#333333", lw=1.4, zorder=4)
    c.set_ylim(0, 3.6); c.set_ylabel("fragments gained by recovery (%)", fontsize=8.5); c.set_xticks([0, 1], ["HBV", "HCC"]); c.set_xlim(-0.6, 1.6)
    c.set_title("C  Recovery, PacBio", fontsize=9.8)
    c.text(0.5, 0.97, "bars: median, lines: range;\nabout 1% in both groups (p = 0.091):\nthe ONT gap is a library effect", transform=c.transAxes, ha="center", va="top", fontsize=7.8, color=MUTE)
    c.grid(axis="y", color="#E6E6E6", lw=0.6); c.set_axisbelow(True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.14)
    fig.savefig(f"{OUT}/fig4_qc.png", dpi=300)



# =====================================================================================
# Figure 3.13  Hallmark programmes (Table 3.5); per row: name, metric, hepatocyte or other, (p, q) ONT, (p, q) PacBio
# =====================================================================================
def fig_hallmark():
    rows = [("Bile acid metabolism (97)", "short-fragment ratio", "hep", (.002, .017), (.023, .29)),
            ("Peroxisome (89)", "length entropy", "hep", (.002, .020), (.001, .033)),
            ("KRAS signalling, down (60)", "length entropy", "oth", (.0007, .017), (.005, .047)),
            ("Interferon-α response (92)", "length entropy", "oth", (.002, .020), (.005, .047))]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(rows))[::-1]
    for yy, (name, met, kind, o, p) in zip(y, rows):
        ax.plot([-np.log10(o[0]), -np.log10(p[0])], [yy, yy], color="#C9CED6", lw=2, zorder=1)
        for (pv, q), c, m in [(o, ONT, "o"), (p, PB, "s")]:
            ax.scatter(-np.log10(pv), yy, s=95, marker=m, zorder=3, facecolor=(c if q < 0.05 else "white"), edgecolor=c, lw=1.6)
            ax.text(-np.log10(pv), yy + 0.27, f"q {q:.3f}".rstrip("0") if q >= 0.1 else f"q {q:.3f}", fontsize=7.2, color=MUTE, ha="center", va="bottom")
        ax.text(-0.02, yy + 0.12, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, fontweight="bold", color=(HCC if kind == "hep" else INK))
        ax.text(-0.02, yy - 0.20, met, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=MUTE)
    ax.axvline(-np.log10(0.05), color="#999999", lw=0.9, ls=(0, (4, 3))); ax.text(-np.log10(0.05) + 0.03, len(rows) - 0.42, "p = 0.05", fontsize=7.6, color=MUTE)
    ax.set_xlim(1.0, 3.7); ax.set_ylim(-0.7, len(rows) - 0.25); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("−log10 p  (3,003 sample relabellings; larger = stronger)")
    ax.grid(axis="x", color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=ONT, label="ONT (direct), filled: q < 0.05", markersize=8),
                       Line2D([0], [0], marker="s", ls="", color=PB, label="PacBio (direct), filled: q < 0.05", markersize=8),
                       Line2D([0], [0], marker="s", ls="", mfc="white", mec=PB, mew=1.6, color=PB, label="hollow: q ≥ 0.05", markersize=8)],
              loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.45, -0.17))
    ax.set_title("Hallmark programmes that carry more of the HCC–HBV promoter difference than other HCC-expressed genes", fontsize=9.3, loc="left", x=-0.38)
    ax.text(3.7, len(rows) - 0.42, "orange = hepatocyte programmes", color=HCC, fontsize=8, ha="right", fontweight="bold")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.25)
    fig.savefig(f"{OUT}/fig4_hallmark.png", dpi=300)



# =====================================================================================
# Figure 3.15  Classifier AUC (Tables 3.6, 3.8 and 3.10)
# =====================================================================================
def fig_classifiers():
    sets = ["F-ratio (log R)", "F-size", "F-entropy", "F-motif", "F-motif-lit", "F-combined"]
    # (AUC, p or q); None = not run
    data = {
        "Unadjusted\nONT, direct fragments": [None, (1.00, .002), (.94, .014), (1.00, .002), (.77, .076), (.88, .027)],
        "Adjusted for depth\nONT, direct fragments": [None, (.77, .078), (.73, .10), (.69, .12), (.56, .33), (.85, .037)],
        "Equal depth\nONT, direct fragments": [(1.00, .012), (1.00, .012), (.98, .014), (1.00, .012), (.69, .13), (.94, .016)],
        # PacBio: prediction only. ONT-direct models locked and applied unchanged; R = ONT direction, no fit.
        # "p" here = 0.01 if the bootstrap 95% CI lower bound is above 0.5, else 1 (significance marker only)
        "ONT models applied to\nPacBio, direct (no training)": [(1.00, .01), (.98, .01), (.38, 1), (.94, .01), (1.00, .01), (.98, .01)],
    }
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 4.6), sharey=True)
    y = np.arange(len(sets))[::-1]
    for ax, (t, vals), L in zip(axes, data.items(), "ABCD"):
        for yy, v in zip(y, vals):
            if v is None:
                ax.text(0.52, yy, "not run", fontsize=7.5, color=MUTE, va="center", ha="left"); continue
            auc, p = v; sig = p < 0.05
            ax.barh(yy, auc - 0.5, left=0.5, height=0.62, color=(TEAL if sig else "#FFFFFF"), ec=TEAL, lw=1.3, zorder=3)
            ax.text(min(max(auc, 0.5) + 0.012, 1.08), yy, f"{auc:.2f}", va="center", fontsize=8, color=INK, zorder=7, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
        ax.axvline(0.5, color="#888888", lw=0.8, zorder=2)
        ax.set_xlim(0.3, 1.12); ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0]); ax.tick_params(labelsize=8)
        ax.set_title(f"{L}  {t}", fontsize=9, loc="left", linespacing=1.35)
        ax.set_xlabel("AUC (leave-one-out)", fontsize=8.5)
        ax.grid(axis="x", color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    axes[3].set_xlabel("AUC on PacBio (model not refitted)", fontsize=8.5)
    axes[0].axvline(0.94, color=ONT, lw=1.2, ls=(0, (4, 3)), zorder=5)
    axes[0].text(0.935, len(sets) - 0.45, "fragment count\n+ recovery alone: 0.94", fontsize=7.6, color=ONT, ha="right", va="center")
    axes[0].set_yticks(y); axes[0].set_yticklabels(sets, fontsize=8.5)
    axes[0].set_ylim(-0.6, len(sets) - 0.1 + 0.55)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=TEAL, ec=TEAL, label="significant (A-C: permutation p or q < 0.05; D: bootstrap 95% CI above 0.5)"),
                        Rectangle((0, 0), 1, 1, fc="white", ec=TEAL, lw=1.3, label="not significant")],
               loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.2, wspace=0.12)
    fig.savefig(f"{OUT}/fig4_classifiers.png", dpi=300)


# figure number in the thesis -> (function, output file)
FIGURES = {
    "3.1": (fig_qc, "fig4_qc.png"),
    "3.13": (fig_hallmark, "fig4_hallmark.png"),
    "3.15": (fig_classifiers, "fig4_classifiers.png"),
}


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help="thesis figure numbers to draw, e.g. 3.1 (default: all)")
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
