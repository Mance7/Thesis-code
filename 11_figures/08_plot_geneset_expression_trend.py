#!/usr/bin/env python3
"""Gene-set promoter pools: how the HCC vs HBV change follows expression (bar panels + pool x feature grid).

Figure:
  Figure 3.12  fig4_geneset_trend.png   (a) fsr and (b) entropy <=600 bp change beyond matched random sets per pool,
                                        pools ordered S3 -> S4 -> S5 (HCC expression), then S2, S1, S6; (c) the same as a grid
Input: geneset_volcano_values.csv written by 07_plot_geneset_hallmark_volcanoes.py
       (x = signed change beyond matched random sets, sig = q). Layout after the expression-quantile panel of Esfahani et al. (2022).
Output: fig4_geneset_trend.png in the --out folder.

Usage: python 08_plot_geneset_expression_trend.py --values "results/figures/geneset_volcano_values.csv" -o "results/figures"
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ap = argparse.ArgumentParser(description="Gene-set expression trend figure (Figure 3.12).")
ap.add_argument("--values", required=True,
                help="gene-set values table from 07_plot_geneset_hallmark_volcanoes.py (csv: platform, run, set, metric, x, sig)")
ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
png = os.path.join(a.out, "fig4_geneset_trend.png")
v = pd.read_csv(a.values)
ORDER = ["S3_hcc_tertile_low", "S4_hcc_tertile_mid", "S5_hcc_tertile_high", "S2_hcc_blood_expressed", "S1_hcc_blood_silent", "S6_liver_enriched"]
LAB = ["S3\nHCC low", "S4\nHCC mid", "S5\nHCC high", "S2\nblood-\nexpressed", "S1\nblood-\nsilent", "S6\nliver-\nenriched"]
COLS = [("ONT", "all"), ("ONT", "direct"), ("PacBio", "all"), ("PacBio", "direct")]
# ggplot-like palette: red shades = ONT, green shades = PacBio (light = all, dark = direct)
COL = {("ONT", "all"): "#F8766D", ("ONT", "direct"): "#C9463D", ("PacBio", "all"): "#7CAE00", ("PacBio", "direct"): "#4E7A00"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 11})
def val(pf, run, s, m):
    """signed change and q for one platform, fragment set, pool and metric"""
    r = v[(v.platform == pf) & (v.run == run) & (v.set == s) & (v.metric == m)].iloc[0]
    return float(r.x), float(r.sig)

fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.18)
for k, (m, title) in enumerate([("fsr", "a  Short-fragment ratio (fsr, share ≤246 bp)"), ("entropy_short", "b  Length entropy (≤600 bp)")]):
    ax = fig.add_subplot(gs[0, k]); w = 0.2; xs = np.arange(len(ORDER))
    ax.axvspan(2.5, 5.5, color="#F2F2F2", zorder=0)
    for j, c in enumerate(COLS):
        vals = [val(*c, s, m) for s in ORDER]
        xx = xs + (j - 1.5) * w
        ax.bar(xx, [x for x, _ in vals], w, color=COL[c], label=f"{c[0]} {c[1]}", zorder=2, edgecolor="white", lw=0.4)
        for xi, (x, q) in zip(xx, vals):
            t = f"{x:+.2f}" + ("*" if q < 0.05 else "")
            ax.text(xi, x + (0.02 if x >= 0 else -0.02), t, ha="center", va="bottom" if x >= 0 else "top", rotation=90,
                    fontsize=6.6, fontweight="bold" if q < 0.05 else "normal", color="#222222")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=8)
    ax.set_ylim(-1.25, 0.75) if m == "fsr" else ax.set_ylim(-0.6, 0.65)
    ax.set_ylabel("HCC vs HBV change beyond matched\nrandom promoter sets (d units)")
    ax.set_title(title)
    ax.text(1, ax.get_ylim()[1] * 0.97, "HCC expression  →", ha="center", va="top", fontsize=8, color="#555555")
    ax.text(4, ax.get_ylim()[1] * 0.97, "blood / liver", ha="center", va="top", fontsize=8, color="#555555")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.515), fontsize=8.5)

# c: rows = promoter pool, columns = feature x platform/run (signed change)
ax = fig.add_subplot(gs[1, :])
cols = [(m, c) for m in ["fsr", "entropy_short"] for c in COLS]
M = np.array([[val(*c, s, m)[0] for (m, c) in cols] for s in ORDER])
Q = np.array([[val(*c, s, m)[1] for (m, c) in cols] for s in ORDER])
cmap = LinearSegmentedColormap.from_list("sup", ["#2C7BB6", "#F7F7F7", "#D7301F"])
im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:+.2f}" + ("*" if Q[i, j] < 0.05 else ""), ha="center", va="center", fontsize=8,
                fontweight="bold" if Q[i, j] < 0.05 else "normal")
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([s.replace("-\n", "-").replace("\n", " ") for s in LAB])
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c[0]}\n{c[1]}" for (m, c) in cols], fontsize=8)
ax.axvline(3.5, color="white", lw=3); ax.axhline(2.5, color="white", lw=3)
ax.text(1.5, -0.75, "fsr", ha="center", fontweight="bold"); ax.text(5.5, -0.75, "entropy (≤600 bp)", ha="center", fontweight="bold")
ax.set_title("c  Promoter pool × feature: signed change (blue = lower in HCC, red = higher; * q < 0.05)", pad=18)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("change beyond random sets (d)")
fig.savefig(png, bbox_inches="tight", facecolor="white")
print("wrote", png)
