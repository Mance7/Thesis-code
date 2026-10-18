#!/usr/bin/env python3
"""
05_explain_depth_adjustment.py
------------------------------
Draws Figure 2.5: depth adjustment by regression compared with equal-depth
downsampling (Methods 2.10), from the real ONT direct-fragment counts.

Row 1: share of fragments longer than 2 kb (removed by the adjustment).
Row 2: share of 256-650 bp fragments (weaker after adjustment, but survives).
Columns: (A, E) share against log10 total fragments N with the fitted line
x = a + b*log10 N and each sample's residual; (B, F) raw shares; (C, G) residuals;
(D, H) shares after every sample is downsampled without replacement
(multivariate hypergeometric) to the smallest sample's count. p-values are
two-sided Mann-Whitney U, HCC vs HBV. For illustration the line is fitted once on
all 14 samples; the classifiers fit it inside each training fold
(03_nested_cv_classifier.py --adjust).

Input: tab-separated per-sample fragment counts per size bin (columns sample,
size_bin, n_direct; bins named as in the 11-bin scheme, e.g. 256-420bp).
HBV carriers are the TBR44xx samples; all others are HCC.
Output: <out>/depth_adjustment_explained.png and a table of slopes and p-values
printed to the screen.

Usage:
  python 05_explain_depth_adjustment.py --counts ont_bin_counts_genomewide.tsv -o depth_adjust_explainer/
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu as mw  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--counts", required=True,
                help="path to the per-sample fragment counts per size bin, direct fragments (tsv)")
ap.add_argument("-o", "--out", required=True, help="output folder for the figure")
ap.add_argument("--seed", type=int, default=8, help="random seed of the downsampling (default 8)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

# counts: samples x size bins; N = total fragments per sample; HBV carriers are TBR44xx
d = pd.read_csv(A.counts, sep="\t")
C = d.pivot(index="sample", columns="size_bin", values="n_direct"); cols = list(C.columns)
N = C.sum(axis=1); L = np.log10(N); hcc = ~C.index.str.startswith("TBR44")
n = int(N.min()); rng = np.random.default_rng(A.seed)
# equal depth: every sample downsampled to the smallest total
D = pd.DataFrame([rng.multivariate_hypergeometric(C.loc[s].to_numpy(), n) for s in C.index], index=C.index, columns=cols)
feats = {">2 kb": ["2001-5000bp", ">5000bp"], "256–650 bp": ["256-420bp", "421-650bp"]}
HCC, HBV, INK, MUTE = "#E8703A", "#2F7BD8", "#1F2933", "#6B7280"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left"})
fig, ax = plt.subplots(2, 4, figsize=(13.5, 7.6), gridspec_kw=dict(width_ratios=[1.35, 0.8, 0.8, 0.8]))
def strip(a, v, title, ylab):
    for k, (m, c) in enumerate([(~hcc, HBV), (hcc, HCC)]):
        jit = np.random.default_rng(k).uniform(-0.12, 0.12, m.sum())
        a.scatter(k + jit, v[m], s=36, c=c, ec="white", lw=0.8, zorder=3)
        a.hlines(np.median(v[m]), k - 0.25, k + 0.25, color=INK, lw=1.4)
    p = mw(v[hcc], v[~hcc]).pvalue
    a.set_xticks([0, 1]); a.set_xticklabels(["HBV", "HCC"]); a.set_xlim(-0.6, 1.6)
    a.set_title(f"{title}\n    HCC vs HBV p = {p:.3f}", fontsize=9); a.set_ylabel(ylab)
    a.grid(axis="y", color="#E6E6E6", lw=0.6); a.set_axisbelow(True)
    return p

# one row per feature: fit, raw, residual and equal-depth panels
rows = []
for i, (lab, bins) in enumerate(feats.items()):
    x = 100 * C[bins].sum(axis=1) / N; xd = 100 * D[bins].sum(axis=1) / n
    b, a0 = np.polyfit(L, x, 1); fit = a0 + b * L; r = x - fit
    a = ax[i, 0]
    a.scatter(L[~hcc], x[~hcc], s=36, c=HBV, ec="white", lw=0.8, zorder=3, label="HBV (n=6)")
    a.scatter(L[hcc], x[hcc], s=36, c=HCC, ec="white", lw=0.8, zorder=3, label="HCC (n=8)")
    g = np.linspace(L.min() - 0.05, L.max() + 0.05, 50); a.plot(g, a0 + b * g, color=INK, lw=1.4)
    for s in C.index: a.plot([L[s], L[s]], [fit[s], x[s]], color=MUTE, lw=0.8, ls=":", zorder=2)
    a.set_xlabel("log10 total fragments, N"); a.set_ylabel(f"share of {lab} fragments (%)")
    a.set_title(f"{'AE'[i]}  Fit x = a + b·log10 N   (b = {b:+.1f} % per 10× fragments)\n    dotted line = residual r, the only part the classifier sees", fontsize=9)
    if i == 0: a.legend(frameon=False, fontsize=8)
    L4 = "BCD" if i == 0 else "FGH"
    p1 = strip(ax[i, 1], x, f"{L4[0]}  Raw share x", "share (%)")
    p2 = strip(ax[i, 2], r, f"{L4[1]}  After adjustment, r", "residual (% points)"); ax[i, 2].axhline(0, color=MUTE, lw=0.8)
    p3 = strip(ax[i, 3], xd, f"{L4[2]}  Equal depth ({n/1e6:.1f} M each)", "share (%)")
    rows.append((lab, round(b, 2), p1, p2, p3))
fig.tight_layout(h_pad=2.5); fig.savefig(f"{A.out}/depth_adjustment_explained.png", dpi=250)
print(pd.DataFrame(rows, columns=["feature", "slope", "p_raw", "p_adjusted", "p_equal_depth"]).to_string(index=False))
