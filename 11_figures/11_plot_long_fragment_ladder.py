#!/usr/bin/env python3
"""Draw the figure showing that the lower 1-2 kb length entropy in HCC comes from where fragments fall in the bin,
not from the nucleosome ladder.

Figure:
  Figure 3.21  fig4_long_fragment_ladder.png   direct fragments; top row ONT, bottom row PacBio
    A, D  group-mean length profile after removing the smooth trend (nucleosome ladder); dotted lines every 195 bp,
          grey band = 1001-2000 bp
    B, E  cumulative length distribution within 1001-2000 bp, one line per sample; dashed line at 1,250 bp
    C, F  rarefied entropy (1001-2000 bp) against the share of 1001-2000 bp fragments that are 1001-1250 bp

How the ladder is drawn: each sample's 1-bp histogram is turned into a share of all its fragments, smoothed
with a 31-bp moving average, log-transformed (natural log), and the trend (a 301-bp moving average of the log
density) is subtracted. The detrended curves are averaged per group and shown from 300 to 2200 bp.

Inputs:
  --hist-dir     1-bp length histograms length_hist_<ont|pacbio>_direct.tsv.gz (one row per sample, columns 1..5000, >5000)
  --periodicity  per-sample periodicity table (tsv: run, sample, group, H_rare, frac_1001_1250, ...); rows ont_direct
                 and pacbio_direct give the rarefied entropy and the share near 1 kb for panels C and F
Output: fig4_long_fragment_ladder.png (150 dpi) in the --out folder.

Usage:
  python 11_plot_long_fragment_ladder.py --hist-dir results/length_hist \
      --periodicity results/length_hist/periodicity_per_sample.tsv -o results/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HCC, HBV = "#E8703A", "#2F7BD8"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#EEEEEE", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "axes.titlelocation": "left", "legend.frameon": False,
})
SMOOTH, TREND = 31, 301          # bp: moving average of the density, moving average of the log density (trend)
SHOW = (300, 2200)               # bp range shown in panels A and D
NRL, FIRST_PEAK = 195, 350       # dotted ladder lines: every 195 bp from the di-nucleosome peak
SPLIT = 1250                     # "near 1 kb" = 1001-1250 bp
LABEL = ["TBR3976", "TBR3013", "TBR3032"]
PLATFORMS = [("ont", "ONT", "ABC"), ("pacbio", "PacBio", "DEF")]


def group_of(s):
    """HBV carriers are the TBR44xx samples, all others are HCC"""
    return "HBV" if s.startswith("TBR44") else "HCC"


def detrended(h):
    """log share of fragments per bp minus its 301-bp moving average (one sample, lengths 1..5000)"""
    share = pd.Series(h / h.sum()).rolling(SMOOTH, center=True, min_periods=1).mean()
    y = np.log(share + 1e-12)
    return (y - y.rolling(TREND, center=True, min_periods=1).mean()).to_numpy()


def main():
    ap = argparse.ArgumentParser(description="Long-fragment ladder and crowding figure (Figure 3.21).")
    ap.add_argument("--hist-dir", required=True,
                    help="folder with length_hist_<ont|pacbio>_direct.tsv.gz (1-bp length histograms, one row per sample)")
    ap.add_argument("--periodicity", required=True,
                    help="per-sample periodicity table (tsv: run, sample, H_rare, frac_1001_1250)")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = pd.read_csv(a.periodicity, sep="\t")
    x = np.arange(1, 5001)
    cols = [str(i) for i in x]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, (pl, name, letters) in enumerate(PLATFORMS):
        d = pd.read_csv(os.path.join(a.hist_dir, f"length_hist_{pl}_direct.tsv.gz"), sep="\t", index_col=0)
        grp = pd.Series({s: group_of(s) for s in d.index})
        H = d[cols].to_numpy(float)

        # A / D: nucleosome ladder, group mean of the detrended log profiles
        ax = axes[row, 0]
        show = (x >= SHOW[0]) & (x <= SHOW[1])
        ax.axvspan(1000, 2000, color="#EEEEEE", lw=0, zorder=0)
        for p in range(FIRST_PEAK, SHOW[1], NRL):
            ax.axvline(p, color="#BBBBBB", lw=0.6, ls=":", zorder=1)
        ax.axhline(0, color="#999999", lw=0.8)
        for g, c in [("HBV", HBV), ("HCC", HCC)]:
            m = (grp == g).to_numpy()
            prof = np.mean([detrended(h) for h in H[m]], axis=0)
            ax.plot(x[show], prof[show], color=c, lw=2, label=f"{g} (n={m.sum()})", zorder=3)
        ax.set_xlabel("fragment length (bp)"); ax.set_ylabel("log density minus trend")
        ax.set_title(f"{letters[0]}  {name}: nucleosome ladder (detrended)")
        ax.legend(loc="upper right", fontsize=10)

        # B / E: cumulative share within 1001-2000 bp, one line per sample
        bx = axes[row, 1]
        inbin = (x >= 1001) & (x <= 2000)
        for s, h in zip(d.index, H):
            cum = np.concatenate([[0], np.cumsum(h[inbin]) / h[inbin].sum()])
            bx.plot(np.arange(1000, 2001), cum, color=HCC if grp[s] == "HCC" else HBV, lw=1, alpha=0.85)
        bx.axvline(SPLIT, color="#888888", lw=0.8, ls="--")
        bx.set_xlabel("fragment length (bp)"); bx.set_ylabel("cumulative share of bin")
        bx.set_title(f"{letters[1]}  {name}: lengths within 1001–2000 bp")

        # C / F: rarefied entropy against the share near 1 kb
        cx = axes[row, 2]
        p = per[per.run == f"{pl}_direct"].set_index("sample")
        for g, c in [("HBV", HBV), ("HCC", HCC)]:
            q = p[[group_of(s) == g for s in p.index]]
            cx.scatter(100 * q.frac_1001_1250, q.H_rare, s=30, color=c, label=g, zorder=3)
        for s in LABEL:
            if s in p.index:
                cx.annotate(s, (100 * p.loc[s, "frac_1001_1250"], p.loc[s, "H_rare"]), xytext=(4, 4),
                            textcoords="offset points", fontsize=7.5, color="#444444")
        cx.set_xlabel("% of 1001–2000 bp fragments that are 1001–1250 bp")
        cx.set_ylabel("rarefied entropy, 1001–2000 bp")
        cx.set_title(f"{letters[2]}  {name}: entropy follows the share near 1 kb")
        cx.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    png = os.path.join(a.out, "fig4_long_fragment_ladder.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
