#!/usr/bin/env python3
"""Draw the figure showing that motif diversity score (MDS) differences follow fragment count.

Figure:
  Figure 3.9  fig4_mds_count_artefact.png
    A  HCC - HBV difference in median MDS (bits) for each gene-set pool in the 601-2000 bp bin, against the
       number of promoters in the pool (log scale); orange = BH q < 0.05, grey = q >= 0.05 or not testable
    B  mean normalised MDS per individual promoter (<=600 bp, direct fragments), HBV carriers against HCC,
       with the line of equality

Inputs:
  --pooled      gene-set pooled statistics (one row per pool x metric; columns group_id, metric, n_promoters,
                median_hcc, median_hbv, fdr_bh). Rows with metric "mds__601-2000" are used.
  --individual  individual-promoter statistics, direct fragments (one row per promoter x size bin x metric;
                columns size_bin, metric, mean_HBV_carrier, mean_HCC). Rows "<=600" / "mds_motif" are used.
Pools where too few samples passed the fragment gate have no median for one group and are left out of A.
Output: fig4_mds_count_artefact.png (200 dpi) in the --out folder; a short summary is printed.

Usage:
  python 06_plot_mds_count_artefact.py --pooled "results/geneset_pooled/pooled_stats_alltogether_results5n.csv" \
      --individual "results/individual_promoters/individual_region_stats_direct.csv.gz" \
      -o "results/figures"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORANGE, GREY, BLUE = "#C2410C", "#9AA5B1", "#1D4ED8"   # q < 0.05, not significant, promoters
POOL_METRIC = "mds__601-2000"                           # MDS in the 601-2000 bp bin, gene-set pools
IND_BIN, IND_METRIC = "<=600", "mds_motif"              # only the <=600 bp bin has enough fragments per promoter
Q = 0.05
plt.rcParams.update({"font.family": "DejaVu Sans"})


def main():
    ap = argparse.ArgumentParser(description="MDS count artefact figure (Figure 3.9).")
    ap.add_argument("--pooled", required=True,
                    help="gene-set pooled statistics (csv: group_id, metric, n_promoters, median_hcc, median_hbv, fdr_bh)")
    ap.add_argument("--individual", required=True,
                    help="individual-promoter statistics, direct fragments (csv or csv.gz: size_bin, metric, mean_HBV_carrier, mean_HCC)")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # A: gene-set pools, HCC - HBV gap in median MDS against pool size
    pool = pd.read_csv(a.pooled)
    pool = pool[pool.metric == POOL_METRIC].copy()
    pool["gap"] = pool.median_hcc - pool.median_hbv
    pool = pool.dropna(subset=["gap"])                  # no median in one group: not testable, nothing to plot
    sig = pool.fdr_bh < Q                               # NaN q (too few samples for a test) counts as not significant

    # B: individual promoters, mean normalised MDS per group
    ind = pd.read_csv(a.individual)
    ind = ind[(ind.size_bin == IND_BIN) & (ind.metric == IND_METRIC)].dropna(subset=["mean_HBV_carrier", "mean_HCC"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4))
    ax.scatter(pool.n_promoters[~sig], pool.gap[~sig], s=22, color=GREY, label=f"q ≥ {Q}", zorder=3)
    ax.scatter(pool.n_promoters[sig], pool.gap[sig], s=22, color=ORANGE, label=f"q < {Q}", zorder=3)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Promoters in pool (log scale)")
    ax.set_ylabel("MDS gap, HCC − HBV (bits)")
    ax.set_title("A  Gene-set pools, 601–2000 bp", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    bx.scatter(ind.mean_HBV_carrier, ind.mean_HCC, s=3, color=BLUE, alpha=0.3, linewidths=0)
    bx.plot([0.2, 0.8], [0.2, 0.8], color="black", lw=0.7)
    bx.set_xlim(0.2, 0.8); bx.set_ylim(0.2, 0.8)
    bx.set_xlabel("Mean normalised MDS, HBV carriers")
    bx.set_ylabel("Mean normalised MDS, HCC")
    bx.set_title(f"B  Individual promoters, ≤600 bp (n = {len(ind):,}, direct)", loc="left", fontsize=11)

    for x in (ax, bx):
        x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
    fig.tight_layout()
    png = os.path.join(a.out, "fig4_mds_count_artefact.png")
    fig.savefig(png, dpi=200)
    plt.close(fig)

    # numbers quoted in the text (Section 3.3.4)
    print(f"pools plotted: {len(pool)}, q < {Q}: {int(sig.sum())}; "
          f"Spearman gap vs pool size: {pool.gap.corr(pool.n_promoters, method='spearman'):.2f}")
    print(f"promoters: {len(ind)}, HCC higher in {100 * np.mean(ind.mean_HCC > ind.mean_HBV_carrier):.1f}%, "
          f"medians HCC {ind.mean_HCC.median():.2f} vs HBV {ind.mean_HBV_carrier.median():.2f}")
    print("wrote", png)


if __name__ == "__main__":
    main()
