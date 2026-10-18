#!/usr/bin/env python3
"""
Data-driven check of the merged size bins: peaks and valleys of the pooled length density.

For analyses where the eleven size bins left cells too sparse, the bins were
merged into four: <=600, 601-2000, 2001-5000 and >5000 bp (Methods 2.4). The
first boundary comes from this script:
  1. fragment lengths of all samples are pooled into one 1 bp histogram,
  2. a Gaussian kernel density estimate is fitted to log10(length)
     (each length weighted by its fragment count; kernel SD --bandwidth
     in log10 units),
  3. local maxima (modes) and minima (valleys) of the density are found with
     scipy.signal.find_peaks on the density and on its negative, keeping only
     those with a prominence of at least --prominence-fraction x the highest
     density value.
On the ONT data this gave modes at about 180, 330 and 450 bp, valleys at 246
and 419 bp, and no further structure above about 600 bp. The upper
boundaries (2000 and 5000 bp) were not taken from the density; they keep the
original 1001-2000 bp and >5000 bp findings within single bins.

Input : either the per-sample 1 bp histograms of 05_entropy_size/04_length_histograms.py
        (--hist, length_hist_<name>.tsv.gz; the ">max" column is left out), or
        per-sample fragment tables (--fragments, glob of csv.gz with a length column)
Output: <out>/length_kde_peaks_valleys.csv  one row per mode/valley (type, length_bp,
        log10_length, density, prominence)
        <out>/length_kde_curve.csv           the density on the log10 grid
        <out>/length_kde_peaks_valleys.png   density with modes and valleys marked

  python 03_size_bin_merge_analysis.py \\
      --hist length_hist/length_hist_ont_all.tsv.gz -o size_bin_merge/
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_BP = 50                  # shortest length used in the density
MAX_BP = 5000                # longest length used (the histograms stop here)
PROMINENCE_FRACTION = 0.02   # same rule as the per-sample valley detection of Stage 1
GRID_POINTS = 2000           # points of the log10 length grid
BANDWIDTH = 0.02             # Gaussian kernel SD, in log10(bp) units (about +-5% of the length)
MERGED_EDGES = [600, 2000, 5000]


def pooled_from_hist(path):
    """Sum the per-sample 1 bp histograms into one pooled count per length."""
    h = pd.read_csv(path, sep="\t", index_col=0)
    h = h[[c for c in h.columns if c.isdigit()]]          # drop the ">max" column
    lengths = h.columns.astype(int).to_numpy()
    return lengths, h.sum(axis=0).to_numpy(float), h.shape[0]


def pooled_from_fragments(pattern, length_col, chunksize):
    """Pool the fragment lengths of all files into one 1 bp histogram."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    acc = np.zeros(MAX_BP + 1, np.int64)
    for f in files:
        for ch in pd.read_csv(f, sep=None, engine="python", usecols=[length_col], chunksize=chunksize):
            L = pd.to_numeric(ch[length_col], errors="coerce").dropna().to_numpy(np.int64)
            L = L[(L >= 1) & (L <= MAX_BP)]
            acc += np.bincount(L, minlength=MAX_BP + 1)
        print(f"  read {os.path.basename(f)}")
    lengths = np.arange(1, MAX_BP + 1)
    return lengths, acc[1:].astype(float), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hist", help="per-sample length histogram table (length_hist_<name>.tsv.gz)")
    src.add_argument("--fragments", help="glob of per-sample fragment tables (csv.gz); quote it")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--length-col", default="length", help="fragment length column (with --fragments)")
    ap.add_argument("--min-bp", type=int, default=MIN_BP, help="shortest length used")
    ap.add_argument("--max-bp", type=int, default=MAX_BP, help="longest length used (at most 5000)")
    ap.add_argument("--prominence-fraction", type=float, default=PROMINENCE_FRACTION,
                    help="minimum prominence of a mode/valley, as a fraction of the highest density")
    ap.add_argument("--bandwidth", type=float, default=BANDWIDTH,
                    help="kernel SD in log10(bp) units")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time (with --fragments)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.hist:
        lengths, counts, n_samples = pooled_from_hist(a.hist)
    else:
        lengths, counts, n_samples = pooled_from_fragments(a.fragments, a.length_col, a.chunksize)
    keep = (lengths >= a.min_bp) & (lengths <= min(a.max_bp, MAX_BP)) & (counts > 0)
    lengths, counts = lengths[keep], counts[keep]
    print(f"{n_samples} samples, {counts.sum():,.0f} fragments of {a.min_bp}-{lengths.max()} bp pooled")

    # KDE on log10 length; every distinct length is one point weighted by its count
    x = np.log10(lengths)
    w = counts / counts.sum()
    grid = np.linspace(x.min(), x.max(), GRID_POINTS)
    z = (grid[:, None] - x[None, :]) / a.bandwidth
    dens = (np.exp(-0.5 * z ** 2) * w[None, :]).sum(1) / (a.bandwidth * np.sqrt(2 * np.pi))

    # modes and valleys, prominence-filtered
    prom = a.prominence_fraction * dens.max()
    pk, pk_info = find_peaks(dens, prominence=prom)
    vl, vl_info = find_peaks(-dens, prominence=prom)
    rows = [dict(type="mode", length_bp=round(10 ** grid[i]), log10_length=grid[i], density=dens[i],
                 prominence=p) for i, p in zip(pk, pk_info["prominences"])]
    rows += [dict(type="valley", length_bp=round(10 ** grid[i]), log10_length=grid[i], density=dens[i],
                  prominence=p) for i, p in zip(vl, vl_info["prominences"])]
    tab = pd.DataFrame(rows, columns=["type", "length_bp", "log10_length", "density", "prominence"])
    tab = tab.sort_values("length_bp").reset_index(drop=True)
    tab.to_csv(os.path.join(a.out, "length_kde_peaks_valleys.csv"), index=False)
    pd.DataFrame({"log10_length": grid, "length_bp": 10 ** grid, "density": dens}).to_csv(
        os.path.join(a.out, "length_kde_curve.csv"), index=False)

    last = int(tab["length_bp"].max()) if len(tab) else None
    print(tab.to_string(index=False))
    print(f"\nlast mode/valley at {last} bp: no structure passes the prominence filter above it")

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(10 ** grid, dens, color="#4472c4", lw=1.2)
    ax.scatter(10 ** grid[pk], dens[pk], color="#c0504d", zorder=5, s=25, label="mode")
    for i in vl:
        ax.axvline(10 ** grid[i], color="#70ad47", ls="--", lw=1)
    for e in MERGED_EDGES:
        ax.axvline(e, color="grey", ls=":", lw=1)
    for i in list(pk) + list(vl):
        ax.annotate(f"{round(10 ** grid[i])}", (10 ** grid[i], dens[i]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("fragment length (bp, log scale); green dashed = valleys, grey dotted = merged bin edges")
    ax.set_ylabel("density of log10 length")
    ax.set_title(f"Pooled fragment lengths ({n_samples} samples): KDE modes and valleys")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "length_kde_peaks_valleys.png"), dpi=160)
    print(f"wrote {a.out}/length_kde_peaks_valleys.csv, length_kde_curve.csv, length_kde_peaks_valleys.png")


if __name__ == "__main__":
    main()
