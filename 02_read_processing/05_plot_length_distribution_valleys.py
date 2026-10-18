#!/usr/bin/env python3
"""
Plots per-sample fragment length distributions from the Stage 1 eligible
fragments and detects peak (nucleosomal) and valley positions in the
50-1000 bp range from the data, instead of relying on literature peak
positions (166 bp, ~334 bp, ~501 bp). Nucleosome spacing can vary by
platform and library preparation, so this checks the mono-/di-/
tri-nucleosome size-bin boundaries against our own data.

Method: build a fine-resolution (1bp) histogram per sample, smooth it
(Savitzky-Golay filter) to remove single-bp noise while preserving the
real peak/valley shape, then use scipy.signal.find_peaks on the smoothed
curve (and its negative) to locate local maxima (peaks) and minima
(valleys).

Input:  FRAGMENTS_DIR/*.eligible_fragments.csv.gz (column "length")
Outputs (in --out):
  - all_samples_length_distribution_valleys.png: one panel per sample, peaks/valleys marked
  - peak_valley_positions.csv: detected peak and valley positions per sample
"""

import argparse

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# ---------------------------------------------------------------------------
# paths are set from the command line in main()
FRAGMENTS_DIR = None
OUT_DIR = None

# Range to search for nucleosomal peaks/valleys. Beyond ~1000bp, individual
# nucleosome periodicity is no longer the dominant signal.
MAX_BP = 1000
MIN_BP = 50

SMOOTH_WINDOW = 15   # must be odd; bp-scale smoothing window
SMOOTH_POLYORDER = 3

# Minimum prominence (as a fraction of the smoothed curve's max height) for
# a local max/min to count as a real peak/valley rather than noise.
PROMINENCE_FRACTION = 0.02

# ---------------------------------------------------------------------------


def get_fragment_files():
    return sorted(FRAGMENTS_DIR.glob("*.eligible_fragments.csv.gz"))


def detect_peaks_and_valleys(lengths):
    """Returns (bin_centers, smoothed_density, peak_positions, valley_positions)."""
    bins = np.arange(MIN_BP, MAX_BP + 1, 1)
    counts, edges = np.histogram(lengths, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    smoothed = savgol_filter(counts, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
    smoothed = np.clip(smoothed, 0, None)

    prominence = PROMINENCE_FRACTION * smoothed.max()

    peak_idx, _ = find_peaks(smoothed, prominence=prominence)
    valley_idx, _ = find_peaks(-smoothed, prominence=prominence)

    peak_positions = bin_centers[peak_idx]
    valley_positions = bin_centers[valley_idx]

    return bin_centers, smoothed, peak_positions, valley_positions


def main():
    global FRAGMENTS_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frags",
                        required=True,
                        help="folder with the Stage 1 eligible fragment tables (eligible_fragments.csv.gz)")
    parser.add_argument("-o", "--out",
                        required=True,
                        help="output folder for the figure and the peak/valley table")
    args = parser.parse_args()

    FRAGMENTS_DIR = Path(args.frags)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_fragment_files()
    print(f"Found {len(files)} eligible-fragment tables.\n")

    n_samples = len(files)
    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    summary_rows = []

    for i, f in enumerate(files):
        sample = f.name.split(".")[0]
        df = pd.read_csv(f, compression="gzip")
        lengths = df["length"].values
        lengths = lengths[(lengths >= MIN_BP) & (lengths <= MAX_BP)]

        if len(lengths) < 100:
            print(f"  [!] {sample}: too few fragments in {MIN_BP}-{MAX_BP}bp range, skipping")
            continue

        bin_centers, smoothed, peaks, valleys = detect_peaks_and_valleys(lengths)

        ax = axes[i]
        ax.plot(bin_centers, smoothed, color="#4472c4", linewidth=1)
        ax.scatter(peaks, smoothed[np.searchsorted(bin_centers, peaks)],
                   color="#c0504d", zorder=5, s=25, label="peak")
        for v in valleys:
            ax.axvline(v, color="#70ad47", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(sample, fontsize=10)
        ax.set_xlabel("Fragment length (bp)", fontsize=8)
        ax.set_ylabel("Count (smoothed)", fontsize=8)
        ax.tick_params(labelsize=7)

        print(f"  {sample}: peaks at {[round(p) for p in peaks]}, "
              f"valleys at {[round(v) for v in valleys]}")

        # record up to first 4 peaks / 4 valleys for the summary table
        row = {"sample": sample, "n_fragments_used": len(lengths)}
        for j, p in enumerate(sorted(peaks)[:4]):
            row[f"peak_{j+1}_bp"] = round(p)
        for j, v in enumerate(sorted(valleys)[:4]):
            row[f"valley_{j+1}_bp"] = round(v)
        summary_rows.append(row)

    # hide unused subplot axes
    for j in range(len(files), len(axes)):
        axes[j].axis("off")

    fig.suptitle("ONT fragment length distributions: detected nucleosomal peaks (red) "
                  "and valleys (green dashed)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig_path = OUT_DIR / "all_samples_length_distribution_valleys.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "peak_valley_positions.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nWrote figure: {fig_path}")
    print(f"Wrote summary: {summary_path}\n")
    print(summary_df.to_string(index=False))
    print("\nCompare valley_1/valley_2 positions above with the literature-based bin edges "
          "(151 bp, 325 bp, 501 bp).")


if __name__ == "__main__":
    main()
