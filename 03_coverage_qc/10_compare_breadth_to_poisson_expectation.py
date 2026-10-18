#!/usr/bin/env python3
"""
Whole-genome vs targeted test, criteria (ii) and (iii): compares each
sample's observed breadth of coverage at >= 1x with the breadth expected if
fragments land at random positions in the genome (Poisson sampling):

    expected breadth = 1 - exp(-mean depth)

With whole-genome sequencing the observed breadth follows this curve at every
depth (ratio observed/expected close to 1). A capture panel concentrates the
reads in the targets, so its breadth stays far below the curve and plateaus
as depth increases.

Input : genomecov_breadth.tsv from 09_compute_genomecov_breadth.sh
        (sample, genome_bp, covered_bp, breadth_pct, mean_depth)
        optional --metadata CSV with columns sample, group (point colours only)
Output: OUT/breadth_vs_poisson.csv
        OUT/breadth_vs_poisson.png/.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--breadth", type=Path, required=True,
                   help="per-sample breadth table from 09_compute_genomecov_breadth.sh (tsv)")
    p.add_argument("--metadata", type=Path, default=None,
                   help="optional CSV with columns sample, group, used to colour the points")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.breadth, sep="\t")
    df["observed_breadth_pct"] = df["breadth_pct"]
    df["expected_breadth_pct"] = 100 * (1 - np.exp(-df["mean_depth"]))
    df["observed_over_expected"] = df["observed_breadth_pct"] / df["expected_breadth_pct"]

    if args.metadata is not None:
        meta = pd.read_csv(args.metadata, usecols=["sample", "group"])
        df = df.merge(meta, on="sample", how="left")
    else:
        df["group"] = "all"
    df["group"] = df["group"].fillna("unknown")

    df = df.sort_values("mean_depth")
    out_csv = args.out / "breadth_vs_poisson.csv"
    df.drop(columns=["breadth_pct"]).to_csv(out_csv, index=False)

    r = df["observed_over_expected"]
    print(df[["sample", "group", "mean_depth", "observed_breadth_pct",
              "expected_breadth_pct", "observed_over_expected"]].to_string(index=False))
    print(f"\nBreadth at >=1x: median {df['observed_breadth_pct'].median():.1f}% "
          f"(range {df['observed_breadth_pct'].min():.1f}-{df['observed_breadth_pct'].max():.1f}%)")
    print(f"Mean depth: {df['mean_depth'].min():.2f}x - {df['mean_depth'].max():.2f}x")
    print(f"Observed/expected: median {r.median():.3f} (range {r.min():.3f}-{r.max():.3f})")
    # does the ratio drift with depth? (a capture library would fall further below at high depth)
    rho = df[["mean_depth", "observed_over_expected"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman rho (depth vs ratio): {rho:.2f}")

    # plot: Poisson curve + one point per sample
    fig, ax = plt.subplots(figsize=(6, 4.5))
    depth = np.linspace(0, max(df["mean_depth"].max() * 1.1, 0.5), 300)
    ax.plot(depth, 100 * (1 - np.exp(-depth)), color="grey", lw=1.5,
            label="Poisson expectation, 1 - e$^{-depth}$")
    for group, sub in df.groupby("group"):
        ax.scatter(sub["mean_depth"], sub["observed_breadth_pct"], s=30, label=group, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["sample"], (row["mean_depth"], row["observed_breadth_pct"]),
                    fontsize=6, xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("Mean depth (x)")
    ax.set_ylabel("Breadth of coverage at >= 1x (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Observed vs Poisson-expected breadth (n={len(df)}; "
                 f"median ratio {r.median():.2f})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "breadth_vs_poisson.png", dpi=300)
    fig.savefig(args.out / "breadth_vs_poisson.pdf")
    print(f"Wrote {out_csv} and breadth_vs_poisson.png/.pdf")


if __name__ == "__main__":
    main()
