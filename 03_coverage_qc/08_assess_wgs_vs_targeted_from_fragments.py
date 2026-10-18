#!/usr/bin/env python3
"""
Whole-genome vs targeted test, criterion (i), repeated on the MAPQ >= 20
Stage 1 fragments (coverage from 07_compute_coverage_from_fragments.sh).

For each sample and annotation, region mean depth = length-weighted mean of
the per-interval mean depths in that annotation BED; genome-wide background
= length-weighted mean depth of the 100 kb windows (they tile the genome).
Enrichment ratio = region / background (WGS ~1x, capture panel several-fold
or more).

It also draws the genome-wide coverage profile (mean depth per window along
chr1..chrY) of a few samples, which is where the recurrent coverage spikes
are seen. Pick the samples with --plot-samples, or the first
--n-plot-samples samples are used.

Input : COV_DIR/{sample}/{sample}.{tag}.coverage_mean.bed
Output: OUT/region_enrichment_fragments.csv           per sample x region
        OUT/region_enrichment_fragments_summary.csv   per region type
        OUT/region_enrichment_fragments.png/.pdf
        OUT/coverage_profiles/{sample}.{window}.coverage_profile.png/.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANNOTATIONS = ["promoters", "gene_bodies", "tss_gene", "tss_transcripts",
               "tss_dedup", "tss_collapsed", "exons_all", "exons_merged",
               "cpg_islands"]
CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
TARGETED_MIN_FOLD = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cov-dir", type=Path, required=True,
                   help="folder with {sample}/{sample}.{tag}.coverage_mean.bed from bedtools coverage -mean")
    p.add_argument("--annotations", nargs="+", default=ANNOTATIONS,
                   help="annotation tags to test")
    p.add_argument("--background-window", default="100kb",
                   help="window tag used as the genome-wide background")
    p.add_argument("--profile-window", default="10kb",
                   help="window tag used for the coverage profile plots")
    p.add_argument("--n-plot-samples", type=int, default=2,
                   help="number of samples to draw coverage profiles for (first N by name)")
    p.add_argument("--plot-samples", nargs="+", default=None,
                   help="draw coverage profiles for these samples instead of the first N")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def read_cov(path):
    """bedtools coverage -mean output: chrom, start, end, mean depth."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chrom", "start", "end", "mean"],
                     dtype={"chrom": str})
    return df[df["chrom"].isin(CANONICAL)]


def weighted_mean(df):
    length = df["end"] - df["start"]
    return float((df["mean"] * length).sum() / length.sum())


def plot_profile(df, sample, window, out_dir):
    """Mean depth per window along the genome, chromosomes side by side."""
    df = df.copy()
    df["chrom"] = pd.Categorical(df["chrom"], categories=CANONICAL, ordered=True)
    df = df.sort_values(["chrom", "start"])
    chrom_len = df.groupby("chrom", observed=True)["end"].max()
    offset = chrom_len.cumsum().shift(fill_value=0)
    x = df["start"].to_numpy() + df["chrom"].map(offset).astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = np.where(df["chrom"].cat.codes % 2 == 0, "#3b6ea5", "#9bb7d4")
    ax.scatter(x, df["mean"], s=1, c=colors, rasterized=True)
    ax.axhline(df["mean"].median(), color="red", lw=0.8, ls="--",
               label=f"median {df['mean'].median():.2f}x")
    ax.set_xticks((offset + chrom_len / 2).to_numpy())
    ax.set_xticklabels([c.replace("chr", "") for c in chrom_len.index], fontsize=8)
    ax.set_xlim(0, chrom_len.sum())
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(f"Mean depth per {window} window")
    top = df.loc[df["mean"].idxmax()]
    ax.set_title(f"{sample}: coverage from MAPQ>=20 fragments "
                 f"(max {top['mean']:.0f}x at {top['chrom']}:{top['start']:,})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    base = out_dir / f"{sample}.{window}.coverage_profile"
    fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples = sorted(d.name for d in args.cov_dir.iterdir()
                     if (d / f"{d.name}.{args.background_window}.coverage_mean.bed").exists())
    if not samples:
        raise RuntimeError(f"No coverage outputs found in {args.cov_dir}")
    print(f"{len(samples)} samples")

    rows = []
    for s in samples:
        background = weighted_mean(read_cov(args.cov_dir / s / f"{s}.{args.background_window}.coverage_mean.bed"))
        for ann in args.annotations:
            f = args.cov_dir / s / f"{s}.{ann}.coverage_mean.bed"
            if not f.exists():
                print(f"  [!] missing {f}")
                continue
            region = weighted_mean(read_cov(f))
            rows.append({"sample": s, "region": ann, "region_mean_depth": region,
                         "genome_mean_depth": background,
                         "enrichment_ratio": region / background if background > 0 else np.nan})

    df = pd.DataFrame(rows)
    summary = (df.groupby("region")["enrichment_ratio"]
                 .agg(["count", "median", "min", "max"]).reset_index())
    summary["looks_targeted"] = summary["median"] >= TARGETED_MIN_FOLD
    base = args.out / "region_enrichment_fragments"
    df.to_csv(f"{base}.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nAll region ratios: {df['enrichment_ratio'].min():.2f} - "
          f"{df['enrichment_ratio'].max():.2f}x (WGS ~1x; capture panel >= {TARGETED_MIN_FOLD}x)")

    # enrichment box plot
    order = [a for a in args.annotations if a in set(df["region"])]
    fig, ax = plt.subplots(figsize=(1.0 * len(order) + 2, 4.5))
    ax.boxplot([df.loc[df["region"] == r, "enrichment_ratio"] for r in order],
               tick_labels=order, showfliers=False)
    for i, r in enumerate(order, start=1):
        y = df.loc[df["region"] == r, "enrichment_ratio"]
        ax.scatter([i] * len(y), y, s=12, color="black", alpha=0.6, zorder=3)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_ylabel("Region mean depth / genome-wide mean depth")
    ax.set_title(f"Region coverage enrichment (MAPQ>=20 fragments, n={len(samples)})")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)

    # coverage profiles of the chosen samples
    plot_list = args.plot_samples if args.plot_samples else samples[:args.n_plot_samples]
    prof_dir = args.out / "coverage_profiles"
    prof_dir.mkdir(exist_ok=True)
    for s in plot_list:
        f = args.cov_dir / s / f"{s}.{args.profile_window}.coverage_mean.bed"
        if not f.exists():
            print(f"  [!] no {args.profile_window} coverage for {s}, skipped")
            continue
        plot_profile(read_cov(f), s, args.profile_window, prof_dir)
        print(f"  profile plotted: {s}")
    print(f"Wrote {base}.csv / _summary.csv / .png and profiles in {prof_dir}")


if __name__ == "__main__":
    main()
