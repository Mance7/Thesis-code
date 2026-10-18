#!/usr/bin/env python3
"""
Finds the recurrent coverage spikes: genome windows with very high coverage
in several samples.

For each sample, a window is a spike bin if its mean depth (from the MAPQ >= 20
fragments, 07_compute_coverage_from_fragments.sh) is at least --fold times
that sample's median window depth (median over windows with depth > 0, so the
threshold scales with each sample's sequencing depth). A spike bin is called
recurrent if it is a spike in at least --min-samples samples (3 by default).

Input : COV_DIR/{sample}/{sample}.{window}.coverage_mean.bed
Output: OUT/spike_bins_per_sample.csv   every spike bin of every sample
        OUT/recurrent_spikes.csv        bins shared by >= min-samples samples
                                        (chrom,start,end,n_samples,samples,
                                         max_depth,median_fold,max_fold)
        OUT/recurrent_spikes.bed        same bins as BED4 (name = n_samples)
"""

import argparse
from pathlib import Path

import pandas as pd

CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cov-dir", type=Path, required=True,
                   help="folder with {sample}/{sample}.{window}.coverage_mean.bed")
    p.add_argument("--window", default="10kb",
                   help="window tag of the coverage files to scan")
    p.add_argument("--fold", type=float, default=10.0,
                   help="spike = window depth >= fold x the sample's median window depth")
    p.add_argument("--min-samples", type=int, default=3,
                   help="a spike bin is recurrent if it is a spike in at least this many samples")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.cov_dir.glob(f"*/*.{args.window}.coverage_mean.bed"))
    if not files:
        raise RuntimeError(f"No *.{args.window}.coverage_mean.bed files in {args.cov_dir}")
    n_total = len(files)
    print(f"{n_total} samples, {args.window} windows, spike = >= {args.fold}x sample median")

    spikes = []
    for f in files:
        sample = f.parent.name
        df = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "depth"], dtype={"chrom": str})
        df = df[df["chrom"].isin(CANONICAL)]
        median = df.loc[df["depth"] > 0, "depth"].median()
        hits = df[df["depth"] >= args.fold * median].copy()
        hits["sample"] = sample
        hits["sample_median_depth"] = median
        hits["fold"] = hits["depth"] / median
        spikes.append(hits)
        print(f"  {sample}: median {median:.3f}x, {len(hits)} spike bins, max {df['depth'].max():.1f}x")

    per_sample = pd.concat(spikes, ignore_index=True)
    per_sample.to_csv(args.out / "spike_bins_per_sample.csv", index=False)

    # count in how many samples each bin is a spike
    rec = (per_sample.groupby(["chrom", "start", "end"])
           .agg(n_samples=("sample", "nunique"),
                samples=("sample", lambda s: ",".join(sorted(s))),
                max_depth=("depth", "max"),
                median_fold=("fold", "median"),
                max_fold=("fold", "max"))
           .reset_index())
    rec = rec[rec["n_samples"] >= args.min_samples].copy()
    rec["n_samples_total"] = n_total
    rec["chrom"] = pd.Categorical(rec["chrom"], categories=CANONICAL, ordered=True)
    rec = rec.sort_values(["chrom", "start"])
    rec["chrom"] = rec["chrom"].astype(str)

    rec.to_csv(args.out / "recurrent_spikes.csv", index=False)
    rec[["chrom", "start", "end", "n_samples"]].to_csv(
        args.out / "recurrent_spikes.bed", sep="\t", header=False, index=False)

    print(f"\n{len(rec)} recurrent spike bins (>= {args.min_samples} of {n_total} samples); "
          f"{(rec['n_samples'] == n_total).sum()} present in all samples")
    print(rec.sort_values("max_depth", ascending=False).head(10).to_string(index=False))
    print(f"Wrote {args.out}/recurrent_spikes.csv and .bed")


if __name__ == "__main__":
    main()
