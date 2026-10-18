#!/usr/bin/env python3
"""
Summarises the recurrent coverage spikes per chromosome: how many spike bins
each chromosome has, how many of them overlap a problem-region track, how
many are present in every sample, and the highest depth. Shows where the
spikes that the tracks do not explain are concentrated.

Input : recurrent_spikes_annotated.csv from 12_intersect_spikes_with_blacklist.py
        recurrent_spikes.csv from 11_extract_spike_coordinates.py (for depths)
Output: OUT/spikes_per_chromosome.csv
        OUT/spikes_per_chromosome.png/.pdf  (stacked bars: matched / unmatched)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotated", type=Path, required=True,
                   help="spikes with track overlaps (csv from 12_intersect_spikes_with_blacklist.py)")
    p.add_argument("--spikes", type=Path, required=True,
                   help="recurrent spikes with depths (csv from 11_extract_spike_coordinates.py)")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(args.annotated, dtype={"chrom": str})
    spikes = pd.read_csv(args.spikes, dtype={"chrom": str})
    df = ann.merge(spikes[["chrom", "start", "end", "max_depth", "n_samples_total"]],
                   on=["chrom", "start", "end"], how="left")
    n_total = int(df["n_samples_total"].max())

    per_chrom = (df.groupby("chrom")
                 .agg(n_spikes=("start", "size"),
                      n_matched=("any_match", "sum"),
                      n_universal=("n_samples", lambda s: int((s == n_total).sum())),
                      max_depth=("max_depth", "max"),
                      median_n_samples=("n_samples", "median"))
                 .reindex([c for c in CANONICAL if c in set(df["chrom"])])
                 .reset_index())
    per_chrom["n_unmatched"] = per_chrom["n_spikes"] - per_chrom["n_matched"]
    per_chrom["pct_matched"] = 100 * per_chrom["n_matched"] / per_chrom["n_spikes"]
    per_chrom.to_csv(args.out / "spikes_per_chromosome.csv", index=False)

    print(per_chrom.to_string(index=False))
    print(f"\nTotal: {int(per_chrom['n_spikes'].sum())} spikes, "
          f"{int(per_chrom['n_matched'].sum())} in a problem-region track, "
          f"{int(per_chrom['n_unmatched'].sum())} not")

    # stacked bars per chromosome
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(per_chrom))
    ax.bar(x, per_chrom["n_matched"], color="#3b6ea5", label="in problem-region track")
    ax.bar(x, per_chrom["n_unmatched"], bottom=per_chrom["n_matched"], color="#d98c3f",
           label="not in any track")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("chr", "") for c in per_chrom["chrom"]])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Recurrent spike bins")
    ax.set_title(f"Recurrent coverage spikes per chromosome (>= 3 of {n_total} samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "spikes_per_chromosome.png", dpi=300)
    fig.savefig(args.out / "spikes_per_chromosome.pdf")
    print(f"Wrote {args.out}/spikes_per_chromosome.csv / .png / .pdf")


if __name__ == "__main__":
    main()
