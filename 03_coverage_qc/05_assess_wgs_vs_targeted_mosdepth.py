#!/usr/bin/env python3
"""
Whole-genome vs targeted test, criterion (i), from the mosdepth outputs.

For each sample and each annotation (promoters, exons, gene bodies, TSS,
CpG islands) we take the mean depth inside the regions (mosdepth
"total_region" row) and divide it by the genome-wide background depth (the
"total" row of the whole-genome run). A capture panel gives several-fold to
fifty-fold enrichment in the targeted regions; whole-genome sequencing gives
a ratio close to 1.

Input : QC_ROOT/{sample}/{subdir}/{sample}.mosdepth.summary.txt
        QC_ROOT/{sample}/{subdir}/{sample}.{annotation}.mosdepth.summary.txt
Output: OUT/region_enrichment_mosdepth_{subdir}.csv          per sample x region
        OUT/region_enrichment_mosdepth_{subdir}_summary.csv  per region type
        OUT/region_enrichment_mosdepth_{subdir}.png/.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANNOTATIONS = ["promoters", "gene_bodies", "tss_gene", "tss_transcripts",
               "tss_dedup", "tss_collapsed", "exons_all", "exons_merged",
               "cpg_islands"]

# enrichment a capture panel would give at the very least
TARGETED_MIN_FOLD = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qc-root", type=Path, required=True,
                   help="folder with one sub-folder per sample holding the mosdepth outputs")
    p.add_argument("--subdir", default="alignment_qc",
                   help="mosdepth folder inside each sample folder "
                        "(alignment_qc or alignment_qc_mapq20)")
    p.add_argument("--annotations", nargs="+", default=ANNOTATIONS,
                   help="annotation names (mosdepth prefixes) to test")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def summary_mean(path, row):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("chrom")
    return float(df.loc[row, "mean"])


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(args.qc_root.iterdir()):
        qc = d / args.subdir
        sample = d.name
        global_summary = qc / f"{sample}.mosdepth.summary.txt"
        if not global_summary.exists():
            continue
        background = summary_mean(global_summary, "total")
        for ann in args.annotations:
            f = qc / f"{sample}.{ann}.mosdepth.summary.txt"
            if not f.exists():
                print(f"  [!] missing {f}")
                continue
            region_mean = summary_mean(f, "total_region")
            rows.append({"sample": sample, "region": ann,
                         "region_mean_depth": region_mean,
                         "genome_mean_depth": background,
                         "enrichment_ratio": region_mean / background if background > 0 else float("nan")})

    if not rows:
        raise RuntimeError(f"No mosdepth summaries found under {args.qc_root}/*/{args.subdir}")

    df = pd.DataFrame(rows)
    summary = (df.groupby("region")["enrichment_ratio"]
                 .agg(["count", "median", "min", "max"])
                 .reset_index())
    summary["looks_targeted"] = summary["median"] >= TARGETED_MIN_FOLD

    base = args.out / f"region_enrichment_mosdepth_{args.subdir}"
    df.to_csv(f"{base}.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nAll region ratios: {df['enrichment_ratio'].min():.2f} - "
          f"{df['enrichment_ratio'].max():.2f}x (WGS ~1x; capture panel >= {TARGETED_MIN_FOLD}x)")

    # one box per region type, one dot per sample
    order = [a for a in args.annotations if a in set(df["region"])]
    fig, ax = plt.subplots(figsize=(1.0 * len(order) + 2, 4.5))
    ax.boxplot([df.loc[df["region"] == r, "enrichment_ratio"] for r in order],
               tick_labels=order, showfliers=False)
    for i, r in enumerate(order, start=1):
        y = df.loc[df["region"] == r, "enrichment_ratio"]
        ax.scatter([i] * len(y), y, s=12, color="black", alpha=0.6, zorder=3)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_ylabel("Region mean depth / genome-wide mean depth")
    ax.set_title(f"Region coverage enrichment (mosdepth, {args.subdir}, n={df['sample'].nunique()})")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.pdf")
    print(f"Wrote {base}.csv / _summary.csv / .png / .pdf")


if __name__ == "__main__":
    main()
