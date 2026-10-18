#!/usr/bin/env python3
"""
Sanity check of the mosdepth outputs before they are used for the
whole-genome vs targeted test.

For every sample it checks that:
  1. every expected run finished (summary.txt present, regions/thresholds
     files are complete gzip files);
  2. the genome-wide mean depth of the whole-genome run agrees with the
     length-weighted mean of the 10 kb and 100 kb windows (they measure the
     same thing, so a large difference means a broken or partial run);
  3. the windows cover (almost) the whole genome length;
  4. no chromosome has a mean depth far from the sample's autosome median
     (flags samples with aberrant chromosome coverage).

Input : QC_ROOT/{sample}/{subdir}/{sample}[.{tag}].mosdepth.summary.txt,
        .regions.bed.gz, .thresholds.bed.gz  (from 01/02, or 03 with
        --subdir alignment_qc_mapq20)
Output: OUT/mosdepth_sanity_{subdir}.csv       one row per sample
        OUT/mosdepth_chrom_depth_{subdir}.csv  chromosome mean depth / median
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATIONS = ["promoters", "gene_bodies", "tss_gene", "tss_transcripts",
               "tss_dedup", "tss_collapsed", "exons_all", "exons_merged",
               "cpg_islands"]
WINDOWS = ["10kb", "100kb"]
CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qc-root", type=Path, required=True,
                   help="folder with one sub-folder per sample holding the mosdepth outputs")
    p.add_argument("--subdir", default="alignment_qc",
                   help="mosdepth folder inside each sample folder "
                        "(alignment_qc = no MAPQ cutoff, alignment_qc_mapq20 = MAPQ >= 20)")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder for the sanity tables")
    p.add_argument("--max-mean-diff", type=float, default=0.05,
                   help="largest allowed relative difference between whole-genome "
                        "and window mean depth")
    p.add_argument("--min-window-fraction", type=float, default=0.95,
                   help="windows must cover at least this fraction of the genome length")
    p.add_argument("--chrom-fold", type=float, default=1.5,
                   help="flag a chromosome whose mean depth is more than this fold "
                        "above or below the autosome median")
    return p.parse_args()


def prefix(qc_dir, sample, tag):
    """{sample}.{tag} for window/annotation runs, {sample} for the global run."""
    return qc_dir / (f"{sample}.{tag}" if tag else sample)


def gzip_ok(path):
    """True if the file exists and decompresses to the end."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def read_summary(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df.set_index("chrom")


def check_sample(qc_dir, sample, args):
    row = {"sample": sample}
    problems = []

    # 1. completeness of every run
    for tag in [""] + WINDOWS + ANNOTATIONS:
        name = tag or "global"
        p = prefix(qc_dir, sample, tag)
        summary = Path(str(p) + ".mosdepth.summary.txt")
        if not summary.exists() or summary.stat().st_size == 0:
            problems.append(f"missing summary ({name})")
            continue
        if tag:
            for suffix in ["regions.bed.gz", "thresholds.bed.gz"]:
                if not gzip_ok(Path(f"{p}.{suffix}")):
                    problems.append(f"missing/corrupt {suffix} ({name})")

    # 2. + 3. global mean vs window means, window coverage of the genome
    global_summary = Path(str(prefix(qc_dir, sample, "")) + ".mosdepth.summary.txt")
    chrom_rows = []
    if global_summary.exists():
        s = read_summary(global_summary)
        canon = s.loc[s.index.isin(CANONICAL)]
        genome_len = canon["length"].sum()
        genome_mean = canon["bases"].sum() / genome_len
        row["genome_length"] = int(genome_len)
        row["genome_mean_depth"] = genome_mean
        row["total_mean_depth"] = s.loc["total", "mean"] if "total" in s.index else np.nan

        for w in WINDOWS:
            regions = Path(str(prefix(qc_dir, sample, w)) + ".regions.bed.gz")
            if not gzip_ok(regions):
                continue
            # 3-column window BED -> mosdepth writes chrom, start, end, mean
            r = pd.read_csv(regions, sep="\t", header=None)
            r = r[r[0].isin(CANONICAL)]
            length = r[2] - r[1]
            mean_col = r.columns[-1]
            win_mean = (r[mean_col] * length).sum() / length.sum()
            row[f"{w}_mean_depth"] = win_mean
            row[f"{w}_rel_diff"] = abs(win_mean - genome_mean) / genome_mean if genome_mean > 0 else np.nan
            row[f"{w}_genome_fraction"] = length.sum() / genome_len
            row[f"{w}_frac_zero_windows"] = (r[mean_col] == 0).mean()
            if row[f"{w}_rel_diff"] > args.max_mean_diff:
                problems.append(f"{w} mean differs from genome mean by {row[f'{w}_rel_diff']:.1%}")
            if row[f"{w}_genome_fraction"] < args.min_window_fraction:
                problems.append(f"{w} windows cover only {row[f'{w}_genome_fraction']:.1%} of genome")

        # 4. per-chromosome depth relative to the autosome median
        auto_median = canon.loc[canon.index.isin(AUTOSOMES), "mean"].median()
        for chrom, r in canon.iterrows():
            rel = r["mean"] / auto_median if auto_median > 0 else np.nan
            chrom_rows.append({"sample": sample, "chrom": chrom,
                               "mean_depth": r["mean"], "rel_to_autosome_median": rel})
            if chrom in AUTOSOMES and (rel > args.chrom_fold or rel < 1 / args.chrom_fold):
                problems.append(f"{chrom} depth {rel:.2f}x autosome median")

    row["n_problems"] = len(problems)
    row["problems"] = "; ".join(problems)
    row["status"] = "OK" if not problems else "CHECK"
    return row, chrom_rows


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(d for d in args.qc_root.iterdir() if (d / args.subdir).is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No {args.subdir}/ folders found in {args.qc_root}")
    print(f"Checking {len(sample_dirs)} samples ({args.subdir})")

    rows, chrom_rows = [], []
    for d in sample_dirs:
        row, crow = check_sample(d / args.subdir, d.name, args)
        rows.append(row)
        chrom_rows.extend(crow)
        print(f"  {d.name:12s} {row['status']}  {row['problems']}")

    pd.DataFrame(rows).to_csv(args.out / f"mosdepth_sanity_{args.subdir}.csv", index=False)
    pd.DataFrame(chrom_rows).to_csv(args.out / f"mosdepth_chrom_depth_{args.subdir}.csv", index=False)
    n_bad = sum(r["status"] != "OK" for r in rows)
    print(f"\n{len(rows) - n_bad}/{len(rows)} samples passed all checks. Tables in {args.out}")


if __name__ == "__main__":
    main()
