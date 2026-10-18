#!/usr/bin/env python3
"""
01_sample_covariates.py
-----------------------
Sample-level covariates (Methods 2.3.3), one row per sample, from the
blacklist-filtered fragment files with end motifs. They are used to test for
confounding (Methods 2.10) and in the TBR3976 case study.

Columns of sample_covariates.csv:
  n_fragments          total fragments after filtering
  n_direct             direct fragments (source == 'direct')
  n_recovered          fragments recovered from concatemers (all other sources)
  recovery_gain_pct    100 * n_recovered / n_direct (the "gain" of Table 2.1)
  pct_recovered        100 * n_recovered / n_fragments (as in covariates.csv of
                       02_build_classifier_features.py)
  mean_length          mean fragment length (bp)
  mean_depth           sum of fragment lengths / --genome-size, i.e. fragment
                       count x mean length / 2.9 Gb (as in Methods 2.12)
  frac_gt_1kb          share of fragments longer than 1,000 bp
  frac_gt_7_5kb        share of fragments longer than 7,500 bp
  frac_c_start         share of valid-motif fragments whose 5' end motif starts with C
                       (DNASE1L3-associated; genomic DNA from leukocyte lysis lacks it)
  frac_c_start_le600   the same among fragments of up to 600 bp
  frac_chrY            share of fragments on chromosome Y (check against recorded sex)

Input: one fragment table per sample (csv.gz) with columns chrom, length, source,
motif_5p, motif_valid; the sample name is taken from the file name (text before
the first '.'). Files are read in chunks; samples run in parallel (-t).

Usage:
  python 01_sample_covariates.py \
      --fragments "fragments/*.fragments_with_motifs.csv.gz" \
      -o sample_covariates/ -t 14
"""
import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

GENOME_SIZE = 2.9e9   # bp, as used for the depth estimate in Methods 2.12


def truthy(s):
    """motif_valid may be stored as True/False text or 1/0."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def one_sample(path, a):
    """Stream one fragment file and return its counts."""
    c = dict(n=0, n_direct=0, sum_len=0, gt_1kb=0, gt_7_5kb=0, chrY=0,
             n_valid=0, c_start=0, n_valid_le600=0, c_start_le600=0)
    cols = [a.chrom_col, a.length_col, a.source_col, a.motif_col, a.valid_col]
    for ch in pd.read_csv(path, usecols=cols, chunksize=a.chunksize, dtype={a.chrom_col: str}):
        L = pd.to_numeric(ch[a.length_col], errors="coerce").to_numpy()
        c["n"] += len(ch)
        c["n_direct"] += int((ch[a.source_col].astype(str).str.strip() == "direct").sum())
        c["sum_len"] += float(np.nansum(L))
        c["gt_1kb"] += int((L > 1000).sum())
        c["gt_7_5kb"] += int((L > 7500).sum())
        c["chrY"] += int((ch[a.chrom_col] == "chrY").sum())
        # C-start share among fragments with a valid end motif
        valid = truthy(ch[a.valid_col]).to_numpy()
        cstart = ch[a.motif_col].astype(str).str.upper().str.startswith("C").to_numpy()
        short = L <= 600
        c["n_valid"] += int(valid.sum())
        c["c_start"] += int((valid & cstart).sum())
        c["n_valid_le600"] += int((valid & short).sum())
        c["c_start_le600"] += int((valid & cstart & short).sum())
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True,
                    help="glob (quoted) of per-sample fragments with motifs (csv.gz), blacklist-filtered")
    ap.add_argument("-o", "--out", required=True, help="output folder; sample_covariates.csv is written there")
    ap.add_argument("--genome-size", type=float, default=GENOME_SIZE,
                    help="genome size (bp) for the mean depth (default 2.9e9)")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chrom-col", default="chrom", help="chromosome column")
    ap.add_argument("--length-col", default="length", help="fragment length column")
    ap.add_argument("--source-col", default="source", help="column with 'direct' or the concatemer source")
    ap.add_argument("--motif-col", default="motif_5p", help="5' end-motif column")
    ap.add_argument("--valid-col", default="motif_valid", help="motif validity column")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time")
    ap.add_argument("-t", "--threads", type=int, default=4, help="samples processed in parallel")
    a = ap.parse_args()

    files = sorted(glob.glob(a.fragments))
    if not files:
        raise SystemExit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)
    names = []
    for f in files:
        m = re.match(a.sample_regex, os.path.basename(f))
        names.append(m.group(1) if m else os.path.basename(f))

    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        counts = list(ex.map(one_sample, files, [a] * len(files)))

    rows = []
    for s, c in zip(names, counts):
        n = c["n"]
        n_rec = n - c["n_direct"]
        div = lambda x, y: x / y if y else np.nan
        rows.append(dict(
            sample=s, n_fragments=n, n_direct=c["n_direct"], n_recovered=n_rec,
            recovery_gain_pct=100 * div(n_rec, c["n_direct"]),
            pct_recovered=100 * div(n_rec, n),
            mean_length=div(c["sum_len"], n),
            mean_depth=c["sum_len"] / a.genome_size,
            frac_gt_1kb=div(c["gt_1kb"], n), frac_gt_7_5kb=div(c["gt_7_5kb"], n),
            frac_c_start=div(c["c_start"], c["n_valid"]),
            frac_c_start_le600=div(c["c_start_le600"], c["n_valid_le600"]),
            frac_chrY=div(c["chrY"], n)))
        print(f"  {s}: {n:,} fragments")
    out = pd.DataFrame(rows)
    path = os.path.join(a.out, "sample_covariates.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
