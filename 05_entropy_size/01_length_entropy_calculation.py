#!/usr/bin/env python3
"""
Fragment-length Shannon entropy per sample, size bin and region.

For every cell (sample x size bin x region) the fragment lengths are counted
at 1 bp resolution inside the bin's range and the normalised Shannon entropy
is computed (Methods 2.6):

    H = -sum(p_i * log2(p_i)) / log2(K)

p_i is the fraction of the cell's fragments with length i, and K is the
number of possible integer lengths in the bin (e.g. K = 1000 for
1001-2000 bp). H is near 1 when lengths are spread evenly over the bin and
near 0 when they pile up at one length. For the open-ended >5000 bp bin, K is
set by the largest fragment length observed in the input files, capped at
50,000 bp.

Regions: genome_wide (all fragments) plus the region flags is_promoter,
is_gene_body, is_tss, is_exon and is_cpg_island, so 11 bins x 6 regions =
66 cells per sample. A cell with fewer than --min-frags fragments (100, the
same threshold as the motif analysis) gets entropy = missing.

This is the plug-in estimate. The fragment-count correction (Miller-Madow
and rarefied entropy) is done in 03_entropy_rarefaction_check.py, and group
tests on these values in 02_length_entropy_stats.py.

Input : one region-tagged fragment table per sample (csv/tsv, .gz ok) with a
        length column and the boolean is_<region> columns
Output: <out>/length_entropy_per_sample_cell.csv  long table: sample, size_bin,
        region, n_fragments, K, passes_min_frags, entropy
        <out>/length_entropy_wide.csv  one row per sample, one column per
        cell named <size_bin>__<region>

  python 01_length_entropy_calculation.py \\
      --fragments "fragments/*.fragments_with_motifs.csv.gz" \\
      -o entropy/ -t 14
"""
import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# Eleven size bins of Methods 2.4: (label, lowest length, highest length);
# None = open-ended
SIZE_BINS = [("<90bp", 1, 89), ("90-125bp", 90, 125), ("126-135bp", 126, 135),
             ("136-150bp", 136, 150), ("151-255bp", 151, 255), ("256-420bp", 256, 420),
             ("421-650bp", 421, 650), ("651-1000bp", 651, 1000), ("1001-2000bp", 1001, 2000),
             ("2001-5000bp", 2001, 5000), (">5000bp", 5001, None)]
REGION_COLS = ["is_promoter", "is_gene_body", "is_tss", "is_exon", "is_cpg_island"]
OPEN_CAP = 50000          # largest K for the >5000 bp bin
MIN_FRAGS = 100           # fragments per cell, as in the motif analysis


def detect_sep(path):
    """Comma or tab, whichever appears more often in the header line."""
    with pd.io.common.get_handle(path, "r", compression="infer") as h:
        first = h.handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def as_bool(s):
    """Region flags may come back as bool, 'True'/'False' strings or 1/0."""
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"]).to_numpy()


def add_hist(acc, lengths):
    """Add a bincount of the lengths to the running histogram, growing it if needed."""
    b = np.bincount(lengths)
    if len(b) > len(acc):
        acc = np.concatenate([acc, np.zeros(len(b) - len(acc), np.int64)])
    acc[:len(b)] += b
    return acc


def read_sample(args):
    """Read one sample in chunks; return its length histogram per region and its longest fragment."""
    path, sample, length_col, region_cols, chunksize = args
    sep = detect_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0).columns
    regions = [r for r in region_cols if r in header]
    missing = [r for r in region_cols if r not in header]
    if missing:
        print(f"  [!] {sample}: no column {missing}, those regions are skipped", file=sys.stderr)
    hists = {"genome_wide": np.zeros(0, np.int64)}
    for r in regions:
        hists[r] = np.zeros(0, np.int64)
    max_len = 0
    for ch in pd.read_csv(path, sep=sep, usecols=[length_col] + regions, chunksize=chunksize):
        L = pd.to_numeric(ch[length_col], errors="coerce").to_numpy()
        ok = np.isfinite(L) & (L >= 1)
        ch, L = ch[ok], L[ok].astype(np.int64)
        if not len(L):
            continue
        max_len = max(max_len, int(L.max()))
        hists["genome_wide"] = add_hist(hists["genome_wide"], L)
        for r in regions:
            hists[r] = add_hist(hists[r], L[as_bool(ch[r])])
    return sample, hists, max_len


def norm_entropy(counts, K):
    """Normalised plug-in Shannon entropy of a length histogram with K possible lengths."""
    counts = counts[counts > 0]
    n = counts.sum()
    if n == 0 or K <= 1:
        return np.nan
    p = counts / n
    return float(-(p * np.log2(p)).sum() / np.log2(K))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True,
                    help="glob of per-sample region-tagged fragment tables (csv.gz); quote it")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--sample-regex", default=r"^([^.]+)",
                    help="regex on the file name; group 1 = sample id")
    ap.add_argument("--length-col", default="length", help="fragment length column")
    ap.add_argument("--region-cols", default=",".join(REGION_COLS),
                    help="comma-separated boolean region columns (genome_wide is always added)")
    ap.add_argument("--min-frags", type=int, default=MIN_FRAGS,
                    help="cells with fewer fragments get entropy = missing")
    ap.add_argument("--open-cap", type=int, default=OPEN_CAP,
                    help="largest K for the open-ended >5000 bp bin")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time per sample")
    ap.add_argument("-t", "--threads", type=int, default=max(1, int((os.cpu_count() or 2) * 0.5)),
                    help="samples read in parallel")
    a = ap.parse_args()

    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)
    region_cols = [c for c in a.region_cols.split(",") if c]

    jobs = []
    for f in files:
        m = re.match(a.sample_regex, os.path.basename(f))
        jobs.append((f, m.group(1) if m else os.path.basename(f), a.length_col, region_cols, a.chunksize))

    # 1. length histograms per sample and region
    hists, observed_max = {}, 0
    with ProcessPoolExecutor(max_workers=max(1, min(a.threads, len(jobs)))) as ex:
        for sample, h, mx in ex.map(read_sample, jobs):
            hists[sample] = h
            observed_max = max(observed_max, mx)
            print(f"  {sample}: {int(h['genome_wide'].sum()):,} fragments, longest {mx:,} bp")

    # 2. entropy per cell
    rows = []
    for lab, lo, hi in SIZE_BINS:
        top = hi if hi is not None else min(max(observed_max, lo), a.open_cap)
        K = top - lo + 1
        for sample in sorted(hists):
            for region, h in hists[sample].items():
                # counts at lengths lo..hi (open bin: every length >= lo)
                c = h[lo:(hi + 1 if hi is not None else len(h))] if len(h) > lo else np.zeros(0, np.int64)
                n = int(c.sum())
                ok = n >= a.min_frags
                rows.append(dict(sample=sample, size_bin=lab, region=region, n_fragments=n, K=K,
                                 passes_min_frags=ok, entropy=norm_entropy(c, K) if ok else np.nan))
    long = pd.DataFrame(rows)
    long_path = os.path.join(a.out, "length_entropy_per_sample_cell.csv")
    long.to_csv(long_path, index=False)

    # 3. wide table, one column per cell (feature table for later steps)
    long["cell"] = long["size_bin"] + "__" + long["region"]
    order = [f"{lab}__{r}" for lab, _, _ in SIZE_BINS for r in ["genome_wide"] + region_cols]
    wide = long.pivot(index="sample", columns="cell", values="entropy")
    wide = wide.reindex(columns=[c for c in order if c in wide.columns])
    wide_path = os.path.join(a.out, "length_entropy_wide.csv")
    wide.to_csv(wide_path)

    n_cells = long.groupby("cell")["passes_min_frags"].all()
    print(f"\n>5000bp bin: K = {K} (longest fragment {observed_max:,} bp, cap {a.open_cap:,})")
    print(f"cells where every sample has >= {a.min_frags} fragments: {int(n_cells.sum())} of {len(n_cells)}")
    print(f"wrote {long_path}\nwrote {wide_path}")


if __name__ == "__main__":
    main()
