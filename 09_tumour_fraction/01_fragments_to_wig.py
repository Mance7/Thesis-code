#!/usr/bin/env python3
"""
01_fragments_to_wig.py
----------------------
Fragment counts per 1 Mb bin, written as an ichorCNA-style WIG file
(Methods 2.12). Each fragment is counted once, in the bin that holds its
midpoint, so the counts come from the same blacklist-filtered fragments as
every other analysis. For the thesis, direct fragments only
(--keep source=direct), ONT and PacBio.

Input:  per-sample eligible fragment tables (csv.gz with chrom, start, end) and the
        reference .fai (chromosome lengths; chr1-22, X, Y are used).
Output: <out>/<sample>.1000kb.wig, fixedStep blocks per chromosome with
        ceil(length / bin) bins, as HMMcopy readCounter writes them.
Samples are processed in parallel (-t).

Usage:
  python 01_fragments_to_wig.py --fragments "filtered_mid/*.eligible_fragments.csv.gz" \
      --fai GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai \
      --out tumour_fraction/wig_ont/ --keep source=direct -t 14
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]


def read_fai(path):
    fai = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "len"])
    return {c: int(l) for c, l in zip(fai.chrom, fai.len) if c in CHROMS}


def one_sample(path, sample, lens, a):
    # ceil(len / bin) bins per chromosome, as HMMcopy readCounter writes them
    counts = {c: np.zeros(-(-lens[c] // a.bin), dtype=np.int64) for c in lens}
    kcol, kvals = (a.keep.split("=", 1)[0], set(a.keep.split("=", 1)[1].split(","))) if a.keep else (None, None)
    cols = [a.chrom_col, a.start_col, a.end_col] + ([kcol] if kcol else [])
    n = 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=a.chunksize):
        if kcol:
            ch = ch[ch[kcol].astype(str).isin(kvals)]
        ch = ch[ch[a.chrom_col].isin(counts.keys())]
        mid = ((ch[a.start_col].to_numpy(np.int64) + ch[a.end_col].to_numpy(np.int64)) // 2) // a.bin
        for c, idx in pd.Series(mid).groupby(ch[a.chrom_col].to_numpy()):
            arr = counts[c]
            idx = idx.to_numpy()
            idx = idx[idx < len(arr)]
            np.add.at(arr, idx, 1)
            n += len(idx)
    out = os.path.join(a.out, f"{sample}.{a.bin // 1000}kb.wig")
    with open(out, "w") as fh:
        for c in CHROMS:
            if c not in counts:
                continue
            fh.write(f"fixedStep chrom={c} start=1 step={a.bin} span={a.bin}\n")
            fh.write("\n".join(map(str, counts[c])) + "\n")
    return sample, n, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample eligible fragment tables (csv.gz)")
    ap.add_argument("--fai", required=True, help="path to the reference .fai index (chromosome lengths)")
    ap.add_argument("-o", "--out", required=True, help="output folder for the WIG files")
    ap.add_argument("--bin", type=int, default=1_000_000, help="bin size in bp (default 1 Mb)")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("-t", "--threads", type=int, default=4, help="samples processed in parallel")
    a = ap.parse_args()
    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)
    lens = read_fai(a.fai)
    if len(lens) < 22:
        sys.exit(f"only {len(lens)} chr1-22/X/Y names found in {a.fai}; expected 'chr' names")
    jobs = [(f, re.match(a.sample_regex, os.path.basename(f)).group(1)) for f in files]
    res = Parallel(n_jobs=a.threads)(delayed(one_sample)(f, s, lens, a) for f, s in jobs)
    for s, n, o in res:
        print(f"  {s}: {n:,} fragments -> {o}")


if __name__ == "__main__":
    main()
