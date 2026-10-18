#!/usr/bin/env python3
"""Per-sample fragment-length histograms at 1 bp resolution.

Writes one small table: rows = samples, columns = lengths 1..--max-len
(fragments longer than --max-len are counted in the last column, ">max").
From it, nucleosome-repeat periodicity in the 1001-2000 bp bin, rarefied
entropy, the KS distance between groups (05_ks_size_profiles.py), the size
bin fold and window checks (06_size_bin_fold_and_window.py,
07_size_bins_locked_ont.py) and any size ratio can be computed without
reading the fragment files again.

Input : per-sample fragment tables (*.eligible_fragments.csv.gz) with a length column
Output: <out>/length_hist_<name>.tsv.gz

Samples are processed in parallel (-t). The thesis ran it four times
(ONT / PacBio, all / direct fragments):
  python 04_length_histograms.py --fragments "fragments/*.eligible_fragments.csv.gz" \\
      --out length_hist --name ont_direct --keep source=direct -t 14
  (names ont_all, ont_direct, pacbio_all, pacbio_direct; PacBio input:
   pacbio_fragments/*.eligible_fragments.csv.gz)
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def one_sample(path, sample, a):
    """Length histogram of one sample (optionally only rows passing --keep)."""
    hist = np.zeros(a.max_len + 2, dtype=np.int64)          # index 0 unused, last = > max_len
    kcol, kvals = (a.keep.split("=", 1)[0], set(a.keep.split("=", 1)[1].split(","))) if a.keep else (None, None)
    cols = [a.length_col] + ([kcol] if kcol else [])
    for ch in pd.read_csv(path, usecols=cols, chunksize=a.chunksize):
        if kcol:
            ch = ch[ch[kcol].astype(str).isin(kvals)]
        L = ch[a.length_col].to_numpy(np.int64)
        L = L[L > 0]
        hist += np.bincount(np.minimum(L, a.max_len + 1), minlength=a.max_len + 2)
    return sample, hist[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob of per-sample fragment tables (csv.gz); quote it")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--name", required=True, help="output file stem, e.g. ont_direct")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("--max-len", type=int, default=5000, help="longest length with its own column; longer fragments go to >max")
    ap.add_argument("--length-col", default="length", help="fragment length column")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="regex on the file name; group 1 = sample id")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time per sample")
    ap.add_argument("-t", "--threads", type=int, default=4, help="samples processed in parallel")
    a = ap.parse_args()
    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, re.match(a.sample_regex, os.path.basename(f)).group(1)) for f in files]
    res = Parallel(n_jobs=a.threads)(delayed(one_sample)(f, s, a) for f, s in jobs)
    cols = [str(i) for i in range(1, a.max_len + 1)] + [f">{a.max_len}"]
    df = pd.DataFrame([h for _, h in res], index=[s for s, _ in res], columns=cols)
    df.index.name = "sample"
    out = os.path.join(a.out, f"length_hist_{a.name}.tsv.gz")
    df.to_csv(out, sep="\t", compression="gzip")
    for s, h in res:
        print(f"  {s}: {h.sum():,} fragments, 1001-2000 bp: {h[1000:2000].sum():,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
