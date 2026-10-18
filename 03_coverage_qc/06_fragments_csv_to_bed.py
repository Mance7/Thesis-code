#!/usr/bin/env python3
"""
Converts each sample's Stage 1 fragment table into a sorted BED3 file, so
that coverage can be computed with bedtools on exactly the fragments used in
every downstream analysis (MAPQ >= 20, canonical chromosomes, concatemer
recovery, clip and SV rules already applied in Stage 1). mosdepth needs a
BAM, which these fragments do not have.

Coordinates in the fragment table are 0-based half-open (pysam), the same as
BED, so start/end are written unchanged. Rows are sorted in the chromosome
order of the genome file, which is what `bedtools coverage -sorted -g` and
`bedtools genomecov` expect.

Input : FRAG_DIR/{sample}.eligible_fragments.csv.gz
        (chrom,start,end,length,strand,mapq,source,has_incidental_sa)
Output: OUT/{sample}.fragments.sorted.bed
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

SUFFIX = ".eligible_fragments.csv.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frag-dir", type=Path, required=True,
                   help="folder with the Stage 1 fragment tables ({sample}.eligible_fragments.csv.gz)")
    p.add_argument("-g", "--genome", type=Path, required=True,
                   help="chromosome sizes file (chrom<TAB>length); sets the sort order")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder for the BED files")
    p.add_argument("--samples", nargs="+", default=None,
                   help="only convert these samples (default: all in --frag-dir)")
    p.add_argument("-t", "--threads", type=int, default=max(1, int((os.cpu_count() or 1) * 0.5)),
                   help="number of samples converted in parallel")
    return p.parse_args()


def load_chrom_order(path):
    sizes = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "length"])
    return list(sizes["chrom"])


def convert(job):
    frag_file, out_file, chrom_order = job
    df = pd.read_csv(frag_file, usecols=["chrom", "start", "end"],
                     dtype={"chrom": str, "start": "int64", "end": "int64"})
    n_in = len(df)

    # keep only chromosomes known to the genome file (Stage 1 already kept chr1-22, X, Y)
    df = df[df["chrom"].isin(chrom_order)]
    df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
    df = df.sort_values(["chrom", "start", "end"], kind="mergesort")

    tmp = out_file.with_suffix(".bed.tmp")
    df.to_csv(tmp, sep="\t", header=False, index=False)
    tmp.rename(out_file)
    return out_file.name, n_in, len(df)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chrom_order = load_chrom_order(args.genome)

    files = sorted(args.frag_dir.glob(f"*{SUFFIX}"))
    if args.samples:
        files = [f for f in files if f.name[:-len(SUFFIX)] in set(args.samples)]
    if not files:
        raise RuntimeError(f"No *{SUFFIX} files found in {args.frag_dir}")

    jobs = [(f, args.out / f"{f.name[:-len(SUFFIX)]}.fragments.sorted.bed", chrom_order)
            for f in files]
    print(f"Converting {len(jobs)} fragment tables with {args.threads} workers")
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        for name, n_in, n_out in ex.map(convert, jobs):
            print(f"  {name}: {n_out:,} / {n_in:,} fragments written")
    print(f"Done. BED files in {args.out}")


if __name__ == "__main__":
    main()
