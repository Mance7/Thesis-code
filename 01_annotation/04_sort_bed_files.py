#!/usr/bin/env python3
"""
Sort every BED file in the resources folder by natural chromosome order,
then start and end.

Rows on non-canonical chromosomes (anything other than chr1-22, X, Y, M) are
dropped. The folder layout under --resources (promoters/, tss/, exons/, ...)
is kept in --out, so the originals are not overwritten.

Input:  folder with the BED files from 01_making_bed_resources.py and
        03_canonicalize_cpg_islands.py
Output: sorted copies of every *.bed file, same relative paths, in --out
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--resources", required=True,
                    help="folder searched (recursively) for *.bed files")
parser.add_argument("-o", "--out", required=True,
                    help="output folder for the sorted BED files")
args = parser.parse_args()

ROOT = Path(args.resources)
OUT = Path(args.out)

chrom_order = {
    f"chr{i}": i for i in range(1, 23)
}
chrom_order["chrX"] = 23
chrom_order["chrY"] = 24
chrom_order["chrM"] = 25


def sort_bed(path, out_path):
    print(f"Sorting {path.name}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype={0: str}
    )

    # keep only canonical chromosomes
    df = df[df[0].isin(chrom_order)]

    df["chrom_order"] = df[0].map(chrom_order)

    # stable sort so rows with equal coordinates keep their input order
    df = df.sort_values(
        ["chrom_order", 1, 2],
        kind="mergesort"
    )

    df = df.drop(columns="chrom_order")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_path,
        sep="\t",
        header=False,
        index=False
    )


for bed in sorted(ROOT.rglob("*.bed")):
    # do not pick up files already written to an output folder inside ROOT
    if OUT.resolve() in bed.resolve().parents:
        continue
    sort_bed(bed, OUT / bed.relative_to(ROOT))

print("\nDone.")
