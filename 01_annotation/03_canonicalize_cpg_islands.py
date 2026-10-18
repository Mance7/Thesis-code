#!/usr/bin/env python3
"""
Canonicalize and sort the UCSC CpG island track for GRCh38.

Keeps only intervals on the canonical chromosomes (chr1-22, X, Y, M; chrM is
removed later by 05_validate_clip_beds_no_chrM.py) and sorts them in natural
chromosome order, then by start and end.

Input:  UCSC CpG Islands track for GRCh38 as a BED-like file (chrom, start, end, ...)
Output: cpg_islands.bed (same columns, canonical chromosomes, sorted)
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cpg", required=True,
                    help="path to the UCSC CpG islands track for GRCh38 (bed, tab separated, no header)")
parser.add_argument("-o", "--out",
                    required=True,
                    help="output BED file of canonical, sorted CpG islands")
args = parser.parse_args()

cpg_file = args.cpg
output = Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)

canonical = {
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
    "chr18", "chr19", "chr20", "chr21", "chr22",
    "chrX", "chrY", "chrM"
}

df = pd.read_csv(
    cpg_file,
    sep="\t",
    header=None
)

# drop alt, unplaced and random contigs
df = df[df[0].isin(canonical)]

# natural chromosome order (chr1, chr2, ... chr22, chrX, chrY, chrM)
chrom_order = {c: i for i, c in enumerate([
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
    "chr18", "chr19", "chr20", "chr21", "chr22",
    "chrX", "chrY", "chrM"
])}

df["order"] = df[0].map(chrom_order)

df = df.sort_values(["order", 1, 2])

df.drop(columns="order").to_csv(
    output,
    sep="\t",
    index=False,
    header=False
)

print(f"{len(df):,} CpG islands written to {output}")
