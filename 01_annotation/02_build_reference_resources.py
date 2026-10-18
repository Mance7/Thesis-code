#!/usr/bin/env python3
"""
Build basic reference resources from the GRCh38 FASTA.

Outputs (in --out):
  GRCh38.chrom.sizes          length of each canonical chromosome (chr1-22, X, Y);
                              used to clip the annotation BED files to chromosome ends
  Genome_10kb_windows.bed     fixed 10 kb windows across the canonical chromosomes
  Genome_100kb_windows.bed    fixed 100 kb windows across the canonical chromosomes
  reference_resources.json    list of the files above

Input: GRCh38 no-alt analysis set FASTA with its .fai index (samtools faidx)
"""

import argparse
import json
import os

import pysam

#########################################
# SETTINGS
#########################################

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reference",
                    required=True,
                    help="path to the GRCh38 reference genome (fasta, indexed)")
parser.add_argument("-o", "--out", required=True,
                    help="output folder for chromosome sizes and window BED files")
args = parser.parse_args()

REFERENCE = args.reference

OUTDIR = args.out

#########################################

os.makedirs(OUTDIR, exist_ok=True)

CANONICAL = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
    "chr8", "chr9", "chr10", "chr11", "chr12", "chr13",
    "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr20", "chr21", "chr22", "chrX", "chrY"
]


###########################################################
# chromosome sizes
###########################################################

print("Reading reference...")

fasta = pysam.FastaFile(REFERENCE)

chrom_sizes = {}

for chrom, length in zip(fasta.references, fasta.lengths):

    if chrom in CANONICAL:
        chrom_sizes[chrom] = length

sizes_file = os.path.join(
    OUTDIR,
    "GRCh38.chrom.sizes"
)

with open(sizes_file, "w") as out:

    for chrom in CANONICAL:

        out.write(f"{chrom}\t{chrom_sizes[chrom]}\n")

print("Chromosome sizes written.")


###########################################################
# window maker
###########################################################

def make_windows(window_size, outfile):
    """Write consecutive windows of window_size bp; the last window of each chromosome is shorter."""

    with open(outfile, "w") as out:

        for chrom in CANONICAL:

            L = chrom_sizes[chrom]

            start = 0

            while start < L:

                end = min(start + window_size, L)

                out.write(
                    f"{chrom}\t{start}\t{end}\n"
                )

                start += window_size


###########################################################
# 10 kb
###########################################################

print("Creating 10 kb windows...")

bed10 = os.path.join(
    OUTDIR,
    "Genome_10kb_windows.bed"
)

make_windows(10000, bed10)

###########################################################
# 100 kb
###########################################################

print("Creating 100 kb windows...")

bed100 = os.path.join(
    OUTDIR,
    "Genome_100kb_windows.bed"
)

make_windows(100000, bed100)


###########################################################
# metadata
###########################################################

metadata = {

    "reference": REFERENCE,

    "resources": {

        "chrom_sizes": {

            "path": sizes_file,

            "description": "Canonical chromosome sizes"

        },

        "windows_10kb": {

            "path": bed10,

            "window_size": 10000,

            "description": "Genome-wide fixed windows"

        },

        "windows_100kb": {

            "path": bed100,

            "window_size": 100000,

            "description": "Genome-wide fixed windows"

        }

    }

}

with open(
    os.path.join(
        OUTDIR,
        "reference_resources.json"
    ),
    "w"
) as f:

    json.dump(metadata, f, indent=4)

print()
print("Finished.")
