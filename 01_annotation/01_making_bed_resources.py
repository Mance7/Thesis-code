#!/usr/bin/env python3
"""
Build the region BED files used to annotate fragments, from the GENCODE v49 GTF.

What it makes (all under --out):
  tss/        tss_transcripts.bed     1-bp start of every transcript (strand aware)
              tss_deduplicated.bed    same, one row per unique position and strand
              tss_collapsed.bed       deduplicated TSSs merged into intervals
              tss_gene_level.bed      1-bp start of every gene record
  promoters/  promoters_transcript.bed  -2000/+500 bp around each transcript TSS
              promoters_gene_level.bed  -2000/+500 bp around each gene-level TSS
  gene_bodies/ gene_bodies.bed, gene_bodies_gene_level.bed   GENCODE gene records
  exons/      exons_all.bed           every GENCODE exon
              exons_gene_merged.bed   exons merged within each gene
  metadata.tsv  annotation version, promoter window and feature counts

"Upstream" and "downstream" follow the strand of the gene: on the - strand the
TSS is the feature end and the window is mirrored.

Input:  GENCODE v49 comprehensive annotation GTF
Output: the BED files above (tab separated, no header)
"""

import argparse
from pathlib import Path

import pandas as pd
import pyranges as pr

# ==========================
# CONFIG
# ==========================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gtf",
                    required=True,
                    help="path to the GENCODE v49 annotation (gtf)")
parser.add_argument("--promoter-up", type=int, default=2000,
                    help="bp upstream of the TSS included in the promoter window")
parser.add_argument("--promoter-down", type=int, default=500,
                    help="bp downstream of the TSS included in the promoter window")
parser.add_argument("-o", "--out", required=True,
                    help="output folder for the BED resources")
args = parser.parse_args()

GTF = args.gtf

PROMOTER_UP = args.promoter_up
PROMOTER_DOWN = args.promoter_down

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

for d in [
    "promoters",
    "gene_bodies",
    "exons",
    "tss",
]:
    (OUT / d).mkdir(exist_ok=True)

# ==========================
# READ GTF
# ==========================

print("Reading GTF...")

gtf = pr.read_gtf(GTF)

genes = gtf[gtf.Feature == "gene"]
transcripts = gtf[gtf.Feature == "transcript"]
exons = gtf[gtf.Feature == "exon"]

# ==========================
# TSS (one per transcript)
# ==========================

print("Building TSS...")

tss = transcripts.df.copy()

plus = tss.Strand == "+"
minus = ~plus

# + strand: TSS is the first base; - strand: TSS is the last base
tss.loc[plus, "End"] = tss.loc[plus, "Start"] + 1
tss.loc[minus, "Start"] = tss.loc[minus, "End"] - 1

tss = tss[[
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "gene_id",
    "gene_name",
    "transcript_id"
]]

tss = tss.sort_values(["Chromosome", "Start"])

tss.to_csv(
    OUT / "tss" / "tss_transcripts.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Deduplicated TSS
# ==========================

# many transcripts share the same start; keep one row per position and strand
tss_dedup = tss.drop_duplicates(
    subset=["Chromosome", "Start", "End", "Strand"]
)

tss_dedup.to_csv(
    OUT / "tss" / "tss_deduplicated.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Collapsed (merged) TSS
# ==========================

pr.PyRanges(tss_dedup).merge().df.to_csv(
    OUT / "tss" / "tss_collapsed.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Gene-level TSS
# ==========================

gene_tss = genes.df.copy()

plus = gene_tss.Strand == "+"
minus = ~plus

gene_tss.loc[plus, "End"] = gene_tss.loc[plus, "Start"] + 1
gene_tss.loc[minus, "Start"] = gene_tss.loc[minus, "End"] - 1

gene_tss = gene_tss[[
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "gene_id",
    "gene_name"
]]

gene_tss.to_csv(
    OUT / "tss" / "tss_gene_level.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Promoters
# ==========================

print("Building promoters...")

# transcript level
prom = transcripts.df.copy()

plus = prom.Strand == "+"
minus = ~plus

# + strand: [TSS - up, TSS + down); - strand: mirrored around the feature end
prom.loc[plus, "Start"] = prom.loc[plus, "Start"] - PROMOTER_UP
prom.loc[plus, "End"] = prom.loc[plus, "Start"] + PROMOTER_UP + PROMOTER_DOWN

prom.loc[minus, "End"] = prom.loc[minus, "End"] + PROMOTER_UP
prom.loc[minus, "Start"] = prom.loc[minus, "End"] - PROMOTER_UP - PROMOTER_DOWN

# windows near the start of a chromosome cannot go below 0
prom.Start = prom.Start.clip(lower=0)

prom[[
    "Chromosome", "Start", "End",
    "Strand", "gene_id",
    "gene_name", "transcript_id"
]].sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "promoters" / "promoters_transcript.bed",
    sep="\t",
    index=False,
    header=False
)

# gene level (the promoter set used in the analysis)
prom = genes.df.copy()

plus = prom.Strand == "+"
minus = ~plus

prom.loc[plus, "Start"] = prom.loc[plus, "Start"] - PROMOTER_UP
prom.loc[plus, "End"] = prom.loc[plus, "Start"] + PROMOTER_UP + PROMOTER_DOWN

prom.loc[minus, "End"] = prom.loc[minus, "End"] + PROMOTER_UP
prom.loc[minus, "Start"] = prom.loc[minus, "End"] - PROMOTER_UP - PROMOTER_DOWN

prom.Start = prom.Start.clip(lower=0)

prom[[
    "Chromosome", "Start", "End",
    "Strand", "gene_id",
    "gene_name"
]].sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "promoters" / "promoters_gene_level.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Gene bodies
# ==========================

print("Building gene bodies...")

genes.df.sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "gene_bodies" / "gene_bodies.bed",
    sep="\t",
    index=False,
    header=False
)

genes.df.drop_duplicates(
    subset="gene_id"
).sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "gene_bodies" / "gene_bodies_gene_level.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Exons
# ==========================

print("Building exons...")

exons.df.sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "exons" / "exons_all.bed",
    sep="\t",
    index=False,
    header=False
)

# overlapping exons of the same gene (from different transcripts) become one interval
merged = pr.PyRanges(
    exons.df[[
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "gene_id"
    ]]
).merge(by="gene_id")

merged.df.sort_values(
    ["Chromosome", "Start"]
).to_csv(
    OUT / "exons" / "exons_gene_merged.bed",
    sep="\t",
    index=False,
    header=False
)

# ==========================
# Metadata
# ==========================

meta = pd.DataFrame([
    ["annotation", "GENCODE v49"],
    ["promoter_upstream", PROMOTER_UP],
    ["promoter_downstream", PROMOTER_DOWN],
    ["gene_count", len(genes.df)],
    ["transcript_count", len(transcripts.df)],
    ["exon_count", len(exons.df)]
], columns=["Key", "Value"])

meta.to_csv(
    OUT / "metadata.tsv",
    sep="\t",
    index=False
)

print("Done.")
