#!/usr/bin/env python3
"""
Make the gene-level promoter BED used by the promoter and gene-set analyses.

One promoter per GENCODE v49 gene record: the window runs from 2,000 bp upstream
to 500 bp downstream of the gene-level TSS, strand aware (TSS = gene start on the
+ strand, gene end on the - strand; on the - strand the window is mirrored).
Windows that would start before position 0 are clipped at 0.

Genes kept: gene_type "protein_coding" on the canonical chromosomes
(chr1-22, chrX, chrY, chrM). chrM and chrY promoters are kept here and removed
later in the analysis.

Input:  GENCODE v49 comprehensive annotation GTF (plain or .gz)
Output: promoters_gencode_v49_gene_level.bed, tab separated, no header:
        chrom, start, end, gene_id (Ensembl, with version), gene_name, strand
        sorted by chromosome (chr1..chr22, chrX, chrY, chrM) and start
"""

import argparse
from pathlib import Path

import pandas as pd

CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
CHROM_ORDER = {c: i for i, c in enumerate(CANONICAL)}

GTF_COLUMNS = ["chrom", "source", "feature", "start", "end",
               "score", "strand", "frame", "attributes"]


def get_attribute(attributes, key):
    """Pull one value (e.g. gene_id "ENSG...") out of the GTF attribute column."""
    return attributes.str.extract(rf'{key} "([^"]+)"', expand=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gtf", required=True,
                        help="path to the GENCODE v49 annotation (gtf or gtf.gz)")
    parser.add_argument("--gene-type", default="protein_coding",
                        help="GENCODE gene_type of the genes kept")
    parser.add_argument("--upstream", type=int, default=2000,
                        help="bp upstream of the TSS included in the promoter window")
    parser.add_argument("--downstream", type=int, default=500,
                        help="bp downstream of the TSS included in the promoter window")
    parser.add_argument("-o", "--out",
                        required=True,
                        help="output promoter BED file")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Reading GTF...")
    gtf = pd.read_csv(args.gtf, sep="\t", comment="#", header=None, names=GTF_COLUMNS,
                      dtype={"chrom": str}, low_memory=False)
    genes = gtf[(gtf["feature"] == "gene") & gtf["chrom"].isin(CHROM_ORDER)].copy()
    genes["gene_id"] = get_attribute(genes["attributes"], "gene_id")
    genes["gene_name"] = get_attribute(genes["attributes"], "gene_name")
    genes["gene_type"] = get_attribute(genes["attributes"], "gene_type")
    print(f"  {len(genes):,} gene records on canonical chromosomes")

    genes = genes[genes["gene_type"] == args.gene_type].copy()
    print(f"  {len(genes):,} with gene_type {args.gene_type}")

    # 0-based TSS: GTF start - 1 on the + strand, GTF end - 1 on the - strand
    plus = genes["strand"] == "+"
    tss = (genes["start"] - 1).where(plus, genes["end"] - 1)

    # windows of up + down bp, same convention as 01_making_bed_resources.py:
    # + strand: [TSS - up, TSS + down); - strand: [TSS + 1 - down, TSS + 1 + up)
    start = (tss - args.upstream).where(plus, tss + 1 - args.downstream)
    end = (tss + args.downstream).where(plus, tss + 1 + args.upstream)

    prom = pd.DataFrame({
        "chrom": genes["chrom"],
        "start": start.clip(lower=0),
        "end": end,
        "gene_id": genes["gene_id"],
        "gene_name": genes["gene_name"],
        "strand": genes["strand"],
    })
    prom["order"] = prom["chrom"].map(CHROM_ORDER)
    prom = prom.sort_values(["order", "start", "end"], kind="mergesort").drop(columns="order")

    prom.to_csv(out, sep="\t", index=False, header=False)
    print(f"{len(prom):,} promoters written to {out}")
    print(prom["chrom"].value_counts().reindex(CANONICAL).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
