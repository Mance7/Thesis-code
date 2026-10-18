#!/usr/bin/env python3
"""
Extract transcription start sites (TSS) from the GENCODE v49 GTF.

For every transcript record the TSS is its first base on its own strand:
the feature start on the + strand and the feature end on the - strand.
Each TSS is written as a 1-bp BED interval. Only canonical chromosomes
(chr1-22, X, Y) are kept. Identical positions (shared by several
transcripts) are deduplicated, and adjacent or overlapping TSSs are then
merged into one interval, as bedtools merge does.

Outputs (in --out):
  tss_transcripts_canonical.bed   chrom, start, end, strand, gene_id, gene_name, transcript_id
                                  (one row per transcript)
  tss_merged.bed                  chrom, start, end (deduplicated and merged; the TSS region flag)
  tss_gene_level.tsv              gene_id, gene_name, chrom, strand, tss (0-based TSS of each
                                  gene record, used for TSS-anchored promoter metrics)

Input: GENCODE v49 comprehensive annotation GTF (plain or .gz)
"""

import argparse
from pathlib import Path

import pandas as pd

CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_ORDER = {c: i for i, c in enumerate(CANONICAL)}

GTF_COLUMNS = ["chrom", "source", "feature", "start", "end",
               "score", "strand", "frame", "attributes"]


def get_attribute(attributes, key):
    """Pull one value (e.g. gene_id "ENSG...") out of the GTF attribute column."""
    return attributes.str.extract(rf'{key} "([^"]+)"', expand=False)


def read_gtf(path, features):
    """Read the GTF rows of the requested feature types on canonical chromosomes."""
    gtf = pd.read_csv(path, sep="\t", comment="#", header=None, names=GTF_COLUMNS,
                      dtype={"chrom": str}, low_memory=False)
    gtf = gtf[gtf["feature"].isin(features) & gtf["chrom"].isin(CHROM_ORDER)].copy()
    return gtf


def tss_position(df):
    """0-based TSS position: GTF start - 1 on the + strand, GTF end - 1 on the - strand."""
    return (df["start"] - 1).where(df["strand"] == "+", df["end"] - 1)


def sort_bed(df):
    df = df.assign(order=df["chrom"].map(CHROM_ORDER))
    return df.sort_values(["order", "start", "end"], kind="mergesort").drop(columns="order")


def merge_intervals(df):
    """Merge overlapping or book-ended intervals per chromosome (bedtools merge behaviour)."""
    merged = []
    for chrom, sub in df.groupby("chrom", sort=False):
        cur_start, cur_end = None, None
        for s, e in zip(sub["start"].to_numpy(), sub["end"].to_numpy()):
            if cur_start is None:
                cur_start, cur_end = s, e
            elif s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((chrom, cur_start, cur_end))
                cur_start, cur_end = s, e
        if cur_start is not None:
            merged.append((chrom, cur_start, cur_end))
    return pd.DataFrame(merged, columns=["chrom", "start", "end"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gtf", required=True,
                        help="path to the GENCODE v49 annotation (gtf or gtf.gz)")
    parser.add_argument("-o", "--out", required=True,
                        help="output folder for the TSS files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Reading GTF...")
    gtf = read_gtf(args.gtf, {"gene", "transcript"})

    # ---- transcript-level TSS ----
    tx = gtf[gtf["feature"] == "transcript"].copy()
    tx["tss"] = tss_position(tx)
    tss = pd.DataFrame({
        "chrom": tx["chrom"],
        "start": tx["tss"],
        "end": tx["tss"] + 1,
        "strand": tx["strand"],
        "gene_id": get_attribute(tx["attributes"], "gene_id"),
        "gene_name": get_attribute(tx["attributes"], "gene_name"),
        "transcript_id": get_attribute(tx["attributes"], "transcript_id"),
    })
    tss = sort_bed(tss)
    tss.to_csv(out / "tss_transcripts_canonical.bed", sep="\t", index=False, header=False)
    print(f"  {len(tss):,} transcript TSSs on canonical chromosomes")

    # ---- deduplicate and merge ----
    # one row per position (strand ignored: the region flag is strand blind)
    dedup = tss.drop_duplicates(subset=["chrom", "start", "end"])[["chrom", "start", "end"]]
    merged = merge_intervals(dedup)
    merged.to_csv(out / "tss_merged.bed", sep="\t", index=False, header=False)
    print(f"  {len(dedup):,} unique TSS positions -> {len(merged):,} merged intervals")

    # ---- gene-level TSS table ----
    genes = gtf[gtf["feature"] == "gene"].copy()
    gene_tss = pd.DataFrame({
        "gene_id": get_attribute(genes["attributes"], "gene_id"),
        "gene_name": get_attribute(genes["attributes"], "gene_name"),
        "chrom": genes["chrom"],
        "strand": genes["strand"],
        "tss": tss_position(genes),
    })
    gene_tss.to_csv(out / "tss_gene_level.tsv", sep="\t", index=False)
    print(f"  {len(gene_tss):,} gene-level TSSs")

    print(f"Done. Files in {out}")


if __name__ == "__main__":
    main()
