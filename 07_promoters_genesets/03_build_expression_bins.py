#!/usr/bin/env python3
"""
03_build_expression_bins.py
---------------------------
Ranks genes by their expression in one reference tissue (default Whole
Blood, the main normal source of plasma cfDNA) and puts each gene into one
of 30 equal-frequency bins, from lowest (1) to highest (30) expression
(Methods 2.8). The bins are used to match random promoter sets to a gene
set's blood expression (07_geneset_random_null.py) and to order promoter
pools by blood expression (09_geneset_pool_trend.py).

About 60% of genes have zero whole-blood TPM. Ties are broken by row order
(rank method "first"), so bins 1-19 all hold zero-TPM genes and their order
inside that range carries no information. The script sets no expression
threshold; the blood-silent cut (TPM < 1) is made in
04_build_hcc_expressed_subsets.py.

Input
-----
A GTEx gene median TPM file (.gct or .gct.gz; GTEx v11 for the thesis):
two header lines (version, then "n_genes n_tissues"), then a header row
Name, Description, <tissue columns...>, then one row per gene. Name is the
versioned Ensembl gene id (e.g. ENSG00000141510.18), Description the symbol.

Output
------
<out>/blood_expression_bins.tsv, tab-separated: gene_id (versioned, as in
the GTEx file), gene_id_noversion, gene_symbol, tpm, expression_bin
(1 = lowest .. n_bins = highest); one row per gene with a value.

Usage
-----
python 03_build_expression_bins.py \
    --expression-file GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz \
    --tissue "Whole Blood" --n-bins 30 -o gene_sets/
"""

import argparse
import gzip
import os

import numpy as np
import pandas as pd


def open_maybe_gzip(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_gct(path):
    """Read a GTEx .gct table (first two lines are GCT metadata)."""
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="infer")
    if list(df.columns[:2]) != ["Name", "Description"]:
        raise SystemExit(f"Expected GCT columns starting with Name, Description; got {list(df.columns[:2])}. "
                         f"Is this a GTEx gene median TPM .gct file?")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expression-file", required=True,
                    help="path to the GTEx gene median TPM table (.gct or .gct.gz)")
    ap.add_argument("--tissue", default="Whole Blood",
                    help="GTEx tissue column to rank on (default: 'Whole Blood')")
    ap.add_argument("--n-bins", type=int, default=30, help="number of expression bins (default 30)")
    ap.add_argument("--bin-method", choices=["quantile", "equal-width-log"], default="quantile",
                    help="quantile (default, used for the thesis): each bin holds about the same number of "
                         "genes. equal-width-log: each bin spans the same width of log(1 + TPM).")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder; blood_expression_bins.tsv is written there")
    args = ap.parse_args()

    df = load_gct(args.expression_file)
    if args.tissue not in df.columns:
        close = [c for c in df.columns if args.tissue.lower() in c.lower()]
        raise SystemExit(f"Tissue column '{args.tissue}' not found. "
                         f"Closest matches: {close[:10] if close else 'none'}. "
                         f"Available columns: {list(df.columns)}")

    out = df[["Name", "Description", args.tissue]].rename(
        columns={"Name": "gene_id", "Description": "gene_symbol", args.tissue: "tpm"})
    out["gene_id_noversion"] = out["gene_id"].str.split(".").str[0]
    out = out[out["tpm"].notna()].copy()

    if args.bin_method == "quantile":
        # rank-based: tied genes (e.g. the many genes at tpm = 0) are spread over the lowest bins in row order
        out["expression_bin"] = pd.qcut(out["tpm"].rank(method="first"), args.n_bins,
                                        labels=range(1, args.n_bins + 1)).astype(int)
    else:
        log_tpm = np.log1p(out["tpm"])
        out["expression_bin"] = pd.cut(log_tpm, args.n_bins, labels=range(1, args.n_bins + 1)).astype(int)

    out = out[["gene_id", "gene_id_noversion", "gene_symbol", "tpm", "expression_bin"]]
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "blood_expression_bins.tsv")
    out.to_csv(out_path, sep="\t", index=False)

    print(f"[info] ranked {len(out)} genes on '{args.tissue}' expression into {args.n_bins} bins "
          f"({args.bin_method})")
    print(f"[info] bin 1 (lowest) tpm range: {out[out.expression_bin==1]['tpm'].min():.3g}-"
          f"{out[out.expression_bin==1]['tpm'].max():.3g}, "
          f"n={int((out.expression_bin==1).sum())}")
    top = args.n_bins
    print(f"[info] bin {top} (highest) tpm range: {out[out.expression_bin==top]['tpm'].min():.3g}-"
          f"{out[out.expression_bin==top]['tpm'].max():.3g}, "
          f"n={int((out.expression_bin==top).sum())}")
    print(f"[info] wrote {out_path}")


if __name__ == "__main__":
    main()
