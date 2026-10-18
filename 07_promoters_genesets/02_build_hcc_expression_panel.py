#!/usr/bin/env python3
"""
02_build_hcc_expression_panel.py
--------------------------------
Builds the "HCC-expressed" gene set (Methods 2.8): genes with a median
expression of at least 1 pTPM across the liver cancer samples of the TCGA
cancer sample RNA data in the Human Protein Atlas v25.1 (Ensembl 109).
For the thesis this gave 11,379 genes. This is a much larger set than the
28-gene driver panel of 01_build_hcc_gene_panel.py.

Input
-----
The HPA RNA cancer sample file (rna_cancer_sample.tsv or .tsv.zip), one row
per gene x sample. Columns are found by name (case-insensitive): a gene
identifier column (Gene), a cancer-type column (Cancer) and an expression
column (pTPM in v25.1). The --*-col options name them if the file differs.
The values are first reduced to one median per gene across the liver cancer
samples; --already-aggregated skips this for a file that already has one row
per gene x cancer type.

Output
------
<out>/hcc_expression_panel.tsv, tab-separated: gene_id,
median_ntpm_liver_cancer (the median expression value, pTPM for v25.1),
source = "hcc_expressed"; one row per gene that passes --min-ntpm, sorted
from highest to lowest expression. The plain gene list used by the gene-set
scripts (hcc_expressed.txt, one Ensembl id per line, no header) is its first
column:  tail -n +2 hcc_expression_panel.tsv | cut -f1 > hcc_expressed.txt

Usage
-----
python 02_build_hcc_expression_panel.py --hpa-file rna_cancer_sample.tsv.zip \
    --cancer-value "liver cancer" --min-ntpm 1.0 -o gene_sets/
"""

import argparse
import os

import numpy as np
import pandas as pd


def find_col(columns, candidates):
    """First column whose name matches one of the candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpa-file", required=True,
                    help="path to the HPA v25.1 RNA cancer sample table (tsv or tsv.zip)")
    ap.add_argument("--cancer-value", default="liver cancer",
                    help="value of the cancer-type column that marks the liver cancer samples "
                         "(matched case-insensitively)")
    ap.add_argument("--min-ntpm", type=float, default=1.0,
                    help="minimum median expression across liver cancer samples to count a gene as "
                         "expressed (pTPM in HPA v25.1; default 1)")
    ap.add_argument("--gene-id-col", default=None, help="name of the gene identifier column (default: found by name)")
    ap.add_argument("--gene-symbol-col", default=None, help="name of the gene symbol column (not needed for the output)")
    ap.add_argument("--cancer-col", default=None, help="name of the cancer-type column (default: found by name)")
    ap.add_argument("--value-col", default=None, help="name of the expression column (default: found by name)")
    ap.add_argument("--already-aggregated", action="store_true",
                    help="the file has one row per gene x cancer type (no per-sample rows), "
                         "so no median across samples is taken")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder; hcc_expression_panel.tsv is written there")
    args = ap.parse_args()

    df = pd.read_csv(args.hpa_file, sep="\t", compression="infer")

    id_col = args.gene_id_col or find_col(df.columns, ["Gene", "gene_id", "Ensembl"])
    cancer_col = args.cancer_col or find_col(df.columns, ["Cancer", "cancer_type", "cancer"])
    value_col = args.value_col or find_col(df.columns, ["nTPM", "TPM", "value", "FPKM", "pTPM"])

    missing = [n for n, v in [("gene id", id_col),
                              ("cancer type", cancer_col), ("expression value", value_col)] if v is None]
    if missing:
        raise SystemExit(f"Could not find column(s) for: {missing}. "
                         f"Columns in the file: {list(df.columns)}. "
                         f"Name them with --gene-id-col/--cancer-col/--value-col.")
    print(f"[info] using columns: id={id_col}, cancer={cancer_col}, value={value_col}")

    # keep the liver cancer rows only
    hcc = df[df[cancer_col].astype(str).str.lower() == args.cancer_value.lower()]
    if hcc.empty:
        available = sorted(df[cancer_col].astype(str).unique())
        raise SystemExit(f"No rows matched --cancer-value '{args.cancer_value}'. "
                         f"Values in the file: {available}")
    print(f"[info] {len(hcc)} rows match cancer type '{args.cancer_value}' "
          f"({hcc[id_col].nunique()} unique genes, "
          f"{'per-sample' if not args.already_aggregated else 'pre-aggregated'})")

    # one value per gene: the median across liver cancer samples
    if args.already_aggregated:
        agg = hcc.groupby([id_col], as_index=False)[value_col].first()
    else:
        agg = hcc.groupby([id_col], as_index=False)[value_col].median()
    agg = agg.rename(columns={value_col: "median_ntpm_liver_cancer"})

    expressed = agg[agg["median_ntpm_liver_cancer"] >= args.min_ntpm].copy()
    expressed["source"] = "hcc_expressed"
    expressed = expressed.rename(columns={id_col: "gene_id"})
    expressed = expressed[["gene_id", "median_ntpm_liver_cancer", "source"]] \
        .sort_values("median_ntpm_liver_cancer", ascending=False)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "hcc_expression_panel.tsv")
    expressed.to_csv(out_path, sep="\t", index=False)
    print(f"\n[info] {len(expressed)} / {agg[id_col].nunique() if id_col in agg else len(agg)} genes pass "
          f"--min-ntpm {args.min_ntpm} in {args.cancer_value}")
    print(f"[info] wrote {out_path}")


if __name__ == "__main__":
    main()
