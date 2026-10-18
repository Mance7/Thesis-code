#!/usr/bin/env python3
"""
04_build_hcc_expressed_subsets.py
---------------------------------
Splits the HCC-expressed promoter set into the six subsets that were fixed
before any of them was tested (pre-registered family H5b, Methods 2.8):
  S1_hcc_blood_silent     HCC-expressed and silent in blood (GTEx whole-blood TPM < 1)
  S2_hcc_blood_expressed  HCC-expressed and expressed in blood (TPM >= 1)
  S3_hcc_tertile_low      lowest third of the set by HCC expression
  S4_hcc_tertile_mid      middle third
  S5_hcc_tertile_high     highest third (S3-S5 test a dose-response)
  S6_liver_enriched       HCC-expressed and liver-enriched (HPA tissue specificity)
The blood-silent cut (TPM < 1) is our own choice for this analysis; the
blood-bin script (03_build_expression_bins.py) defines no threshold.

Inputs (local files, nothing is downloaded):
  --hcc-expressed   the HCC-expressed gene list (hcc_expressed.txt, one Ensembl id per line)
  --hcc-expr        HCC expression per gene: csv/tsv with a gene id column and a value column,
                    e.g. the HPA cancer sample table with --hcc-expr-filter (median per gene)
  --blood           blood expression per gene: blood_expression_bins.tsv from
                    03_build_expression_bins.py (value column tpm, silent = tpm < 1), or a bin
                    file with --blood-silent-bins listing the bins that mean "silent"
  --liver-enriched  gene id list, or the HPA proteinatlas.tsv (genes "Tissue enriched" or
                    "Group enriched" in liver are taken)

Outputs in --out:
  S1_...txt to S6_...txt   one gene id per line (version suffix removed)
  subsets_summary.csv      size and median HCC expression of every subset
  gene_sets_arg.txt        the NAME=file list for --gene-sets of 07_geneset_random_null.py

Usage:
  python 04_build_hcc_expressed_subsets.py --hcc-expressed hcc_expressed.txt \
      --hcc-expr rna_cancer_sample.tsv --hcc-expr-gene-col Gene --hcc-expr-value-col pTPM \
      --hcc-expr-filter "Cancer=liver cancer" \
      --blood blood_expression_bins.tsv --liver-enriched proteinatlas.tsv -o h5b_subsets/
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd


def strip_ver(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def read_any(path):
    return pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")


def read_list(path):
    return {strip_ver(l.split()[0]) for l in open(path) if l.strip() and not l.startswith("#")}


def read_liver(path):
    """Plain gene_id list, or HPA proteinatlas.tsv / search export (filtered to liver-enriched)."""
    first = open(path, encoding="utf-8", errors="replace").readline()
    if "RNA tissue specificity" not in first:
        return read_list(path)
    hp = pd.read_csv(path, sep="\t", encoding_errors="replace")
    spec = hp["RNA tissue specificity"].astype(str)
    ntpm = hp.get("RNA tissue specific nTPM", pd.Series([""] * len(hp))).astype(str).str.lower()
    keep = spec.isin(["Tissue enriched", "Group enriched"]) & ntpm.str.contains(r"\bliver\b")
    print(f"  HPA file: {int(keep.sum())} liver tissue/group-enriched genes", file=sys.stderr)
    return {strip_ver(g) for g in hp.loc[keep, "Ensembl"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hcc-expressed", required=True, help="path to the HCC-expressed gene list (one gene id per line)")
    ap.add_argument("--hcc-expr", required=True, help="path to a table of HCC expression per gene (csv/tsv)")
    ap.add_argument("--hcc-expr-gene-col", default=None, help="gene id column of --hcc-expr (default: first column)")
    ap.add_argument("--hcc-expr-value-col", default=None, help="expression column of --hcc-expr (default: second column)")
    ap.add_argument("--hcc-expr-filter", default=None,
                    help="for a long per-sample table (HPA rna_cancer_sample.tsv): COLUMN=VALUE to keep, "
                         "e.g. 'Cancer=Liver hepatocellular carcinoma'; values are then the per-gene median")
    ap.add_argument("--blood", required=True, help="path to blood expression per gene (blood_expression_bins.tsv)")
    ap.add_argument("--blood-silent-bins", default=None, help="comma-separated bin labels meaning silent in blood (used instead of --blood-silent-max when given)")
    ap.add_argument("--blood-value-col", default="tpm", help="blood expression column of --blood (default: tpm)")
    ap.add_argument("--blood-silent-max", type=float, default=1.0, help="blood value below this = silent (default 1 TPM)")
    ap.add_argument("--liver-enriched", required=True, help="path to a liver-enriched gene list or the HPA proteinatlas.tsv")
    ap.add_argument("--min-genes", type=int, default=50, help="warn when a subset is smaller than this")
    ap.add_argument("-o", "--out", required=True, help="output folder for the subset files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base = read_list(a.hcc_expressed)

    he = read_any(a.hcc_expr)
    gcol = a.hcc_expr_gene_col or he.columns[0]; vcol = a.hcc_expr_value_col or he.columns[1]
    if a.hcc_expr_filter:
        fc, fv = a.hcc_expr_filter.split("=", 1)
        he = he[he[fc].astype(str).str.strip() == fv.strip()]
        if he.empty:
            sys.exit(f"--hcc-expr-filter matched no rows; values in {fc}: {sorted(read_any(a.hcc_expr)[fc].astype(str).unique())[:30]}")
    hexp = pd.Series(pd.to_numeric(he[vcol], errors="coerce").to_numpy(), index=he[gcol].map(strip_ver))
    hexp = hexp.dropna().groupby(level=0).median()      # one value per gene (median over samples for a long table)

    # blood state per gene: 'silent' or 'expressed'
    bl = read_any(a.blood); bg = bl.columns[0]
    if a.blood_silent_bins is not None:
        silent_bins = {b.strip() for b in a.blood_silent_bins.split(",")}
        bcol = bl.columns[1]
        bstate = dict(zip(bl[bg].map(strip_ver), np.where(bl[bcol].astype(str).str.strip().isin(silent_bins), "silent", "expressed")))
    elif a.blood_value_col and a.blood_silent_max is not None:
        v = pd.to_numeric(bl[a.blood_value_col], errors="coerce")
        bstate = dict(zip(bl[bg].map(strip_ver), np.where(v < a.blood_silent_max, "silent", "expressed")))
    else:
        sys.exit("give --blood-silent-bins, or --blood-value-col with --blood-silent-max")

    liver = read_liver(a.liver_enriched)

    # HCC expression tertiles inside the set (ties broken by order)
    in_expr = sorted(g for g in base if g in hexp.index)
    t = pd.qcut(hexp.loc[in_expr].rank(method="first"), 3, labels=["low", "mid", "high"])
    subsets = {
        "S1_hcc_blood_silent": sorted(g for g in base if bstate.get(g) == "silent"),
        "S2_hcc_blood_expressed": sorted(g for g in base if bstate.get(g) == "expressed"),
        "S3_hcc_tertile_low": sorted(t.index[t == "low"]),
        "S4_hcc_tertile_mid": sorted(t.index[t == "mid"]),
        "S5_hcc_tertile_high": sorted(t.index[t == "high"]),
        "S6_liver_enriched": sorted(base & liver),
    }
    # one gene list per subset, plus a summary table
    rows = []
    for name, genes in subsets.items():
        with open(os.path.join(a.out, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(genes) + ("\n" if genes else ""))
        rows.append(dict(subset=name, n_genes=len(genes),
                         median_hcc_expr=float(hexp.reindex(genes).median()) if genes else np.nan))
        if len(genes) < a.min_genes:
            print(f"[warn] {name} has only {len(genes)} genes", file=sys.stderr)
    S = pd.DataFrame(rows)
    S.loc[len(S)] = dict(subset="(base set)", n_genes=len(base), median_hcc_expr=float(hexp.reindex(sorted(base)).median()))
    S.loc[len(S)] = dict(subset="(no HCC expression value)", n_genes=len(base) - len(in_expr), median_hcc_expr=np.nan)
    S.loc[len(S)] = dict(subset="(no blood state)", n_genes=sum(g not in bstate for g in base), median_hcc_expr=np.nan)
    S.to_csv(os.path.join(a.out, "subsets_summary.csv"), index=False)
    print(S.to_string(index=False))
    arg = ",".join(f"{n}={os.path.abspath(os.path.join(a.out, n + '.txt'))}" for n in subsets)
    open(os.path.join(a.out, "gene_sets_arg.txt"), "w").write(arg + "\n")
    print(f"\n--gene-sets string written to {os.path.join(a.out, 'gene_sets_arg.txt')}")


if __name__ == "__main__":
    main()
