#!/usr/bin/env python3
"""
01_build_hcc_gene_panel.py
--------------------------
Writes the HCC driver/pathway gene panel used as the "Pathway" gene set
(Methods 2.8). The panel is 28 consensus driver genes of hepatocellular
carcinoma from TCGA-LIHC (Cancer Genome Atlas Research Network, 2017) and
the Chinese Liver Cancer Atlas (494 genomes; Chen et al., 2024), grouped
into six pathways. It is built in, so no input file is needed.

BUILT-IN PANEL (by pathway)
---------------------------
Wnt/beta-catenin signalling:     CTNNB1, AXIN1, AXIN2, APC
TP53 / cell cycle:               TP53, RB1, CCND1, CDKN2A, MYC
Chromatin remodelling (SWI/SNF): ARID1A, ARID2, SMARCA4
Oxidative stress (NRF2):         NFE2L2, KEAP1
PI3K/mTOR:                       TSC1, TSC2, PIK3CA
Other recurrent HCC drivers:     ALB, LZTR1, EEF1A1, SF3B1, RPS6KA3,
                                 KMT2B, IL6ST, JAK1, APOB, CPS1, GPC3

For the thesis the panel was used as it is, without expansion from a
pathway database. The script can optionally add genes from an MSigDB GMT
file (--gmt) or a list of extra symbols (--extra-genes); both are off by
default and were not used.

Input:  none (optional --gmt file with gene symbols)
Output: <out>/hcc_gene_panel.tsv, tab-separated with a header:
        gene_symbol, source (the pathway, "user_supplied" or "gmt:<name>").
        A short summary is printed to the screen.

Usage:
    python 01_build_hcc_gene_panel.py -o gene_sets/
"""

import argparse
import os
from collections import defaultdict

SEED_PATHWAYS = {
    "Wnt/beta-catenin signaling": ["CTNNB1", "AXIN1", "AXIN2", "APC"],
    "TP53/cell-cycle": ["TP53", "RB1", "CCND1", "CDKN2A", "MYC"],
    "Chromatin remodeling (SWI/SNF)": ["ARID1A", "ARID2", "SMARCA4"],
    "Oxidative stress (NRF2)": ["NFE2L2", "KEAP1"],
    "PI3K/mTOR": ["TSC1", "TSC2", "PIK3CA"],
    "Other recurrent HCC drivers": [
        "ALB", "LZTR1", "EEF1A1", "SF3B1", "RPS6KA3", "KMT2B",
        "IL6ST", "JAK1", "APOB", "CPS1", "GPC3",
    ],
}


def parse_gmt(path):
    """GMT format: one pathway per line, tab-separated:
    pathway_name <TAB> description_or_url <TAB> gene1 <TAB> gene2 ..."""
    pathways = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _, *genes = parts
            pathways[name] = [g for g in genes if g]
    return pathways


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True,
                    help="output folder; hcc_gene_panel.tsv is written there")
    ap.add_argument("--gmt", default=None,
                    help="optional MSigDB-format GMT file (gene symbols) to add pathway genes from "
                         "(not used for the thesis)")
    ap.add_argument("--gmt-pathways", nargs="+", default=None,
                    help="only take GMT pathways whose name contains one of these words "
                         "(case-insensitive). Default: every GMT pathway that holds at least one panel gene.")
    ap.add_argument("--extra-genes", nargs="+", default=None,
                    help="extra gene symbols to add directly (not used for the thesis)")
    args = ap.parse_args()

    # one row per gene; the first pathway a gene appears in is kept
    gene_pathway = {}
    for pathway, genes in SEED_PATHWAYS.items():
        for g in genes:
            gene_pathway.setdefault(g, pathway)
    n_seed = len(gene_pathway)
    print(f"[info] built-in panel: {n_seed} genes across {len(SEED_PATHWAYS)} pathways")

    if args.extra_genes:
        for g in args.extra_genes:
            gene_pathway.setdefault(g.upper(), "user_supplied")
        print(f"[info] added {len(args.extra_genes)} extra genes")

    if args.gmt:
        gmt_pathways = parse_gmt(args.gmt)
        print(f"[info] loaded {len(gmt_pathways)} pathways from {args.gmt}")
        seed_genes = set(gene_pathway.keys())
        n_added = 0
        for name, genes in gmt_pathways.items():
            if args.gmt_pathways:
                if not any(sub.lower() in name.lower() for sub in args.gmt_pathways):
                    continue
            else:
                if not (set(genes) & seed_genes):
                    continue
            for g in genes:
                if g not in gene_pathway:
                    gene_pathway[g] = f"gmt:{name}"
                    n_added += 1
        print(f"[info] GMT expansion added {n_added} genes")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "hcc_gene_panel.tsv")
    with open(out_path, "w") as fh:
        fh.write("gene_symbol\tsource\n")
        for gene in sorted(gene_pathway):
            fh.write(f"{gene}\t{gene_pathway[gene]}\n")

    by_source = defaultdict(int)
    for src in gene_pathway.values():
        by_source[src] += 1
    print(f"\n[info] wrote {out_path}: {len(gene_pathway)} genes")
    for src, n in sorted(by_source.items()):
        print(f"    {src}: {n}")


if __name__ == "__main__":
    main()
