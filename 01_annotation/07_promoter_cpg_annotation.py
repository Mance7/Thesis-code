#!/usr/bin/env python3
"""
GC and CpG content of every promoter, used to build CpG-matched random promoter sets.

For each promoter in a BED file (chrom, start, end, gene_id), reads the sequence from a
local reference FASTA and reports:
  length, n_N, gc_frac, cpg_count, cpg_oe (observed/expected CpG, Gardiner-Garden & Frommer 1987),
  cpg_class  HCP / ICP / LCP after Weber et al. 2007 (Nat Genet 39:457):
             HCP: CpG o/e > 0.75 and GC > 0.55; LCP: CpG o/e < 0.48; ICP otherwise.
             Weber classify by their best 500 bp window; here the maximum over 500 bp windows
             (step 50 bp) is used for the HCP test and the whole-promoter o/e for the LCP test
             ([--whole] uses the whole promoter for both).
  gc_tertile, cpg_oe_tertile  label-blind tertiles over all promoters
The output is used as the CpG stratum for the gene-set random-set null
(geneset_random_null.py --strata-file promoter_cpg.csv --strata-cols cpg_class).

The sequence is read from the same FASTA used for alignment (no .fai index needed).
Chromosomes are processed in parallel (-t).

Usage:
  python 07_promoter_cpg_annotation.py --fasta GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
      --promoters promoters_gene_level.bed --out promoter_cpg.csv -t 16
"""
import argparse, gzip, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


def strip_ver(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def read_fasta(path, wanted):
    """Yield (chrom, sequence) for chromosomes in `wanted`, one at a time."""
    op = gzip.open if path.endswith(".gz") else open
    name, buf = None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(buf).upper()
                name, buf = line[1:].split()[0], []
            elif name in wanted:
                buf.append(line.strip())
    if name in wanted:
        yield name, "".join(buf).upper()


def stats_one(seq):
    L = len(seq); n_N = seq.count("N"); Le = L - n_N
    c, g = seq.count("C"), seq.count("G"); cg = seq.count("CG")
    gc = (c + g) / Le if Le else np.nan
    oe = cg * Le / (c * g) if c and g else 0.0
    return L, n_N, gc, cg, oe


def max_window(seq, w=500, step=50):
    best_gc, best_oe = 0.0, 0.0
    for i in range(0, max(1, len(seq) - w + 1), step):
        _, _, gc, _, oe = stats_one(seq[i:i + w])
        if np.isfinite(gc) and gc > 0.55 and oe > best_oe:
            best_gc, best_oe = gc, oe
    return best_gc, best_oe


def work(args):
    chrom, seq, rows, whole = args
    out = []
    for gene, s, e in rows:
        sub = seq[max(0, s):e]
        L, n_N, gc, cg, oe = stats_one(sub)
        if whole:
            hcp = gc > 0.55 and oe > 0.75
        else:
            _, w_oe = max_window(sub)
            hcp = w_oe > 0.75
        cls = "HCP" if hcp else ("LCP" if oe < 0.48 else "ICP")
        out.append((gene, chrom, s, e, L, n_N, gc, cg, oe, cls))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="path to the GRCh38 reference genome (fasta or fasta.gz)")
    ap.add_argument("--promoters", required=True, help="gene-level promoter BED: chrom, start, end, gene_id")
    ap.add_argument("-o", "--out", required=True, help="output table of GC/CpG content per promoter (csv)")
    ap.add_argument("--whole", action="store_true", help="classify HCP on the whole promoter, not the best 500 bp window")
    ap.add_argument("-t", "--threads", type=int, default=8, help="chromosomes processed in parallel")
    a = ap.parse_args()

    with open(a.promoters) as _fh:
        _first = next(l for l in _fh if l.strip() and not l.startswith(("#", "track", "browser")))
    if len(_first.rstrip("\n").split("\t")) < 4:
        raise SystemExit(f"--promoters {a.promoters} has only {len(_first.split(chr(9)))} tab-separated columns; "
                         "it needs chrom, start, end, gene_id (Ensembl). Use the gene-level promoter BED that keeps gene IDs "
                         "(the same file used for the gene-set null).")
    prom = pd.read_csv(a.promoters, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chrom", "start", "end", "gene_id"], comment="#")
    prom["gene_id"] = prom["gene_id"].map(strip_ver)
    prom = prom.drop_duplicates("gene_id")
    by_chrom = {c: list(zip(g.gene_id, g.start.astype(int), g.end.astype(int))) for c, g in prom.groupby("chrom")}
    print(f"{len(prom):,} promoters on {len(by_chrom)} chromosomes", file=sys.stderr)

    res = []
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        futs = []
        for chrom, seq in read_fasta(a.fasta, set(by_chrom)):
            futs.append(ex.submit(work, (chrom, seq, by_chrom[chrom], a.whole)))
            print(f"  {chrom}: {len(by_chrom[chrom]):,} promoters queued", file=sys.stderr)
        for f in futs:
            res.extend(f.result())
    df = pd.DataFrame(res, columns=["gene_id", "chrom", "start", "end", "length", "n_N",
                                    "gc_frac", "cpg_count", "cpg_oe", "cpg_class"])
    missing = set(prom.gene_id) - set(df.gene_id)
    if missing:
        print(f"[warn] {len(missing)} promoters on chromosomes absent from the FASTA", file=sys.stderr)
    for col in ["gc_frac", "cpg_oe"]:
        df[col.replace("_frac", "") + "_tertile"] = pd.qcut(df[col].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    df.to_csv(a.out, index=False)
    print(df["cpg_class"].value_counts().to_string(), file=sys.stderr)
    print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
