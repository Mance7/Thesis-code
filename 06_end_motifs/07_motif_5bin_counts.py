#!/usr/bin/env python3
"""
07_motif_5bin_counts.py

5' end-motif counts per sample in 5 size bins, genome-wide, for ONT and PacBio (all and direct fragments).
Bins: <=255 | 256-650 | 651-2000 | 2001-5000 | >5000 bp  (all edges are real 11-bin edges).
ONT  : re-bins the per-sample 11-bin motif tables (motif_region/<all|direct>/per_sample/*.motif_freq_region.csv)
PacBio: counts straight from the fragment files (*.fragments_with_motifs.csv.gz, column motif_5p, one end per fragment)
Only A/C/G/T 4-mers are kept. Pandas only (no scipy), so it runs anywhere.

Optional --pacbio-promoter-out: also writes the PacBio fragments whose midpoint sits in a promoter window
(columns for 09_geneset_pool_trend.py), in small gz parts.

  python 07_motif_5bin_counts.py --ont-motif-dir "filtered_mid/motif_region" --pacbio-dir "absolute/filtered_mid/size_bins/end_motifs" \
      -o "motif5_counts" --threads 2 [--promoters promoters_gencode_v49_gene_level.bed --pacbio-promoter-out pb_prom_frags]
"""
import argparse, glob, os, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

KMERS = set("".join(p) for p in itertools.product("ACGT", repeat=4))
MAP11 = {"<90bp": "<=255bp", "90-125bp": "<=255bp", "126-135bp": "<=255bp", "136-150bp": "<=255bp",
         "151-255bp": "<=255bp", "256-420bp": "256-650bp", "421-650bp": "256-650bp",
         "651-1000bp": "651-2000bp", "1001-2000bp": "651-2000bp", "2001-5000bp": "2001-5000bp", ">5000bp": ">5000bp"}
EDGES = [0, 255, 650, 2000, 5000, np.inf]
LAB5 = ["<=255bp", "256-650bp", "651-2000bp", "2001-5000bp", ">5000bp"]


def ont_one(f, src):
    d = pd.read_csv(f)
    d = d[(d.region == "genome_wide") & (d.group_type == "size_bin") & d.motif.isin(KMERS)]
    d = d.assign(bin=d["bin"].map(MAP11))
    out = d.groupby(["sample", "bin", "motif"], as_index=False)["count"].sum()
    return out.assign(platform="ONT", source=src)


def promoter_index(bed):
    p = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    idx = {}
    for c, g in p.groupby("chrom"):          # merged windows per chromosome, for a fast midpoint test
        g = g.sort_values("start"); s, e = [], []
        for a, b in zip(g.start.values, g.end.values):
            if s and a <= e[-1]:
                e[-1] = max(e[-1], b)
            else:
                s.append(a); e.append(b)
        idx[c] = (np.array(s), np.array(e))
    return idx


def pacbio_one(f, prom_bed, prom_out, part_rows):
    cols = ["sample", "chrom", "start", "end", "length", "source", "motif_5p"]
    d = pd.read_csv(f, usecols=cols)
    s = d["sample"].iloc[0]
    if prom_out:                               # fragments with midpoint inside a promoter window, all motifs kept
        idx = promoter_index(prom_bed)
        mid = (d.start.values + d.end.values) // 2
        keep = np.zeros(len(d), bool)
        for c, (st, en) in idx.items():
            m = (d.chrom.values == c)
            if not m.any():
                continue
            i = np.searchsorted(st, mid[m], side="right") - 1
            ok = (i >= 0) & (mid[m] < en[np.clip(i, 0, None)])
            keep[np.where(m)[0][ok]] = True
        p = d[keep]
        for k in range(0, max(len(p), 1), part_rows):
            p.iloc[k:k + part_rows].to_csv(os.path.join(prom_out, f"{s}.part{k // part_rows:02d}.fragments_with_motifs.csv.gz"), index=False)
    d = d[d.motif_5p.isin(KMERS)]
    d["bin"] = pd.cut(d.length, EDGES, labels=LAB5, right=True).astype(str)
    out = []
    for src, sub in [("all", d), ("direct", d[d.source == "direct"])]:
        c = sub.groupby(["bin", "motif_5p"]).size().rename("count").reset_index().rename(columns={"motif_5p": "motif"})
        out.append(c.assign(sample=s, platform="PacBio", source=src))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ont-motif-dir", required=True, help="ONT motif_region folder (has all/ and direct/)")
    ap.add_argument("--pacbio-dir", required=True, help="PacBio folder with *.fragments_with_motifs.csv.gz")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--promoters", help="promoter BED (chrom, start, end, gene)")
    ap.add_argument("--pacbio-promoter-out", help="folder for PacBio promoter fragments (small gz parts)")
    ap.add_argument("--part-rows", type=int, default=150000)
    ap.add_argument("-t", "--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.pacbio_promoter_out:
        os.makedirs(a.pacbio_promoter_out, exist_ok=True)
    ont = [ont_one(f, s) for s in ("all", "direct")
           for f in sorted(glob.glob(os.path.join(a.ont_motif_dir, s, "per_sample", "*.motif_freq_region.csv")))]
    files = sorted(glob.glob(os.path.join(a.pacbio_dir, "*.fragments_with_motifs.csv.gz")))
    with ProcessPoolExecutor(a.threads) as ex:
        pb = list(ex.map(pacbio_one, files, [a.promoters] * len(files), [a.pacbio_promoter_out] * len(files),
                         [a.part_rows] * len(files)))
    res = pd.concat(ont + pb, ignore_index=True)[["platform", "source", "sample", "bin", "motif", "count"]]
    fn = os.path.join(a.out, "motif_5bin_counts_genome_wide.csv.gz")
    res.to_csv(fn, index=False)
    print(res.groupby(["platform", "source"]).agg(samples=("sample", "nunique"), fragments=("count", "sum")).to_string())
    print("wrote", fn)


if __name__ == "__main__":
    main()
