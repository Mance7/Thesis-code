#!/usr/bin/env python3
"""
10_tss_profiles.py
------------------
Fragment profiles around transcription start sites, per gene group
(Methods 2.8, "TSS coverage profiles"; exploratory).

Why: active promoters lose their nucleosome right at the TSS, so cfDNA released
from a tissue shows a coverage dip at the TSSs of the genes that tissue
expresses (Ulz et al. 2016; Snyder et al. 2016). The dip is measured against
the flanks of the same sample, so it does not depend on sequencing depth. If
the extra cfDNA in HCC comes from the liver, the dip at liver-gene TSSs should
be deeper in HCC than in HBV carriers, while the dip at blood-gene TSSs (the
normal main source of cfDNA) should be present in everyone.

For every sample and gene group, the script sums over the group's TSSs
(strand-oriented, upstream on the left) in --bin (10) bp bins over +/- --flank
(3,000) bp:
  cov_short   coverage by fragments <= --short-max (250) bp
  cov_long    coverage by fragments  > --short-max bp
  mid_short   midpoints of fragments <= --short-max bp (nucleosome positions)
  ends        both fragment ends, all lengths (cut sites)
Each fragment is counted at every TSS of the group it reaches (no assignment
to one promoter). Values are divided by the number of TSSs in the group (mean
profile per TSS). The dip score (mean cov_short at -150..+150 bp divided by the
mean at 2-3 kb on both sides) and the group tests are computed from this table
afterwards.

TSS: from a GENCODE v49 GTF (gene features; + strand = gene start, - strand =
gene end, as in the promoter definition of Methods 2.5) or from --tss (tsv:
chrom, pos (0-based), strand, gene_id).

Output (--out folder): tss_profiles_<name>.tsv.gz, one row per sample x group x
kind, one column per bin centre (bp from the TSS); tss_groups_<name>.tsv with the
number of TSSs found per group.

Usage:
  python 10_tss_profiles.py --fragments "filtered_mid/*.eligible_fragments.csv.gz" \
      --gtf gencode.v49.annotation.gtf \
      --groups "LIVER=h5b_subsets/S6_liver_enriched.txt,BLOOD_SPECIFIC=BLOOD_SPECIFIC.txt,SILENT_BOTH=SILENT_BOTH.txt" \
      --out tss/ --name ont_direct --keep source=direct -t 14
"""
import argparse, glob, gzip, os, re, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

KINDS = ["cov_short", "cov_long", "mid_short", "ends"]


def strip_ver(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def read_tss(a, wanted):
    rows = []
    if a.tss:
        t = pd.read_csv(a.tss, sep="\t", header=None, comment="#", names=["chrom", "pos", "strand", "gene"])
        t["gene"] = t.gene.map(strip_ver)
        return t[t.gene.isin(wanted)].drop_duplicates("gene")
    op = gzip.open if a.gtf.endswith(".gz") else open
    with op(a.gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if len(p) < 9 or p[2] != "gene":
                continue
            m = re.search(r'gene_id "([^"]+)"', p[8])
            if not m:
                continue
            g = strip_ver(m.group(1))
            if g in wanted:
                pos = int(p[3]) - 1 if p[6] == "+" else int(p[4]) - 1
                rows.append((p[0], pos, p[6], g))
    return pd.DataFrame(rows, columns=["chrom", "pos", "strand", "gene"]).drop_duplicates("gene")


def one_sample(path, sample, a, tss_by_chrom, n_tss):
    W, B = a.flank, a.bin
    nb = 2 * W // B
    cov = {k: np.zeros(n_tss * (nb + 1), np.int64) for k in ("cov_short", "cov_long")}
    cnt = {k: np.zeros(n_tss * nb, np.int64) for k in ("mid_short", "ends")}
    kcol, kvals = (a.keep.split("=", 1)[0], set(a.keep.split("=", 1)[1].split(","))) if a.keep else (None, None)
    cols = [a.chrom_col, a.start_col, a.end_col] + ([kcol] if kcol else [])
    n_read = 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=a.chunksize, dtype={a.chrom_col: str}):
        if kcol:
            ch = ch[ch[kcol].astype(str).isin(kvals)]
        n_read += len(ch)
        for c, g in ch.groupby(a.chrom_col):
            if c not in tss_by_chrom:
                continue
            pos, gidx, minus = tss_by_chrom[c]
            s = g[a.start_col].to_numpy(np.int64); e = g[a.end_col].to_numpy(np.int64)
            lo = np.searchsorted(pos, s - W, "right"); hi = np.searchsorted(pos, e + W, "left")
            n = np.maximum(hi - lo, 0)
            if n.sum() == 0:
                continue
            fi = np.repeat(np.arange(len(s)), n)
            ti = lo[fi] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
            P = pos[ti]; mn = minus[ti]; T = gidx[ti]
            rs = s[fi] - P; re_ = e[fi] - P                              # half-open [rs, re_) relative to TSS
            rs, re_ = np.where(mn, 1 - re_, rs), np.where(mn, 1 - rs, re_)   # flip minus strand
            L = re_ - rs
            short = L <= a.short_max
            b0 = np.clip((rs + W) // B, 0, nb); b1 = np.clip((re_ - 1 + W) // B + 1, 0, nb)
            ok = b1 > b0
            for k, sel in (("cov_short", ok & short), ("cov_long", ok & ~short)):
                cov[k] += np.bincount(T[sel] * (nb + 1) + b0[sel], minlength=n_tss * (nb + 1))
                cov[k] -= np.bincount(T[sel] * (nb + 1) + b1[sel], minlength=n_tss * (nb + 1))
            mid = (rs + re_ - 1) // 2 + W
            sel = short & (mid >= 0) & (mid < 2 * W)
            cnt["mid_short"] += np.bincount(T[sel] * nb + mid[sel] // B, minlength=n_tss * nb)
            for x in (rs + W, re_ - 1 + W):
                sel = (x >= 0) & (x < 2 * W)
                cnt["ends"] += np.bincount(T[sel] * nb + x[sel] // B, minlength=n_tss * nb)
    out = {k: np.cumsum(v.reshape(n_tss, nb + 1), 1)[:, :nb] for k, v in cov.items()}
    out.update({k: v.reshape(n_tss, nb) for k, v in cnt.items()})
    return sample, out, n_read


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample eligible fragment tables (csv.gz)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gtf", help="path to the GENCODE v49 annotation GTF (.gtf or .gtf.gz)")
    src.add_argument("--tss", help="path to a TSS table instead of the GTF (chrom, pos, strand, gene_id)")
    ap.add_argument("--groups", required=True, help="NAME=gene_list,NAME=gene_list (Ensembl ids, one per line)")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--name", required=True, help="run name used in the output file names, e.g. ont_direct")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("--flank", type=int, default=3000, help="bp on each side of the TSS")
    ap.add_argument("--bin", type=int, default=10, help="bin width (bp)")
    ap.add_argument("--short-max", type=int, default=250, help="longest short fragment (bp)")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("-t", "--threads", type=int, default=4, help="samples processed in parallel")
    a = ap.parse_args()
    if (2 * a.flank) % a.bin:
        sys.exit("--bin must divide 2*--flank")
    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)

    groups = {}
    for item in [x for x in a.groups.split(",") if x.strip()]:
        nm, p = item.split("=", 1)
        groups[nm] = {strip_ver(l.split()[0]) for l in open(p) if l.strip() and not l.lower().startswith("gene")}
    wanted = set().union(*groups.values())
    tss = read_tss(a, wanted).sort_values(["chrom", "pos"]).reset_index(drop=True)
    gi = {g: i for i, g in enumerate(tss.gene)}
    tss_by_chrom = {c: (d.pos.to_numpy(np.int64), d.index.to_numpy(), (d.strand == "-").to_numpy())
                    for c, d in tss.groupby("chrom")}
    info = []
    for nm, gs in groups.items():
        idx = [gi[g] for g in gs if g in gi]
        groups[nm] = np.array(idx, int)
        info.append(dict(group=nm, n_genes_listed=len(gs), n_tss=len(idx)))
        print(f"  group {nm}: {len(gs)} genes, {len(idx)} with a TSS")
    pd.DataFrame(info).to_csv(os.path.join(a.out, f"tss_groups_{a.name}.tsv"), sep="\t", index=False)

    jobs = [(f, re.match(a.sample_regex, os.path.basename(f)).group(1)) for f in files]
    res = Parallel(n_jobs=a.threads)(delayed(one_sample)(f, s, a, tss_by_chrom, len(tss)) for f, s in jobs)
    nb = 2 * a.flank // a.bin
    centers = -a.flank + a.bin * np.arange(nb) + a.bin / 2
    recs = []
    for s, out, n_read in res:
        for nm, idx in groups.items():
            for k in KINDS:
                prof = out[k][idx].sum(0) / max(len(idx), 1)
                recs.append([s, nm, k] + list(prof))
        print(f"  {s}: {n_read:,} fragments read")
    df = pd.DataFrame(recs, columns=["sample", "group", "kind"] + [f"{c:g}" for c in centers])   # wide: one row per profile
    o = os.path.join(a.out, f"tss_profiles_{a.name}.tsv.gz")
    df.to_csv(o, sep="\t", index=False, compression="gzip", float_format="%.5g")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
