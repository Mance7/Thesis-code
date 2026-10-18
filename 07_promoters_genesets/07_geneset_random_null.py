#!/usr/bin/env python3
"""
07_geneset_random_null.py
-------------------------
Random-set null for the gene-set pooled metrics (pre-registered H5 and H5b,
Methods 2.8). Is a gene-set result specific to the genes in the set, or would
any similar group of promoters give the same HCC-vs-HBV difference?

For each gene set, the fragments of all its promoters are pooled per sample
before the metric is computed (gene-set pooling), and the group difference
(Cohen's d, Mann-Whitney p) is compared with the same statistic for
--n-random random promoter sets of the same size, drawn with the same make-up
of strata: blood-expression bin (H5), and with --strata-file / --count-tertiles
also promoter CpG class and a label-blind fragment-count tertile (H5b).

Fragment-to-promoter rule: each fragment of up to --max-len (600) bp is
assigned to the single promoter that contains its midpoint (if promoters
overlap, the one with the nearest start at or before the midpoint), so every
fragment is counted once. This differs from the region flags of Methods 2.5,
which use any overlap.

Metrics (fragments <= --max-len, default 600 bp):
  entropy_short  normalised Shannon entropy of the 1 bp lengths, K = max_len
  fsr            n(length <= --short-max, default 246) / n(length <= max_len)
A pooled value with fewer than --min-frags (20) fragments is set to missing;
a set is tested when each group has at least 3 samples with a value.

Speed: one pass per sample builds a sparse promoter x length histogram, so
each random set is just a sum of rows.

Output (--out folder):
  geneset_random_null.csv  one row per (set, metric): observed d and MWU p, median N
                           per group, and
                             emp_p_one_sided  share of random sets with d at least as extreme
                                              in the observed direction
                             emp_p_abs        share of random sets with |d| >= |d_obs| (the reported p)
                             pctile_p         rank of the observed MWU p among random-set p-values
                           (each as (b + 1) / (B + 1), B = valid random sets),
                           q_bh_abs (BH over the sets of one run, per metric; the H5b family),
                           n_strata and n_short_strata (strata with fewer promoters than the set
                           needs; all of their promoters are then taken)
  null_d_<set>_<metric>.txt  the random-set d values

Inputs:
  --promoters  BED: chrom, start, end, gene_id (4th column, version suffix ignored)
  --gene-sets  NAME=file,... with one gene id per line
  --expr-bins  blood_expression_bins.tsv from 03_build_expression_bins.py (promoters
               without a bin form their own stratum)
  --strata-file  gene_id-keyed table of extra matching columns (CpG class from
               promoter_cpg_annotation.py)

Usage (H5b run; repeat with --keep source=direct):
  python 07_geneset_random_null.py --fragments "filtered_mid/*.eligible_fragments.csv.gz" \
      --metadata ont_samples.csv --promoters promoters_gencode_v49_gene_level.bed \
      --expr-bins blood_expression_bins.tsv \
      --gene-sets "Expressed=hcc_expressed.txt,$(cat h5b_subsets/gene_sets_arg.txt)" \
      --strata-file promoter_cpg.csv --strata-cols cpg_class --count-tertiles \
      --n-random 1000 -t 32 --out geneset_null_h5b_all/
"""
import argparse, glob, os, re, sys, warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse, stats

warnings.filterwarnings("ignore")


def strip_ver(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def read_any(path, usecols=None):
    with pd.io.common.get_handle(path, "r", compression="infer") as h:
        first = h.handle.readline()
    return pd.read_csv(path, sep="\t" if "\t" in first else ",", usecols=usecols, encoding_errors="replace")


def norm_entropy_rows(H, K):
    """H: sets x lengths count matrix -> normalised entropy per row."""
    N = H.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(N > 0, H / N, 0)
        lp = np.where(p > 0, np.log2(p), 0)
    return -(p * lp).sum(1) / np.log2(K)


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def sample_hist(f, s, a, by_chrom, ends, P):
    """Stream one fragment file into a sparse promoter x length histogram.
    A fragment (1..max_len bp) is counted for the promoter whose window holds its midpoint:
    the promoter with the largest start <= midpoint, kept only if its end is past the midpoint."""
    with pd.io.common.get_handle(f, "r", compression="infer") as h:
        first = h.handle.readline()
    sep = "\t" if "\t" in first else ","
    cols = [a.chrom_col, a.start_col, a.end_col, a.length_col]
    if a.keep:
        kcol, kvals = a.keep.split("=", 1); kvals = [v.strip() for v in kvals.split(",")]
        cols.append(kcol)
    H = sparse.csr_matrix((P, a.max_len), dtype=np.int64); n = 0
    for df in pd.read_csv(f, sep=sep, usecols=list(dict.fromkeys(cols)), chunksize=a.chunksize, encoding_errors="replace"):
        if a.keep:
            df = df[df[kcol].astype(str).str.strip().isin(kvals)]
        df = df[(df[a.length_col] >= 1) & (df[a.length_col] <= a.max_len)]
        rows, cols_ = [], []
        for c, g in df.groupby(a.chrom_col):
            if c not in by_chrom:
                continue
            order, starts = by_chrom[c]
            mid = ((g[a.start_col].to_numpy() + g[a.end_col].to_numpy()) // 2)
            j = np.searchsorted(starts, mid, side="right") - 1
            ok = j >= 0
            pid = np.where(ok, order[np.clip(j, 0, None)], -1)
            ok &= (pid >= 0) & (ends[np.clip(pid, 0, None)] > mid)
            rows.append(pid[ok]); cols_.append(g[a.length_col].to_numpy()[ok] - 1)
        if rows:
            r = np.concatenate(rows); cc = np.concatenate(cols_)
            H = H + sparse.csr_matrix((np.ones(len(r), np.int64), (r, cc)), shape=(P, a.max_len))
            n += len(r)
    return s, H, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample eligible fragment tables (csv.gz)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("--promoters", required=True, help="path to the gene-level promoter BED (chrom, start, end, gene_id)")
    ap.add_argument("--gene-sets", required=True, help="NAME=file,NAME=file gene lists (one gene id per line)")
    ap.add_argument("--expr-bins", help="path to blood_expression_bins.tsv (gene id + expression bin)")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end"); ap.add_argument("--length-col", default="length")
    ap.add_argument("--max-len", type=int, default=600); ap.add_argument("--short-max", type=int, default=246)
    ap.add_argument("--min-frags", type=int, default=20, help="minimum fragments per sample per pool, as in the pooled analysis")
    ap.add_argument("--n-random", type=int, default=1000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel workers (samples read in parallel; random sets scored in parallel)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--strata-file", help="gene_id-keyed csv/tsv with extra matching columns")
    ap.add_argument("--strata-cols", default="cpg_class", help="comma-separated columns of --strata-file to match on")
    ap.add_argument("--count-tertiles", action="store_true", help="also match on label-blind total fragment-count tertile")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = read_any(a.metadata); meta.columns = [c.strip().lower() for c in meta.columns]
    grp = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    with open(a.promoters) as _fh:
        _first = next(l for l in _fh if l.strip() and not l.startswith(("#", "track", "browser")))
    if len(_first.rstrip("\n").split("\t")) < 4:
        raise SystemExit(f"--promoters {a.promoters} has only {len(_first.split(chr(9)))} tab-separated columns; "
                         "it needs chrom, start, end, gene_id (Ensembl), as in the gene-level promoter BED.")
    prom = pd.read_csv(a.promoters, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "gene"], comment="#")
    prom["gene"] = prom["gene"].map(strip_ver)
    prom = prom.drop_duplicates("gene").reset_index(drop=True)
    P = len(prom)
    gene_idx = {g: i for i, g in enumerate(prom.gene)}
    by_chrom = {c: (g.index.to_numpy()[np.argsort(g.start.to_numpy())], np.sort(g.start.to_numpy()))
                for c, g in prom.groupby("chrom")}
    ends = prom.end.to_numpy()

    if a.expr_bins:
        eb = read_any(a.expr_bins)
        if "expression_bin" in eb.columns:  # build_expression_bins.py output (gene_id, gene_id_noversion, gene_symbol, tpm, expression_bin)
            eb = eb[["gene_id" if "gene_id" in eb.columns else eb.columns[0], "expression_bin"]]
        eb = eb.iloc[:, :2].copy(); eb.columns = ["gene", "expr_bin"]
        print(f"  expression bins: {eb.expr_bin.nunique()} bins for {len(eb)} genes")
        ebm = dict(zip(eb.gene.map(strip_ver), eb.expr_bin))
        stratum = np.array([str(ebm.get(g, "NA")) for g in prom.gene], dtype=object)
    else:
        stratum = np.array(["all"] * P, dtype=object)

    if a.strata_file:
        sf = read_any(a.strata_file); gcol = sf.columns[0]
        for col in [c.strip() for c in a.strata_cols.split(",") if c.strip()]:
            m_ = dict(zip(sf[gcol].map(strip_ver), sf[col].astype(str)))
            stratum = np.array([f"{st}|{m_.get(g, 'NA')}" for st, g in zip(stratum, prom.gene)], dtype=object)
            print(f"  matching also on {col}: {pd.Series([m_.get(g, 'NA') for g in prom.gene]).value_counts().to_dict()}")

    sets = {}
    for item in a.gene_sets.split(","):
        name, path = item.split("=", 1)
        genes = [strip_ver(l) for l in open(path) if l.strip()]
        idx = np.array(sorted({gene_idx[g] for g in genes if g in gene_idx}))
        print(f"  set {name}: {len(genes)} genes, {len(idx)} matched to promoters")
        sets[name] = idx

    # per-sample sparse promoter x length histograms (lengths 1..max_len), samples in parallel;
    # each fragment goes to the promoter that contains its midpoint (see sample_hist)
    jobs = []
    for f in sorted(glob.glob(a.fragments)):
        m = re.match(a.sample_regex, os.path.basename(f)); s = m.group(1) if m else os.path.basename(f)
        if grp.get(s) in (a.case, a.control):
            jobs.append((f, s))
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        res_ = list(ex.map(sample_hist, [f for f, _ in jobs], [s for _, s in jobs],
                           [a] * len(jobs), [by_chrom] * len(jobs), [ends] * len(jobs), [P] * len(jobs)))
    samples, hists = [], []
    for s, H, n in res_:
        samples.append(s); hists.append(H.tocsr())
        print(f"  {s} ({grp[s]}): {n:,} fragments <= {a.max_len} bp in promoters")
    y = np.array([grp[s] == a.case for s in samples])
    if a.count_tertiles:
        tot = np.asarray(sum(H.sum(1) for H in hists)).ravel()      # both groups pooled: label-blind
        cuts = np.quantile(tot, [1 / 3, 2 / 3])
        tert = np.digitize(tot, cuts)
        stratum = np.array([f"{st}|n{t}" for st, t in zip(stratum, tert)], dtype=object)
        print(f"  matching also on fragment-count tertile (cuts {cuts[0]:.0f}, {cuts[1]:.0f} fragments)")

    def pooled_metrics(member_matrix):
        """member_matrix: sets x P sparse 0/1 -> dict metric -> samples x sets arrays."""
        def one(H):
            S = (member_matrix @ H).toarray().astype(float)     # sets x lengths
            N = S.sum(1)
            e = norm_entropy_rows(S, a.max_len); fr = S[:, :a.short_max].sum(1) / np.where(N > 0, N, np.nan)
            e[N < a.min_frags] = np.nan; fr[N < a.min_frags] = np.nan
            return e, fr, N
        with ThreadPoolExecutor(max_workers=max(1, a.threads)) as ex:
            outs = list(ex.map(one, hists))
        ent, fsr, n = zip(*outs)
        return {"entropy_short": np.array(ent), "fsr": np.array(fsr)}, np.array(n)

    def group_stats(V):
        """V: samples x sets -> (d, p) per set."""
        d, p = [], []
        for j in range(V.shape[1]):
            c, h = V[y, j], V[~y, j]
            c, h = c[~np.isnan(c)], h[~np.isnan(h)]
            if len(c) < 3 or len(h) < 3:
                d.append(np.nan); p.append(np.nan); continue
            d.append(cohens_d(c, h)); p.append(stats.mannwhitneyu(c, h, alternative="two-sided").pvalue)
        return np.array(d), np.array(p)

    strata = {s: np.where(stratum == s)[0] for s in np.unique(stratum)}
    out = []
    for name, idx in sets.items():
        if len(idx) == 0:
            continue
        # random sets with the same stratum composition
        comp = pd.Series(stratum[idx]).value_counts()
        n_short = int(sum(k > len(strata[s]) for s, k in comp.items()))
        rand = np.zeros((a.n_random, P), np.int8)
        for r_i in range(a.n_random):
            pick = np.concatenate([rng.choice(strata[s], size=min(k, len(strata[s])), replace=False) for s, k in comp.items()])
            rand[r_i, pick] = 1
        obs_m = sparse.csr_matrix((np.ones(len(idx)), (np.zeros(len(idx), int), idx)), shape=(1, P))
        obs_vals, obs_n = pooled_metrics(obs_m)
        rnd_vals, _ = pooled_metrics(sparse.csr_matrix(rand))
        for metric in obs_vals:
            d0, p0 = group_stats(obs_vals[metric]); d0, p0 = d0[0], p0[0]
            dr, pr = group_stats(rnd_vals[metric]); ok = ~np.isnan(dr)
            dr, pr = dr[ok], pr[ok]; B = len(dr)
            one = np.sum(dr >= d0) if d0 >= 0 else np.sum(dr <= d0)
            out.append(dict(set=name, metric=metric, n_promoters=len(idx), obs_d=d0, obs_mwu_p=p0,
                            median_N_case=float(np.median(obs_n[y, 0])), median_N_control=float(np.median(obs_n[~y, 0])),
                            n_random_valid=B, random_d_median=float(np.median(dr)) if B else np.nan,
                            emp_p_one_sided=(one + 1) / (B + 1), emp_p_abs=(np.sum(np.abs(dr) >= abs(d0)) + 1) / (B + 1),
                            pctile_p=(np.sum(pr <= p0) + 1) / (B + 1), n_strata=len(comp), n_short_strata=n_short))
            np.savetxt(os.path.join(a.out, f"null_d_{name}_{metric}.txt"), dr, fmt="%.4f")
        print(f"  {name}: done")
    res = pd.DataFrame(out)
    if len(res):
        res["q_bh_abs"] = np.nan
        for _, ix in res.groupby("metric").groups.items():
            pv = res.loc[ix, "emp_p_abs"].to_numpy(float); o = np.argsort(pv); n_ = len(pv)
            q = np.minimum.accumulate((pv[o] * n_ / np.arange(1, n_ + 1))[::-1])[::-1]
            qq = np.empty(n_); qq[o] = np.minimum(q, 1); res.loc[ix, "q_bh_abs"] = qq
    res.to_csv(os.path.join(a.out, "geneset_random_null.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
