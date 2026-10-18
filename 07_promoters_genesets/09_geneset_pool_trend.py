#!/usr/bin/env python3
"""
09_geneset_pool_trend.py
------------------------
Direction and significance of the HCC-vs-HBV difference for every promoter
pool across expression (Methods 2.8, Appendix A; exploratory). Pools are the
30 blood-expression bins, optionally HCC-expressed x blood bin
(--cross-expressed), and named gene sets (Expressed, S1-S6, ...). For each
pool the fragments of all its promoters are pooled per sample, each feature is
computed per sample, and HCC is compared with HBV: group means, log2 fold
change, Cohen's d, Mann-Whitney p and BH q. The output feeds the volcano plots
(log2FC vs -log10 p) and the expression-trend heatmap.

Fragment-to-promoter rule: as in 07_geneset_random_null.py, each fragment is
assigned to the single promoter that contains its midpoint (counted once).
The fsr, the <=600 bp entropy and the motif features use fragments of up to
600 bp; R and the longer size bins use the longer fragments assigned the same
way. Region flags elsewhere (Methods 2.5) use any overlap.

Features:
  fsr            n(<= 246) / n(<= 600)       (a ratio, so no depth correction is needed)
  R              n(256-650) / n(> 2000)      (the size ratio of the thesis; missing if fewer
                                              than --min-r-den fragments > 2 kb)
  wps            pooled windowed protection score per fragment: [sum over the pool's
                 promoters of (120-180 bp fragments spanning the promoter-window midpoint
                 - those with an end within +/-60 bp)] / n(120-180 bp fragments).
                 Dividing by the fragment count keeps it from scaling with depth.
Per size bin (<=600, 601-2000, 2001-5000, >5000; names end in _le600, _601_2000, _2001_5000, _gt5000):
  entropy_<bin>  normalised length entropy inside the bin (K = bin width; >5000 uses K = long-max - 5000)
  mds_<bin>      normalised entropy of the 256 5' end 4-mers (motif files only)
  Both depend on fragment count, so the reported values are rarefied (each sample drawn
  down to the smallest count among samples for that pool and bin, --n-rare draws);
  the unrarefied values are kept as *_raw.
Motif features (only if the fragment files have a 5' end-motif column, e.g. motif_5p; fragments <= 600 bp):
  m_CCCA m_CCAG m_CCTG m_GGAG m_AGTT m_ATTT   share of each 4-mer among valid 4-mers
  c_end          CCCA + CCAG + CCTG share

Outputs in --out: per_sample_pool_features.csv.gz, pool_stats.csv (q_all = BH over every
pool x feature; q_within_kind_feature = BH over the pools of one kind, per feature),
trend_vs_blood_bin.csv (Spearman rho of log2FC against blood bin), run_info.txt.

Usage (one run = one fragment set; --keep source=direct for direct fragments):
  python 09_geneset_pool_trend.py --fragments "filtered_mid/*.fragments_with_motifs.csv.gz" \
      --metadata ont_samples.csv --promoters promoters_gencode_v49_gene_level.bed \
      --expr-bins blood_expression_bins.tsv \
      --gene-sets "Expressed=hcc_expressed.txt,$(cat h5b_subsets/gene_sets_arg.txt)" \
      --cross-expressed hcc_expressed.txt -o pool_trend/ont_all -t 14 [--keep source=direct]
"""
import argparse, glob, os, re, warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np
import pandas as pd
from scipy import sparse, stats

warnings.filterwarnings("ignore")
KMERS = ["".join(p) for p in product("ACGT", repeat=4)]
KIDX = {k: i for i, k in enumerate(KMERS)}
NAMED = ["CCCA", "CCAG", "CCTG", "GGAG", "AGTT", "ATTT"]
SB = [(1, 600, "le600"), (601, 2000, "601_2000"), (2001, 5000, "2001_5000"), (5001, 10**12, "gt5000")]
SB_LO = np.array([b[0] for b in SB])


def strip_ver(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def sniff(path):
    with pd.io.common.get_handle(path, "r", compression="infer") as h:
        first = h.handle.readline()
    sep = "\t" if "\t" in first else ","
    return sep, [c.strip() for c in first.rstrip("\n").split(sep)]


def read_any(path):
    sep, _ = sniff(path)
    return pd.read_csv(path, sep=sep, encoding_errors="replace")


def sample_hist(f, s, a, by_chrom, ends, P, motif_col, pstart, W):
    sep, header = sniff(f)
    cols = [a.chrom_col, a.start_col, a.end_col, a.length_col]
    if motif_col:
        cols.append(motif_col)
    if a.keep:
        kcol, kvals = a.keep.split("=", 1); kvals = [v.strip() for v in kvals.split(",")]
        cols.append(kcol)
    # same midpoint rule as 07_geneset_random_null.py; all lengths are kept here
    L = a.long_max + 1                     # columns 0..long_max-1 = lengths 1..long_max, last = > long_max
    nb = len(SB)
    HL = sparse.csr_matrix((P, L), dtype=np.int64)
    HM = sparse.csr_matrix((P, 256 * nb), dtype=np.int64)       # bin * 256 + kmer
    HP = sparse.csr_matrix((P, W * nb), dtype=np.int64)         # bin * W + start offset in the promoter window (not used by the features)
    WP = np.zeros((P, 3), np.int64)                              # wps: spanning, end near, n 120-180 bp
    n = 0
    for df in pd.read_csv(f, sep=sep, usecols=list(dict.fromkeys(cols)), chunksize=a.chunksize, encoding_errors="replace"):
        if a.keep:
            df = df[df[kcol].astype(str).str.strip().isin(kvals)]
        df = df[df[a.length_col] >= 1]
        R, C, RM, CM, RP, CP = [], [], [], [], [], []
        for c, g in df.groupby(a.chrom_col):
            if c not in by_chrom:
                continue
            order, starts = by_chrom[c]
            st_ = g[a.start_col].to_numpy(); en_ = g[a.end_col].to_numpy()
            mid = (st_ + en_) // 2
            j = np.searchsorted(starts, mid, side="right") - 1
            ok = j >= 0
            pid = np.where(ok, order[np.clip(j, 0, None)], -1)
            ok &= (pid >= 0) & (ends[np.clip(pid, 0, None)] > mid)
            ln = g[a.length_col].to_numpy()
            b = np.searchsorted(SB_LO, ln, side="right") - 1
            R.append(pid[ok]); C.append(np.minimum(ln[ok], L) - 1)
            pidc = np.clip(pid, 0, None)
            off = st_ - pstart[pidc]
            okp = ok & (off >= 0) & (off < np.minimum(ends[pidc] - pstart[pidc], W))
            RP.append(pid[okp]); CP.append(b[okp] * W + off[okp])
            w = ok & (ln >= a.wps_min_len) & (ln <= a.wps_max_len)
            if w.any():
                pm = (pstart[pid[w]] + ends[pid[w]]) // 2
                sw, ew = st_[w], en_[w]
                span = (sw < pm) & (ew > pm)
                near = ((sw >= pm - a.wps_window) & (sw <= pm + a.wps_window)) | ((ew >= pm - a.wps_window) & (ew <= pm + a.wps_window))
                np.add.at(WP[:, 0], pid[w], span.astype(np.int64)); np.add.at(WP[:, 1], pid[w], near.astype(np.int64))
                np.add.at(WP[:, 2], pid[w], 1)
            if motif_col:
                mi = g[motif_col].astype(str).str.upper().map(KIDX).fillna(-1).astype(np.int64).to_numpy()
                okm = ok & (mi >= 0)
                RM.append(pid[okm]); CM.append(b[okm] * 256 + mi[okm])
        def add(H, RR, CC, shape):
            if RR:
                r = np.concatenate(RR); cc = np.concatenate(CC)
                if len(r):
                    H = H + sparse.csr_matrix((np.ones(len(r), np.int64), (r, cc)), shape=shape)
            return H
        n += sum(len(x) for x in R)
        HL = add(HL, R, C, (P, L)); HM = add(HM, RM, CM, (P, 256 * nb)); HP = add(HP, RP, CP, (P, W * nb))
    return s, HL, HM, HP, sparse.csr_matrix(WP), n


def norm_entropy(M, K):
    N = M.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(N > 0, M / N, 0)
        lp = np.where(p > 0, np.log2(p), 0)
    return -(p * lp).sum(1) / np.log2(K)


def features(SL, SM, SP, SW, a, has_motif, W):
    """Pooled matrices (pools x columns) -> dict feature -> array(pools)."""
    s = SL[:, :a.max_len]                       # lengths 1..600
    Ns = s.sum(1); Nall = SL.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = {
            "fsr": s[:, :a.short_max].sum(1) / Ns,
        }
        f["fsr"][Ns < a.min_frags] = np.nan
        # primary size ratio R (thesis definition): n(256-650) / n(> 2000)
        num = SL[:, 255:650].sum(1); den = SL[:, 2000:].sum(1)
        R = num / den; R[(den < a.min_r_den) | (num < a.min_frags)] = np.nan
        f["R"] = R
        for k, (lo, hi, nm) in enumerate(SB):
            hi_ = min(hi, a.long_max)
            M = SL[:, lo - 1:hi_]
            e = norm_entropy(M, hi_ - lo + 1); e[M.sum(1) < (a.min_frags if k == 0 else a.min_frags_long)] = np.nan
            f[f"entropy_{nm}"] = e
        nw = SW[:, 2]
        wps = (SW[:, 0] - SW[:, 1]) / nw; wps[nw < a.min_frags] = np.nan
        f["wps"] = wps
    if has_motif:
        S0 = SM[:, :256]; Nm = S0.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            for m in NAMED:
                f[f"m_{m}"] = S0[:, KIDX[m]] / Nm
            f["c_end"] = S0[:, [KIDX[m] for m in NAMED[:3]]].sum(1) / Nm
            for k in [x for x in f if x.startswith("m_")] + ["c_end"]:
                f[k][Nm < a.min_frags] = np.nan
            for k, (_, _, nm) in enumerate(SB):
                Mk = SM[:, k * 256:(k + 1) * 256]
                md = norm_entropy(Mk, 256); md[Mk.sum(1) < (a.min_frags if k == 0 else a.min_frags_long)] = np.nan
                f[f"mds_{nm}"] = md
    return f


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = ~np.isnan(p)
    pv = p[ok]; o = np.argsort(pv); n = len(pv)
    if n:
        qq = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        t = np.empty(n); t[o] = np.minimum(qq, 1); q[ok] = t
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample fragment tables (csv.gz; with motif_5p for the motif features)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("--promoters", required=True, help="BED: chrom, start, end, gene_id")
    ap.add_argument("--expr-bins", required=True, help="gene id + blood expression bin (1 = lowest)")
    ap.add_argument("--gene-sets", default="", help="NAME=file,NAME=file (one gene id per line)")
    ap.add_argument("--cross-expressed", help="also make pools HCC-expressed x blood bin from this gene list")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end"); ap.add_argument("--length-col", default="length")
    ap.add_argument("--motif-col", default="auto", help="5' end-motif column; 'auto' tries motif_5p/motif/end_motif; 'none' skips motifs")
    ap.add_argument("--max-len", type=int, default=600); ap.add_argument("--short-max", type=int, default=246)
    ap.add_argument("--long-max", type=int, default=50000, help="lengths above this share one histogram column")
    ap.add_argument("--wps-min-len", type=int, default=120); ap.add_argument("--wps-max-len", type=int, default=180)
    ap.add_argument("--wps-window", type=int, default=60)
    ap.add_argument("--min-r-den", type=int, default=5, help="R needs at least this many > 2000 bp fragments per sample per pool")
    ap.add_argument("--n-rare", type=int, default=10, help="rarefaction draws per sample x pool x bin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-frags", type=int, default=20, help="per sample per pool, <= max-len fragments")
    ap.add_argument("--min-frags-long", type=int, default=20, help="per sample per pool, 601-2000 bp fragments")
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("-t", "--threads", type=int, default=8, help="parallel workers (samples read in parallel)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = read_any(a.metadata); meta.columns = [c.strip().lower() for c in meta.columns]
    grp = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    prom = pd.read_csv(a.promoters, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "gene"], comment="#")
    prom["gene"] = prom["gene"].map(strip_ver)
    prom = prom.drop_duplicates("gene").reset_index(drop=True); P = len(prom)
    gidx = {g: i for i, g in enumerate(prom.gene)}
    by_chrom = {c: (g.index.to_numpy()[np.argsort(g.start.to_numpy())], np.sort(g.start.to_numpy())) for c, g in prom.groupby("chrom")}
    ends = prom.end.to_numpy(); pstart = prom.start.to_numpy()
    W = int(np.max(ends - pstart))
    print(f"{P:,} promoters; widest window {W} bp (fragment-start window)", flush=True)

    eb = read_any(a.expr_bins)
    gcol = next((c for c in eb.columns if c.lower() in ("gene_id_noversion", "gene_id", "gene")), eb.columns[0])
    bcol = next((c for c in eb.columns if "bin" in c.lower()), eb.columns[-1])
    ebm = dict(zip(eb[gcol].map(strip_ver), pd.to_numeric(eb[bcol], errors="coerce")))
    pbin = np.array([ebm.get(g, np.nan) for g in prom.gene], float)
    bins = sorted(int(b) for b in np.unique(pbin[~np.isnan(pbin)]))
    print(f"blood bins: {len(bins)} ({bins[0]}-{bins[-1]}), {np.sum(~np.isnan(pbin)):,} of {P:,} promoters binned", flush=True)

    pools = []                                  # (name, kind, order, idx)
    for b in bins:
        pools.append((f"B{b:02d}", "blood_bin", b, np.where(pbin == b)[0]))
    if a.cross_expressed:
        ex = {strip_ver(l.split()[0]) for l in open(a.cross_expressed) if l.strip()}
        isx = np.array([g in ex for g in prom.gene])
        for b in bins:
            pools.append((f"E_B{b:02d}", "expressed_x_bin", b, np.where((pbin == b) & isx)[0]))
    for k, item in enumerate([x for x in a.gene_sets.split(",") if x.strip()]):
        name, path = item.split("=", 1)
        genes = {strip_ver(l.split()[0]) for l in open(path) if l.strip()}
        pools.append((name.strip(), "gene_set", k, np.array(sorted(gidx[g] for g in genes if g in gidx), int)))
    for nm, kind, _, idx in pools:
        if kind == "gene_set":
            print(f"  pool {nm}: {len(idx):,} promoters", flush=True)
    rows, cols = [], []
    for i, (_, _, _, idx) in enumerate(pools):
        rows += [i] * len(idx); cols += list(idx)
    Mem = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pools), P))

    jobs = []
    for f in sorted(glob.glob(a.fragments)):
        m = re.match(a.sample_regex, os.path.basename(f)); s = m.group(1) if m else os.path.basename(f)
        if grp.get(s) in (a.case, a.control):
            jobs.append((f, s))
    if not jobs:
        raise SystemExit("no fragment files matched samples in the metadata")
    _, header = sniff(jobs[0][0])
    motif_col = None
    if a.motif_col == "auto":
        motif_col = next((c for c in ("motif_5p", "motif", "end_motif", "motif5p") if c in header), None)
    elif a.motif_col != "none":
        motif_col = a.motif_col if a.motif_col in header else None
    print(f"{len(jobs)} samples; columns: {header}; motif column: {motif_col or 'none (motif features skipped)'}", flush=True)

    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        res = list(ex.map(sample_hist, [f for f, _ in jobs], [s for _, s in jobs], [a] * len(jobs),
                          [by_chrom] * len(jobs), [ends] * len(jobs), [P] * len(jobs), [motif_col] * len(jobs),
                          [pstart] * len(jobs), [W] * len(jobs)))
    long_rows = []
    pooled = {}                                   # sample -> pooled (lengths, motifs, start offsets)
    for s, HL, HM, HP, HW, n in res:
        SL = (Mem @ HL).toarray().astype(float); SM = (Mem @ HM).toarray().astype(float)
        SP = (Mem @ HP).toarray().astype(float); SW = (Mem @ HW).toarray().astype(float)
        F = features(SL, SM, SP, SW, a, motif_col is not None, W)
        pooled[s] = (SL, SM, SP)
        print(f"  {s} ({grp[s]}): {n:,} fragments in promoters", flush=True)
        for i, (nm, kind, order, idx) in enumerate(pools):
            base = dict(sample=s, group=grp[s], pool=nm, pool_kind=kind, pool_order=order, n_promoters=len(idx),
                        n_short=SL[i, :a.max_len].sum(), n_all=SL[i].sum())
            for k, v in F.items():
                if k.startswith(("entropy_", "mds_")):
                    k = k + "_raw"                # count-dependent: the rarefied version below is the one to read
                long_rows.append({**base, "feature": k, "value": v[i]})

    # Rarefied versions of the count-dependent metrics (length entropy, MDS): each sample's pooled counts
    # are drawn down (without replacement) to the smallest count among samples for that pool and bin, --n-rare times,
    # and the entropy averaged, so HCC's larger fragment numbers cannot raise or lower the value by themselves.
    rng = np.random.default_rng(a.seed)
    samp = list(pooled)
    def rarefied(get, K, minf, name):
        for i, (nm, kind, order, idx) in enumerate(pools):
            rows = {s_: get(s_, i) for s_ in samp}
            ns = {s_: r.sum() for s_, r in rows.items()}
            okk = [s_ for s_ in samp if ns[s_] >= minf]
            tgt = int(min(ns[s_] for s_ in okk)) if okk else 0
            for s_ in samp:
                v = np.nan
                if s_ in okk and tgt >= minf:
                    c = rows[s_]; nzi = c > 0; cc = c[nzi].astype(np.int64)
                    v = float(np.mean([norm_entropy(rng.multivariate_hypergeometric(cc, tgt, method="marginals")[None, :].astype(float), K)[0]
                                       for _ in range(a.n_rare)]))
                    if name.startswith("pref_end"):
                        v = 1 - v
                long_rows.append(dict(sample=s_, group=grp[s_], pool=nm, pool_kind=kind, pool_order=order, n_promoters=len(idx),
                                      n_short=pooled[s_][0][i, :a.max_len].sum(), n_all=pooled[s_][0][i].sum(),
                                      feature=name, value=v, rarefied_to=tgt))
    for k, (lo, hi, nm) in enumerate(SB):
        hi_ = min(hi, a.long_max); minf = a.min_frags if k == 0 else a.min_frags_long
        rarefied(lambda s_, i, lo=lo, hi_=hi_: pooled[s_][0][i, lo - 1:hi_], hi_ - lo + 1, minf, f"entropy_{nm}")
        if motif_col:
            rarefied(lambda s_, i, k=k: pooled[s_][1][i, k * 256:(k + 1) * 256], 256, minf, f"mds_{nm}")
    print("rarefied metrics done", flush=True)
    ps = pd.DataFrame(long_rows)
    ps.to_csv(os.path.join(a.out, "per_sample_pool_features.csv.gz"), index=False)

    out = []
    for (pool, kind, order, feat), g in ps.groupby(["pool", "pool_kind", "pool_order", "feature"], sort=False):
        c = g.loc[g.group == a.case, "value"].dropna().to_numpy(); h = g.loc[g.group == a.control, "value"].dropna().to_numpy()
        r = dict(pool=pool, pool_kind=kind, pool_order=order, feature=feat, n_promoters=int(g.n_promoters.iloc[0]),
                 n_case=len(c), n_control=len(h), median_n_short_case=float(g.loc[g.group == a.case, "n_short"].median()),
                 median_n_short_control=float(g.loc[g.group == a.control, "n_short"].median()))
        if len(c) >= 3 and len(h) >= 3:
            mc, mh = c.mean(), h.mean()
            r.update(mean_case=mc, mean_control=mh, diff=mc - mh,
                     log2fc=np.log2(mc / mh) if mc > 0 and mh > 0 else np.nan,   # wps can be negative: use d or diff for it
                     d=cohens_d(c, h), p_mwu=stats.mannwhitneyu(c, h, alternative="two-sided").pvalue)
        out.append(r)
    st = pd.DataFrame(out)
    st["q_all"] = bh(st.get("p_mwu", pd.Series(np.nan, index=st.index)))           # BH over every pool x feature of this run
    st["q_within_kind_feature"] = np.nan                                            # BH over the pools of one kind, per feature
    for _, ix in st.groupby(["pool_kind", "feature"]).groups.items():
        st.loc[ix, "q_within_kind_feature"] = bh(st.loc[ix, "p_mwu"])
    st.to_csv(os.path.join(a.out, "pool_stats.csv"), index=False)

    tr = []                                     # descriptive trend: log2FC vs blood bin
    for (kind, feat), g in st[st.pool_kind != "gene_set"].groupby(["pool_kind", "feature"]):
        g = g.dropna(subset=["log2fc"])
        if len(g) >= 5:
            rho, p = stats.spearmanr(g.pool_order, g.log2fc)
            tr.append(dict(pool_kind=kind, feature=feat, n_bins=len(g), spearman_rho=rho, p=p))
    pd.DataFrame(tr).to_csv(os.path.join(a.out, "trend_vs_blood_bin.csv"), index=False)
    with open(os.path.join(a.out, "run_info.txt"), "w") as fh:
        fh.write(f"samples {len(jobs)}; keep={a.keep}; motif column={motif_col}; pools {len(pools)}; "
                 f"max_len {a.max_len}; short_max {a.short_max}; min_frags {a.min_frags}\n")
    print("done:", a.out, flush=True)


if __name__ == "__main__":
    main()
