#!/usr/bin/env python3
"""
08_promoter_contribution_rank.py
--------------------------------
Which promoters carry the HCC-vs-HBV difference of a pooled promoter set?
(Methods 2.8, "Which promoters and programmes carry the difference";
exploratory.)

For a universe of promoters (the HCC-expressed set) the pooled metrics are
computed per sample exactly as in 07_geneset_random_null.py: fragments of up
to --max-len (600) bp, each assigned to the single promoter that contains its
midpoint, pooled over the universe; entropy_short and fsr (share <= --short-max,
246 bp). Each promoter then gets a leave-one-out influence:

    influence_i = sign(d_full) * (d_full - d_without_i)

i.e. how much the group difference (Cohen's d, HCC vs HBV) shrinks when that
promoter's fragments are removed from every sample. Positive values push the
difference in its observed direction.

Test: for each --test-sets gene list (the H5b subsets, or the 50 MSigDB
Hallmark sets restricted to the universe), the mean influence of its
promoters is compared with the same statistic under every relabelling of the
samples (all C(14,6) = 3,003 for 8 vs 6; random draws if there are more than
--max-exact). p = (b + 1) / (B + 1), one-sided (larger influence than under
random labels). The relabelling calibrates for promoters that dominate the
pool through fragment count alone. BH across test sets per metric.

The per-promoter list is exploratory: with 8 vs 6 samples no single promoter
can be significant after correction, so the top promoters are candidates only.

Needs 07_geneset_random_null.py in the same folder (its fragment reader is used).

Usage:
  python 08_promoter_contribution_rank.py --fragments "filtered_mid/*.eligible_fragments.csv.gz" \
      --metadata ont_samples.csv --promoters promoters_gencode_v49_gene_level.bed \
      --universe hcc_expressed.txt \
      --test-sets "S1=h5b_subsets/S1_hcc_blood_silent.txt,S5=h5b_subsets/S5_hcc_tertile_high.txt,S6=h5b_subsets/S6_liver_enriched.txt" \
      --out promoter_rank_all/ -t 14 [--keep source=direct]

Outputs in --out:
  promoter_influence.csv   one row per universe promoter: gene id, gene name (if the
                           BED has a 5th column), median fragments per group, pooled
                           group fsr, influence for both metrics, ranks, test-set membership
  set_enrichment.csv       per test set and metric: n, observed mean influence,
                           null median, p_perm, q_bh
  top_promoters.csv        top --top-n promoters by fsr influence and by entropy influence
  run_info.txt
"""
import argparse, glob, importlib.util, itertools, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# fragment reader and helpers from 07_geneset_random_null.py (same folder); registered under a
# plain module name so the worker processes can find its functions
_spec = importlib.util.spec_from_file_location(
    "geneset_random_null", os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_geneset_random_null.py"))
gnull = importlib.util.module_from_spec(_spec)
sys.modules["geneset_random_null"] = gnull
_spec.loader.exec_module(gnull)


def loo_metrics(H, members, max_len, short_max):
    """H: sparse P x L histogram of one sample. Returns full-pool (entropy, fsr, N)
    and leave-one-out arrays over the universe promoters `members`."""
    Hu = H[members].tocsr()
    T = np.asarray(Hu.sum(0)).ravel().astype(float)          # pooled histogram
    N = T.sum(); Sshort = T[:short_max].sum()
    n_i = np.asarray(Hu.sum(1)).ravel().astype(float)
    s_i = np.asarray(Hu[:, :short_max].sum(1)).ravel().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsr_full = Sshort / N if N > 0 else np.nan
        fsr_loo = (Sshort - s_i) / (N - n_i)
    # entropy: sum_k x log x terms, updated only where row i is non-zero
    xlogx = lambda x: np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.0)
    A = xlogx(T).sum()
    corr = np.zeros(len(members))
    coo = Hu.tocoo()
    if coo.nnz:
        tk = T[coo.col]; new = tk - coo.data
        np.add.at(corr, coo.row, xlogx(new) - xlogx(tk))
    K = np.log2(max_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_full = (np.log2(N) - A / N) / K if N > 0 else np.nan
        Nl = N - n_i
        ent_loo = np.where(Nl > 0, (np.log2(np.where(Nl > 0, Nl, 1)) - (A + corr) / np.where(Nl > 0, Nl, 1)) / K, np.nan)
    return ent_full, fsr_full, N, ent_loo, fsr_loo, n_i, s_i


def d_rows(M, y):
    """Cohen's d (case - control) for each row of M (rows x samples) under boolean labels y."""
    a, b = M[:, y], M[:, ~y]
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(1, ddof=1), b.var(1, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a.mean(1) - b.mean(1)) / sp


def influences(full, loo, y):
    """full: samples vector; loo: P x samples. Returns d_full, influence per promoter."""
    d0 = d_rows(full[None, :], y)[0]
    dl = d_rows(loo, y)
    return d0, np.sign(d0) * (d0 - dl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample eligible fragment tables (csv.gz)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("--promoters", required=True, help="path to the gene-level promoter BED (chrom, start, end, gene_id[, name])")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--universe", help="gene list defining the pooled set (default: all promoters in the BED)")
    ap.add_argument("--test-sets", default="", help="NAME=file,NAME=file gene lists tested for enrichment of influence")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from the file name")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end"); ap.add_argument("--length-col", default="length")
    ap.add_argument("--max-len", type=int, default=600); ap.add_argument("--short-max", type=int, default=246)
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct'")
    ap.add_argument("--max-exact", type=int, default=20000, help="enumerate all labellings up to this many, else sample")
    ap.add_argument("--n-perm", type=int, default=5000, help="random labellings if not exact")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel workers (samples are read in parallel)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = gnull.read_any(a.metadata); meta.columns = [c.strip().lower() for c in meta.columns]
    grp = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    with open(a.promoters) as fh:
        first = next(l for l in fh if l.strip() and not l.startswith(("#", "track", "browser")))
    ncol = len(first.rstrip("\n").split("\t"))
    if ncol < 4:
        raise SystemExit(f"--promoters has {ncol} columns; needs chrom, start, end, gene_id (Ensembl)")
    usecols = [0, 1, 2, 3] + ([4] if ncol >= 5 else [])
    prom = pd.read_csv(a.promoters, sep="\t", header=None, usecols=usecols, comment="#")
    prom.columns = ["chrom", "start", "end", "gene"] + (["name"] if ncol >= 5 else [])
    prom["gene"] = prom["gene"].map(gnull.strip_ver)
    prom = prom.drop_duplicates("gene").reset_index(drop=True)
    P = len(prom)
    gene_idx = {g: i for i, g in enumerate(prom.gene)}
    by_chrom = {c: (g.index.to_numpy()[np.argsort(g.start.to_numpy())], np.sort(g.start.to_numpy()))
                for c, g in prom.groupby("chrom")}
    ends = prom.end.to_numpy()

    def load_list(path):
        return [gnull.strip_ver(l.split("\t")[0]) for l in open(path) if l.strip() and not l.lower().startswith("gene")]

    if a.universe:
        uni = np.array(sorted({gene_idx[g] for g in load_list(a.universe) if g in gene_idx}))
    else:
        uni = np.arange(P)
    print(f"universe: {len(uni)} promoters")
    tests = {}
    for item in [x for x in a.test_sets.split(",") if x.strip()]:
        name, path = item.split("=", 1)
        gs = {gene_idx[g] for g in load_list(path) if g in gene_idx}
        pos = np.where(np.isin(uni, list(gs)))[0]
        tests[name] = pos
        print(f"  test set {name}: {len(gs)} promoters, {len(pos)} in universe")

    jobs = []
    for f in sorted(glob.glob(a.fragments)):
        m = re.match(a.sample_regex, os.path.basename(f)); s = m.group(1) if m else os.path.basename(f)
        if grp.get(s) in (a.case, a.control):
            jobs.append((f, s))
    if not jobs:
        raise SystemExit("no fragment files matched samples of the two groups in --metadata")
    with ProcessPoolExecutor(max_workers=max(1, min(a.threads, len(jobs)))) as ex:
        res_ = list(ex.map(gnull.sample_hist, [f for f, _ in jobs], [s for _, s in jobs],
                           [a] * len(jobs), [by_chrom] * len(jobs), [ends] * len(jobs), [P] * len(jobs)))
    samples = [r[0] for r in res_]
    y = np.array([grp[s] == a.case for s in samples])
    nS = len(samples)
    E_full = np.zeros(nS); F_full = np.zeros(nS); Ntot = np.zeros(nS)
    E_loo = np.zeros((len(uni), nS)); F_loo = np.zeros((len(uni), nS))
    n_i = np.zeros((len(uni), nS)); s_i = np.zeros((len(uni), nS))
    for k, (s, H, n) in enumerate(res_):
        E_full[k], F_full[k], Ntot[k], E_loo[:, k], F_loo[:, k], n_i[:, k], s_i[:, k] = loo_metrics(
            H.tocsr(), uni, a.max_len, a.short_max)
        print(f"  {s} ({grp[s]}): {int(Ntot[k]):,} fragments in the universe pool")

    # observed influences
    obs = {}
    for metric, full, loo in [("fsr", F_full, F_loo), ("entropy_short", E_full, E_loo)]:
        d0, inf = influences(full, loo, y)
        obs[metric] = (d0, inf)
        print(f"  pooled {metric}: d = {d0:.3f}; sum of influences = {np.nansum(inf):.3f}")

    # label permutations
    n_case = int(y.sum())
    combos = list(itertools.combinations(range(nS), n_case)) if \
        len(list(itertools.islice(itertools.combinations(range(nS), n_case), a.max_exact + 1))) <= a.max_exact else None
    if combos is not None:
        labs = combos; mode = f"exact, {len(combos)} labellings"
    else:
        rng = np.random.default_rng(a.seed)
        labs = [tuple(sorted(rng.choice(nS, n_case, replace=False))) for _ in range(a.n_perm)]; mode = f"random, {a.n_perm} labellings"
    print(f"permutation null: {mode}")
    null = {m: {t: [] for t in tests} for m in obs}
    for lab in labs:
        yy = np.zeros(nS, bool); yy[list(lab)] = True
        for metric, full, loo in [("fsr", F_full, F_loo), ("entropy_short", E_full, E_loo)]:
            _, inf = influences(full, loo, yy)
            for t, pos in tests.items():
                null[metric][t].append(np.nanmean(inf[pos]) if len(pos) else np.nan)

    rows = []
    for metric in obs:
        inf = obs[metric][1]
        for t, pos in tests.items():
            o = np.nanmean(inf[pos]) if len(pos) else np.nan
            nl = np.array(null[metric][t]); nl = nl[~np.isnan(nl)]
            rows.append(dict(metric=metric, test_set=t, n_promoters=len(pos), obs_mean_influence=o,
                             null_median=float(np.median(nl)) if len(nl) else np.nan,
                             p_perm=(np.sum(nl >= o) + 1) / (len(nl) + 1) if len(nl) else np.nan,
                             share_of_promoters=len(pos) / len(uni)))
    enr = pd.DataFrame(rows)
    if len(enr):
        enr["q_bh"] = np.nan
        for _, ix in enr.groupby("metric").groups.items():
            pv = enr.loc[ix, "p_perm"].to_numpy(float); o_ = np.argsort(pv); n_ = len(pv)
            q = np.minimum.accumulate((pv[o_] * n_ / np.arange(1, n_ + 1))[::-1])[::-1]
            qq = np.empty(n_); qq[o_] = np.minimum(q, 1); enr.loc[ix, "q_bh"] = qq
    enr.to_csv(os.path.join(a.out, "set_enrichment.csv"), index=False)

    tab = prom.iloc[uni][["chrom", "start", "end", "gene"] + (["name"] if "name" in prom else [])].reset_index(drop=True)
    tab["median_frags_case"] = np.median(n_i[:, y], 1); tab["median_frags_control"] = np.median(n_i[:, ~y], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tab["pooled_fsr_case"] = s_i[:, y].sum(1) / n_i[:, y].sum(1)
        tab["pooled_fsr_control"] = s_i[:, ~y].sum(1) / n_i[:, ~y].sum(1)
    for metric in obs:
        tab[f"influence_{metric}"] = obs[metric][1]
        tab[f"rank_{metric}"] = tab[f"influence_{metric}"].rank(ascending=False, method="min")
    for t, pos in tests.items():
        tab[f"in_{t}"] = False; tab.loc[pos, f"in_{t}"] = True
    tab = tab.sort_values("rank_fsr")
    tab.to_csv(os.path.join(a.out, "promoter_influence.csv"), index=False)
    top = pd.concat([tab.nsmallest(a.top_n, "rank_fsr").assign(ranked_by="fsr"),
                     tab.nsmallest(a.top_n, "rank_entropy_short").assign(ranked_by="entropy_short")])
    top.to_csv(os.path.join(a.out, "top_promoters.csv"), index=False)

    with open(os.path.join(a.out, "run_info.txt"), "w") as fh:
        fh.write(f"samples: {nS} ({n_case} {a.case}, {nS - n_case} {a.control}); keep={a.keep}\n")
        fh.write(f"universe: {len(uni)} promoters; max_len={a.max_len}; short_max={a.short_max}\n")
        fh.write(f"null: {mode}; smallest attainable p = {1 / (len(labs) + 1):.2e}\n")
        for metric in obs:
            fh.write(f"pooled {metric}: d = {obs[metric][0]:.3f}\n")
    print(enr.to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
