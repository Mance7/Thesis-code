#!/usr/bin/env python3
"""
End-motif tests with CLR, moderated t, hierarchical FDR and pre-specified families.

The revised end-motif analysis of Methods 2.7 and 2.9:

Layer 1  CLR transform of each sample-cell's motif composition,
         CLR_j = log(c_j + 0.5) - mean_k log(c_k + 0.5), and a limma-style
         moderated t (empirical-Bayes variance shrinkage over all motifs in
         the cell, Smyth 2004). Mann-Whitney p and Cliff's delta are kept as
         sensitivity columns; "robust" = same sign and |Cliff's delta| >= 0.5.
Layer 2  optional merge to the 4-bin scheme, region exclusion, label-blind
         independent filtering (mean raw count >= 5, optional CLR variance quantile).
Layer 3  hierarchical testing: stage 1 = one exact-permutation test per
         (bin, region) cell on sum(mod_t^2) using every possible label split
         (3,003 for 8 vs 6); stage 2 = per-motif BH inside selected cells at
         level q*R/M (Benjamini & Bogomolov 2014).
Layer 4  pre-specified families read from --families (simple text table):
             family_name <TAB> size_bin <TAB> region <TAB> motif <TAB> direction(+|-|two)
         Each family is BH-corrected on its own; one-sided where a direction is given.
Also     the full global screen (all cells x motifs, BH) for transparency.

Input: long table with one row per (sample, size_bin, region, motif). Either a
count column (--count-col) or frequency + cell size (--freq-col as percent,
--n-col) from which counts are reconstructed.
Output (--out): motif_tests_all.csv, motif_cell_stage1.csv,
motif_prespecified_families.csv (with --families), summary.txt

Usage, four merged bins (<=600, 601-2000, 2001-5000, >5000 bp; Results 3.3):
  python 05_motif_reduced_family_stats.py \\
      --table motif_region/direct/motif_frequencies_region_stratified_long.csv --metadata metadata/ont_samples.csv \\
      --out motif_reduced/ --bin-col bin --freq-col frequency_pct --n-col n_fragments --merge-4bin --threads 20
  (the families of this run use the 4-bin labels, in the same five-column format)
Five bins (table from 04_remap_motif_bins.py, families in motif_families_5bin.tsv):
  python 05_motif_reduced_family_stats.py --table motif5/ONT_direct/table/motif_frequencies_5bin_long.csv \\
      --metadata metadata/ont_samples.csv \\
      --out motif5/ONT_direct/stats --bin-col bin --count-col count \\
      --families motif_families_5bin.tsv --threads 20
"""
import argparse, itertools, os, sys, warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import stats, special

warnings.filterwarnings("ignore", category=RuntimeWarning)

FOUR_BIN = {"<90bp": "<=600bp", "90-125bp": "<=600bp", "126-135bp": "<=600bp", "136-150bp": "<=600bp",
            "151-255bp": "<=600bp", "256-420bp": "<=600bp", "421-650bp": "601-2000bp",
            "651-1000bp": "601-2000bp", "1001-2000bp": "601-2000bp", "2001-5000bp": "2001-5000bp",
            ">5000bp": ">5000bp"}
# Note: 421-650bp straddles 600 bp. Its fragments go to 601-2000bp here; exact
# 600 bp edges would need the long table rebuilt from the fragment files.


# ---------- statistics ----------
def trigamma_inv(x):
    """Inverse trigamma by Newton's method (as in limma::trigammaInverse)."""
    x = np.asarray(x, float)
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if np.all(np.abs(-dif / y) < 1e-8):
            break
    return y


def squeeze_var(s2, d):
    """Empirical-Bayes prior (d0, s0^2) for residual variances s2 with df d (Smyth 2004)."""
    s2 = np.maximum(s2, 1e-12)
    z = np.log(s2)
    e = z - special.digamma(d / 2) + np.log(d / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, d / 2)
    if evar > 0:
        d0 = 2 * trigamma_inv(evar)
        s02 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s02 = np.inf, np.exp(emean)
    return d0, s02


def moderated_t(X, y):
    """X: samples x features (CLR values); y: boolean case labels. Returns t, df, p, diff."""
    a, b = X[y], X[~y]
    n1, n2 = len(a), len(b)
    d = n1 + n2 - 2
    diff = a.mean(0) - b.mean(0)
    s2 = ((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)
    s2 = s2 / d
    d0, s02 = squeeze_var(s2, d)
    if np.isinf(d0):
        s2t, dft = np.full_like(s2, s02), np.inf
    else:
        s2t, dft = (d0 * s02 + d * s2) / (d0 + d), d + d0
    t = diff / np.sqrt(s2t * (1 / n1 + 1 / n2))
    p = 2 * (stats.norm.sf(np.abs(t)) if np.isinf(dft) else stats.t.sf(np.abs(t), dft))
    return t, dft, p, diff


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); m = ok.sum()
    if m == 0:
        return out
    o = np.argsort(p[ok]); r = p[ok][o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; tmp = np.empty(m); tmp[o] = np.minimum(r, 1); out[ok] = tmp
    return out


def clr(counts, pseudo=0.5):
    lc = np.log(counts + pseudo)
    return lc - lc.mean(1, keepdims=True)


def _count_ge(X, n, splits, obs):
    ge = 0
    for idx in splits:
        yp = np.zeros(n, bool); yp[list(idx)] = True
        ge += np.nansum(moderated_t(X, yp)[0] ** 2) >= obs - 1e-12
    return int(ge)


def exact_perm_p(X, y, max_perm=5000, rng=None, threads=1):
    """Stage-1 cell test: statistic sum(mod_t^2); p over all label splits (or a random subset if too many).
    Splits are scored in parallel chunks across `threads` workers."""
    n, k = len(y), int(y.sum())
    obs = np.nansum(moderated_t(X, y)[0] ** 2)
    total = special.comb(n, k, exact=True)
    exact = total <= max_perm
    if exact:
        splits = list(itertools.combinations(range(n), k))
    else:
        splits = [tuple(rng.choice(n, k, replace=False)) for _ in range(max_perm)]
    chunks = [splits[i::max(1, threads)] for i in range(max(1, threads))]
    if threads > 1:
        ge = sum(Parallel(n_jobs=threads)(delayed(_count_ge)(X, n, c, obs) for c in chunks if c))
    else:
        ge = _count_ge(X, n, splits, obs)
    cnt = len(splits)
    return obs, (ge + (0 if exact else 1)) / (cnt + (0 if exact else 1)), cnt


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="motif long table (csv), one row per sample, bin, region, motif")
    ap.add_argument("--metadata", required=True,
                    help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--sample-col", default="sample"); ap.add_argument("--bin-col", default="size_bin")
    ap.add_argument("--region-col", default="region"); ap.add_argument("--motif-col", default="motif")
    ap.add_argument("--count-col", help="raw count column (alternative to --freq-col + --n-col)")
    ap.add_argument("--freq-col", help="frequency column, in percent")
    ap.add_argument("--n-col", help="fragments per sample-cell column")
    ap.add_argument("--merge-4bin", action="store_true", help="merge the 11 bins into <=600, 601-2000, 2001-5000, >5000 bp")
    ap.add_argument("--exclude-regions", default="is_cpg_island",
                    help="comma-separated regions left out (thesis: is_cpg_island; '' keeps all)")
    ap.add_argument("--min-frags", type=int, default=100, help="per sample-cell, as in the original pipeline")
    ap.add_argument("--min-mean-count", type=float, default=5, help="label-blind filter: mean raw count across samples")
    ap.add_argument("--var-quantile", type=float, default=0.0, help="drop motifs below this CLR-variance quantile (label-blind), e.g. 0.25")
    ap.add_argument("--min-per-group", type=int, default=3, help="samples per group needed to test a cell")
    ap.add_argument("--families", help="tsv: family, size_bin, region, motif, direction")
    ap.add_argument("--q", type=float, default=0.05, help="FDR level")
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--max-perm", type=int, default=5000, help="all label splits are used when there are at most this many")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel workers for the stage-1 permutation test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.table, sep=None, engine="python")
    meta = pd.read_csv(a.metadata, sep=None, engine="python", encoding_errors="replace")
    meta.columns = [c.strip().lower() for c in meta.columns]
    grp = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))
    df = df.rename(columns={a.sample_col: "sample", a.bin_col: "size_bin", a.region_col: "region", a.motif_col: "motif"})
    df["sample"] = df["sample"].astype(str).str.strip()
    df = df[df["size_bin"].astype(str) != "background"]     # genome-wide background rows are a denominator, not a test cell
    if a.count_col:
        df["count"] = df[a.count_col].astype(float)
        ncell = df.groupby(["sample", "size_bin", "region"])["count"].transform("sum")
        df["n_cell"] = ncell
    else:
        if not (a.freq_col and a.n_col):
            sys.exit("give --count-col, or --freq-col (percent) and --n-col")
        df["count"] = np.rint(df[a.freq_col].astype(float) / 100 * df[a.n_col].astype(float))
        df["n_cell"] = df[a.n_col].astype(float)
    if a.merge_4bin:
        df["size_bin"] = df["size_bin"].map(FOUR_BIN).fillna(df["size_bin"])
        df = df.groupby(["sample", "size_bin", "region", "motif"], as_index=False).agg(count=("count", "sum"))
        df["n_cell"] = df.groupby(["sample", "size_bin", "region"])["count"].transform("sum")
    excl = [r for r in a.exclude_regions.split(",") if r]
    df = df[~df.region.isin(excl)]
    df["group"] = df["sample"].map(grp)
    df = df[df.group.isin([a.case, a.control])]

    rows, cell_rows = [], []
    for (b, r), g in df.groupby(["size_bin", "region"], sort=False):
        g = g[g.n_cell >= a.min_frags]
        M = g.pivot_table(index="sample", columns="motif", values="count", aggfunc="sum", fill_value=0)
        if M.empty:
            continue
        y = np.array([grp[s] == a.case for s in M.index])
        if y.sum() < a.min_per_group or (~y).sum() < a.min_per_group:
            continue
        keep = M.mean(axis=0) >= a.min_mean_count                       # label-blind
        Xc = clr(M.to_numpy(float))                                 # CLR over all motifs, then filter
        X = pd.DataFrame(Xc, index=M.index, columns=M.columns).loc[:, keep]
        if a.var_quantile > 0 and X.shape[1] > 4:
            v = X.var(axis=0); X = X.loc[:, v >= v.quantile(a.var_quantile)]
        Xv = X.to_numpy()
        t, dft, p, diff = moderated_t(Xv, y)
        for j, m in enumerate(X.columns):
            ca, cb = Xv[y, j], Xv[~y, j]
            rows.append(dict(size_bin=b, region=r, motif=m, n_case=int(y.sum()), n_control=int((~y).sum()),
                             clr_diff=diff[j], mod_t=t[j], df=dft, p_modt=p[j],
                             p_mwu=stats.mannwhitneyu(ca, cb, alternative="two-sided").pvalue,
                             cliffs_delta=cliffs_delta(ca, cb),
                             mean_count=M[m].mean()))
        obs, pc, nperm = exact_perm_p(Xv, y, a.max_perm, rng, a.threads)
        cell_rows.append(dict(size_bin=b, region=r, n_motifs=X.shape[1], n_case=int(y.sum()),
                              n_control=int((~y).sum()), stat_sum_t2=obs, p_cell=pc, n_perm=nperm))
        print(f"  {b:>12} {r:<14} motifs={X.shape[1]:3d}  cell p={pc:.4g} ({nperm} perms)")

    res = pd.DataFrame(rows); cells = pd.DataFrame(cell_rows)
    if res.empty:
        sys.exit("nothing testable; check column names and --min-frags")

    # global screen (exploratory)
    res["padj_global_modt"] = bh(res.p_modt); res["padj_global_mwu"] = bh(res.p_mwu)

    # hierarchical
    cells["padj_cell"] = bh(cells.p_cell)
    sel = cells[cells.padj_cell < a.q]
    R, Mcells = len(sel), len(cells)
    res["hier_selected_cell"] = False; res["padj_within_cell"] = np.nan; res["hier_sig"] = False
    for _, c in sel.iterrows():
        m = (res.size_bin == c.size_bin) & (res.region == c.region)
        res.loc[m, "hier_selected_cell"] = True
        pa = bh(res.loc[m, "p_modt"].to_numpy())
        res.loc[m, "padj_within_cell"] = pa
        res.loc[m, "hier_sig"] = pa < a.q * R / Mcells
    res["robust"] = (np.sign(res.clr_diff) == np.sign(res.cliffs_delta)) & (res.cliffs_delta.abs() >= 0.5)

    # pre-specified families
    fam_out = []
    if a.families:
        fam = pd.read_csv(a.families, sep="\t", comment="#", header=None,
                          names=["family", "size_bin", "region", "motif", "direction"])
        for fname, f in fam.groupby("family"):
            sub = f.merge(res, on=["size_bin", "region", "motif"], how="left")
            ps = []
            for _, s in sub.iterrows():
                if pd.isna(s.mod_t):
                    ps.append(np.nan); continue
                if s.direction in ("+", "-"):
                    sgn = 1 if s.direction == "+" else -1
                    tail = stats.norm.sf(sgn * s.mod_t) if np.isinf(s.df) else stats.t.sf(sgn * s.mod_t, s.df)
                    ps.append(tail)
                else:
                    ps.append(s.p_modt)
            sub["p_family"] = ps; sub["padj_family"] = bh(sub.p_family)
            fam_out.append(sub)
            print(f"  family {fname}: {sub.p_family.notna().sum()} tests, {int((sub.padj_family < a.q).sum())} significant")
    famdf = pd.concat(fam_out) if fam_out else pd.DataFrame()

    res.sort_values("p_modt").to_csv(os.path.join(a.out, "motif_tests_all.csv"), index=False)
    cells.sort_values("p_cell").to_csv(os.path.join(a.out, "motif_cell_stage1.csv"), index=False)
    if len(famdf):
        famdf.to_csv(os.path.join(a.out, "motif_prespecified_families.csv"), index=False)
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        fh.write(f"cells tested: {Mcells}; selected at stage 1 (BH<{a.q}): {R}\n")
        fh.write(f"motif tests (global screen): {len(res)}\n")
        fh.write(f"global BH<{a.q}: modt={int((res.padj_global_modt < a.q).sum())}, mwu={int((res.padj_global_mwu < a.q).sum())}\n")
        fh.write(f"hierarchical significant: {int(res.hier_sig.sum())} (robust: {int((res.hier_sig & res.robust).sum())})\n")
        if len(famdf):
            for fname, f in famdf.groupby("family"):
                fh.write(f"family {fname}: {int((f.padj_family < a.q).sum())}/{f.p_family.notna().sum()} significant\n")
    print(open(os.path.join(a.out, "summary.txt")).read())


if __name__ == "__main__":
    main()
