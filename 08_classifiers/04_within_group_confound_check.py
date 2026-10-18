#!/usr/bin/env python3
"""
04_within_group_confound_check.py
---------------------------------
Is a classifier feature a proxy for depth or concatemer recovery? (Methods 2.10)

Concatemer recovery separates HCC from HBV carriers completely in this cohort,
so a between-group comparison cannot tell a biological feature from a
recovery or depth proxy. Inside one group the label is fixed, so a feature
that follows log10 n_fragments or pct_recovered within HCC and within HBV is
following the covariate, not the disease.

For every feature and covariate, both are centred on their group means
(pooled within-group correlation) and the script reports:
  r_within      pooled within-group Pearson r
  p_perm        two-sided permutation p, the covariate shuffled inside each group (--n-perm)
  q_bh          Benjamini-Hochberg over all features x covariates of one table
A summary row per table and covariate gives the share of features with
|r_within| >= --r-flag (0.5) and q < 0.10.

Inputs: feature tables and covariates.csv from 02_build_classifier_features.py
(n_fragments is turned into log10_n_fragments).
Outputs in --out: <folder>_<table>_within_group.csv per table and within_group_summary.csv.

Usage:
  python 04_within_group_confound_check.py \
      --features feats/direct/F_size.csv feats/direct/F_entropy_rare.csv feats/direct/F_motif_lit.csv feats/direct/F_motif.csv \
      --covariates feats/all/covariates.csv --metadata ont_samples.csv --out confound_check_direct/ -t 20
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_wide(path):
    df = pd.read_csv(path, sep=None, engine="python")
    idx = "sample" if "sample" in df.columns else df.columns[0]
    df = df.set_index(idx)
    df.index = df.index.astype(str).str.strip()
    return df.apply(pd.to_numeric, errors="coerce")


def centre(v, g):
    out = v.copy()
    for k in np.unique(g):
        m = g == k
        out[m] = v[m] - np.nanmean(v[m])
    return out


def one_feature(x, covs, g, n_perm, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for cname, c in covs.items():
        ok = np.isfinite(x) & np.isfinite(c)
        if ok.sum() < 6 or np.nanstd(x[ok]) == 0:
            rows.append((cname, np.nan, np.nan)); continue
        xs, cs, gs = x[ok], c[ok], g[ok]
        xc, cc = centre(xs, gs), centre(cs, gs)
        r = float(np.corrcoef(xc, cc)[0, 1])
        null = np.empty(n_perm)
        for b in range(n_perm):
            cp = cs.copy()
            for k in np.unique(gs):                      # shuffle inside each group only
                m = np.where(gs == k)[0]
                cp[m] = cp[rng.permutation(m)]
            null[b] = np.corrcoef(xc, centre(cp, gs))[0, 1]
        p = (np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / (n_perm + 1)
        rows.append((cname, r, p))
    return rows


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); n = len(pv)
        qq = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        tmp = np.empty(n); tmp[o] = np.minimum(qq, 1); q[ok] = tmp
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True, help="one or more wide feature tables (csv)")
    ap.add_argument("--covariates", required=True, help="path to covariates.csv (n_fragments, pct_recovered)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--n-perm", type=int, default=2000, help="within-group permutations per feature and covariate")
    ap.add_argument("--r-flag", type=float, default=0.5, help="|r_within| at or above this (with q < 0.10) flags a feature")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("-t", "--threads", type=int, default=4, help="features processed in parallel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = pd.read_csv(a.metadata, sep=None, engine="python", encoding_errors="replace")
    m.columns = [c.strip().lower() for c in m.columns]
    grp = dict(zip(m["sample"].astype(str).str.strip(), m["group"].astype(str).str.strip()))
    C = load_wide(a.covariates)
    if "n_fragments" in C.columns:
        C["log10_n_fragments"] = np.log10(C["n_fragments"].clip(lower=1))
        C = C.drop(columns="n_fragments")

    summary = []
    for path in a.features:
        F = load_wide(path)
        F = F[[s in grp and grp[s] in (a.case, a.control) and s in C.index for s in F.index]]
        F = F.loc[:, F.notna().mean() >= 0.5]
        g = np.array([grp[s] for s in F.index])
        covs = {c: C.loc[F.index, c].to_numpy(float) for c in C.columns}
        res = Parallel(n_jobs=a.threads)(
            delayed(one_feature)(F[c].to_numpy(float), covs, g, a.n_perm, a.seed + i)
            for i, c in enumerate(F.columns))
        rows = [(f, cn, r, p) for f, rr in zip(F.columns, res) for cn, r, p in rr]
        out = pd.DataFrame(rows, columns=["feature", "covariate", "r_within", "p_perm"])
        out["q_bh"] = bh(out["p_perm"])
        name = os.path.splitext(os.path.basename(path))[0]
        tag = os.path.basename(os.path.dirname(os.path.abspath(path)))
        out.to_csv(os.path.join(a.out, f"{tag}_{name}_within_group.csv"), index=False)
        for cn in C.columns:
            o = out[out.covariate == cn]
            flag = ((o.r_within.abs() >= a.r_flag) & (o.q_bh < 0.10)).sum()
            summary.append(dict(table=f"{tag}/{name}", covariate=cn, n_features=len(o),
                                median_abs_r=float(o.r_within.abs().median()),
                                n_flagged=int(flag), pct_flagged=100.0 * flag / max(len(o), 1)))
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(a.out, "within_group_summary.csv"), index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
