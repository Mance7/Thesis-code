#!/usr/bin/env python3
"""
HCC vs HBV carrier tests of fragment-length entropy, one test per cell.

Reads the per-sample entropy table of 01_length_entropy_calculation.py and,
for every size bin x region cell (66 cells), compares HCC with HBV carriers
(Methods 2.6 and 2.9):
  - two-sided Mann-Whitney U test (exact for small groups)
  - Cohen's d with pooled SD, and its exact 95% confidence interval from the
    noncentral t distribution
  - Cliff's delta (rank-based effect size)
  - Benjamini-Hochberg FDR over all tested cells together
Positive d / delta = entropy higher in HCC.

A sample enters a cell only if it has >= --min-frags fragments there (100,
the same gate as the motif analysis); a cell is tested only if both groups
then have >= --min-per-group samples (3).

Input : length_entropy_per_sample_cell.csv (01_length_entropy_calculation.py)
        metadata csv with columns sample, group
Output: <out>/length_entropy_stats.csv  one row per cell
        <out>/length_entropy_stats_summary.txt  cells with FDR < --alpha

  python 02_length_entropy_stats.py --entropy entropy/length_entropy_per_sample_cell.csv \\
      --metadata metadata/ont_samples.csv -o entropy_stats/
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq


def read_csv_any_encoding(path):
    """Metadata exported from Excel may not be UTF-8; try the usual encodings in turn."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def cohens_d(a, b):
    """Difference of means divided by the pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def d_confint(d, n1, n2, level=0.95):
    """Exact CI of Cohen's d by inverting the noncentral t distribution."""
    if not np.isfinite(d):
        return np.nan, np.nan
    df = n1 + n2 - 2
    scale = np.sqrt(n1 * n2 / (n1 + n2))
    t_obs = d * scale
    lo_q, hi_q = (1 + level) / 2, (1 - level) / 2
    f = lambda nc, q: stats.nct.cdf(t_obs, df, nc) - q   # decreases with nc

    def root(q):
        # widen a bracket around t_obs step by step (very distant nc can give NaN)
        lo, hi, step = t_obs - 1, t_obs + 1, 1.0
        while not f(lo, q) > 0:
            step *= 2; lo = t_obs - step
            if step > 1e4:
                return np.nan
        step = 1.0
        while not f(hi, q) < 0:
            step *= 2; hi = t_obs + step
            if step > 1e4:
                return np.nan
        return brentq(f, lo, hi, args=(q,))
    return root(lo_q) / scale, root(hi_q) / scale


def cliffs_delta(a, b):
    """P(a > b) - P(a < b) over all pairs."""
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def bh(p):
    """Benjamini-Hochberg adjusted p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.minimum(ranked, 1)
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entropy", required=True,
                    help="per-sample entropy table, length_entropy_per_sample_cell.csv")
    ap.add_argument("--metadata", required=True, help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--min-frags", type=int, default=100, help="fragments a sample needs in a cell")
    ap.add_argument("--min-per-group", type=int, default=3, help="samples per group needed to test a cell")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level for the summary")
    ap.add_argument("--case", default="HCC")
    ap.add_argument("--control", default="HBV_carrier")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ent = pd.read_csv(a.entropy)
    meta = read_csv_any_encoding(a.metadata)
    meta.columns = [c.strip().lower() for c in meta.columns]
    group = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))
    ent["sample"] = ent["sample"].astype(str).str.strip()
    ent["group"] = ent["sample"].map(group)
    ent = ent[ent["group"].isin([a.case, a.control])]

    # fragment-count gate: below it the cell is missing for that sample
    ent.loc[ent["n_fragments"] < a.min_frags, "entropy"] = np.nan

    rows = []
    for (b, r), g in ent.groupby(["size_bin", "region"], sort=False):
        g = g.dropna(subset=["entropy"])
        x = g.loc[g["group"] == a.case, "entropy"].to_numpy()
        y = g.loc[g["group"] == a.control, "entropy"].to_numpy()
        rec = dict(size_bin=b, region=r, n_case=len(x), n_control=len(y),
                   mean_case=x.mean() if len(x) else np.nan, mean_control=y.mean() if len(y) else np.nan,
                   median_n_fragments_case=g.loc[g["group"] == a.case, "n_fragments"].median(),
                   median_n_fragments_control=g.loc[g["group"] == a.control, "n_fragments"].median(),
                   tested=False)
        if len(x) >= a.min_per_group and len(y) >= a.min_per_group:
            d = cohens_d(x, y)
            lo, hi = d_confint(d, len(x), len(y))
            rec.update(tested=True, mwu_U=stats.mannwhitneyu(x, y, alternative="two-sided").statistic,
                       p_mwu=stats.mannwhitneyu(x, y, alternative="two-sided").pvalue,
                       cohens_d=d, d_ci_low=lo, d_ci_high=hi, cliffs_delta=cliffs_delta(x, y))
        rows.append(rec)
    res = pd.DataFrame(rows)
    if "p_mwu" not in res:
        raise SystemExit("no cell had enough samples in both groups")
    res["padj_bh"] = bh(res["p_mwu"])           # one family: all tested cells
    res["direction"] = np.where(res["cohens_d"] > 0, f"{a.case} higher",
                                np.where(res["cohens_d"] < 0, f"{a.case} lower", ""))
    out_path = os.path.join(a.out, "length_entropy_stats.csv")
    res.to_csv(out_path, index=False)

    sig = res[res["padj_bh"] < a.alpha].sort_values("padj_bh")
    cols = ["size_bin", "region", "n_case", "n_control", "mean_case", "mean_control",
            "cohens_d", "cliffs_delta", "p_mwu", "padj_bh"]
    txt = (f"cells: {len(res)}; tested (>= {a.min_per_group} per group, >= {a.min_frags} fragments): "
           f"{int(res['tested'].sum())}; FDR < {a.alpha}: {len(sig)}\n\n")
    txt += sig[cols].to_string(index=False) if len(sig) else "none"
    with open(os.path.join(a.out, "length_entropy_stats_summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
