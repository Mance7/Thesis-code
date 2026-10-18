#!/usr/bin/env python3
"""
04_test_gse71378.py

Pre-specified tests of the GSE71378 external validation (Methods 2.13), run on the
outputs of 02_extract_gse71378_sizes_motifs.py. Liver cancer is the case group; all
tests are one-sided exact Mann-Whitney tests in the discovery direction (higher in
liver cancer), with Benjamini-Hochberg correction over the three tests:
  P1  R_sr = n(256-600 bp) / n(100-250 bp)
  P2  C-end share = (CCCA + CCAG + CCTG) / all 5' 4-mers, fragments <= 600 bp
  P3  rarefied 151-255 bp entropy (10,000 fragments, mean of 100 draws, as in 2.6)
Comparisons: primary = liver cancer vs the 6 non-cancer runs of the short-read batch,
secondary = vs all 12 non-cancer runs, exploratory = vs the other cancers of the
short-read batch.

Input : DIR/gse_summary.tsv, gse_hist.tsv.gz, gse_motifs.tsv.gz; runs TSV (run, group, batch)
Output: OUT/gse71378_tests.csv (one row per comparison x metric)
        OUT/gse71378_per_run.csv (per-run R_sr, C-end share and entropy)
"""
import argparse, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description="GSE71378 pre-specified tests (R_sr, C-end share, 151-255 bp entropy)")
ap.add_argument("--dir", required=True,
                help="folder with gse_summary.tsv, gse_hist.tsv.gz and gse_motifs.tsv.gz")
ap.add_argument("--runs", default=str(HERE / "runs_gse71378.tsv"), help="runs TSV with columns run, group, batch")
ap.add_argument("-o", "--out", required=True, help="output folder")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

def bh(p):
    """Benjamini-Hochberg q values (NaN p values are ignored)"""
    p = np.asarray(p, float); n = np.sum(~np.isnan(p)); o = np.argsort(np.where(np.isnan(p), np.inf, p))
    q = np.full(len(p), np.nan); prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]; prev = min(prev, p[i] * n / (rank + 1)); q[i] = prev
    return q

d, runs, outp = args.dir, args.runs, os.path.join(args.out, "gse71378")
summ = pd.read_csv(f"{d}/gse_summary.tsv", sep="\t").set_index("run")
hist = pd.read_csv(f"{d}/gse_hist.tsv.gz", sep="\t").set_index("run")
mot = pd.read_csv(f"{d}/gse_motifs.tsv.gz", sep="\t").set_index("run")
meta = pd.read_csv(runs, sep="\t").set_index("run")
df = meta.join(summ[["fragments", "R_sr", "share_256_650"]], how="inner")

# P2: share of the C-end motifs among all 5' end 4-mers (fragments <= 600 bp)
C = ["CCCA", "CCAG", "CCTG"]
df["cend_share"] = mot.reindex(df.index)[C].sum(1) / mot.reindex(df.index).sum(1)
# P3: rarefied Shannon entropy of the 151-255 bp length profile
rng = np.random.default_rng(71378)
def rare_entropy(row, lo=151, hi=255, n=10_000, draws=100):
    c = row[[str(i) for i in range(lo, hi + 1)]].to_numpy(float)
    if c.sum() < n:
        return np.nan
    out = []
    for _ in range(draws):
        s = rng.multivariate_hypergeometric(c.astype(np.int64), n)
        p = s[s > 0] / n
        out.append(-(p * np.log2(p)).sum())
    return float(np.mean(out))
df["H_151_255"] = [rare_entropy(hist.loc[r]) for r in df.index]

def test(case, ctrl, metric):
    """one-sided exact Mann-Whitney test, case greater than controls"""
    x, y = df.loc[case, metric].dropna(), df.loc[ctrl, metric].dropna()
    if len(x) == 0 or len(y) == 0:
        return dict(metric=metric, n_case=len(x), n_ctrl=len(y), p=np.nan)
    p = mannwhitneyu(x, y, alternative="greater", method="exact").pvalue
    return dict(metric=metric, n_case=len(x), n_ctrl=len(y), case_median=x.median(), ctrl_median=y.median(),
                case_values=",".join(f"{v:.4g}" for v in x), ctrl_range=f"{y.min():.4g}-{y.max():.4g}",
                case_above_all_ctrl=int((x > y.max()).sum()), p_one_sided=p)

# groups: primary controls = non-cancer runs of the short-read batch
liver = df.index[df.group == "liver_cancer"]
sb = df.batch == "short_reads"
prim_ctrl = df.index[sb & df.group.isin(["healthy", "noncancer_disease"])]
all_ctrl = df.index[df.group.isin(["healthy", "noncancer_disease"])]
oth = df.index[sb & (df.group == "other_cancer")]
rows = []
for name, ctrl in [("primary: vs non-cancer, same batch", prim_ctrl), ("secondary: vs all non-cancer", all_ctrl),
                   ("exploratory: vs other cancers, same batch", oth)]:
    res = [dict(comparison=name, **test(liver, ctrl, m)) for m in ("R_sr", "cend_share", "H_151_255")]
    q = bh([r.get("p_one_sided", np.nan) for r in res])
    for r, qq in zip(res, q):
        r["q_BH"] = qq
    rows += res
out = pd.DataFrame(rows)
out.to_csv(f"{outp}_tests.csv", index=False)
df.to_csv(f"{outp}_per_run.csv")
pd.set_option("display.width", 250, "display.max_columns", 20)
print(out.to_string(index=False))
print(df.groupby(["group", "batch"])[["fragments", "R_sr", "cend_share", "H_151_255"]].median().to_string())
