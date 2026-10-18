#!/usr/bin/env python3
"""
Original 256-motif screen: HCC vs HBV carrier tests per size bin x region x motif.

This is the first end-motif screen of the thesis (Methods 2.7, "original
screen"), run on the 11 size bins x 6 regions (genome_wide, is_promoter,
is_gene_body, is_tss, is_exon, is_cpg_island) x 256 motifs.

1. O/E ratio. For each sample, the frequency of a motif in a cell is divided
   by that sample's own genome-wide background frequency of the motif (all
   fragments, all bins, all regions pooled):
       O/E = frequency_pct(bin, region, motif) / background_freq_pct(motif)
   1.0 = as often as the sample's overall composition predicts. A motif not
   seen in a cell of a sample that passes the gate counts as frequency 0.
2. Gate. A sample enters a cell only with >= --min-frags (100) fragments in
   it; a cell is tested only with >= --min-per-group (3) samples per group.
3. Tests per (bin, region, motif): two-sided Mann-Whitney U and one-way ANOVA
   (equivalent to a two-group t-test) on the O/E values, Cohen's d with
   pooled SD (positive = higher in HCC).
4. Benjamini-Hochberg FDR in two scopes: global (all tested bin x region x
   motif rows together; primary) and per bin (within each size bin).
5. Confidence tier, scored 0-3 on: MWU FDR < --alpha, ANOVA FDR < --alpha and
   |d| >= 0.8 (Cohen 1988); 3 = High confidence, 2 = Confident, 1 = Moderate,
   0 = none. Scored for the global and for the per-bin scope.
6. Label-blind noise flag: mean raw count of the motif across all samples in
   the cell >= --min-mean-count (5). Rows below it stay in the screen but are
   flagged (passes_min_mean_count = False) and were left out of the classifier
   candidate pool.
Also the motif diversity score per cell, MDS = -sum(p log2 p) / log2(256),
tested the same way (MWU, Cohen's d, BH over cells).

Groups come from the metadata file (sample, group), never from sample names.
Cells are tested in parallel (--workers).

Input : motif_frequencies_region_stratified_long.csv (02_aggregate_motif_frequencies_region_stratified.py)
        metadata csv with columns sample, group
Output: <out>/motif_stats_region_stratified.csv   one row per (bin, region, motif)
        <out>/mds_stats_region_stratified.csv     one row per (bin, region)
        <out>/motif_stats_summary.txt

  python 03_motif_stats_region_stratified.py \\
      --table motif_region/all/motif_frequencies_region_stratified_long.csv \\
      --metadata metadata/ont_samples.csv -o motif_stats_region/ --workers 14
"""
import argparse
import itertools
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=getattr(stats, "ConstantInputWarning", RuntimeWarning))

MOTIFS = ["".join(m) for m in itertools.product("ACGT", repeat=4)]
BIN_ORDER = ["<90bp", "90-125bp", "126-135bp", "136-150bp", "151-255bp", "256-420bp", "421-650bp",
             "651-1000bp", "1001-2000bp", "2001-5000bp", ">5000bp"]
TIER_NAMES = {3: "High confidence", 2: "Confident", 1: "Moderate", 0: "none"}


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


def mwu_p(x, y):
    try:
        return stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    except ValueError:
        return np.nan


def test_cell(task):
    """All 256 motif tests (and the MDS test) of one (bin, region) cell."""
    b, r, freq, counts, bg, y, min_mean_count = task
    # freq, counts: samples x 256 (percent and raw count); bg: samples x 256 background percent
    with np.errstate(divide="ignore", invalid="ignore"):
        oe = freq / bg
    rows = []
    for j, motif in enumerate(MOTIFS):
        v = oe[:, j]
        x, z = v[y], v[~y]
        x, z = x[np.isfinite(x)], z[np.isfinite(z)]
        rec = dict(size_bin=b, region=r, motif=motif, n_case=len(x), n_control=len(z),
                   mean_oe_case=x.mean() if len(x) else np.nan, mean_oe_control=z.mean() if len(z) else np.nan,
                   mean_freq_pct_case=freq[y, j].mean(), mean_freq_pct_control=freq[~y, j].mean(),
                   mean_raw_count=counts[:, j].mean())
        rec["passes_min_mean_count"] = rec["mean_raw_count"] >= min_mean_count
        if len(x) >= 2 and len(z) >= 2:
            rec["p_mwu"] = mwu_p(x, z)
            rec["p_anova"] = stats.f_oneway(x, z).pvalue
            rec["cohens_d"] = cohens_d(x, z)
        rows.append(rec)
    # motif diversity score of every sample in the cell
    n = counts.sum(1, keepdims=True)
    p = np.where(n > 0, counts / np.maximum(n, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mds = -np.where(p > 0, p * np.log2(p), 0).sum(1) / np.log2(256)
    mds_row = dict(size_bin=b, region=r, n_case=int(y.sum()), n_control=int((~y).sum()),
                   mean_mds_case=mds[y].mean(), mean_mds_control=mds[~y].mean(),
                   p_mwu=mwu_p(mds[y], mds[~y]), cohens_d=cohens_d(mds[y], mds[~y]))
    return rows, mds_row


def tier(res, scope, alpha, d_min):
    """Score 0-3 and tier name for one FDR scope ('global' or 'perbin')."""
    score = ((res[f"padj_{scope}_mwu"] < alpha).astype(int)
             + (res[f"padj_{scope}_anova"] < alpha).astype(int)
             + (res["cohens_d"].abs() >= d_min).astype(int))
    score = score.where(res["p_mwu"].notna(), 0)
    return score, score.map(TIER_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True,
                    help="motif long table, motif_frequencies_region_stratified_long.csv")
    ap.add_argument("--metadata", required=True, help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--min-frags", type=int, default=100, help="fragments a sample needs in a cell")
    ap.add_argument("--min-per-group", type=int, default=3, help="samples per group needed to test a cell")
    ap.add_argument("--min-mean-count", type=float, default=5,
                    help="label-blind flag: mean raw motif count across samples in the cell")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level used for the tiers")
    ap.add_argument("--d-min", type=float, default=0.8, help="|Cohen's d| counted as a large effect")
    ap.add_argument("--case", default="HCC")
    ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("-t", "--threads", "--workers", dest="workers", type=int,
                    default=max(1, int((os.cpu_count() or 2) * 0.5)), help="cells tested in parallel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = read_csv_any_encoding(a.metadata)
    meta.columns = [c.strip().lower() for c in meta.columns]
    group = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    df = pd.read_csv(a.table)
    df["sample"] = df["sample"].astype(str).str.strip()
    df = df[df["sample"].map(group).isin([a.case, a.control])]
    if "count" not in df.columns:
        # older tables: rebuild raw counts from percent x cell size
        df["count"] = np.rint(df["frequency_pct"] * df["n_fragments"] / 100)

    # per-sample genome-wide background frequency of every motif
    bgdf = df[df["group_type"] == "background"]
    bg = (bgdf.pivot_table(index="sample", columns="motif", values="frequency_pct", aggfunc="sum")
          .reindex(columns=MOTIFS))
    cells = df[df["group_type"] == "size_bin"]

    tasks = []
    for (b, r), g in cells.groupby(["bin", "region"], sort=False):
        g = g[g["n_fragments"] >= a.min_frags]
        if g.empty:
            continue
        freq = g.pivot_table(index="sample", columns="motif", values="frequency_pct", aggfunc="sum",
                             fill_value=0).reindex(columns=MOTIFS, fill_value=0)
        counts = g.pivot_table(index="sample", columns="motif", values="count", aggfunc="sum",
                               fill_value=0).reindex(columns=MOTIFS, fill_value=0)
        samples = [s for s in freq.index if s in bg.index]
        y = np.array([group[s] == a.case for s in samples])
        if y.sum() < a.min_per_group or (~y).sum() < a.min_per_group:
            continue
        tasks.append((b, r, freq.loc[samples].to_numpy(float), counts.loc[samples].to_numpy(float),
                      bg.loc[samples].to_numpy(float), y, a.min_mean_count))
    if not tasks:
        sys.exit("no testable cell; check --min-frags, --min-per-group and the metadata")
    print(f"{len(tasks)} testable (bin, region) cells, {a.workers} workers")

    rows, mds_rows = [], []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for r_cell, m_row in ex.map(test_cell, tasks):
            rows.extend(r_cell)
            mds_rows.append(m_row)
    res = pd.DataFrame(rows)

    # FDR: global (primary) and within each size bin
    res["padj_global_mwu"] = bh(res["p_mwu"])
    res["padj_global_anova"] = bh(res["p_anova"])
    res["padj_perbin_mwu"] = np.nan
    res["padj_perbin_anova"] = np.nan
    for b, idx in res.groupby("size_bin").groups.items():
        res.loc[idx, "padj_perbin_mwu"] = bh(res.loc[idx, "p_mwu"])
        res.loc[idx, "padj_perbin_anova"] = bh(res.loc[idx, "p_anova"])
    res["score_global"], res["tier_global"] = tier(res, "global", a.alpha, a.d_min)
    res["score_perbin"], res["tier_perbin"] = tier(res, "perbin", a.alpha, a.d_min)
    res["direction"] = np.where(res["cohens_d"] > 0, f"{a.case} higher",
                                np.where(res["cohens_d"] < 0, f"{a.case} lower", ""))
    res["bin_order"] = res["size_bin"].map({b: i for i, b in enumerate(BIN_ORDER)})
    res = res.sort_values(["p_mwu", "bin_order"]).drop(columns="bin_order")
    res.to_csv(os.path.join(a.out, "motif_stats_region_stratified.csv"), index=False)

    mds = pd.DataFrame(mds_rows)
    mds["padj_bh"] = bh(mds["p_mwu"])
    mds.sort_values("p_mwu").to_csv(os.path.join(a.out, "mds_stats_region_stratified.csv"), index=False)

    tested = res["p_mwu"].notna()
    lines = [f"cells tested: {len(tasks)}; motif tests: {int(tested.sum())}",
             f"global FDR < {a.alpha}: MWU {int((res.padj_global_mwu < a.alpha).sum())}, "
             f"ANOVA {int((res.padj_global_anova < a.alpha).sum())}",
             f"per-bin FDR < {a.alpha}: MWU {int((res.padj_perbin_mwu < a.alpha).sum())}, "
             f"ANOVA {int((res.padj_perbin_anova < a.alpha).sum())}",
             "tiers (global): " + str(res.loc[tested, "tier_global"].value_counts().to_dict()),
             "tiers (per bin): " + str(res.loc[tested, "tier_perbin"].value_counts().to_dict()),
             f"rows passing the mean raw count >= {a.min_mean_count} flag: {int(res.passes_min_mean_count.sum())}",
             f"MDS cells with FDR < {a.alpha}: {int((mds.padj_bh < a.alpha).sum())} of {len(mds)}"]
    with open(os.path.join(a.out, "motif_stats_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
