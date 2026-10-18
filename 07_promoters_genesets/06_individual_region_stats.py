#!/usr/bin/env python3
"""
06_individual_region_stats.py
-----------------------------
Motif diversity score (MDS) per individual promoter and its group test
(Methods 2.7 and 2.8, level 2). The other per-promoter metrics come from
05_extra_fragmentomic_metrics.py; this script adds MDS so it can be compared
with the region-pooled MDS. Exploratory analysis.

Input: the promoter-tagged fragment tables (one row per fragment and promoter
it overlaps, any overlap of at least 1 bp), with motif_5p / motif_valid
columns. Files that already have a size_bin column keep their bins (rows with
size_bin "background" are dropped); otherwise lengths are cut into the four
merged bins (<=600, 601-2000, 2001-5000, >5000 bp).

MDS = -sum(p log2 p) / log2(256) over the valid 5' end motifs of a cell
(Jiang et al. 2020). With few fragments MDS reads high by chance (5 random
4-mers rarely repeat), so check the n column of any hit.

Gates:
  --min-frags              a sample gives a value for a (promoter, bin) cell only if it
                           has at least this many fragments there (default 5)
  --min-samples-per-group  a cell is tested only if each group has at least this
                           many samples with a value (default 3)

Test per (promoter, size bin): two-sided Mann-Whitney U (HCC vs HBV_carrier), Cohen's d
(pooled SD) and Cliff's delta, BH-FDR (--fdr-scope, default one round over all tests).
Work is split by region_id over --threads processes; BH is applied once on the
combined table, so the result does not depend on the number of workers.
--cutoff-scan writes how many cells survive at other --min-frags values.

Outputs (prefix --out-prefix):
  <prefix>_per_sample_metrics_long.csv.gz   one row per sample x promoter x bin
  <prefix>_stats.csv.gz                     one row per tested cell
  <prefix>_cutoff_scan.csv                  only with --cutoff-scan

Usage:
python 06_individual_region_stats.py \
    --tagged-glob "promoter_tagged/*.csv.gz" --metadata ont_samples.csv \
    --min-frags 5 --min-samples-per-group 3 \
    --out-prefix promoter_mds/all -t 16
python 06_individual_region_stats.py \
    --tagged-glob "promoter_tagged/*.csv.gz" --metadata ont_samples.csv \
    --source direct --out-prefix promoter_mds/direct -t 16
"""

import argparse
import glob
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

MOTIF_METRICS = ["mds_motif"]
ID_COLS = ["gene_name", "region_chrom", "region_start", "region_end"]


def default_workers():
    return max(1, (multiprocessing.cpu_count() or 2) // 2)


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    # O(n1*n2) is fine here: n1,n2 are per-group SAMPLE counts (<=~20), not fragment counts.
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (n1 * n2)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(fdr, 0, 1)
    return out


def load_all_tagged(glob_pat):
    files = sorted(glob.glob(glob_pat))
    if not files:
        sys.exit(f"No files matched {glob_pat}")
    def sep_of(f):
        import gzip
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            first = fh.readline()
        return "\t" if first.count("\t") > first.count(",") else ","
    dfs = [pd.read_csv(f, sep=sep_of(f), low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)


def assign_bins(df, edges, labels):
    if "size_bin" in df.columns:      # files are already binned: keep their bins
        df = df[df["size_bin"].astype(str) != "background"]
        return df
    df = df.copy()
    df["size_bin"] = pd.cut(df["length"], bins=edges, labels=labels,
                             right=True, include_lowest=True)
    return df


def build_bin_ranges(edges, labels):
    """label -> (low, high, is_open_ended); the last bin is open-ended."""
    ranges = {}
    for i, lab in enumerate(labels):
        low, high = edges[i], edges[i + 1]
        is_open = (i == len(labels) - 1)
        ranges[lab] = (low, high, is_open)
    return ranges


def mds_of_motifs(motif_series):
    """Normalised Shannon entropy of the motif counts over 256 possible 4-mers."""
    m = motif_series.dropna()
    if len(m) == 0:
        return np.nan
    counts = m.value_counts()
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p))
    return H / np.log2(256)


def compute_metrics_chunk(df_chunk, bin_ranges, short_cutoff_bp, has_motifs):
    """Runs in a worker process: MDS per sample x region x size bin (size_bin already assigned)."""
    rows = []
    group_cols = ["sample", "region_id", "size_bin"] + [c for c in ID_COLS if c in df_chunk.columns]
    for keys, sub in df_chunk.groupby(group_cols, observed=True, dropna=False):
        sample, region_id, size_bin = keys[:3]
        extra = dict(zip(group_cols[3:], keys[3:]))
        vals, ns = {}, {}

        if has_motifs and "motif_5p" in sub.columns:
            if "motif_valid" in sub.columns:
                valid_motifs = sub.loc[sub["motif_valid"].astype(bool), "motif_5p"]
            else:
                valid_motifs = sub["motif_5p"]
            vals["mds_motif"] = mds_of_motifs(valid_motifs)
            ns["mds_motif"] = int(valid_motifs.notna().sum())

        for metric, value in vals.items():
            rows.append({
                "sample": sample, "region_id": region_id, **extra, "size_bin": size_bin,
                "metric": metric, "value": value, "n": ns[metric],
            })
    return pd.DataFrame(rows)


def chunk_by_region(df, n_chunks):
    ids = df["region_id"].unique()
    if n_chunks <= 1 or len(ids) <= 1:
        return [df]
    bucket = pd.Series(np.arange(len(ids)) % n_chunks, index=ids)
    df = df.copy()
    df["_chunk"] = df["region_id"].map(bucket)
    return [g.drop(columns="_chunk") for _, g in df.groupby("_chunk")]


def parallel_compute_metrics(df, bin_ranges, short_cutoff_bp, has_motifs, workers):
    chunks = chunk_by_region(df, workers)
    results = []
    if workers <= 1 or len(chunks) <= 1:
        for c in chunks:
            results.append(compute_metrics_chunk(c, bin_ranges, short_cutoff_bp, has_motifs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(compute_metrics_chunk, c, bin_ranges, short_cutoff_bp, has_motifs)
                    for c in chunks]
            for f in as_completed(futs):
                results.append(f.result())
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def survivorship_scan(per_sample_long, meta, thresholds, min_samples_per_group):
    merged = per_sample_long.merge(meta, on="sample", how="left")
    rows = []
    for thr in thresholds:
        gated = merged[merged["n"] >= thr]
        counts = (gated.groupby(["region_id", "size_bin", "metric", "group"], observed=True)["sample"]
                  .nunique().unstack(fill_value=0))
        for col in meta["group"].unique():
            if col not in counts.columns:
                counts[col] = 0
        ok = (counts >= min_samples_per_group).all(axis=1)
        rows.append({"min_frags": thr, "surviving_region_bin_metric_cells": int(ok.sum()),
                      "total_region_bin_metric_cells_seen": int(len(counts))})
    return pd.DataFrame(rows)


def run_tests_chunk(df_chunk, meta, min_frags, min_samples_per_group):
    merged = df_chunk.merge(meta, on="sample", how="left")
    gated = merged[merged["n"] >= min_frags]
    groups = sorted(meta["group"].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {groups}")
    g1, g2 = groups

    results = []
    group_cols = ["region_id", "size_bin", "metric"] + [c for c in ID_COLS if c in gated.columns]
    for keys, sub in gated.groupby(group_cols, observed=True, dropna=False):
        region_id, size_bin, metric = keys[:3]
        extra = dict(zip(group_cols[3:], keys[3:]))
        v1 = sub.loc[sub["group"] == g1, "value"].dropna().values
        v2 = sub.loc[sub["group"] == g2, "value"].dropna().values
        n1, n2 = len(v1), len(v2)
        if n1 < min_samples_per_group or n2 < min_samples_per_group:
            continue
        if np.std(v1) == 0 and np.std(v2) == 0 and np.mean(v1) == np.mean(v2):
            continue
        try:
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        except ValueError:
            continue
        pooled_sd = (np.sqrt(((n1 - 1) * np.var(v1, ddof=1) + (n2 - 1) * np.var(v2, ddof=1))
                              / max(n1 + n2 - 2, 1)) if (n1 > 1 and n2 > 1) else np.nan)
        cohens_d = (np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan
        delta = cliffs_delta(v1, v2)
        results.append({
            "region_id": region_id, **extra, "size_bin": size_bin, "metric": metric,
            f"n_{g1}": n1, f"n_{g2}": n2,
            f"mean_{g1}": np.mean(v1), f"mean_{g2}": np.mean(v2),
            "mwu_p": p, "cohens_d": cohens_d, "cliffs_delta": delta,
        })
    return pd.DataFrame(results)


def parallel_run_tests(per_sample_long, meta, min_frags, min_samples_per_group, workers, fdr_scope="all-combined"):
    chunks = chunk_by_region(per_sample_long, workers)
    results = []
    if workers <= 1 or len(chunks) <= 1:
        for c in chunks:
            results.append(run_tests_chunk(c, meta, min_frags, min_samples_per_group))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_tests_chunk, c, meta, min_frags, min_samples_per_group)
                    for c in chunks]
            for f in as_completed(futs):
                results.append(f.result())
    res = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not res.empty:
        # BH applied once on the full combined table, whatever the number of chunks.
        # '_family' pools motif_freq_XXXX metrics (none here, so per-family = per-metric;
        # kept for the same options as 05_extra_fragmentomic_metrics.py).
        res["_family"] = res["metric"].where(~res["metric"].str.startswith("motif_freq_"), "motif_freq")
        if fdr_scope == "all-combined":
            res["fdr_global"] = bh_fdr(res["mwu_p"].values)
        elif fdr_scope == "per-metric":  # every exact metric name gets its own BH pool
            res["fdr_global"] = np.nan
            for metric, idx in res.groupby("metric").groups.items():
                res.loc[idx, "fdr_global"] = bh_fdr(res.loc[idx, "mwu_p"].values)
        else:  # per-family (same as per-metric here)
            res["fdr_global"] = np.nan
            for fam, idx in res.groupby("_family").groups.items():
                res.loc[idx, "fdr_global"] = bh_fdr(res.loc[idx, "mwu_p"].values)
        res = res.drop(columns="_family")
        res = res.sort_values("mwu_p")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagged-glob", required=True, help="glob (quoted) of promoter-tagged fragment tables with motif columns (csv/tsv, .gz ok)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("--region-name", default="promoter", help="name of the region type (label only)")
    ap.add_argument("--min-frags", type=int, default=5,
                     help="minimum fragments a sample must have in a (region, bin) cell to give a value (default 5)")
    ap.add_argument("--min-samples-per-group", type=int, default=3,
                     help="minimum samples per group passing --min-frags for a cell to be tested (default 3)")
    ap.add_argument("--short-cutoff-bp", type=int, default=150, help="not used by MDS (kept for the shared function signature)")
    ap.add_argument("--bin-edges", default="0,600,2000,5000,50000", help="size-bin edges, used only if the files have no size_bin column")
    ap.add_argument("--bin-labels", default="<=600,601-2000,2001-5000,>5000", help="labels of those bins")
    ap.add_argument("--no-motifs", action="store_true",
                     help="skip MDS (the script then stops, since MDS is its only metric)")
    ap.add_argument("--out-prefix", required=True, help="output path prefix, e.g. promoter_mds/all (folder is created)")
    ap.add_argument("--cutoff-scan", default="",
                     help="comma-separated --min-frags values for the survival table, e.g. 3,5,7,10,15,20,30,50")
    ap.add_argument("--fdr-scope", choices=["per-family", "per-metric", "all-combined"], default="all-combined",
                     help="all-combined (default): one BH round over every test. per-metric / "
                          "per-family: one round per metric (the same here, as MDS is the only metric).")
    ap.add_argument("-t", "--threads", "--workers", dest="workers", type=int, default=None,
                     help="Parallel worker processes (default: max(1, cpu_count//2))")
    ap.add_argument("--source", default="", help="keep only rows with this value in the 'source' column, e.g. direct (default: all rows)")
    args = ap.parse_args()

    workers = args.workers if args.workers is not None else default_workers()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    print(f"[info] using {workers} worker process(es)", file=sys.stderr)

    edges = [float(x) for x in args.bin_edges.split(",")]
    labels = args.bin_labels.split(",")
    if len(labels) != len(edges) - 1:
        sys.exit("--bin-labels must have exactly one fewer entry than --bin-edges")
    bin_ranges = build_bin_ranges(edges, labels)

    meta = pd.read_csv(args.metadata, sep=None, engine="python")
    if not {"sample", "group"} <= set(meta.columns):
        sys.exit("--metadata must have columns: sample, group")

    print("[info] loading tagged fragments...", file=sys.stderr)
    df = load_all_tagged(args.tagged_glob)
    df = assign_bins(df, edges, labels)
    has_motifs = ("motif_5p" in df.columns) and not args.no_motifs
    if args.source and "source" in df.columns:
        df = df[df["source"].astype(str) == args.source]
    if not has_motifs:
        sys.exit("[error] no motif_5p column: MDS is the only metric of this script.")

    print(f"[info] computing per-sample metrics ({workers} workers)...", file=sys.stderr)
    per_sample = parallel_compute_metrics(df, bin_ranges, args.short_cutoff_bp, has_motifs, workers)
    per_sample.to_csv(f"{args.out_prefix}_per_sample_metrics_long.csv.gz", index=False, compression="gzip")
    print(f"[info] {len(per_sample)} (sample,region,bin,metric) rows -> "
          f"{args.out_prefix}_per_sample_metrics_long.csv.gz", file=sys.stderr)

    if args.cutoff_scan:
        thresholds = [int(x) for x in args.cutoff_scan.split(",")]
        print(f"[info] running cutoff scan over {thresholds}...", file=sys.stderr)
        scan = survivorship_scan(per_sample, meta, thresholds, args.min_samples_per_group)
        scan.to_csv(f"{args.out_prefix}_cutoff_scan.csv", index=False)
        print(scan.to_string(index=False), file=sys.stderr)

    print(f"[info] testing at min-frags={args.min_frags}, "
          f"min-samples-per-group={args.min_samples_per_group} ({workers} workers)...", file=sys.stderr)
    res = parallel_run_tests(per_sample, meta, args.min_frags, args.min_samples_per_group, workers,
                              fdr_scope=args.fdr_scope)
    if res.empty:
        print("[warn] no (region, bin) cell passed the gates; see the cutoff scan output.", file=sys.stderr)
        return

    res.to_csv(f"{args.out_prefix}_stats.csv.gz", index=False, compression="gzip")
    print(f"[info] {len(res)} tests run -> {args.out_prefix}_stats.csv.gz", file=sys.stderr)

    print("\n[info] FDR<0.05 hits by metric:", file=sys.stderr)
    summary = (res.assign(sig=res["fdr_global"] < 0.05)
                  .groupby("metric")
                  .agg(n_tests=("mwu_p", "size"), n_sig=("sig", "sum"),
                       median_n_used=("n_" + sorted(meta["group"].unique())[0], "median"))
                  .reset_index())
    print(summary.to_string(index=False), file=sys.stderr)
    print("[info] MDS at a floor of 5 fragments reads high by chance; check n on any hit.", file=sys.stderr)


if __name__ == "__main__":
    main()