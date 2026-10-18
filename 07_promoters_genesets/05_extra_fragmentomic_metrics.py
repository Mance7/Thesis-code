#!/usr/bin/env python3
"""
05_extra_fragmentomic_metrics.py
--------------------------------
Individual-promoter fragmentomic metrics and group tests (Methods 2.8, level 2:
each of the 19,662 GENCODE v49 gene-level promoters tested on its own
fragments). For every promoter and sample the script computes the metrics
below, then compares HCC with HBV carriers per cell (two-sided Mann-Whitney U)
and corrects with Benjamini-Hochberg and Storey's q-value. Exploratory analysis.

Input fragments come from the promoter tagging step: one row per fragment and
promoter it overlaps (any overlap of at least 1 bp, the region-flag rule of
Methods 2.5), so a fragment can count for more than one promoter here. This is
not the midpoint rule of the gene-set scripts (07-09).

METRICS
-------
fsr               short-fragment ratio per promoter, all size bins pooled
                  (size_bin = "pooled"):
                  fsr = n(length <= --short-max) / n(length <= --long-max)
                  (defaults 246 and 600 bp; 246 bp is the valley between the
                  mono- and di-nucleosome peaks)
long_frac         share of all of a promoter's fragments that are longer than
                  --long-frac-min (default 1000 bp); pooled over bins
motif_freq_XXXX   frequency of each 5' end motif among valid-motif fragments,
                  per promoter x size bin (all observed motifs, or --motifs)
pref_end          preferred-end score per promoter x size bin: 1 - normalised
                  Shannon entropy of fragment start positions at 1 bp
                  resolution. With a TSS column (--promoter-tss-col) the window
                  is TSS +/- --pref-end-window (metric pref_end_tss); without
                  it, the whole promoter window
wps               windowed protection score at the TSS (or the promoter-window
                  midpoint if there is no TSS column), from fragments of
                  --wps-min-len to --wps-max-len bp, pooled over bins:
                  WPS = (fragments spanning the position)
                        - (fragments with an end within +/- --wps-window bp)
wps_per_bin       optional (--wps-per-bin): the same score from each size bin's
                  own fragments
wps_ndr_contrast  optional (add to --metrics; needs a TSS column): mean WPS over
                  the +1 nucleosome window minus mean WPS over the
                  nucleosome-depleted region just upstream of the TSS

A sample contributes a value to a cell only if it has at least --min-frags
fragments there (--min-frags-bin for the per-bin metrics; default 5 for both).
A cell is tested if each group has at least 2 samples with a value.

INPUT COLUMNS
-------------
One row per fragment x promoter:
    sample, promoter_id, promoter_chrom, promoter_start, promoter_end,
    frag_start, frag_end, size_bin, motif_5p, motif_valid (+ tss, source)
Other column names can be given with the --*-col options. With one file per
sample and no sample column, the sample name is taken from the file name
(text before the first '.').

OUTPUT
------
--out (a .csv.gz path or a folder; a folder gets promoter_extra_metrics_stats.csv.gz):
  one row per promoter x size bin x metric: region_chrom, region_id,
  region_start, region_end, region_mid, size_bin, metric, mwu_p, fdr_global,
  qvalue_storey, storey_pi0, n_case, n_control
<out>_diagnostic.csv: per metric, cells tested out of the possible maximum,
  smallest p and the number of BH / Storey hits.

USAGE
-----
python 05_extra_fragmentomic_metrics.py \
    --fragments "promoter_tagged/*.csv.gz" --metadata ont_samples.csv \
    --out extra_metrics_all/ -t 32
python 05_extra_fragmentomic_metrics.py \
    --fragments "promoter_tagged/*.csv.gz" --metadata ont_samples.csv \
    --source direct --out extra_metrics_direct/ -t 32
"""

import argparse
import glob as globmod
import gzip
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (same as statsmodels fdr_bh)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    fdr = ranked * n / (np.arange(n) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    out = np.empty(n)
    out[order] = fdr
    return out


def storey_pi0(pvals, lambdas=None):
    """Proportion of true null tests, pi0 (Storey & Tibshirani 2003):
    pi0(lambda) = #{p > lambda} / (m * (1 - lambda)) over a grid of lambda.
    We take the mean of this curve over the grid instead of a spline
    extrapolated to the largest lambda, because Mann-Whitney p-values from
    small groups are very discrete and the single-point estimate becomes
    unstable. Storey's method assumes continuous, uniform null p-values, which
    small-sample Mann-Whitney only approximates, so pi0 is a rough diagnostic."""
    p = np.asarray(pvals, dtype=float)
    if lambdas is None:
        lambdas = np.arange(0.05, 0.91, 0.05)
    pi0_hat = np.array([np.mean(p > l) / (1 - l) for l in lambdas])
    pi0 = float(np.nanmean(pi0_hat))
    if not np.isfinite(pi0):
        pi0 = 1.0  # fall back to pi0 = 1 (plain BH) rather than NaN q-values
    return float(np.clip(pi0, 0.0, 1.0))


def storey_qvalues(pvals):
    """Storey's q-value: same monotone running-min construction as BH, but
    scaled by the estimated pi0 instead of assuming pi0=1."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m < 20:
        pi0 = 1.0  # fewer than 20 tests: pi0 fixed at 1, i.e. plain BH
    else:
        pi0 = storey_pi0(p)
    if not np.isfinite(pi0):
        pi0 = 1.0
    order = np.argsort(p)
    ranked = p[order]
    q = pi0 * ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out, pi0


def detect_sep(path):
    """Guess the delimiter (tab or comma) from the header line."""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode) as fh:
        first_line = fh.readline()
    tab_count, comma_count = first_line.count("\t"), first_line.count(",")
    return "\t" if tab_count > comma_count else ","


def _read_one(args):
    p, file_sep, sample_col, source = args
    d = pd.read_csv(p, sep=file_sep)
    if source != "all":
        if "source" not in d.columns:
            raise SystemExit(f"--source {source} needs a 'source' column; {p} has none")
        d = d[d["source"].astype(str).str.strip() == source]
    if sample_col not in d.columns:
        # single-sample-per-file layout: derive sample from filename
        d[sample_col] = os.path.basename(p).split(".")[0]
    return d


def load_fragments(path, cols, sep=None, threads=1, source="all"):
    paths = globmod.glob(path) if any(ch in path for ch in "*?[]") else [path]
    if not paths:
        raise SystemExit(f"No files matched --fragments {path}")
    file_sep = sep or detect_sep(paths[0])
    print(f"[info] reading --fragments with delimiter {repr(file_sep)} "
          f"({'auto-detected' if sep is None else 'from --sep'})")
    jobs = [(p, file_sep, cols["sample"], source) for p in paths]
    if threads > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(threads, len(jobs))) as ex:
            frames = list(ex.map(_read_one, jobs))
    else:
        frames = [_read_one(j) for j in jobs]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing expected columns {missing} in fragments input. "
            f"Columns found: {list(df.columns)}. "
            f"Name the columns with the --*-col options, or set --sep if the "
            f"delimiter was guessed wrong."
        )
    return df


_MWU = {}


def _mwu_cells(i):
    """Worker: one promoter chunk of the long table -> list of per-cell result dicts."""
    chunk = _MWU["chunks"][i]
    group_col, case_label, control_label = _MWU["group_col"], _MWU["case"], _MWU["control"]
    key_cols = _MWU["key_cols"]
    results = []
    for keys, cell in chunk.groupby(key_cols, dropna=False):
        case_vals = cell.loc[cell[group_col] == case_label, "value"].dropna()
        ctrl_vals = cell.loc[cell[group_col] == control_label, "value"].dropna()
        if len(case_vals) < 2 or len(ctrl_vals) < 2:
            continue  # too few samples in a group: not tested
        try:
            _, p = mannwhitneyu(case_vals, ctrl_vals, alternative="two-sided")
        except ValueError:
            continue  # e.g. all-identical values in both groups
        row = dict(zip(key_cols, keys))
        row["mwu_p"] = p
        row["n_case"] = len(case_vals)
        row["n_control"] = len(ctrl_vals)
        results.append(row)
    return results


def _split_by_promoter(df, n_chunks):
    """Chunks that never split a promoter (all metrics of a promoter stay together)."""
    if n_chunks <= 1:
        return [df]
    codes = pd.factorize(df["promoter_id"])[0] % n_chunks
    return [g for _, g in df.groupby(codes, sort=True)]


def run_mwu_fdr(long_df, metadata, group_col, case_label, control_label,
                 fdr_scope, threads=1):
    """long_df: one row per (promoter_id, size_bin, metric, sample, value).
    Returns one row per (promoter_id, size_bin, metric) with mwu_p, fdr_global,
    plus region coordinate columns carried through."""
    long_df = long_df.merge(metadata, on="sample", how="left")
    if long_df[group_col].isna().any():
        missing_samples = long_df.loc[long_df[group_col].isna(), "sample"].unique()
        raise SystemExit(f"Samples not found in --metadata: {list(missing_samples)}")

    key_cols = ["promoter_id", "region_id", "promoter_chrom", "promoter_start",
                "promoter_end", "size_bin", "metric"]
    chunks = _split_by_promoter(long_df, max(1, threads) * 4)
    _MWU.update(chunks=chunks, group_col=group_col, case=case_label, control=control_label,
                key_cols=key_cols)
    if threads > 1 and len(chunks) > 1:
        # fork: the chunks are inherited by the workers, not pickled
        with ProcessPoolExecutor(max_workers=threads, mp_context=mp.get_context("fork")) as ex:
            parts = list(ex.map(_mwu_cells, range(len(chunks))))
    else:
        parts = [_mwu_cells(i) for i in range(len(chunks))]
    results = [r for part in parts for r in part]

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # same row order as a single global groupby (sorted cell keys), whatever the chunking
    out = out.sort_values(key_cols, na_position="last", kind="mergesort").reset_index(drop=True)
    # '_family' puts all motif_freq_XXXX metrics into one pool (as in the motif
    # screen: one BH round over all motif cells), while fsr, pref_end and wps
    # each stay their own family. Used only with --fdr-scope per-family.
    out["_family"] = out["metric"].where(~out["metric"].str.startswith("motif_freq_"), "motif_freq")
    out["fdr_global"] = np.nan
    out["qvalue_storey"] = np.nan
    out["storey_pi0"] = np.nan
    if fdr_scope == "all-combined":
        out["fdr_global"] = bh_fdr(out["mwu_p"])
        q, pi0 = storey_qvalues(out["mwu_p"].values)
        out["qvalue_storey"], out["storey_pi0"] = q, pi0
    elif fdr_scope == "per-metric":  # every exact metric name gets its own BH pool
        for metric, idx in out.groupby("metric").groups.items():
            out.loc[idx, "fdr_global"] = bh_fdr(out.loc[idx, "mwu_p"])
            q, pi0 = storey_qvalues(out.loc[idx, "mwu_p"].values)
            out.loc[idx, "qvalue_storey"], out.loc[idx, "storey_pi0"] = q, pi0
    else:  # per-family: all motif_freq_* pooled together, fsr/pref_end/wps separate
        for fam, idx in out.groupby("_family").groups.items():
            out.loc[idx, "fdr_global"] = bh_fdr(out.loc[idx, "mwu_p"])
            q, pi0 = storey_qvalues(out.loc[idx, "mwu_p"].values)
            out.loc[idx, "qvalue_storey"], out.loc[idx, "storey_pi0"] = q, pi0
    out = out.drop(columns="_family")

    out = out.rename(columns={
        "promoter_chrom": "region_chrom",
        "promoter_start": "region_start",
        "promoter_end": "region_end",
    })
    out["region_mid"] = ((out["region_start"] + out["region_end"]) // 2).astype(int)
    return out[["region_chrom", "region_id", "region_start", "region_end", "region_mid",
                "size_bin", "metric", "mwu_p", "fdr_global", "qvalue_storey", "storey_pi0",
                "n_case", "n_control"]]


def compute_fsr(frags, cols, short_max, long_max, min_frags):
    d = frags[frags[cols["frag_len"]] <= long_max].copy()
    grp = d.groupby([cols["sample"], "promoter_id", "promoter_chrom",
                      "promoter_start", "promoter_end"])
    n_total = grp.size().rename("n_total")
    n_short = d[d[cols["frag_len"]] <= short_max].groupby(
        [cols["sample"], "promoter_id", "promoter_chrom",
         "promoter_start", "promoter_end"]).size().rename("n_short")
    agg = pd.concat([n_total, n_short], axis=1).fillna(0).reset_index()
    agg = agg[agg["n_total"] >= min_frags]
    agg["value"] = agg["n_short"] / agg["n_total"]
    agg["size_bin"] = "pooled"
    agg["metric"] = "fsr"
    agg = agg.rename(columns={cols["sample"]: "sample"})
    return agg[["sample", "promoter_id", "promoter_chrom", "promoter_start",
                "promoter_end", "size_bin", "metric", "value"]]


def compute_long_frac(frags, cols, long_frac_min, min_frags):
    """Share of a promoter's fragments (all lengths in the denominator, unlike
    fsr) that are longer than long_frac_min. Pooled over size bins, because
    the ratio needs the whole length range in one calculation."""
    d = frags.copy()
    grp = d.groupby([cols["sample"], "promoter_id", "promoter_chrom",
                      "promoter_start", "promoter_end"])
    n_total = grp.size().rename("n_total")
    n_long = d[d[cols["frag_len"]] > long_frac_min].groupby(
        [cols["sample"], "promoter_id", "promoter_chrom",
         "promoter_start", "promoter_end"]).size().rename("n_long")
    agg = pd.concat([n_total, n_long], axis=1).fillna(0).reset_index()
    agg = agg[agg["n_total"] >= min_frags]
    agg["value"] = agg["n_long"] / agg["n_total"]
    agg["size_bin"] = "pooled"
    agg["metric"] = "long_frac"
    agg = agg.rename(columns={cols["sample"]: "sample"})
    return agg[["sample", "promoter_id", "promoter_chrom", "promoter_start",
                "promoter_end", "size_bin", "metric", "value"]]


def compute_motif_freq(frags, cols, motifs, min_frags):
    """One frequency per (sample, promoter_id, size_bin, motif). If motifs is
    None, every 5' end motif seen among valid-motif fragments is used. Motifs
    absent from a cell are filled in as zeros (cross-join), so the tests see
    true zeros instead of missing values."""
    d = frags.copy()
    valid = d[d[cols["motif_valid"]].astype(bool)].copy()
    group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                  "promoter_start", "promoter_end", cols["size_bin"]]

    n_valid = valid.groupby(group_cols).size().rename("n_valid").reset_index()
    n_valid = n_valid[n_valid["n_valid"] >= min_frags]
    if n_valid.empty:
        return pd.DataFrame()

    if motifs is None:
        motifs = sorted(valid[cols["motif_5p"]].dropna().unique().tolist())
        print(f"[info] motif_freq: discovered {len(motifs)} distinct observed motifs "
              f"(computing all of them, not a curated subset)")

    counts = valid.groupby(group_cols + [cols["motif_5p"]]).size().rename("n_motif").reset_index()

    motif_df = pd.DataFrame({cols["motif_5p"]: motifs})
    full = n_valid.merge(motif_df, how="cross")
    full = full.merge(counts, on=group_cols + [cols["motif_5p"]], how="left")
    full["n_motif"] = full["n_motif"].fillna(0)
    full["value"] = full["n_motif"] / full["n_valid"]
    full["metric"] = "motif_freq_" + full[cols["motif_5p"]].astype(str)
    full = full.rename(columns={cols["sample"]: "sample", cols["size_bin"]: "size_bin"})
    return full[["sample", "promoter_id", "promoter_chrom", "promoter_start",
                 "promoter_end", "size_bin", "metric", "value"]]


def _norm_entropy(counts, n_possible):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0 or n_possible <= 1:
        return np.nan
    p = counts / counts.sum()
    h = -(p * np.log2(p)).sum()
    return h / np.log2(n_possible)


def compute_pref_end(frags, cols, min_frags, window_half_width=None):
    """Preferred-end score: 1 - normalised entropy of fragment start positions.

    window_half_width=None: the whole promoter window (promoter_start to
    promoter_end, 2,500 bp for the -2000/+500 promoters).
    window_half_width=<int> (needs the TSS column): the window
    [tss - window_half_width, tss + window_half_width] (metric pref_end_tss)."""
    use_tss = window_half_width is not None
    if use_tss and cols.get("tss") not in frags.columns:
        raise SystemExit("pref_end with a TSS window needs a TSS column "
                         "(--promoter-tss-col).")

    rows = []
    if use_tss:
        group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                      "promoter_start", "promoter_end", cols["tss"], cols["size_bin"]]
    else:
        group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                      "promoter_start", "promoter_end", cols["size_bin"]]

    for keys, cell in frags.groupby(group_cols):
        n = len(cell)
        if n < min_frags:
            continue
        if use_tss:
            # keys = (sample, promoter_id, chrom, start, end, tss, size_bin)
            tss = keys[5]
            win_start, win_end = tss - window_half_width, tss + window_half_width
            promoter_chrom, promoter_start, promoter_end, size_bin = keys[2], win_start, win_end, keys[6]
        else:
            win_start, win_end = keys[3], keys[4]
            promoter_chrom, promoter_start, promoter_end, size_bin = keys[2], keys[3], keys[4], keys[5]

        window = int(win_end - win_start)
        starts = cell[cols["frag_start"]].values
        # keep only fragment starts inside the window (clipping them to the
        # edges would pile them into the edge positions and fake a preferred end)
        starts = starts[(starts >= win_start) & (starts < win_end)]
        if len(starts) < min_frags:
            continue
        # 1 bp resolution histogram of start positions within the window
        offsets = (starts - win_start).astype(np.int64)
        counts = np.bincount(offsets, minlength=max(window, 1))
        norm_h = _norm_entropy(counts, max(window, 2))
        if np.isnan(norm_h):
            continue
        rows.append({
            "sample": keys[0], "promoter_id": keys[1], "promoter_chrom": promoter_chrom,
            "promoter_start": promoter_start, "promoter_end": promoter_end, "size_bin": size_bin,
            "metric": "pref_end" if not use_tss else "pref_end_tss", "value": 1.0 - norm_h,
        })
    return pd.DataFrame(rows)


def compute_wps(frags, cols, wps_min_len, wps_max_len, wps_window, min_frags, per_bin=False):
    """Windowed protection score at one position per promoter.

    per_bin=False (standard): fragments of wps_min_len to wps_max_len bp,
    pooled over all size bins (metric wps).
    per_bin=True: each size bin's own fragments, length window ignored
    (metric wps_per_bin; not the usual WPS definition).

    Position: the TSS if the TSS column is present, otherwise the midpoint
    of the promoter window.
    """
    use_tss = cols.get("tss") in frags.columns if cols.get("tss") else False
    tss_col = [cols["tss"]] if use_tss else []
    if not use_tss:
        print("[warn] wps/wps_per_bin: no TSS column found, using the promoter-window "
              "midpoint as the position (see --promoter-tss-col).")

    def pos_for(keys, base_idx):
        if use_tss:
            return keys[base_idx + 1]
        p_start, p_end = keys[base_idx - 1], keys[base_idx]
        return (p_start + p_end) // 2

    rows = []
    if per_bin:
        group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                      "promoter_start", "promoter_end"] + tss_col + [cols["size_bin"]]
        for keys, cell in frags.groupby(group_cols):
            if len(cell) < min_frags:
                continue
            pos = pos_for(keys, 4)
            size_bin = keys[-1]
            starts = cell[cols["frag_start"]].values
            ends = cell[cols["frag_end"]].values
            spanning = int(((starts < pos) & (ends > pos)).sum())
            endpoint_near = int((((starts >= pos - wps_window) & (starts <= pos + wps_window)) |
                                 ((ends >= pos - wps_window) & (ends <= pos + wps_window))).sum())
            rows.append({
                "sample": keys[0], "promoter_id": keys[1], "promoter_chrom": keys[2],
                "promoter_start": keys[3], "promoter_end": keys[4], "size_bin": size_bin,
                "metric": "wps_per_bin", "value": float(spanning - endpoint_near),
            })
        return pd.DataFrame(rows)

    d = frags[(frags[cols["frag_len"]] >= wps_min_len) &
              (frags[cols["frag_len"]] <= wps_max_len)].copy()
    group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                  "promoter_start", "promoter_end"] + tss_col
    for keys, cell in d.groupby(group_cols):
        if len(cell) < min_frags:
            continue
        pos = pos_for(keys, 4)
        starts = cell[cols["frag_start"]].values
        ends = cell[cols["frag_end"]].values
        spanning = int(((starts < pos) & (ends > pos)).sum())
        endpoint_near = int((((starts >= pos - wps_window) & (starts <= pos + wps_window)) |
                             ((ends >= pos - wps_window) & (ends <= pos + wps_window))).sum())
        rows.append({
            "sample": keys[0], "promoter_id": keys[1], "promoter_chrom": keys[2],
            "promoter_start": keys[3], "promoter_end": keys[4], "size_bin": "pooled",
            "metric": "wps", "value": float(spanning - endpoint_near),
        })
    return pd.DataFrame(rows)


def compute_wps_ndr_contrast(frags, cols, wps_min_len, wps_max_len, wps_window,
                              ndr_start, ndr_end, nuc_start, nuc_end, scan_step, min_frags):
    """Needs a TSS column. WPS is computed every scan_step bp across the
    nucleosome-depleted region (ndr_start..ndr_end relative to the TSS, e.g.
    -200..0) and the +1 nucleosome (nuc_start..nuc_end, e.g. 0..200), and
    summarised as one value per (sample, promoter):
        contrast = mean(WPS in nuc window) - mean(WPS in NDR window)
    Positive = protected downstream and open upstream, the pattern of an
    active promoter; near zero = no phasing seen."""
    if cols.get("tss") not in frags.columns:
        raise SystemExit("wps_ndr_contrast needs a TSS column (--promoter-tss-col).")
    d = frags[(frags[cols["frag_len"]] >= wps_min_len) &
              (frags[cols["frag_len"]] <= wps_max_len)].copy()
    group_cols = [cols["sample"], "promoter_id", "promoter_chrom",
                  "promoter_start", "promoter_end", cols["tss"]]
    ndr_offsets = np.arange(ndr_start, ndr_end, scan_step)
    nuc_offsets = np.arange(nuc_start, nuc_end, scan_step)

    rows = []
    for keys, cell in d.groupby(group_cols):
        if len(cell) < min_frags:
            continue
        tss = keys[-1]
        starts = cell[cols["frag_start"]].values
        ends = cell[cols["frag_end"]].values

        def wps_at(pos):
            spanning = ((starts < pos) & (ends > pos)).sum()
            near = (((starts >= pos - wps_window) & (starts <= pos + wps_window)) |
                    ((ends >= pos - wps_window) & (ends <= pos + wps_window))).sum()
            return spanning - near

        ndr_vals = [wps_at(tss + off) for off in ndr_offsets]
        nuc_vals = [wps_at(tss + off) for off in nuc_offsets]
        if not ndr_vals or not nuc_vals:
            continue
        contrast = float(np.mean(nuc_vals) - np.mean(ndr_vals))
        rows.append({
            "sample": keys[0], "promoter_id": keys[1], "promoter_chrom": keys[2],
            "promoter_start": keys[3], "promoter_end": keys[4], "size_bin": "pooled",
            "metric": "wps_ndr_contrast", "value": contrast,
        })
    return pd.DataFrame(rows)


_CALC = {}


def _compute_chunk(i):
    """Worker: all requested metrics for one chunk of promoters (every metric groups by promoter,
    so splitting by promoter gives the same result as one pass)."""
    g = _CALC
    frags, cols, a = g["chunks"][i], g["cols"], g["args"]
    out = []
    if "fsr" in a.metrics:
        out.append(compute_fsr(frags, cols, a.short_max, a.long_max, a.min_frags))
    if "long_frac" in a.metrics:
        out.append(compute_long_frac(frags, cols, a.long_frac_min, a.min_frags))
    if "motif_freq" in a.metrics:
        out.append(compute_motif_freq(frags, cols, g["motifs"], g["min_frags_bin"]))
    if "pref_end" in a.metrics:
        out.append(compute_pref_end(frags, cols, g["min_frags_bin"],
                                    window_half_width=a.pref_end_window if g["tss_available"] else None))
    if "wps" in a.metrics:
        out.append(compute_wps(frags, cols, a.wps_min_len, a.wps_max_len, a.wps_window,
                               a.min_frags, per_bin=False))
        if a.wps_per_bin:
            out.append(compute_wps(frags, cols, a.wps_min_len, a.wps_max_len, a.wps_window,
                                   g["min_frags_bin"], per_bin=True))
    if "wps_ndr_contrast" in a.metrics:
        out.append(compute_wps_ndr_contrast(
            frags, cols, a.wps_min_len, a.wps_max_len, a.wps_window,
            a.ndr_start, a.ndr_end, a.nuc_start, a.nuc_end, a.wps_scan_step, a.min_frags))
    out = [t for t in out if not t.empty]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True,
                     help="path or glob (quoted) of promoter-tagged fragment tables (csv/tsv, .gz ok)")
    ap.add_argument("--source", default="all",
                     help="'all' (default) or the value of the 'source' column to keep, e.g. 'direct'")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group (csv)")
    ap.add_argument("--sep", default=None,
                     help="delimiter of --fragments and --metadata (',' or '\\t'); "
                          "default: guessed from the header line")
    ap.add_argument("--out", required=True, help="output stats file (*_stats.csv.gz) or output folder")
    ap.add_argument("--min-frags", type=int, default=5,
                     help="minimum fragments per sample and promoter for the pooled metrics "
                          "(fsr, long_frac, wps, wps_ndr_contrast); default 5")
    ap.add_argument("--min-frags-bin", type=int, default=5,
                     help="minimum fragments per sample, promoter and size bin for the per-bin "
                          "metrics (motif_freq_*, pref_end, wps_per_bin); default 5")
    ap.add_argument("--short-max", type=int, default=246, help="fsr numerator: fragments up to this length (bp)")
    ap.add_argument("--long-max", type=int, default=600, help="fsr denominator: fragments up to this length (bp)")
    ap.add_argument("--long-frac-min", type=int, default=1000,
                     help="long_frac = share of all of a promoter's fragments longer than this (bp)")
    ap.add_argument("--motifs", nargs="+", default=None,
                     help="only these end motifs (e.g. CCAG GGAG); default: every motif seen in the data")
    ap.add_argument("--promoter-level", choices=["gene", "transcript"], default=None,
                     help="optional label written as a 'promoter_level' column and added to the "
                          "output file name, so gene- and transcript-level runs do not overwrite each other")
    ap.add_argument("--wps-min-len", type=int, default=100, help="WPS: shortest fragment used (bp)")
    ap.add_argument("--wps-max-len", type=int, default=280, help="WPS: longest fragment used (bp)")
    ap.add_argument("--wps-window", type=int, default=60, help="WPS: fragment ends within +/- this many bp count against protection")
    ap.add_argument("--wps-per-bin", action="store_true",
                     help="also compute WPS per size bin from that bin's own fragments (metric wps_per_bin)")
    ap.add_argument("--promoter-tss-col", default="tss",
                     help="column with the TSS coordinate. If present in --fragments, pref_end uses a "
                          "TSS-centred window and wps is taken at the TSS; needed for wps_ndr_contrast. "
                          "If absent, pref_end uses the whole promoter window and wps its midpoint.")
    ap.add_argument("--pref-end-window", type=int, default=200,
                     help="half-width (bp) of the TSS-centred pref_end window (only with a TSS column)")
    ap.add_argument("--ndr-start", type=int, default=-200,
                     help="wps_ndr_contrast: start (bp from the TSS) of the nucleosome-depleted region")
    ap.add_argument("--ndr-end", type=int, default=0, help="wps_ndr_contrast: end of the nucleosome-depleted region")
    ap.add_argument("--nuc-start", type=int, default=0,
                     help="wps_ndr_contrast: start (bp from the TSS) of the +1 nucleosome window")
    ap.add_argument("--nuc-end", type=int, default=200, help="wps_ndr_contrast: end of the +1 nucleosome window")
    ap.add_argument("--wps-scan-step", type=int, default=20,
                     help="wps_ndr_contrast: step (bp) between WPS positions")
    ap.add_argument("--metrics", nargs="+",
                     default=["fsr", "long_frac", "motif_freq", "pref_end", "wps"],
                     help="metric families to compute; wps_ndr_contrast (needs a TSS column) is off by default")
    ap.add_argument("--group-col", default="group", help="group column of --metadata")
    ap.add_argument("--case-label", default="HCC")
    ap.add_argument("--control-label", default="HBV_carrier")
    ap.add_argument("-t", "--threads", "--workers", dest="threads", type=int, default=1,
                     help="worker processes (default 1). Files are read in parallel, then metrics and "
                          "tests run in parallel over promoter chunks; the output is the same for any value, "
                          "because rows are sorted by cell and BH/Storey run once on the full table.")
    ap.add_argument("--fdr-scope", choices=["per-family", "per-metric", "all-combined"], default="per-metric",
                     help="per-metric (default): one BH round per metric name. per-family: all "
                          "motif_freq_* in one round, fsr/pref_end/wps each separate. all-combined: one "
                          "round over every test of the run.")
    # column names of the fragment tables
    ap.add_argument("--sample-col", default="sample")
    ap.add_argument("--promoter-id-col", default="promoter_id")
    ap.add_argument("--promoter-chrom-col", default="promoter_chrom")
    ap.add_argument("--promoter-start-col", default="promoter_start")
    ap.add_argument("--promoter-end-col", default="promoter_end")
    ap.add_argument("--frag-start-col", default="frag_start")
    ap.add_argument("--frag-end-col", default="frag_end")
    ap.add_argument("--frag-len-col", default=None,
                     help="fragment length column (default: frag_end - frag_start)")
    ap.add_argument("--size-bin-col", default="size_bin")
    ap.add_argument("--motif-5p-col", default="motif_5p")
    ap.add_argument("--motif-valid-col", default="motif_valid")
    args = ap.parse_args()

    if args.sep == "\\t":  # the shell passes a literal backslash-t
        args.sep = "\t"

    # check the output path before the long computation
    if os.path.isdir(args.out) or args.out.endswith(("/", os.sep)):
        os.makedirs(args.out, exist_ok=True)
        args.out = os.path.join(args.out, "promoter_extra_metrics_stats.csv.gz")
        print(f"[info] --out is a directory; writing {args.out}")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise SystemExit(f"Cannot write to {out_dir}")

    cols = {
        "sample": args.sample_col,
        "frag_start": args.frag_start_col,
        "frag_end": args.frag_end_col,
        "size_bin": args.size_bin_col,
        "motif_5p": args.motif_5p_col,
        "motif_valid": args.motif_valid_col,
        "tss": args.promoter_tss_col,
    }
    required_cols = {
        "sample": args.sample_col,
        "promoter_id": args.promoter_id_col,
        "promoter_chrom": args.promoter_chrom_col,
        "promoter_start": args.promoter_start_col,
        "promoter_end": args.promoter_end_col,
        "frag_start": args.frag_start_col,
        "frag_end": args.frag_end_col,
        "size_bin": args.size_bin_col,
        "motif_5p": args.motif_5p_col,
        "motif_valid": args.motif_valid_col,
    }

    frags = load_fragments(args.fragments, required_cols, sep=args.sep, threads=args.threads, source=args.source)
    frags = frags.rename(columns={
        args.promoter_id_col: "promoter_id",
        args.promoter_chrom_col: "promoter_chrom",
        args.promoter_start_col: "promoter_start",
        args.promoter_end_col: "promoter_end",
    })
    cols["frag_len"] = args.frag_len_col or "_frag_len"
    if args.frag_len_col is None:
        frags["_frag_len"] = frags[args.frag_end_col] - frags[args.frag_start_col]

    metadata_sep = args.sep or detect_sep(args.metadata)
    metadata = pd.read_csv(args.metadata, sep=metadata_sep)
    if args.sample_col not in metadata.columns:
        raise SystemExit(f"--metadata must have a '{args.sample_col}' column")
    metadata = metadata.rename(columns={args.sample_col: "sample"})

    tss_available = bool(cols.get("tss")) and cols["tss"] in frags.columns
    if args.promoter_tss_col and not tss_available:
        print(f"[warn] --promoter-tss-col '{args.promoter_tss_col}' not found in --fragments -- "
              f"pref_end/wps will use their non-TSS fallback behavior.")

    min_frags_bin = args.min_frags_bin if args.min_frags_bin is not None else args.min_frags
    if args.min_frags_bin is None:
        print(f"[info] --min-frags-bin not set: using --min-frags ({args.min_frags}) for "
              f"the per-bin metrics too.")

    if "motif_freq" in args.metrics and args.motifs is None:
        # discover motifs once over ALL fragments, so every chunk uses the same motif list
        motifs = sorted(frags.loc[frags[cols["motif_valid"]].astype(bool), cols["motif_5p"]]
                        .dropna().unique().tolist())
        print(f"[info] motif_freq: discovered {len(motifs)} distinct observed motifs "
              f"(all of them are tested)")
    else:
        motifs = args.motifs
    print(f"[info] computing {', '.join(args.metrics)} on {args.threads} worker process(es) ...")
    n_chunks = max(1, args.threads) * 4 if args.threads > 1 else 1
    chunks = _split_by_promoter(frags, n_chunks)
    _CALC.update(chunks=chunks, cols=cols, args=args, motifs=motifs,
                 min_frags_bin=min_frags_bin, tss_available=tss_available)
    if args.threads > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=args.threads, mp_context=mp.get_context("fork")) as ex:
            long_tables = list(ex.map(_compute_chunk, range(len(chunks))))
    else:
        long_tables = [_compute_chunk(i) for i in range(len(chunks))]
    long_tables = [t for t in long_tables if not t.empty]
    if not long_tables:
        raise SystemExit("No metric produced any rows -- check --min-frags and input schema.")
    long_df = pd.concat(long_tables, ignore_index=True)
    long_df["region_id"] = long_df["promoter_id"]
    print(f"[info] {len(long_df)} (sample, promoter, size_bin, metric) rows before stats")

    stats = run_mwu_fdr(long_df, metadata, args.group_col, args.case_label,
                         args.control_label, args.fdr_scope, threads=args.threads)
    if stats.empty:
        raise SystemExit("No (promoter, size_bin, metric) cell had >=2 samples per group -- nothing to test.")

    out_path = args.out
    if args.promoter_level:
        stats["promoter_level"] = args.promoter_level
        base, ext = out_path.split(".", 1) if "." in os.path.basename(out_path) else (out_path, "")
        out_path = f"{base}_{args.promoter_level}." + ext if ext else f"{out_path}_{args.promoter_level}"

    stats.to_csv(out_path, index=False, compression="infer")

    # --- diagnostic: cells tested vs. the possible maximum, per metric -----
    # Separates "dropped for lack of fragments/samples" from "tested but not
    # significant after FDR" (small min_p, few hits, most cells tested), which
    # points at multiple-testing dilution rather than depth.
    n_promoters_total = frags["promoter_id"].nunique()
    n_bins_total = frags[args.size_bin_col].nunique()

    def theoretical_max(metric_name):
        # fsr, long_frac and wps are pooled over bins (one cell per promoter);
        # pref_end, wps_per_bin and motif_freq_* have one cell per promoter x bin
        if metric_name in ("fsr", "wps", "long_frac", "wps_ndr_contrast"):
            return n_promoters_total
        return n_promoters_total * n_bins_total

    diag_rows = []
    print()
    for metric, sub in stats.groupby("metric"):
        n_sig_bh = int((sub["fdr_global"] < 0.05).sum())
        n_sig_storey = int((sub["qvalue_storey"] < 0.05).sum())
        pi0 = sub["storey_pi0"].iloc[0] if "storey_pi0" in sub.columns else float("nan")
        theo_max = theoretical_max(metric)
        n_tested = len(sub)
        pct_tested = (n_tested / theo_max * 100) if theo_max else float("nan")
        diag_rows.append({
            "metric": metric, "n_promoters_total": n_promoters_total,
            "n_bins_total": n_bins_total, "theoretical_max_cells": theo_max,
            "n_cells_tested": n_tested, "pct_of_theoretical_tested": pct_tested,
            "min_p": sub["mwu_p"].min(), "n_fdr_sig_bh": n_sig_bh,
            "n_fdr_sig_storey": n_sig_storey, "storey_pi0": pi0,
        })
        print(f"[info] {metric}: {n_tested}/{theo_max} cells tested ({pct_tested:.1f}%), "
              f"min p={sub['mwu_p'].min():.2e} -- BH: {n_sig_bh} sig, "
              f"Storey: {n_sig_storey} sig (pi0={pi0:.3f})")

    # one summary line for all motif_freq_* metrics together
    motif_rows = [r for r in diag_rows if r["metric"].startswith("motif_freq_")]
    if motif_rows:
        fam_tested = sum(r["n_cells_tested"] for r in motif_rows)
        fam_theo = sum(r["theoretical_max_cells"] for r in motif_rows)
        fam_min_p = min(r["min_p"] for r in motif_rows)
        fam_sig_bh = sum(r["n_fdr_sig_bh"] for r in motif_rows)
        fam_sig_storey = sum(r["n_fdr_sig_storey"] for r in motif_rows)
        print(f"[info] motif_freq (all {len(motif_rows)} motifs combined): "
              f"{fam_tested}/{fam_theo} cells tested ({fam_tested/fam_theo*100:.1f}%), "
              f"BH: {fam_sig_bh} sig, Storey: {fam_sig_storey} sig, min p={fam_min_p:.2e}")

    diag_df = pd.DataFrame(diag_rows)

    def strip_known_ext(p):
        for ext in (".csv.gz", ".csv"):
            if p.endswith(ext):
                return p[: -len(ext)]
        return p.rsplit(".", 1)[0] if "." in os.path.basename(p) else p

    diag_path = strip_known_ext(out_path) + "_diagnostic.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"\n[info] wrote {out_path}")
    print(f"[info] wrote {diag_path}")


if __name__ == "__main__":
    main()