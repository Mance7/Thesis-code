#!/usr/bin/env python3
"""
Fragment-length entropy corrected for fragment count (rarefaction), with group tests.

Why: the plug-in Shannon entropy of a K-category histogram estimated from N
fragments is biased low, and when N < K it cannot exceed log2(N)/log2(K)
after normalisation. The 1001-2000 bp bin has K = 1000 possible integer
lengths and long fragments are sparse, so a group with fewer long fragments
shows lower entropy by counting alone (Methods 2.6).

For every (sample, size_bin, region) cell this script reports:
  N               fragments in the cell
  H_plugin        normalised plug-in entropy (as in 01_length_entropy_calculation.py)
  H_mm            Miller-Madow bias-corrected entropy, normalised the same way
  H_rare          mean normalised entropy after rarefying every sample in that cell
                  to the same N (the smallest N among samples passing --min-frags),
                  averaged over --n-rare random subsamples without replacement
                  (the primary estimator of the thesis)
and, per cell, tests HCC vs HBV for each estimator (two-sided MWU + Cohen's d,
BH-FDR across cells per estimator), plus the diagnostics of count dependence:
  N_ratio         median N HCC / median N HBV
  p_N             MWU p for N between groups
  rho_HN_*        Spearman correlation of H_plugin with N (pooled, within HCC, within HBV)

Verdict column (per cell, for cells significant with H_plugin):
  "survives"       still FDR < alpha after rarefaction, same direction
  "attenuated"     same direction, nominal p < 0.05 but not FDR
  "count-driven"   loses significance after rarefaction

Input: one fragment table per sample (csv/tsv, optionally .gz) with a length column
and boolean region columns (is_promoter, is_gene_body, is_tss, is_exon,
is_cpg_island). "genome_wide" = all fragments. Alternatively --regions-bed restricts
to fragments overlapping a BED (e.g. a gene-set promoter pool), with the region name
from --regions-name.
Output (--out): entropy_per_sample_cell.csv, entropy_rarefaction_tests.csv,
entropy_rarefaction_summary.txt

Usage:
  python 03_entropy_rarefaction_check.py \\
      --fragments 'fragments/*.fragments_with_motifs.csv.gz' \\
      --metadata metadata/ont_samples.csv --out rarefaction_out/ \\
      [--bins 11bin|4bin|"0,90,125,..."] [--min-frags 100] [--n-rare 100] [--seed 1] [--keep source=direct]

  # gene-set pool, e.g. the 28-promoter pathway pool, merged bins:
  python 03_entropy_rarefaction_check.py --fragments '...' --metadata ont_samples.csv \\
      --regions-bed pathway_promoters.bed --regions-name Pathway --bins 4bin --out pool_out/
"""
import argparse, glob, os, re, sys, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

BIN_SCHEMES = {
    # (label, low_inclusive, high_inclusive); high None = open-ended
    "11bin": [("<90bp", 1, 89), ("90-125bp", 90, 125), ("126-135bp", 126, 135),
              ("136-150bp", 136, 150), ("151-255bp", 151, 255), ("256-420bp", 256, 420),
              ("421-650bp", 421, 650), ("651-1000bp", 651, 1000), ("1001-2000bp", 1001, 2000),
              ("2001-5000bp", 2001, 5000), (">5000bp", 5001, None)],
    "4bin": [("<=600bp", 1, 600), ("601-2000bp", 601, 2000), ("2001-5000bp", 2001, 5000),
             (">5000bp", 5001, None)],
}
OPEN_CAP = 50000  # same cap as 01_length_entropy_calculation.py


def read_table(path, usecols=None):
    """Read csv/tsv with the encoding fallback used across the pipeline."""
    with (pd.io.common.get_handle(path, "r", compression="infer")) as h:
        first = h.handle.readline()
    sep = "\t" if "\t" in first else ","
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, usecols=usecols, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def parse_bins(spec):
    if spec in BIN_SCHEMES:
        return BIN_SCHEMES[spec]
    edges = [int(x) for x in spec.split(",")]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append((f"{lo + 1}-{hi}bp", lo + 1, hi))
    out.append((f">{edges[-1]}bp", edges[-1] + 1, None))
    return out


def load_bed(path):
    bed = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"], comment="#")
    merged = {}
    for c, g in bed.sort_values(["c", "s"]).groupby("c"):
        s, e = g["s"].to_numpy(), g["e"].to_numpy()
        ms, me = [s[0]], [e[0]]
        for a, b in zip(s[1:], e[1:]):
            if a <= me[-1]:
                me[-1] = max(me[-1], b)
            else:
                ms.append(a); me.append(b)
        merged[c] = (np.array(ms), np.array(me))
    return merged


def overlaps(chrom, start, end, merged, mode):
    """Vectorised overlap test of fragments against merged, sorted intervals."""
    hit = np.zeros(len(chrom), dtype=bool)
    for c in np.unique(chrom):
        if c not in merged:
            continue
        ms, me = merged[c]
        idx = np.where(chrom == c)[0]
        if mode == "midpoint":
            q = (start[idx] + end[idx]) // 2
            j = np.searchsorted(ms, q, side="right") - 1
            hit[idx] = (j >= 0) & (me[np.clip(j, 0, None)] > q)
        else:
            j = np.searchsorted(ms, end[idx], side="left") - 1
            hit[idx] = (j >= 0) & (me[np.clip(j, 0, None)] > start[idx])
    return hit


def n_categories(lo, hi, observed_max):
    if hi is None:
        hi = min(max(observed_max, lo), OPEN_CAP)
    return hi - lo + 1


def norm_entropy(counts, K):
    counts = counts[counts > 0]
    N = counts.sum()
    if N == 0 or K <= 1:
        return np.nan, np.nan
    p = counts / N
    H = -(p * np.log2(p)).sum()
    H_mm = H + (len(counts) - 1) / (2 * N * np.log(2))  # Miller-Madow, in bits
    return H / np.log2(K), H_mm / np.log2(K)


def rarefied_entropy(counts, n_target, K, n_rep, rng):
    """Mean normalised entropy of n_rep subsamples of size n_target drawn without
    replacement from a length histogram (counts per length)."""
    counts = counts[counts > 0].astype(np.int64)
    if counts.sum() < n_target or n_target <= 0:
        return np.nan
    subs = rng.multivariate_hypergeometric(counts, n_target, size=n_rep)
    return float(np.mean([norm_entropy(row, K)[0] for row in subs]))


def add_hist(acc, x):
    """Accumulate a bincount of x into acc, growing acc if needed."""
    b = np.bincount(x)
    if len(b) > len(acc):
        acc = np.concatenate([acc, np.zeros(len(b) - len(acc), np.int64)])
    acc[:len(b)] += b
    return acc


def load_sample(f, sample, a, region_cols, merged):
    """Stream one fragment file; return {region: length histogram} and the max length."""
    cols = [a.length_col] + region_cols
    if merged is not None:
        cols += [a.chrom_col, a.start_col, a.end_col]
    kcol = kvals = None
    if a.keep:
        kcol, kvals = a.keep.split("=", 1)
        kvals = [v.strip() for v in kvals.split(",")]
        cols.append(kcol)
    with pd.io.common.get_handle(f, "r", compression="infer") as h:
        first = h.handle.readline()
    sep = "\t" if "\t" in first else ","
    hists, n, mx = {}, 0, 0
    for df in pd.read_csv(f, sep=sep, usecols=list(dict.fromkeys(cols)), chunksize=a.chunksize, encoding_errors="replace"):
        if kcol:
            df = df[df[kcol].astype(str).str.strip().isin(kvals)]
        L = pd.to_numeric(df[a.length_col], errors="coerce").to_numpy()
        ok = np.isfinite(L) & (L >= 1)
        df, L = df[ok], L[ok].astype(np.int64)
        if not len(L):
            continue
        n += len(L); mx = max(mx, int(L.max()))
        masks = {}
        if merged is not None:
            masks[a.regions_name] = overlaps(df[a.chrom_col].astype(str).to_numpy(), df[a.start_col].to_numpy(int),
                                             df[a.end_col].to_numpy(int), merged, a.overlap_mode)
        else:
            masks["genome_wide"] = np.ones(len(L), bool)
            for rc in region_cols:
                masks[rc] = df[rc].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).to_numpy()
        for reg, m in masks.items():
            hists[reg] = add_hist(hists.get(reg, np.zeros(0, np.int64)), L[m])
    return sample, hists, n, mx


def cell_counts(hist, lo, hi):
    if hist is None or len(hist) <= lo:
        return np.zeros(0, np.int64)
    return hist[lo:(hi + 1 if hi is not None else len(hist))]


def entropy_cell(task):
    """One (bin, region) cell: estimators for every sample. Seeded per cell, so results
    do not depend on the number of threads."""
    lab, lo, hi, K, reg, per_sample, min_frags, n_rare, seed = task
    rng = np.random.default_rng(seed)
    Ns = {s: int(c.sum()) for s, c in per_sample.items()}
    passing = [s for s in per_sample if Ns[s] >= min_frags]
    n_target = min(Ns[s] for s in passing) if passing else 0
    out = []
    for s, c in per_sample.items():
        hp, hmm = norm_entropy(c, K) if Ns[s] else (np.nan, np.nan)
        hr = rarefied_entropy(c, n_target, K, n_rare, rng) if (s in passing and n_target > 0) else np.nan
        out.append(dict(sample=s, size_bin=lab, region=reg, K=K, N=Ns[s], passes_min_frags=s in passing,
                        N_rarefied_to=n_target, H_plugin=hp, H_mm=hmm, H_rare=hr))
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m); tmp[order] = np.minimum(ranked, 1)
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob of per-sample region-tagged fragment tables (csv.gz); quote it")
    ap.add_argument("--metadata", required=True, help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="regex on file basename; group 1 = sample id")
    ap.add_argument("--length-col", default="length")
    ap.add_argument("--region-cols", default="is_promoter,is_gene_body,is_tss,is_exon,is_cpg_island")
    ap.add_argument("--regions-bed", help="BED file; restrict to fragments overlapping it instead of using the region columns")
    ap.add_argument("--regions-name", default="pool")
    ap.add_argument("--overlap-mode", choices=["any", "midpoint"], default="any")
    ap.add_argument("--chrom-col", default="chrom"); ap.add_argument("--start-col", default="start"); ap.add_argument("--end-col", default="end")
    ap.add_argument("--keep", help="only fragments where COLUMN is one of VALUES, e.g. 'source=direct' "
                                   "(to test whether concatemer-recovered fragments drive a result)")
    ap.add_argument("--bins", default="11bin", help="11bin, 4bin, or comma-separated edges")
    ap.add_argument("--min-frags", type=int, default=100, help="fragments a sample needs in a cell")
    ap.add_argument("--n-rare", type=int, default=100, help="random subsamples averaged for H_rare")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel workers (samples are read in parallel, then cells)")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time per sample (memory control)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bins = parse_bins(a.bins)
    meta = read_table(a.metadata)
    meta.columns = [c.strip().lower() for c in meta.columns]
    group = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    region_cols = [] if a.regions_bed else [c for c in a.region_cols.split(",") if c]
    merged = load_bed(a.regions_bed) if a.regions_bed else None

    # pass 1: per-sample length histograms per region, samples read in parallel
    jobs = []
    for f in files:
        m = re.match(a.sample_regex, os.path.basename(f))
        sample = m.group(1) if m else os.path.basename(f)
        if sample not in group:
            print(f"  skip {sample}: not in metadata", file=sys.stderr); continue
        jobs.append((f, sample))
    hists, observed_max = {}, 0
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        futs = [ex.submit(load_sample, f, s, a, region_cols, merged) for f, s in jobs]
        for fu in as_completed(futs):
            sample, h, n, mx = fu.result()
            hists[sample] = h; observed_max = max(observed_max, mx)
            print(f"  loaded {sample} ({group[sample]}): {n:,} fragments")
    samples_sorted = sorted(hists)

    # pass 2: estimators, cells in parallel
    regions = sorted({r for h in hists.values() for r in h})
    tasks, ci = [], 0
    for lab, lo, hi in bins:
        K = n_categories(lo, hi, observed_max)
        for reg in regions:
            per_sample = {s: cell_counts(hists[s].get(reg), lo, hi) for s in samples_sorted}
            tasks.append((lab, lo, hi, K, reg, per_sample, a.min_frags, a.n_rare, a.seed * 100003 + ci)); ci += 1
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        for out in ex.map(entropy_cell, tasks):
            rows.extend(out)
    for r in rows:
        r["group"] = group[r["sample"]]
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(a.out, "entropy_per_sample_cell.csv"), index=False)

    # pass 3: tests
    res = []
    for (lab, reg), g in cells.groupby(["size_bin", "region"], sort=False):
        g = g[g.passes_min_frags]
        c, h = g[g.group == a.case], g[g.group == a.control]
        rec = dict(size_bin=lab, region=reg, n_case=len(c), n_control=len(h),
                   N_med_case=c.N.median(), N_med_control=h.N.median(),
                   N_ratio=c.N.median() / h.N.median() if len(h) else np.nan,
                   N_rarefied_to=g.N_rarefied_to.iloc[0] if len(g) else np.nan)
        if len(c) >= 3 and len(h) >= 3:
            rec["p_N"] = stats.mannwhitneyu(c.N, h.N, alternative="two-sided").pvalue
            rec["rho_HN_all"] = stats.spearmanr(g.N, g.H_plugin).statistic
            rec["rho_HN_case"] = stats.spearmanr(c.N, c.H_plugin).statistic
            rec["rho_HN_control"] = stats.spearmanr(h.N, h.H_plugin).statistic
            for est in ("H_plugin", "H_mm", "H_rare"):
                cc, hh = c[est].dropna(), h[est].dropna()
                if len(cc) >= 3 and len(hh) >= 3:
                    rec[f"p_{est}"] = stats.mannwhitneyu(cc, hh, alternative="two-sided").pvalue
                    rec[f"d_{est}"] = cohens_d(cc, hh)
        res.append(rec)
    res = pd.DataFrame(res)
    for est in ("H_plugin", "H_mm", "H_rare"):
        if f"p_{est}" in res:
            res[f"padj_{est}"] = bh(res[f"p_{est}"])

    def verdict(r):
        if not (r.get("padj_H_plugin", 1) < a.alpha):
            return ""
        same = np.sign(r.get("d_H_rare", np.nan)) == np.sign(r.get("d_H_plugin", np.nan))
        if r.get("padj_H_rare", 1) < a.alpha and same:
            return "survives"
        if r.get("p_H_rare", 1) < 0.05 and same:
            return "attenuated"
        return "count-driven"
    res["verdict"] = res.apply(verdict, axis=1)
    res.to_csv(os.path.join(a.out, "entropy_rarefaction_tests.csv"), index=False)

    sig = res[res.verdict != ""]
    cols = ["size_bin", "region", "N_med_case", "N_med_control", "p_N", "rho_HN_all",
            "d_H_plugin", "padj_H_plugin", "d_H_mm", "padj_H_mm", "d_H_rare", "padj_H_rare", "verdict"]
    cols = [c for c in cols if c in sig]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
        print("\nCells significant with the original plug-in estimator:")
        print(sig[cols].to_string(index=False) if len(sig) else "  none")
    with open(os.path.join(a.out, "entropy_rarefaction_summary.txt"), "w") as fh:
        fh.write(sig[cols].to_string(index=False) if len(sig) else "none")
        fh.write("\n\nverdict counts: " + str(sig.verdict.value_counts().to_dict()) + "\n")
    print(f"\nwrote {a.out}/entropy_per_sample_cell.csv, entropy_rarefaction_tests.csv, entropy_rarefaction_summary.txt")


if __name__ == "__main__":
    main()
