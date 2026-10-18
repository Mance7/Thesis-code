#!/usr/bin/env python3
"""
End-motif tests at the level of single promoters (Methods 2.8, Results 3.3.3-3.4.1).

Each of the 19,662 promoter windows is tested on its own fragments.
The fragment files are read with the delimiter detected automatically.

What it computes:
  * motif frequency of each of the 256 5' end motifs per (promoter, size_bin, sample)
    = count / valid-motif fragments in that cell; tested HCC vs HBV per (promoter, bin,
    motif) with a Mann-Whitney U (normal approximation, tie and continuity corrected)
  * MDS per (promoter, size_bin, sample), tested the same way (BH within MDS)
Gate: a sample enters a cell only with >= --min-frags valid-motif fragments there; a test
needs >= --min-per-group samples in each group.

Motifs that are hit at more promoters than chance (family F3)
  For every (size_bin, motif, direction) the script counts promoters with p < --alpha in
  that direction, then repeats the WHOLE count under every relabelling of the samples
  (or --n-perm random ones; same gate, same tests). Output per (bin, motif, direction):
  observed hits, null mean / 99th percentile, empirical p = (#perm >= obs + 1)/(n_perm + 1),
  and BH over all bin x motif x direction rows. This permutation test keeps the
  correlation between promoters and motifs, so its BH q-values control the FDR of
  "which motifs differ at more promoters than chance". Motifs with q < 0.10 form family
  F3 for the region-pooled cross-check.

Outputs (--out-prefix P):
  P_motif_tests_nominal.csv.gz   every (promoter, bin, motif) test with p < --write-p
  P_motif_hitcount_F3.csv        per (bin, motif, direction): hits, null, emp p, BH q, F3 flag
  P_motif_concordance_input.csv  per (bin, motif): tests, median and mean HCC-HBV difference,
                                 share of promoters with HCC > HBV (for the cross-check)
  P_mds_tests.csv.gz             MDS test per (promoter, bin) with BH q
  P_cells_per_bin.csv            per bin: cells, and cells where every sample passes --min-frags
  P_run_info.txt

Usage:
  python 06_individual_promoter_motif_stats.py \\
      --fragments "individual_promoters/transcript_binned_final/*.fragments(promoters)_binned_gene_name.csv.gz" \\
      --metadata metadata/ont_samples.csv --out-prefix promoter_motif/promoter_indiv_all --threads 16
  # direct fragments only:
  ... --keep source=direct --out-prefix promoter_motif/promoter_indiv_direct
"""
import argparse, glob, itertools, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats

MOTIFS = ["".join(m) for m in itertools.product("ACGT", repeat=4)]
M_INDEX = {m: i for i, m in enumerate(MOTIFS)}


def sniff_sep(path):
    with pd.io.common.get_handle(path, "r", compression="infer") as h:
        first = h.handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def truthy(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def _load_one(args):
    f, a, keep = args
    need = {a.sample_col, a.region_col, a.bin_col, a.motif_col, "motif_valid", a.gene_col} | set(keep)
    sep = sniff_sep(f)
    parts = []
    for ch in pd.read_csv(f, sep=sep, usecols=lambda c: c in need, chunksize=2_000_000, low_memory=False):
        if a.sample_col not in ch.columns:
            # one file per sample: sample name = file name up to the first dot
            ch[a.sample_col] = os.path.basename(f).split(".")[0]
        for c in [a.region_col, a.bin_col, a.motif_col]:
            if c not in ch.columns:
                raise SystemExit(f"{f}: column '{c}' missing; columns are {list(ch.columns)}. Use the --*-col flags.")
        for col, vals in keep.items():
            ch = ch[ch[col].astype(str).str.strip().isin(vals)]
        if "motif_valid" in ch.columns:
            ch = ch[truthy(ch["motif_valid"])]
        mi = ch[a.motif_col].astype(str).str.upper().map(M_INDEX)
        ch = ch.assign(mi=mi).dropna(subset=["mi"])
        parts.append(pd.DataFrame({
            "sample": ch[a.sample_col].astype(str).str.strip(),
            "region": ch[a.region_col].astype(str),
            "gene": ch[a.gene_col].astype(str) if a.gene_col in ch.columns else "",
            "bin": ch[a.bin_col].astype(str),
            "mi": ch["mi"].astype(np.int16)}))
    return f, (pd.concat(parts, ignore_index=True) if parts else None)


def load(files, a, keep):
    out = []
    with ProcessPoolExecutor(max_workers=max(1, min(a.threads, len(files)))) as ex:
        for f, d in ex.map(_load_one, [(f, a, keep) for f in files]):
            if d is not None:
                out.append(d)
            print(f"  read {os.path.basename(f)}", file=sys.stderr)
    return pd.concat(out, ignore_index=True)


def mds(counts):
    """counts: (..., 256) -> normalised Shannon entropy over 256 motifs."""
    n = counts.sum(-1, keepdims=True)
    p = np.where(n > 0, counts / np.maximum(n, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0).sum(-1)
    return h / 8.0


def work(args):
    """One chunk of (region, bin) cells. Returns hit counts for all label sets + observed rows."""
    (cell_keys, C, labels_all, n_bins, bin_of_cell, a) = args
    # C: (n_cells, n_samples, 256) counts; labels_all: (B+1, n_samples) 1 = case, row 0 = observed
    Bp1 = labels_all.shape[0]
    hits = np.zeros((Bp1, n_bins, 256, 2), dtype=np.int32)      # dir 0 = HCC higher, 1 = HCC lower
    zcrit = stats.norm.isf(a.alpha / 2)
    rows, mds_rows, conc = [], [], []
    for ci in range(C.shape[0]):
        cnt = C[ci]
        n_s = cnt.sum(1)
        inc = n_s >= a.min_frags
        if a.require_all and not inc.all():
            continue
        if inc.sum() < 2 * a.min_per_group:
            continue
        L = labels_all[:, inc]                                   # (B+1, k)
        n1 = L.sum(1); n2 = L.shape[1] - n1
        okp = (n1 >= a.min_per_group) & (n2 >= a.min_per_group)
        if not okp[0]:
            continue
        F = cnt[inc] / n_s[inc, None]                            # (k, 256) frequencies
        k = F.shape[0]
        # columns with any variation
        var = F.max(0) > F.min(0)
        b = bin_of_cell[ci]
        # MDS test (observed labels only)
        m = mds(cnt[inc].astype(float))
        y = L[0].astype(bool)
        if m[y].std() + m[~y].std() > 0 or m[y].mean() != m[~y].mean():
            try:
                pm = stats.mannwhitneyu(m[y], m[~y], alternative="two-sided").pvalue
            except ValueError:
                pm = np.nan
            mds_rows.append((cell_keys[ci][0], cell_keys[ci][1], cell_keys[ci][2], int(y.sum()), int((~y).sum()),
                             float(m[y].mean()), float(m[~y].mean()), pm))
        if not var.any():
            continue
        Fv = F[:, var]
        R = stats.rankdata(Fv, axis=0)                           # (k, V)
        c = (Fv[:, None, :] == Fv[None, :, :]).sum(1)            # tie group size of each element
        T = (c ** 2).sum(0) - k                                  # sum over tie groups of (t^3 - t)
        R1 = L @ R                                               # (B+1, V) rank sums of the case group
        U = R1 - (n1 * (n1 + 1) / 2)[:, None]
        mu = (n1 * n2 / 2)[:, None]
        sig = np.sqrt((n1 * n2 / 12)[:, None] * ((k + 1) - T[None, :] / (k * (k - 1))))
        dev = U - mu
        sigok = sig > 0
        hit = (np.abs(dev) - 0.5 > zcrit * sig) & sigok & okp[:, None]
        vidx = np.where(var)[0]
        up = hit & (dev > 0); dn = hit & (dev < 0)
        hits[:, b, vidx, 0] += up
        hits[:, b, vidx, 1] += dn
        # observed per-test output
        z0 = np.where(sigok[0], (np.abs(dev[0]) - 0.5).clip(min=0) / np.where(sigok[0], sig[0], 1), 0)
        p0 = np.where(sigok[0], 2 * stats.norm.sf(z0), 1.0)
        diff = Fv[y].mean(0) - Fv[~y].mean(0)
        conc.append(np.c_[np.full(len(vidx), b), vidx, diff, p0])
        sel = p0 < a.write_p
        for j in np.where(sel)[0]:
            rows.append((cell_keys[ci][0], cell_keys[ci][1], cell_keys[ci][2], MOTIFS[vidx[j]], int(n1[0]), int(n2[0]),
                         float(Fv[y, j].mean()), float(Fv[~y, j].mean()), float(diff[j]), float(p0[j])))
    conc = np.vstack(conc) if conc else np.zeros((0, 4))
    return hits, rows, mds_rows, conc


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); n = len(pv)
        qq = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        t = np.empty(n); t[o] = np.minimum(qq, 1); q[ok] = t
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob of promoter-tagged fragment files with motifs (csv or tsv, .gz ok)")
    ap.add_argument("--metadata", required=True,
                    help="csv with columns sample, group")
    ap.add_argument("--out-prefix", required=True, help="output path prefix (folder/name); the folder is created")
    ap.add_argument("--sample-col", default="sample"); ap.add_argument("--region-col", default="region_id")
    ap.add_argument("--gene-col", default="gene_name"); ap.add_argument("--bin-col", default="size_bin")
    ap.add_argument("--motif-col", default="motif_5p")
    ap.add_argument("--keep", action="append", default=[], help="COL=V1,V2 row filter, e.g. source=direct")
    ap.add_argument("--min-frags", type=int, default=5, help="per sample-cell floor (valid-motif fragments)")
    ap.add_argument("--min-per-group", type=int, default=3, help="samples per group needed for a test")
    ap.add_argument("--downsample", action="store_true",
                    help="randomly downsample every sample to the fragment count of the smallest sample "
                         "(after --keep) before any test, removing the HCC/HBV depth difference")
    ap.add_argument("--allow-missing", dest="require_all", action="store_false",
                    help="also test cells where some samples are below --min-frags. Default: a cell is "
                         "used only if EVERY sample passes, so every relabelling has the same group sizes and the "
                         "same attainable p; otherwise low-depth HBV samples drop out under the observed labels "
                         "only and the hit-count null is biased upwards")
    ap.add_argument("--alpha", type=float, default=0.01, help="nominal p threshold that defines a 'hit'")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="random relabellings; ignored when the exact enumeration is used (see --max-exact)")
    ap.add_argument("--max-exact", type=int, default=20000,
                    help="enumerate ALL distinct labellings when there are at most this many "
                         "(8 HCC / 6 HBV = 3003): exact permutation p, smallest possible p = 1/3003")
    ap.add_argument("--write-p", type=float, default=0.05, help="write per-test rows with p below this")
    ap.add_argument("--f3-q", type=float, default=0.10, help="BH q below which a motif joins family F3")
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-t", "--threads", type=int, default=8, help="parallel workers")
    a = ap.parse_args()
    t0 = time.time()
    keep = {}
    for k in a.keep:
        col, vals = k.split("=", 1); keep[col] = [v.strip() for v in vals.split(",")]
    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)

    meta = pd.read_csv(a.metadata, sep=None, engine="python", encoding_errors="replace")
    meta.columns = [c.strip().lower() for c in meta.columns]
    grp = dict(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    df = load(files, a, keep)
    df = df[df["sample"].map(grp).isin([a.case, a.control])]
    if a.downsample:
        # equal depth: every sample keeps the same number of valid-motif fragments (the smallest sample's),
        # drawn at random with a fixed seed, so depth no longer differs between groups
        n_min = int(df["sample"].value_counts().min())
        rng_ds = np.random.default_rng(a.seed)
        keep_idx = np.concatenate([rng_ds.choice(ix, size=n_min, replace=False)
                                   for _, ix in sorted(df.groupby("sample").indices.items())])
        df = df.iloc[np.sort(keep_idx)].reset_index(drop=True)
        print(f"downsampled every sample to {n_min:,} valid-motif fragments (seed {a.seed})", file=sys.stderr)
    samples = sorted(df["sample"].unique())
    y = np.array([grp[s] == a.case for s in samples], int)
    print(f"{len(df):,} valid-motif fragments, {len(samples)} samples ({y.sum()} {a.case})", file=sys.stderr)

    bins = sorted(df["bin"].unique())
    bin_id = {b: i for i, b in enumerate(bins)}
    s_id = {s: i for i, s in enumerate(samples)}
    cells = df.groupby(["region", "bin"], sort=False).ngroup().to_numpy()
    df["cell"] = cells
    keys_df = df.drop_duplicates("cell").sort_values("cell")[["region", "gene", "bin"]]
    cell_keys = list(keys_df.itertuples(index=False, name=None))
    bin_of_cell = keys_df["bin"].map(bin_id).to_numpy()
    n_cells = len(cell_keys)
    si = df["sample"].map(s_id).to_numpy(); mi = df["mi"].to_numpy().astype(int)
    print(f"{n_cells:,} (promoter, bin) cells, bins {bins}", file=sys.stderr)
    per_cs = np.bincount(cells.astype(np.int64) * len(samples) + si.astype(np.int64),
                         minlength=n_cells * len(samples)).reshape(n_cells, len(samples))
    n_pass = (per_cs >= a.min_frags).sum(1)
    gate = pd.DataFrame({"bin": [bins[b] for b in bin_of_cell], "all": n_pass == len(samples)})
    gate_tab = gate.groupby("bin")["all"].agg(["sum", "size"]).rename(columns={"sum": "all_pass", "size": "cells"})
    gate_tab.to_csv(f"{a.out_prefix}_cells_per_bin.csv")
    print("cells with every sample >= --min-frags, per bin (also in _cells_per_bin.csv):\n" + gate_tab.to_string(),
          file=sys.stderr)

    n_lab = math.comb(len(y), int(y.sum()))
    if n_lab <= a.max_exact:
        # exact: every distinct labelling once, observed first; emp p = (#labellings >= obs) / n_lab
        perms = []
        for idx in itertools.combinations(range(len(y)), int(y.sum())):
            v = np.zeros(len(y), int); v[list(idx)] = 1
            if not np.array_equal(v, y):
                perms.append(v)
        labels = np.vstack([y] + perms).astype(float)
        a.n_perm = len(perms); mode = f"exact ({n_lab} labellings)"
    else:
        rng = np.random.default_rng(a.seed)
        labels = np.vstack([y] + [rng.permutation(y) for _ in range(a.n_perm)]).astype(float)
        mode = f"random ({a.n_perm} relabellings)"
    print(f"F3 null: {mode}; smallest attainable emp p = {1 / (a.n_perm + 1):.2e}", file=sys.stderr)

    # sparse -> dense per chunk
    order = np.argsort(cells, kind="stable")
    cells_s, si_s, mi_s = cells[order], si[order], mi[order]
    n_chunks = max(1, a.threads * 4)
    bounds = np.linspace(0, n_cells, n_chunks + 1).astype(int)

    def chunk_args():
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo:
                continue
            l = np.searchsorted(cells_s, lo); r = np.searchsorted(cells_s, hi)
            C = np.zeros((hi - lo, len(samples), 256), dtype=np.int32)
            np.add.at(C, (cells_s[l:r] - lo, si_s[l:r], mi_s[l:r]), 1)
            yield (cell_keys[lo:hi], C, labels, len(bins), bin_of_cell[lo:hi], a)

    hits = np.zeros((a.n_perm + 1, len(bins), 256, 2), dtype=np.int64)
    rows, mds_rows, conc = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        for i, (h, r, mr, cc) in enumerate(ex.map(work, chunk_args())):
            hits += h; rows += r; mds_rows += mr; conc.append(cc)
            print(f"  chunk {i + 1}/{n_chunks} done ({time.time() - t0:.0f}s)", file=sys.stderr)

    # per-test nominal rows
    T = pd.DataFrame(rows, columns=["region_id", "gene_name", "size_bin", "motif", f"n_{a.case}", f"n_{a.control}",
                                    f"freq_{a.case}", f"freq_{a.control}", "diff", "p"])
    T.sort_values("p").to_csv(f"{a.out_prefix}_motif_tests_nominal.csv.gz", index=False, compression="gzip")

    # F3 hit-count permutation test
    out = []
    for b, bl in enumerate(bins):
        for j, mname in enumerate(MOTIFS):
            for d, dname in enumerate(["HCC_higher", "HCC_lower"]):
                obs = int(hits[0, b, j, d]); null = hits[1:, b, j, d]
                out.append((bl, mname, dname, obs, float(null.mean()), float(np.percentile(null, 99)) if len(null) else np.nan,
                            (np.sum(null >= obs) + 1) / (len(null) + 1) if len(null) else np.nan))
    H = pd.DataFrame(out, columns=["size_bin", "motif", "direction", "hits_obs", "hits_null_mean", "hits_null_p99", "emp_p"])
    H = H[(H.hits_obs > 0) | (H.hits_null_mean > 0)].copy()
    H["q_bh"] = bh(H["emp_p"])
    H["F3"] = H["q_bh"] < a.f3_q
    k_need = math.ceil(len(H) / (a.n_perm + 1) / a.f3_q)
    print(f"[note] {len(H)} bin x motif x direction rows: at the smallest emp p, at least {k_need} rows "
          f"must sit at that floor together before any reaches BH q<{a.f3_q}", file=sys.stderr)
    H.sort_values(["q_bh", "emp_p"]).to_csv(f"{a.out_prefix}_motif_hitcount_F3.csv", index=False)

    # concordance input
    CC = np.vstack(conc) if conc else np.zeros((0, 4))
    if len(CC):
        cdf = pd.DataFrame(CC, columns=["b", "m", "diff", "p"])
        g = cdf.groupby(["b", "m"])
        cs = pd.DataFrame({"n_tests": g.size(), "median_diff": g["diff"].median(), "mean_diff": g["diff"].mean(),
                           "share_HCC_higher": g["diff"].apply(lambda v: float((v > 0).mean()))}).reset_index()
        cs["size_bin"] = cs["b"].astype(int).map(dict(enumerate(bins)))
        cs["motif"] = cs["m"].astype(int).map(dict(enumerate(MOTIFS)))
        cs[["size_bin", "motif", "n_tests", "median_diff", "mean_diff", "share_HCC_higher"]].to_csv(
            f"{a.out_prefix}_motif_concordance_input.csv", index=False)

    Mdf = pd.DataFrame(mds_rows, columns=["region_id", "gene_name", "size_bin", f"n_{a.case}", f"n_{a.control}",
                                          f"mds_{a.case}", f"mds_{a.control}", "p"])
    Mdf["q_bh"] = bh(Mdf["p"])
    Mdf.sort_values("p").to_csv(f"{a.out_prefix}_mds_tests.csv.gz", index=False, compression="gzip")

    with open(f"{a.out_prefix}_run_info.txt", "w") as fh:
        fh.write(f"files={len(files)} keep={keep} samples={len(samples)} cells={n_cells} bins={bins} "
                 f"min_frags={a.min_frags} min_per_group={a.min_per_group} alpha={a.alpha} n_perm={a.n_perm} "
                 f"seed={a.seed} f3_q={a.f3_q} null={mode} require_all={a.require_all} downsample={a.downsample}\n")
    n_f3 = int(H["F3"].sum())
    print(f"\nmotif tests written (p<{a.write_p}): {len(T):,}; MDS tests: {len(Mdf):,} "
          f"(BH<0.05: {int((Mdf.q_bh < 0.05).sum())})", file=sys.stderr)
    print(f"F3 motifs (hit-count permutation test, BH q<{a.f3_q}): {n_f3}", file=sys.stderr)
    print(H.sort_values("q_bh").head(20).to_string(index=False), file=sys.stderr)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
