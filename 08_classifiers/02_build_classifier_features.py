#!/usr/bin/env python3
"""
02_build_classifier_features.py
-------------------------------
Builds the pre-specified, label-free feature tables of the H6 classifiers
(Methods 2.10), one wide csv per feature set (rows = samples, first column =
sample), ready for 03_nested_cv_classifier.py --features. Nothing here looks
at the group labels.

  F_size.csv            share of fragments in each of the 11 size bins (genome-wide)
  F_entropy_plugin.csv  normalised plug-in length entropy in each of the 11 bins
  F_entropy_rare.csv    the same, rarefied to the smallest per-bin N across samples
                        (mean of --n-rare subsamples); the primary estimator (Methods 2.6)
  F_motif.csv           CLR of the 256 5' end-motif counts in each of the 4 merged bins
                        (1,024 columns; the classifier compresses them with --pca 5)
  F_motif_lit.csv       CLR of the literature motifs (--lit-motifs) + MDS, per merged bin
  F_combined_<est>.csv  F_size + F_entropy_<est> + F_motif_lit (est = --entropy-estimator)
  covariates.csv        n_fragments and pct_recovered (share of fragments whose source is
                        not 'direct'), for the covariate baseline and the depth adjustment
  build_info.txt        the settings of the run

CLR_j = log(c_j + 0.5) - mean_k log(c_k + 0.5) (Methods 2.7). Entropy uses the same
definitions as the entropy analysis (05_entropy_size): K = number of integer lengths
in the bin, the open >5000 bp bin capped at 50,000 bp, at least --min-frags (100)
fragments per bin.

--keep source=direct builds every set from direct (non-recovered) fragments only.
--downsample (equal-depth models) first draws every sample's length histogram and motif
table down, without replacement, to the smallest sample's total, so every feature comes
from the same number of fragments; --downsample-to N uses a fixed N instead (the PacBio
depth, for ONT models locked at the depth they are applied to).
Files are read in chunks, so 30 M-row samples fit in memory.

Usage:
  python 02_build_classifier_features.py \
      --fragments "filtered_mid/*.fragments_with_motifs.csv.gz" --out feats/all/ -t 8
  python 02_build_classifier_features.py \
      --fragments "filtered_mid/*.fragments_with_motifs.csv.gz" --out feats/direct/ \
      --keep source=direct -t 8
  python 02_build_classifier_features.py \
      --fragments "filtered_mid/*.fragments_with_motifs.csv.gz" --out feats_equal_depth/all/ \
      --downsample -t 20
"""
import argparse, glob, itertools, os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# size bins and entropy: the same definitions as the entropy analysis (entropy_rarefaction_check.py)
BIN_SCHEMES = {
    # (label, low_inclusive, high_inclusive); high None = open-ended
    "11bin": [("<90bp", 1, 89), ("90-125bp", 90, 125), ("126-135bp", 126, 135),
              ("136-150bp", 136, 150), ("151-255bp", 151, 255), ("256-420bp", 256, 420),
              ("421-650bp", 421, 650), ("651-1000bp", 651, 1000), ("1001-2000bp", 1001, 2000),
              ("2001-5000bp", 2001, 5000), (">5000bp", 5001, None)],
    "4bin": [("<=600bp", 1, 600), ("601-2000bp", 601, 2000), ("2001-5000bp", 2001, 5000),
             (">5000bp", 5001, None)],
}
OPEN_CAP = 50000  # longest length counted; the open >5000 bp bin uses K up to this cap


def norm_entropy(counts, K):
    """Normalised plug-in and Miller-Madow entropy of a count vector over K categories."""
    counts = counts[counts > 0]
    N = counts.sum()
    if N == 0 or K <= 1:
        return np.nan, np.nan
    p = counts / N
    H = -(p * np.log2(p)).sum()
    H_mm = H + (len(counts) - 1) / (2 * N * np.log(2))  # Miller-Madow, in bits
    return H / np.log2(K), H_mm / np.log2(K)


MOTIFS = ["".join(m) for m in itertools.product("ACGT", repeat=4)]
M_INDEX = {m: i for i, m in enumerate(MOTIFS)}
B11, B4 = BIN_SCHEMES["11bin"], BIN_SCHEMES["4bin"]


def bin_index(lengths, scheme):
    out = np.full(len(lengths), -1)
    for i, (_, lo, hi) in enumerate(scheme):
        out[(lengths >= lo) & ((lengths <= hi) if hi is not None else True)] = i
    return out


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def sniff_sep(path):
    with pd.io.common.get_handle(path, "r", compression="infer") as h:
        first = h.handle.readline()
    return "\t" if "\t" in first else ","


def process_sample(path, a, keep):
    """Stream one fragment file; return length histogram, motif counts per 4-bin, totals."""
    sep = sniff_sep(path)
    hist = np.zeros(OPEN_CAP + 1, dtype=np.int64)         # length histogram, lengths capped
    motif = np.zeros((len(B4), 256), dtype=np.int64)
    n_all = n_direct = 0
    cols = None
    for chunk in pd.read_csv(path, sep=sep, chunksize=a.chunksize, encoding_errors="replace", low_memory=False):
        if cols is None:
            cols = set(chunk.columns)
            for c in [a.length_col, a.motif_col]:
                if c not in cols:
                    sys.exit(f"{path}: column '{c}' not found; columns are {sorted(cols)}")
        if "source" in cols:
            src = chunk["source"].astype(str).str.strip()
            n_all += len(chunk); n_direct += int((src == "direct").sum())
        else:
            n_all += len(chunk)
        for col, vals in keep.items():
            chunk = chunk[chunk[col].astype(str).str.strip().isin(vals)]
        if chunk.empty:
            continue
        L = pd.to_numeric(chunk[a.length_col], errors="coerce").to_numpy()
        ok = np.isfinite(L) & (L >= 1)
        L = np.minimum(L[ok].astype(np.int64), OPEN_CAP)
        hist += np.bincount(L, minlength=OPEN_CAP + 1)
        sub = chunk.loc[ok]
        if "motif_valid" in sub.columns:
            v = truthy(sub["motif_valid"]).to_numpy()
        else:
            v = np.ones(len(sub), bool)
        mi = sub[a.motif_col].astype(str).str.upper().map(M_INDEX).to_numpy()
        good = v & ~pd.isna(mi)
        b4 = bin_index(L, B4)
        good &= b4 >= 0
        np.add.at(motif, (b4[good], mi[good].astype(int)), 1)
    return hist, motif, n_all, n_direct


def clr(counts, pseudo):
    x = np.log(counts + pseudo)
    return x - x.mean(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, help="glob (quoted) of per-sample fragments with motifs (csv.gz)")
    ap.add_argument("-o", "--out", required=True, help="output folder for the feature tables")
    ap.add_argument("--sample-regex", default=r"^([^.]+)", help="sample id from file basename")
    ap.add_argument("--length-col", default="length", help="fragment length column")
    ap.add_argument("--motif-col", default="motif_5p", help="5' end-motif column")
    ap.add_argument("--keep", action="append", default=[], help="COL=V1,V2 row filter, e.g. source=direct")
    ap.add_argument("--lit-motifs", default="CCCA,CCTG,CCAG",
                    help="literature end motifs for F_motif_lit (Jin et al. 2021)")
    ap.add_argument("--entropy-estimator", choices=["plugin", "rare"], default="rare",
                    help="which entropy goes into F_combined (default rare, the primary estimator)")
    ap.add_argument("--n-rare", type=int, default=100, help="rarefaction subsamples per sample and bin")
    ap.add_argument("--min-frags", type=int, default=100, help="per-bin minimum N for an entropy value")
    ap.add_argument("--pseudocount", type=float, default=0.5, help="CLR pseudocount")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows read at a time")
    ap.add_argument("--seed", type=int, default=1, help="random seed for rarefaction and downsampling")
    ap.add_argument("--downsample", action="store_true",
                    help="equal-depth features: downsample every sample's length histogram and motif table to the "
                         "smallest sample's total (without replacement) before computing any feature")
    ap.add_argument("--downsample-to", type=int, default=0,
                    help="like --downsample, but to this fixed number of fragments (e.g. the PacBio depth, so ONT "
                         "models can be locked at the depth they will be applied to); stops if a sample has fewer")
    ap.add_argument("-t", "--threads", type=int, default=4, help="samples processed in parallel")
    a = ap.parse_args()

    keep = {}
    for k in a.keep:
        col, vals = k.split("=", 1)
        keep[col] = [v.strip() for v in vals.split(",")]
    lit = [m.strip().upper() for m in a.lit_motifs.split(",") if m.strip()]
    bad = [m for m in lit if m not in M_INDEX]
    if bad:
        sys.exit(f"not 4-mers over ACGT: {bad}")

    files = sorted(glob.glob(a.fragments))
    if not files:
        sys.exit(f"no files match {a.fragments}")
    os.makedirs(a.out, exist_ok=True)

    samples, hists, motifs, cov = [], [], [], []
    names = []
    for f in files:
        m = re.match(a.sample_regex, os.path.basename(f))
        names.append(m.group(1) if m else os.path.basename(f))
    with ProcessPoolExecutor(max_workers=max(1, a.threads)) as ex:
        results = list(ex.map(process_sample, files, [a] * len(files), [keep] * len(files)))
    for s, (h, mo, n_all, n_direct) in zip(names, results):
        print(f"[{s}] read", file=sys.stderr)
        samples.append(s); hists.append(h); motifs.append(mo)
        cov.append({"sample": s, "n_fragments": n_all,
                    "pct_recovered": 100.0 * (n_all - n_direct) / n_all if n_all else np.nan})
    H = np.vstack(hists)                       # samples x lengths
    MO = np.stack(motifs)                      # samples x 4 bins x 256
    if a.downsample or a.downsample_to:
        # equal depth: every feature then comes from the same number of fragments in every sample
        dsr = np.random.default_rng(a.seed + 7)
        if a.downsample_to:
            short = [s for s, h, m in zip(samples, H, MO) if h.sum() < a.downsample_to or m.sum() < a.downsample_to]
            if short:
                sys.exit(f"--downsample-to {a.downsample_to}: fewer fragments in {short}")
        nh = a.downsample_to or int(H.sum(1).min()); H = np.vstack([dsr.multivariate_hypergeometric(h, nh) for h in H])
        flat = MO.reshape(len(MO), -1); nm = a.downsample_to or int(flat.sum(1).min())
        MO = np.vstack([dsr.multivariate_hypergeometric(m, nm) for m in flat]).reshape(MO.shape)
        print(f"downsampled to {nh:,} fragments (lengths) and {nm:,} (motifs) per sample", file=sys.stderr)
    lengths = np.arange(H.shape[1])
    obs_max = int(lengths[H.sum(0) > 0].max())
    rng = np.random.default_rng(a.seed)

    # F_size and entropies on the 11-bin scheme
    size, ent_p, ent_r = {}, {}, {}
    totals = H.sum(1)
    for label, lo, hi in B11:
        hi_eff = hi if hi is not None else min(max(obs_max, lo), OPEN_CAP)
        K = hi_eff - lo + 1
        seg = H[:, lo:hi_eff + 1]
        n = seg.sum(1)
        size[f"frac_{label}"] = n / np.where(totals > 0, totals, np.nan)
        ent_p[f"ent_{label}"] = [norm_entropy(seg[i], K)[0] if n[i] >= a.min_frags else np.nan
                                 for i in range(len(samples))]
        ok = n >= a.min_frags
        target = int(n[ok].min()) if ok.any() else 0
        vals = []
        for i in range(len(samples)):
            if not ok[i] or target == 0:
                vals.append(np.nan); continue
            reps = [norm_entropy(rng.multivariate_hypergeometric(seg[i], target), K)[0] for _ in range(a.n_rare)]
            vals.append(float(np.mean(reps)))
        ent_r[f"ent_{label}"] = vals
        print(f"  {label}: K={K} rarefied to N={target}", file=sys.stderr)

    def write(name, d):
        df = pd.DataFrame(d, index=pd.Index(samples, name="sample"))
        df.to_csv(os.path.join(a.out, name))
        return df

    F_size = write("F_size.csv", size)
    F_ent = {"plugin": write("F_entropy_plugin.csv", ent_p), "rare": write("F_entropy_rare.csv", ent_r)}

    # motif sets on the 4-bin scheme
    motif_cols, lit_cols = {}, {}
    for b, (label, _, _) in enumerate(B4):
        C = clr(MO[:, b, :].astype(float), a.pseudocount)
        for j, mname in enumerate(MOTIFS):
            motif_cols[f"clr_{label}_{mname}"] = C[:, j]
        for mname in lit:
            lit_cols[f"clr_{label}_{mname}"] = C[:, M_INDEX[mname]]
        lit_cols[f"mds_{label}"] = [norm_entropy(MO[i, b, :], 256)[0] for i in range(len(samples))]
    write("F_motif.csv", motif_cols)
    F_lit = write("F_motif_lit.csv", lit_cols)

    est = a.entropy_estimator
    pd.concat([F_size, F_ent[est], F_lit], axis=1).to_csv(os.path.join(a.out, f"F_combined_{est}.csv"))
    pd.DataFrame(cov).set_index("sample").to_csv(os.path.join(a.out, "covariates.csv"))
    with open(os.path.join(a.out, "build_info.txt"), "w") as fh:
        fh.write(f"files={len(files)} keep={keep} lit_motifs={lit} entropy_estimator={est} "
                 f"min_frags={a.min_frags} n_rare={a.n_rare} pseudocount={a.pseudocount} seed={a.seed}\n")
    print(f"wrote feature sets for {len(samples)} samples to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
