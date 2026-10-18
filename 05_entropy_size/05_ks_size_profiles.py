#!/usr/bin/env python3
"""Kolmogorov-Smirnov distance between the HCC and HBV fragment-length distributions.

Genome-wide, per platform (ONT, PacBio) and fragment set (all, direct), for
all lengths and for lengths <= 1000 bp (Methods 2.9, Results 3.7).
Each sample's 1 bp histogram is turned into a CDF; the group CDF is the mean
of its samples' CDFs, so every patient weighs the same and depth does not
matter; D = max |CDF_HCC - CDF_HBV|. The p-value is an exact permutation p
over all 3,003 relabellings of the 14 samples.

Input : <hist-dir>/length_hist_{ont,pacbio}_{direct,all}.tsv.gz (04_length_histograms.py)
        metadata csv with columns sample, group
Output: <out>/ks_size_profiles.tsv

Runs in seconds on one core:
  python 05_ks_size_profiles.py --hist-dir length_hist --metadata metadata/ont_samples.csv -o ks/
"""
import argparse, itertools, os
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hist-dir', required=True, help='folder with the length_hist_*.tsv.gz tables')
    ap.add_argument('--metadata', required=True, help='csv with columns sample, group')
    ap.add_argument('--case', default='HCC', help='group name of the cases in --metadata')
    ap.add_argument('-o', '--out', required=True, help='output folder')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.metadata, sep=None, engine='python', encoding_errors='replace')
    meta.columns = [c.strip().lower() for c in meta.columns]
    HCC = set(meta.loc[meta['group'].astype(str).str.strip() == a.case, 'sample'].astype(str).str.strip())

    rows = []
    for plat in ('ont', 'pacbio'):
        for kind in ('direct', 'all'):
            h = pd.read_csv(f'{a.hist_dir}/length_hist_{plat}_{kind}.tsv.gz', sep='\t', index_col=0)
            # the last column ">5000" is treated as length 5001
            lens = np.array([int(c) if c.isdigit() else 5001 for c in h.columns])
            for lo, hi, lab in ((1, 5001, 'all lengths'), (1, 1000, '<=1000 bp')):
                keep = (lens >= lo) & (lens <= hi)
                X = h.loc[:, keep].to_numpy(float)
                C = np.cumsum(X, 1) / X.sum(1, keepdims=True)      # per-sample CDF
                y = np.array([s in HCC for s in h.index])

                def D(lbl):
                    # largest gap between the two mean CDFs, its sign and position
                    diff = C[lbl].mean(0) - C[~lbl].mean(0)
                    i = np.abs(diff).argmax(); return abs(diff[i]), diff[i], i
                d0, sgn, i0 = D(y)
                # exact permutation p over every split of the samples into groups of the same sizes
                n = len(y); k = y.sum(); b = 0; tot = 0
                for comb in itertools.combinations(range(n), k):
                    l = np.zeros(n, bool); l[list(comb)] = True
                    tot += 1; b += D(l)[0] >= d0 - 1e-12
                rows.append(dict(platform=plat, fragments=kind, range=lab, n_hcc=int(k), n_hbv=int(n - k), D=round(d0, 4),
                                 at_bp=int(lens[keep][i0]), hcc_cdf_higher=bool(sgn > 0), perm_p=b / tot, n_perm=tot))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, 'ks_size_profiles.tsv'), sep='\t', index=False)
    print(out.to_string())


if __name__ == '__main__':
    main()
