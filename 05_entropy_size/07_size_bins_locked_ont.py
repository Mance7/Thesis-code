#!/usr/bin/env python3
"""Size-ratio window chosen on ONT only, locked, then tested once on PacBio.

A stricter version of the window check in 06_size_bin_fold_and_window.py
(Discussion 4.3, Figure 4.4). There the denominator is fixed at >2 kb; here the
denominator is also chosen on ONT, and the bins and the ONT cut-off are locked
in a file before PacBio is read. Both pick the same 410-459 bp numerator
(PacBio Hedges g 2.15 with >2 kb, 2.14 with the >=3 kb denominator chosen here).
It checks the size-ratio definition; it is not one of the classifiers of
Section 2.10.

Rule (fixed before running; uses ONT labels only, never PacBio):
  1. Denominator: fragments >= L, L from {1000,1500,2000,2500,3000,4000,5000}.
     Pick the L whose raw per-sample count differs LEAST between HCC and HBV
     on ONT (smallest |Hedges g| of log count) = a within-library yardstick
     that carries no disease signal.
  2. Numerator: every window [a, b] on a 10-bp grid, 60-2000 bp, width >= 50 bp.
     Pick the window with the largest ONT Hedges g of log(window / denominator).
  3. Lock (a, b, L) and the ONT cut-off (midpoint between the highest HBV and
     the lowest HCC log ratio) to locked_bins.json.
  4. Stage 'pacbio' reads only the lock file and reports PacBio AUC, exact
     Mann-Whitney p, Hedges g, and sens/spec at the ONT cut-off.
Inputs: 1-bp histograms <hist-dir>/length_hist_{ont,pacbio}_direct.tsv.gz
        (04_length_histograms.py) and a metadata csv (sample, group)
Outputs (-o): ont_denominator_choice.tsv, ont_numerator_grid.tsv.gz, locked_bins.json,
        pacbio_locked_ratio_per_sample.tsv, pacbio_test.json
Usage: 07_size_bins_locked_ont.py ont --hist-dir length_hist --metadata metadata/ont_samples.csv -o OUT
       07_size_bins_locked_ont.py pacbio --hist-dir length_hist --metadata metadata/ont_samples.csv -o OUT
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

LS = [1000, 1500, 2000, 2500, 3000, 4000, 5000]
GROUP = {}   # sample -> group, filled from --metadata in main()


def load(path):
    d = pd.read_csv(path, sep="\t", index_col=0)
    gt = d.pop(">5000").to_numpy(float)
    h = d.to_numpy(float)
    c = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, 1)], 1)
    g = np.array(["HCC" if GROUP[s] == "HCC" else "HBV" for s in d.index])
    return d.index.to_numpy(), g, c, gt


def win(c, a, b):
    return c[:, b] - c[:, a - 1]


def longc(c, gt, L):
    return c[:, 5000] - c[:, L - 1] + gt if L <= 5000 else gt


def hedges(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp * (1 - 3 / (4 * (nx + ny) - 9))


def stats(v, g):
    x, y = v[g == "HCC"], v[g == "HBV"]
    return dict(g=hedges(x, y), auc=mannwhitneyu(x, y).statistic / (len(x) * len(y)),
                p=mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue,
                hcc_min=x.min(), hbv_max=y.max())


def score(args):
    c, den, g, starts = args
    out = []
    for a in starts:
        for b in range(a + 49, 2001, 10):
            out.append((a, b, hedges(*(lambda v: (v[g == "HCC"], v[g == "HBV"]))(np.log(win(c, a, b) / den)))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ont", "pacbio"], help="ont = choose and lock; pacbio = test the lock")
    ap.add_argument("--hist-dir", required=True,
                    help="folder with length_hist_{ont,pacbio}_direct.tsv.gz")
    ap.add_argument("--metadata", required=True,
                    help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder (holds locked_bins.json)")
    ap.add_argument("--threads", type=int, default=4, help="parallel workers for the window grid")
    a = ap.parse_args()
    meta = pd.read_csv(a.metadata, sep=None, engine="python", encoding_errors="replace")
    meta.columns = [c.strip().lower() for c in meta.columns]
    GROUP.update(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))
    os.makedirs(a.out, exist_ok=True)
    lock = os.path.join(a.out, "locked_bins.json")

    if a.stage == "ont":
        s, g, c, gt = load(os.path.join(a.hist_dir, "length_hist_ont_direct.tsv.gz"))
        den_rows = [dict(L=L, **stats(np.log(longc(c, gt, L)), g)) for L in LS]
        den_tab = pd.DataFrame(den_rows)
        den_tab.to_csv(os.path.join(a.out, "ont_denominator_choice.tsv"), sep="\t", index=False)
        L = int(den_tab.loc[den_tab.g.abs().idxmin(), "L"])
        den = longc(c, gt, L)
        starts = list(range(60, 1952, 10))
        with ProcessPoolExecutor(a.threads) as ex:
            res = ex.map(score, [(c, den, g, starts[i::a.threads]) for i in range(a.threads)])
        grid = pd.DataFrame([r for p in res for r in p], columns=["a", "b", "g_ont"])
        grid.sort_values("g_ont", ascending=False).to_csv(
            os.path.join(a.out, "ont_numerator_grid.tsv.gz"), sep="\t", index=False)
        best = grid.loc[grid.g_ont.idxmax()]
        lo, hi = int(best.a), int(best.b)
        st = stats(np.log(win(c, lo, hi) / den), g)
        cut = (st["hcc_min"] + st["hbv_max"]) / 2
        json.dump(dict(numerator=[lo, hi], denominator_from=L, ont_log_cutoff=cut,
                       ont=st, rule=__doc__.split("Inputs:")[0].strip()),
                  open(lock, "w"), indent=1, default=float)
        print(json.dumps(json.load(open(lock)), indent=1))
    else:
        k = json.load(open(lock))
        lo, hi = k["numerator"]; L = k["denominator_from"]
        s, g, c, gt = load(os.path.join(a.hist_dir, "length_hist_pacbio_direct.tsv.gz"))
        lr = np.log(win(c, lo, hi) / longc(c, gt, L))
        st = stats(lr, g)
        call = lr > k["ont_log_cutoff"]
        st.update(sens_at_ont_cut=float(call[g == "HCC"].mean()), spec_at_ont_cut=float((~call[g == "HBV"]).mean()))
        pd.DataFrame(dict(sample=s, group=g, log_ratio=lr)).to_csv(
            os.path.join(a.out, "pacbio_locked_ratio_per_sample.tsv"), sep="\t", index=False)
        json.dump(dict(locked=k, pacbio=st), open(os.path.join(a.out, "pacbio_test.json"), "w"), indent=1, default=float)
        print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
