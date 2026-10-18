#!/usr/bin/env python3
"""
10_test_zhang_cohort.py

CRA001537 low-pass cohort (Zhang et al. 2020): group comparisons on the fixed short-read size
measures (test only; nothing is refit). R_sr = fragments 256-600 bp / 100-250 bp, fixed before any
external data were seen (Methods 2.13). Short reads cannot see the long-read range (> ~700 bp), so
R_sr is the short-read stand-in and is compared within this cohort only.
Questions:
  main     HCC vs hepatitis (8 vs 17)
  source   HCC vs cirrhosis (tumour vs damaged liver), HCC vs HCC after surgery (unpaired: no patient
           link in GSA), cirrhosis vs hepatitis
  context  HCC vs healthy
  early vs advanced HCC is descriptive only.
Tests: two-sided Mann-Whitney U (permutation p for small groups, else normal approximation with
tie correction), AUC = P(a > b), bootstrap 95% CI of the median difference.

Input : DIR/ev_summary.tsv (from 05_ or 07_extract_external_cohorts*.py, bs mode)
        sample sheet TSV with columns run, group, label, detail (detail = early stage HCC / advanced HCC)
Output: OUT/zhang_tests.tsv, OUT/zhang_per_sample.tsv, OUT/zhang_groups.tsv
"""
import argparse, os, csv, math
import numpy as np

ap = argparse.ArgumentParser(description="CRA001537 cohort: group tests on R_sr and size shares")
ap.add_argument("--dir", required=True, help="extraction folder with ev_summary.tsv")
ap.add_argument("--meta", required=True, help="sample sheet TSV with columns run, group, label, detail")
ap.add_argument("-o", "--out", required=True, help="output folder")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# sample groups and labels from the sheet; groups default to those in ev_summary.tsv
meta = {r["run"]: r for r in csv.DictReader(open(args.meta), delimiter="\t")}
rows = [r for r in csv.DictReader(open(os.path.join(args.dir, "ev_summary.tsv")), delimiter="\t")]
for r in rows:
    m = meta.get(r["run"], {})
    r["group"], r["label"], r["detail"] = m.get("group", r["group"]), m.get("label", ""), m.get("detail", "")
out = args.out
METRICS = ["R_sr", "share_256_650", "share_le150", "share_151_255"]

def mwu(x, y):
    """Mann-Whitney U, two-sided; exact by permutation when small, else normal approximation with tie correction"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    allv = np.r_[x, y]; n1, n2 = len(x), len(y)
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(allv):                       # average ranks for ties
        idx = allv == v; ranks[idx] = ranks[idx].mean()
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    if n1 * n2 <= 400:
        rng = np.random.default_rng(1); cnt = 0; B = 20000
        obs = abs(U - n1 * n2 / 2)
        for _ in range(B):
            p = rng.permutation(len(allv))
            u = ranks[p[:n1]].sum() - n1 * (n1 + 1) / 2
            cnt += abs(u - n1 * n2 / 2) >= obs - 1e-9
        pval = (cnt + 1) / (B + 1)
    else:
        t = np.unique(allv, return_counts=True)[1]
        sd = math.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - (t ** 3 - t).sum() / ((n1 + n2) * (n1 + n2 - 1))))
        z = (U - n1 * n2 / 2) / sd if sd else 0
        pval = math.erfc(abs(z) / math.sqrt(2))
    return U, U / (n1 * n2), pval                   # U, AUC (P(x > y)), p

def boot_diff(x, y, B=5000):
    """bootstrap 95% interval of the difference in medians"""
    rng = np.random.default_rng(2)
    d = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(B)]
    return np.percentile(d, [2.5, 97.5])

g = lambda name, k: [float(r[k]) for r in rows if r["group"] == name]
tests = [("main", "HCC", "hepatitis"), ("source", "HCC", "cirrhosis"), ("source", "HCC", "HCC_post_surgery"),
         ("source", "cirrhosis", "hepatitis"), ("context", "HCC", "healthy")]
with open(os.path.join(out, "zhang_tests.tsv"), "w") as fo:
    fo.write("question\tmetric\tgroup_a\tn_a\tmedian_a\tgroup_b\tn_b\tmedian_b\tdiff_median\tdiff_ci95\tAUC_a_gt_b\tp_two_sided\n")
    for q, A, Bg in tests:
        for k in METRICS:
            x, y = g(A, k), g(Bg, k)
            if len(x) < 2 or len(y) < 2:
                continue
            U, auc, p = mwu(x, y); lo, hi = boot_diff(np.array(x), np.array(y))
            fo.write(f"{q}\t{k}\t{A}\t{len(x)}\t{np.median(x):.6g}\t{Bg}\t{len(y)}\t{np.median(y):.6g}\t"
                     f"{np.median(x) - np.median(y):.6g}\t{lo:.4g}..{hi:.4g}\t{auc:.3f}\t{p:.4g}\n")
with open(os.path.join(out, "zhang_per_sample.tsv"), "w") as fo:
    cols = ["run", "label", "group", "detail", "fragments"] + METRICS
    fo.write("\t".join(cols) + "\n")
    for r in sorted(rows, key=lambda r: (r["group"], r["label"])):
        fo.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
with open(os.path.join(out, "zhang_groups.tsv"), "w") as fo:
    fo.write("group\tn\t" + "\t".join(f"{k}_median\t{k}_IQR" for k in METRICS) + "\n")
    for name in sorted({r["group"] for r in rows}):
        line = [name, str(len(g(name, "R_sr")))]
        for k in METRICS:
            v = np.array(g(name, k)); line += [f"{np.median(v):.6g}", f"{np.percentile(v, 25):.4g}-{np.percentile(v, 75):.4g}"]
        fo.write("\t".join(line) + "\n")
    hcc = [r for r in rows if r["group"] == "HCC"]
    for st in ("early stage HCC", "advanced HCC"):
        v = np.array([float(r["R_sr"]) for r in hcc if r["detail"] == st])
        if len(v):
            fo.write(f"HCC:{st}\t{len(v)}\t{np.median(v):.6g}\t(R_sr only, descriptive)\n")
print(open(os.path.join(out, "zhang_tests.tsv")).read())
print(open(os.path.join(out, "zhang_groups.tsv")).read())
