#!/usr/bin/env python3
"""Draw the external short-read validation figure.

Figure:
  Figure 3.25  fig4_external.png
    (A) size ratio R_sr = (256-600 bp)/(100-250 bp) and (B) C-end motif share in the GSE71378 short-read batch,
        per group; circled = runs that ended early (< 3 million pairs)
    (C) R_sr and (D) share of fragments <= 150 bp in CRA001537 (bisulfite), per group
    (E) mean fragment-length profile of hepatitis, cirrhosis and HCC
    (F) share <= 150 bp within one flowcell

Inputs: the per-run GSE71378 table (R_sr, cend_share, group, batch) and run summary (pairs_read per run), the CRA001537
per-sample table (zhang_per_sample.tsv) and length histograms (ev_hist.tsv.gz), and a sheet with the flowcell of each
CRA001537 run. The GSE71378 p values written under panels A and B are the pre-specified one-sided exact tests of Table 3.13;
the CRA001537 p values are two-sided Mann-Whitney tests.
Output: fig4_external.png in the --out folder.

Usage:
  python 13_plot_external_validation.py --gse-dir results/external_validation/gse71378 \
      --zhang-table results/external_validation/cra001537/zhang_per_sample.tsv \
      --zhang-hist results/external_validation/cra001537/ev_hist.tsv.gz \
      --flowcells runs_flowcell.tsv -o results/figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
ap = argparse.ArgumentParser(description="External validation figure (Figure 3.25).")
ap.add_argument("--gse-dir", required=True,
                help="GSE71378 output folder with gse71378_per_run.csv (per-run R_sr, cend_share, group, batch) and gse_summary.tsv (pairs_read)")
ap.add_argument("--zhang-table", required=True, help="CRA001537 per-sample table (tsv: run, group, R_sr, share_le150, ...)")
ap.add_argument("--zhang-hist", required=True, help="CRA001537 fragment-length histograms (ev_hist.tsv.gz, one row per run)")
ap.add_argument("--flowcells", required=True, help="tsv with columns run, flowcell for the CRA001537 runs")
ap.add_argument("--one-flowcell", default="HH327BBXX", help="flowcell used for panel F (text matched inside the flowcell id)")
ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
A = ap.parse_args()
D, OUT = A.gse_dir, A.out
os.makedirs(OUT, exist_ok=True)
HCC, POST, CTRL, CIR, OTH = "#E8703A", "#F2B38F", "#2F7BD8", "#7FA8E3", "#9A9A9A"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.grid": True, "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.dpi": 300})
d = pd.read_csv(f"{D}/gse71378_per_run.csv", index_col=0)
s = pd.read_csv(f"{D}/gse_summary.tsv", sep="\t").set_index("run")
# runs that ended early (fewer than 3 million pairs) are circled
d["trunc"] = s["pairs_read"] < 3_000_000
# only the short-read batch of GSE71378 is plotted
sb = d[d.batch == "short_reads"]
cols = {"other_cancer": OTH, "noncancer_disease": CTRL, "healthy": CTRL, "liver_cancer": HCC}
lab = [("other_cancer", "Other\ncancers"), ("noncancer_disease", "Non-cancer\ndisease"), ("healthy", "Healthy"), ("liver_cancer", "Liver\ncancer")]
rng = np.random.default_rng(3)
z = pd.read_csv(A.zhang_table, sep="\t")
z = z.drop(columns=[c for c in ("flowcell",) if c in z.columns]).merge(pd.read_csv(A.flowcells, sep="\t")[["run", "flowcell"]], on="run", how="left")
fig, ax = plt.subplots(2, 3, figsize=(8.0, 7.2))
# A, B: GSE71378
for a, m, t in ((ax[0, 0], "R_sr", "A   GSE71378: size ratio R_sr"), (ax[0, 1], "cend_share", "B   GSE71378: C-end motif share")):
    for i, (g, _) in enumerate(lab):
        v = sb[sb.group == g]
        x = i + rng.uniform(-.17, .17, len(v))
        a.scatter(x, v[m], s=22 if g != "other_cancer" else 14, color=cols[g], alpha=.85 if g != "other_cancer" else .6,
                  edgecolor="white", linewidth=.4, zorder=3)
        tr = v[v.trunc]
        a.scatter(x[v.trunc.values], tr[m], s=34, facecolor="none", edgecolor="#222", linewidth=.8, zorder=4)
        a.hlines(v[m].median(), i - .28, i + .28, color="#222", lw=1.2, zorder=4)
    a.set_xticks(range(4)); a.set_xticklabels([l for _, l in lab], fontsize=6.5)
    a.tick_params(axis="x", pad=2)
    a.set_title(t, fontsize=9)
ax[0, 0].set_ylabel("(256–600 bp) / (100–250 bp)"); ax[0, 1].set_ylabel("CCCA + CCAG + CCTG, share of ends")
ax[0, 0].set_yscale("log")
ax[0, 0].set_xlabel("liver cancer vs 6 non-cancer: p = 0.87", fontsize=7); ax[0, 1].set_xlabel("p = 0.19", fontsize=7)
# CRA001537 groups
order = [("healthy", "Healthy", CTRL), ("hepatitis", "Hepat-\nitis", CTRL), ("cirrhosis", "Cirrho-\nsis", CIR),
         ("HCC", "HCC", HCC), ("HCC_post_surgery", "HCC\nafter op", POST)]
def pval(metric, frame=z):
    x = frame[frame.group == "HCC"][metric].values; y = frame[frame.group == "hepatitis"][metric].values
    return mannwhitneyu(x, y, alternative="two-sided").pvalue
def strip(a, met, ylog=False, floor=None):
    for i, (g, l, c) in enumerate(order):
        v = z[z.group == g][met].values.astype(float)
        x = i + rng.uniform(-.2, .2, len(v))
        vv = np.where(v <= 0, floor, v) if ylog else v
        a.scatter(x, vv, s=20, color=c, edgecolor="white", linewidth=.4, zorder=3)
        med = np.median(v)
        a.hlines(floor if (ylog and med <= 0) else med, i - .3, i + .3, color="#222", lw=1.2, zorder=4)
    a.set_xticks(range(5)); a.set_xticklabels([f"{l}\nn={len(z[z.group == g])}" for g, l, _ in order], fontsize=6.3)
# C: R_sr (zeros are drawn on the floor of the log axis)
a = ax[0, 2]; strip(a, "R_sr", True, 4e-6)
a.set_yscale("log"); a.set_ylim(2e-6, 1)
a.set_yticks([4e-6, 1e-4, 1e-2, 1]); a.set_yticklabels(["0", "1e-4", "1e-2", "1"]); a.axhline(6e-6, color="#bbb", lw=.7, ls=":")
a.set_title("C   CRA001537: R_sr", fontsize=9); a.set_ylabel("(256–600 bp) / (100–250 bp)")
a.set_xlabel(f"HCC vs hepatitis: p = {pval('R_sr'):.2f} (lower in HCC)", fontsize=7)
# D: share <= 150 bp
a = ax[1, 0]; strip(a, "share_le150")
a.set_title("D   CRA001537: share ≤150 bp", fontsize=9); a.set_ylabel("fragments ≤150 bp / all fragments")
a.set_xlabel(f"HCC vs hepatitis: p = {pval('share_le150'):.3f}", fontsize=7)
# E: mean length profile per group
H = pd.read_csv(A.zhang_hist, sep="\t", index_col=0); H.columns = H.columns.astype(int)
F = H.div(H.sum(axis=1), axis=0).loc[:, 1:400]
grp = z.set_index("run").group
a = ax[1, 1]; xs = np.arange(1, 401)
for g, l, c in [("hepatitis", "Hepatitis", CTRL), ("cirrhosis", "Cirrhosis", CIR), ("HCC", "HCC", HCC)]:
    y = F.loc[grp.reindex(F.index) == g].mean(axis=0).values
    y = np.convolve(y, np.ones(5) / 5, mode="same")
    a.plot(xs, y * 100, color=c, lw=1.5, label=f"{l} (n={int((grp == g).sum())})")
a.axvline(150, color="#aaa", lw=.8, ls="--"); a.set_xlim(50, 350); a.legend(fontsize=7, loc="upper right")
a.set_title("E   CRA001537: length profile", fontsize=9)
a.set_xlabel("fragment length (bp)"); a.set_ylabel("% of fragments per bp (mean)")
# F: one flowcell only
a = ax[1, 2]; f = z[z.flowcell.astype(str).str.contains(A.one_flowcell)]
grp_f = [("healthy", "Healthy", CTRL), ("hepatitis", "Hepat-\nitis", CTRL), ("HCC", "HCC", HCC), ("HCC_post_surgery", "HCC\nafter op", POST)]
for i, (g, l, c) in enumerate(grp_f):
    v = f[f.group == g].share_le150.values
    x = i + rng.uniform(-.2, .2, len(v))
    a.scatter(x, v, s=22, color=c, edgecolor="white", linewidth=.4, zorder=3)
    a.hlines(np.median(v), i - .3, i + .3, color="#222", lw=1.2, zorder=4)
a.set_xticks(range(4)); a.set_xticklabels([f"{l}\nn={len(f[f.group == g])}" for g, l, _ in grp_f], fontsize=6.3)
a.set_title("F   One flowcell: share ≤150 bp", fontsize=9); a.set_ylabel("fragments ≤150 bp / all fragments")
a.set_xlabel(f"HCC vs hepatitis: p = {pval('share_le150', f):.2f}", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_external.png"); print("ok")
