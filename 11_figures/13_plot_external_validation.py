#!/usr/bin/env python3
"""Draw the external short-read validation figure.

Figure:
  Figure 3.25  fig4_external.png  (A) size ratio R_sr = (256-600 bp)/(100-250 bp) and (B) C-end motif share in the
               GSE71378 short-read batch, per group; circled = runs that ended early (< 3 million pairs);
               (C) R_sr in the CRA001537 bisulfite pilot, one sample per group (Table 3.14)

Inputs: the per-run GSE71378 table (R_sr, cend_share, group, batch), the GSE71378 run summary (pairs_read per run)
and the CRA001537 pilot summary (group, R_sr). The primary-test p values written under panels A and B are the
pre-specified one-sided exact tests of Table 3.13.
Output: fig4_external.png in the --out folder.

Usage:
  python 13_plot_external_validation.py --gse-dir results/external_validation/gse71378 \
      --pilot results/external_validation/cra001537_pilot/pilot_summary.tsv -o results/figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(description="External validation figure (Figure 3.25).")
ap.add_argument("--gse-dir", required=True,
                help="GSE71378 output folder with gse71378_per_run.csv (per-run R_sr, cend_share, group, batch) and gse_summary.tsv (pairs_read)")
ap.add_argument("--pilot", required=True,
                help="CRA001537 pilot summary, one row per group (tsv: group, R_sr)")
ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
A = ap.parse_args()
D, OUT = A.gse_dir, A.out
os.makedirs(OUT, exist_ok=True)
HCC, CTRL, OTH = "#E8703A", "#2F7BD8", "#9A9A9A"
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
fig, ax = plt.subplots(1, 3, figsize=(8.0, 3.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
for a, m, t in ((ax[0], "R_sr", "A   Size ratio R_sr"), (ax[1], "cend_share", "B   C-end motif share")):
    for i, (g, _) in enumerate(lab):
        v = sb[sb.group == g]
        x = i + rng.uniform(-.17, .17, len(v))
        a.scatter(x, v[m], s=22 if g != "other_cancer" else 14, color=cols[g], alpha=.85 if g != "other_cancer" else .6,
                  edgecolor="white", linewidth=.4, zorder=3)
        tr = v[v.trunc]
        a.scatter(x[v.trunc.values], tr[m], s=34, facecolor="none", edgecolor="#222", linewidth=.8, zorder=4)
        a.hlines(v[m].median(), i - .28, i + .28, color="#222", lw=1.2, zorder=4)
    a.set_xticks(range(4)); a.set_xticklabels([l for _, l in lab], fontsize=7)
    a.tick_params(axis="x", pad=2)
    a.set_title(t)
ax[0].set_ylabel("(256–600 bp) / (100–250 bp)"); ax[1].set_ylabel("CCCA + CCAG + CCTG, share of ends")
ax[0].set_yscale("log")
ax[0].set_xlabel("primary test, liver cancer vs 6 non-cancer: p = 0.87", fontsize=7.5)
ax[1].set_xlabel("primary test: p = 0.19", fontsize=7.5)
# C: CRA001537 bisulfite pilot, one sample per group
pil = pd.read_csv(A.pilot, sep="\t")
order = ["healthy", "hepatitis", "HCC"]; pil = pil.set_index("group").loc[order]
ax[2].bar(range(3), pil["R_sr"], color=[CTRL, CTRL, HCC], width=.6, zorder=3)
for i, v in enumerate(pil["R_sr"]): ax[2].text(i, v * 1.05 + .004, f"{v:.3f}", ha="center", fontsize=7.5)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["Healthy", "Hepatitis", "HCC"], fontsize=7.5)
ax[2].set_ylim(0, 0.31); ax[2].set_title("C   Pilot, n = 1 each"); ax[2].set_ylabel("R_sr (bisulfite WGBS)")
ax[2].set_xlabel("one sample per group", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_external.png"); print("ok")
