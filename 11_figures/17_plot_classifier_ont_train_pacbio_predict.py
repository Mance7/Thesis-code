#!/usr/bin/env python3
"""Classifier AUC plot: models trained on ONT (leave-one-out), PacBio prediction only (locked ONT models, no PacBio folds),
at full and equal depth, direct and all fragments. Supplementary to Figure 3.15 and Tables 3.6, 3.8 and 3.10;
not a numbered thesis figure.
Input: classifier_auc_table.csv (columns set, platform, depth, fragments, auc, lo, hi, perm_p).
Output: classifier_ont_train_pacbio_predict.png in the --out folder.

Usage: python 17_plot_classifier_ont_train_pacbio_predict.py --auc-table "results/classifiers/classifier_auc_table.csv" -o "results/figures"
"""
import argparse, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ap = argparse.ArgumentParser(description="ONT-trained / PacBio-predicted classifier AUC plot.")
ap.add_argument("--auc-table", required=True,
                help="AUC table, one row per feature set x platform x depth x fragment set (csv: set, platform, depth, fragments, auc, lo, hi, perm_p)")
ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
ONT, PB, INK, MUTE, GRID = "#333333", "#8B5FBF", "#222222", "#6B6B6B", "#E4E4E4"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
t = pd.read_csv(A.auc_table)
SETS = [("R", "Size ratio R*"), ("F_size", "F-size"), ("F_entropy_rare", "F-entropy"), ("F_motif", "F-motif (256)"),
        ("F_motif_lit", "F-motif-lit"), ("F_combined_rare", "F-combined")]
def sig(r):
    """filled marker: R is a fixed rule; ONT needs permutation p < 0.05; PacBio needs the 95% CI above 0.5"""
    if r.set == "R": return True
    if r.platform == "ONT": return r.perm_p < 0.05
    return r.lo > 0.5
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
titles = {("full", "direct"): "A  Full depth, direct fragments", ("equal", "direct"): "B  Equal depth (140,488 per sample), direct",
          ("full", "all"): "C  Full depth, all fragments", ("equal", "all"): "D  Equal depth (141,034 per sample), all"}
for (i, V) in enumerate(["direct", "all"]):
    for (j, dep) in enumerate(["full", "equal"]):
        ax = axes[i, j]; sub = t[(t.depth == dep) & (t.fragments == V)]
        for k, (s, lab) in enumerate(SETS):
            y0 = len(SETS) - 1 - k
            for off, pl, col in [(0.17, "ONT", ONT), (-0.17, "PacBio", PB)]:
                r = sub[(sub.set == s) & (sub.platform == pl)].iloc[0]; y = y0 + off
                ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
                ax.scatter(r.auc, y, s=46, marker="o" if pl == "ONT" else "s", zorder=3,
                           facecolor=col if sig(r) else "white", edgecolor=col, linewidth=1.6)
                ax.text(1.115, y, f"{r.auc:.2f}", va="center", ha="right", fontsize=8, color=INK)
        ax.axvline(0.5, color=MUTE, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_yticks(range(len(SETS))); ax.set_yticklabels([l for _, l in SETS][::-1])
        ax.set_xlim(0.0, 1.13); ax.set_xticks([0, .25, .5, .75, 1]); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        ax.set_title(titles[(dep, V)], loc="left", fontsize=10, fontweight="bold", color=INK)
        if i == 1: ax.set_xlabel("AUC (dot) with 95% CI (line); 0.5 = chance")
h = [Line2D([], [], marker="o", color=ONT, lw=1.6, markersize=7, label="ONT: trained here, leave-one-out AUC"),
     Line2D([], [], marker="s", color=PB, lw=1.6, markersize=7, label="PacBio: prediction only (ONT model, no PacBio training or folds)"),
     Line2D([], [], marker="o", color=MUTE, mfc="white", lw=0, markersize=7, label="open = not significant")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.04, "Filled = significant (ONT: permutation p < 0.05; PacBio: 95% CI above 0.5). Equal depth: every ONT and PacBio sample cut to the smallest PacBio sample.\n"
         "*R = fixed rule (higher R = HCC), nothing fitted; its bins were chosen after seeing both platforms, so its AUC is descriptive.\n"
         "PacBio: at the ONT 0.5 cut-off all or almost all PacBio samples land on one side, so only the ranking (AUC) transfers.",
         ha="center", va="bottom", fontsize=7.4, color=MUTE)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(os.path.join(A.out, "classifier_ont_train_pacbio_predict.png"))
