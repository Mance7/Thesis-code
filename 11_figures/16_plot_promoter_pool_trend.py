#!/usr/bin/env python3
"""Plots of the promoter-pool trend statistics (pool_stats.csv of the gene-set pool trend script), one folder per run.
This is the per-bin extension of Figure 3.12: the same HCC vs HBV view for each of the 30 blood-expression bins
and for every fragment feature. Not yet a numbered thesis figure.
  volcano_<feature>.png       one file per feature (R, fsr, WPS, entropy, MDS, end motifs), one panel per run:
                              log2 fold change (HCC/HBV) vs -log10 MWU p; circles = blood-expression bins, diamonds = gene sets
  trend_<kind>_<run>.png      rows = pools (blood bins 1-30, then gene sets), columns = features, colour = log2FC, * = q < 0.05
Usage: python 16_plot_promoter_pool_trend.py --runs "ONT all=DIR1,ONT direct=DIR2,PacBio all=DIR3,PacBio direct=DIR4" -o OUT_DIR
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HCC, HBV, INK, MUTE, LINE = "#E8703A", "#2F7BD8", "#1F2933", "#6B7280", "#C9CED6"
CMAP = LinearSegmentedColormap.from_list("hbv_hcc", [HBV, "#F4F4F4", HCC])
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                     "axes.titleweight": "bold", "savefig.dpi": 300})
BINS = ["le600", "601_2000", "2001_5000", "gt5000"]
FEATS = (["R", "fsr", "wps"]
         + [f"entropy_{b}" for b in BINS] + [f"mds_{b}" for b in BINS]
         + ["m_CCCA", "m_CCAG", "m_CCTG", "c_end", "m_GGAG", "m_AGTT", "m_ATTT"])
NICE = {"R": "size ratio R = 256-650 / >2000 bp", "fsr": "fsr (share <=246 of <=600 bp)",
        "wps": "WPS per fragment (120-180 bp)",
        "c_end": "C-end motifs (CCCA+CCAG+CCTG)"}
for _b, _t in zip(BINS, ["<=600", "601-2000", "2001-5000", ">5000"]):
    NICE[f"entropy_{_b}"] = f"length entropy {_t} bp (rarefied)"
    NICE[f"mds_{_b}"] = f"MDS {_t} bp (rarefied)"
SHORTSET = {"Expressed": "all", "S1_hcc_blood_silent": "S1", "S2_hcc_blood_expressed": "S2", "S3_hcc_tertile_low": "S3",
            "S4_hcc_tertile_mid": "S4", "S5_hcc_tertile_high": "S5", "S6_liver_enriched": "S6", "X1_blood_silent_not_liver": "X1"}
SETS = ["S3_hcc_tertile_low", "S4_hcc_tertile_mid", "S5_hcc_tertile_high", "S2_hcc_blood_expressed",
        "S1_hcc_blood_silent", "X1_blood_silent_not_liver", "S6_liver_enriched", "Expressed"]


def xval(d, feat):
    """log2 FC, except WPS (can be negative): HCC - HBV difference."""
    return (d["diff"] if feat == "wps" else d["log2fc"]).to_numpy(float)


def volcano_per_feature(runs, out):
    """One file per feature, one panel per run. Blood bins coloured by bin (light = low blood expression), gene sets as
    labelled diamonds; black ring = BH q < 0.05 (within pool type and feature)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    seq = LinearSegmentedColormap.from_list("bins", ["#D6EDEB", "#1B7F79", "#0B3B38"]); nrm = Normalize(1, 30)
    feats = [f for f in FEATS if any(f in set(st.feature) for st in runs.values())]
    for feat in feats:
        n = len(runs); nc = 2 if n > 1 else 1; nr = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.7 * nr + 0.5), squeeze=False)
        for ax, (name, st) in zip(axes.flat, runs.items()):
            d = st[(st.feature == feat)].dropna(subset=["p_mwu"]).copy(); d["x"] = xval(d, feat)
            d = d.dropna(subset=["x"])
            if d.empty:
                ax.set_title(f"{name}: no data"); ax.axis("off"); continue
            lim = np.nanmax(np.abs(d.x)) * 1.15 or 1
            ax.axvline(0, color=LINE, lw=0.8); ax.axhline(-np.log10(0.05), color=MUTE, lw=0.8, ls="--")
            b = d[d.pool_kind == "blood_bin"]
            ax.scatter(b.x, -np.log10(b.p_mwu), c=seq(nrm(b.pool_order)), s=26, edgecolor=np.where(b.q_within_kind_feature < 0.05, INK, "white"), lw=1.0, zorder=3)
            g = d[d.pool_kind == "gene_set"]
            ax.scatter(g.x, -np.log10(g.p_mwu), marker="D", s=34, color=HCC, edgecolor=np.where(g.q_within_kind_feature < 0.05, INK, "white"), lw=1.0, zorder=4)
            for _, r in g.iterrows():
                ax.annotate(SHORTSET.get(r.pool, r.pool), (r.x, -np.log10(r.p_mwu)), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=INK)
            ax.set_xlim(-lim, lim); ax.set_ylim(0, max(-np.log10(d.p_mwu.min()) * 1.2, 1.6))
            ax.set_title(name, fontsize=9)
            ax.text(0.01, 0.98, "lower in HCC", transform=ax.transAxes, color=HBV, va="top", fontsize=7)
            ax.text(0.99, 0.98, "higher in HCC", transform=ax.transAxes, color=HCC, va="top", ha="right", fontsize=7)
            ax.set_xlabel("HCC - HBV difference" if feat == "wps" else "log2 fold change, HCC / HBV", fontsize=8)
            ax.set_ylabel("-log10 p (Mann-Whitney)", fontsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        cax = fig.add_axes([0.25, 0.035, 0.3, 0.015])
        cb = fig.colorbar(ScalarMappable(nrm, seq), cax=cax, orientation="horizontal"); cax.set_title("blood-expression bin of circles (1 = lowest)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        fig.text(0.6, 0.03, "diamonds = gene sets (S1-S6, X1, all)\nblack ring = q < 0.05", fontsize=7, color=INK, va="center")
        fig.suptitle(NICE.get(feat, feat) + ": promoter pools, HCC vs HBV", x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0.08, 1, 0.96)); fig.savefig(os.path.join(out, f"volcano_{feat}.png")); plt.close(fig)


def trend(st, name, out, kind="blood_bin"):
    """heatmap: rows = pools of one kind (then gene sets), columns = features, colour scaled per feature"""
    feats = [f for f in FEATS if f in set(st.feature)]
    b = st[st.pool_kind == kind].sort_values("pool_order")
    gs = st[(st.pool_kind == "gene_set")]
    rows = list(dict.fromkeys(b.pool)) + [s for s in SETS if s in set(gs.pool)]
    X = np.full((len(rows), len(feats)), np.nan); Q = np.full_like(X, np.nan)
    for i, p in enumerate(rows):
        for j, f in enumerate(feats):
            r = st[(st.pool == p) & (st.feature == f)]
            if len(r):
                X[i, j] = (r["diff"] if f == "wps" else r.log2fc).iloc[0]
                Q[i, j] = r.q_within_kind_feature.iloc[0]
    # each feature has its own scale (fold changes differ a lot between features): colour = log2FC / max |log2FC| of that column
    scale = np.nanmax(np.abs(X), 0); scale[~np.isfinite(scale) | (scale == 0)] = 1
    fig, ax = plt.subplots(figsize=(0.42 * len(feats) + 2.6, 0.2 * len(rows) + 1.8))
    im = ax.imshow(X / scale, cmap=CMAP, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    for i in range(len(rows)):
        for j in range(len(feats)):
            if Q[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([p if not p.startswith(("B", "E_B")) else p.replace("E_B", "E-bin ").replace("B", "bin ") for p in rows], fontsize=6.5)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([f"{f}\n±{s:.2f}" for f, s in zip(feats, scale)], rotation=90, fontsize=6.5)
    nb = len(dict.fromkeys(b.pool))
    if nb < len(rows):
        ax.axhline(nb - 0.5, color="white", lw=3)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("log2 FC, HCC/HBV (scaled per feature;\nnumber under name = column max)", fontsize=7)
    ax.set_title(f"{name}: change with blood expression (bin 1 = lowest), * q < 0.05", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"trend_{kind}_{name.replace(' ', '_')}.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Volcano and trend plots of the promoter-pool statistics.")
    ap.add_argument("--runs", required=True,
                    help="comma-separated NAME=FOLDER pairs; each folder holds pool_stats.csv of one platform / fragment set")
    ap.add_argument("-o", "--out", required=True, help="output folder for the PNG files (created if missing)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = {}
    for item in a.runs.split(","):
        name, d = item.split("=", 1)
        st = pd.read_csv(os.path.join(d.strip(), "pool_stats.csv")); runs[name.strip()] = st
        trend(st, name.strip(), a.out)
        if "expressed_x_bin" in set(st.pool_kind):
            trend(st, name.strip(), a.out, kind="expressed_x_bin")
    volcano_per_feature(runs, a.out)
    print("written to", a.out)
