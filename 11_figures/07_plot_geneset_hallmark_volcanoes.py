#!/usr/bin/env python3
"""Direction and significance of the gene-set promoter-pooled results, drawn from the result tables.

Figures:
  Figure 3.11  fig4_geneset_volcano.png    H5b gene-set null (full HCC-expressed set and S1-S6), ONT and PacBio, all and direct
  Figure 3.14  fig4_hallmark_volcano.png   all 50 Hallmark sets inside HCC-expressed promoters, ONT and PacBio, all and direct
  (not numbered) geneset_trend_heatmap.png subsets ordered by HCC / blood expression, colour = signed excess
                                           (the final version of this view is Figure 3.12, 08_plot_geneset_expression_trend.py)
Also writes geneset_volcano_values.csv, the values behind the plots (input of 08_plot_geneset_expression_trend.py).
x is oriented by direction: left = lower in HCC, right = higher in HCC.

Inputs: the ONT gene-set random-null tables (all and direct fragments) and the Hallmark output folders
(one per platform and fragment set). PacBio gene-set values are typed in from Table 3.9.

Usage:
  python 07_plot_geneset_hallmark_volcanoes.py --geneset-null-all results/fig_inputs/geneset_random_null-all.csv \
      --geneset-null-direct results/fig_inputs/geneset_random_null_direct.csv --hallmark-dir results/hallmark -o results/figures
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HCC, HBV = "#E8703A", "#2F7BD8"
INK, MUTE, LINE = "#1F2933", "#6B7280", "#C9CED6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "axes.titlelocation": "left",
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 300})
A = None   # command-line arguments, set in __main__
CMAP = LinearSegmentedColormap.from_list("hbv_hcc", [HBV, "#F4F4F4", HCC])
SHORT = {"Expressed": "All HCC-expr.", "S1_hcc_blood_silent": "S1 blood-silent", "S2_hcc_blood_expressed": "S2 blood-expr.",
         "S3_hcc_tertile_low": "S3 low", "S4_hcc_tertile_mid": "S4 mid", "S5_hcc_tertile_high": "S5 high",
         "S6_liver_enriched": "S6 liver"}
METRIC = {"fsr": "fsr (share <=246 bp)", "entropy_short": "entropy (<=600 bp)"}

# PacBio gene-set null (Table 3.9): excess = |obs d| - |random median d|;
# q = BH over six subsets, full set = empirical p. Direction taken from the full-set pooled d (fsr -0.91, entropy +1.55),
# because the signed per-subset d was not stored for PacBio.
PB = {  # set: (ent_ex_all, ent_q_all, ent_ex_dir, ent_q_dir, fsr_ex_all, fsr_q_all, fsr_ex_dir, fsr_q_dir)
    "Expressed": (0.05, 0.11, 0.05, 0.089, 0.06, 0.005, 0.06, 0.004),
    "S1_hcc_blood_silent": (0.15, 0.21, 0.14, 0.25, 0.06, 0.37, 0.06, 0.35),
    "S2_hcc_blood_expressed": (0.04, 0.20, 0.04, 0.18, 0.05, 0.019, 0.06, 0.007),
    "S3_hcc_tertile_low": (-0.28, 1.0, -0.27, 0.99, -0.11, 0.94, -0.09, 0.91),
    "S4_hcc_tertile_mid": (0.14, 0.20, 0.15, 0.18, 0.04, 0.37, 0.04, 0.35),
    "S5_hcc_tertile_high": (0.20, 0.20, 0.17, 0.18, 0.26, 0.007, 0.24, 0.007),
    "S6_liver_enriched": (-0.14, 0.81, -0.19, 0.91, 0.14, 0.37, 0.12, 0.35),
}


def place_labels(ax, pts, fs=7):
    """pts: list of (x, y, text, color). Stack labels that would collide (same q is common with exact tests)."""
    placed = []
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    for x, y, t, c in sorted(pts, key=lambda p: (-p[1], p[0])):
        dy = 0
        while any(abs((y + dy) - py) < 0.075 * (y1 - y0) and abs(x - px) < 0.32 * (x1 - x0) for px, py in placed):
            dy -= 0.07 * (y1 - y0)
        placed.append((x, y + dy))
        ha = "right" if x > x0 + 0.72 * (x1 - x0) else "left"
        ax.annotate(t, (x, y), xytext=(x + (-0.012 if ha == "right" else 0.012) * (x1 - x0), y + dy + 0.02 * (y1 - y0)),
                    fontsize=fs, color=c, ha=ha, arrowprops=dict(arrowstyle="-", color=LINE, lw=0.6) if dy else None)


def geneset_table():
    """one row per platform, fragment set, gene set and metric: signed excess (x) and q (sig)"""
    rows = []
    for run, f in [("all", A.geneset_null_all), ("direct", A.geneset_null_direct)]:
        t = pd.read_csv(f)
        for _, r in t.iterrows():
            ex = abs(r.obs_d) - abs(r.random_d_median)
            sig = r.emp_p_abs if r.set == "Expressed" else r.q_bh_abs
            rows.append(dict(platform="ONT", run=run, set=r.set, metric=r.metric, x=np.sign(r.obs_d) * ex, sig=sig, obs_d=r.obs_d))
    for s, v in PB.items():
        for run, (ea, eq, fa, fq) in [("all", (v[0], v[1], v[4], v[5])), ("direct", (v[2], v[3], v[6], v[7]))]:
            rows.append(dict(platform="PacBio", run=run, set=s, metric="entropy_short", x=+ea, sig=eq, obs_d=np.nan))
            rows.append(dict(platform="PacBio", run=run, set=s, metric="fsr", x=-fa, sig=fq, obs_d=np.nan))
    return pd.DataFrame(rows)


def volcano_geneset(T, out):
    """Figure 3.11: one panel per platform x fragment set; circles fsr, squares entropy"""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharey=True)
    ymax = -np.log10(T.sig.min()) * 1.25
    for ax, (pl, run) in zip(axes.flat, [("ONT", "all"), ("ONT", "direct"), ("PacBio", "all"), ("PacBio", "direct")]):
        d = T[(T.platform == pl) & (T.run == run)]
        ax.axhline(-np.log10(0.05), color=MUTE, lw=0.8, ls="--"); ax.axvline(0, color=LINE, lw=0.8)
        lim = max(1.0, np.abs(T.x).max() * 1.15)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, ymax); pts = []
        for m, mk in [("fsr", "o"), ("entropy_short", "s")]:
            dd = d[d.metric == m]
            col = [(HCC if x > 0 else HBV) if s < 0.05 else "#B8BEC6" for x, s in zip(dd.x, dd.sig)]
            ax.scatter(dd.x, -np.log10(dd.sig), c=col, marker=mk, s=46, edgecolor="white", lw=1.2, zorder=3)
            pts += [(r.x, -np.log10(r.sig), SHORT[r.set], INK) for _, r in dd.iterrows() if r.sig < 0.05]
        place_labels(ax, pts)
        ax.set_title(f"{pl}, {run} fragments")
        ax.text(0.02, 0.97, "lower in HCC", transform=ax.transAxes, color=HBV, fontsize=8, va="top")
        ax.text(0.98, 0.97, "higher in HCC", transform=ax.transAxes, color=HCC, fontsize=8, va="top", ha="right")
    for ax in axes[1]:
        ax.set_xlabel("Change beyond matched random promoter sets (Cohen's d units)")
    for ax in axes[:, 0]:
        ax.set_ylabel("-log10 q (vs matched null)")
    h = [plt.Line2D([], [], marker="o", ls="", color=INK, label=METRIC["fsr"]),
         plt.Line2D([], [], marker="s", ls="", color=INK, label=METRIC["entropy_short"]),
         plt.Line2D([], [], marker="o", ls="", color="#B8BEC6", label="q >= 0.05")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Gene-set promoter pools: direction and significance (HCC vs HBV, 8 vs 6)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(os.path.join(out, "fig4_geneset_volcano.png")); plt.close(fig)


def volcano_hallmark(out):
    """Figure 3.14: 50 Hallmark sets; x = signed mean leave-one-out influence x 1000, y = permutation p"""
    KEY = {"BILE_ACID_METABOLISM": "Bile acid", "PEROXISOME": "Peroxisome", "KRAS_SIGNALING_DN": "KRAS down",
           "INTERFERON_ALPHA_RESPONSE": "IFN-alpha"}
    D = {}
    for pl in ("ont", "pacbio"):
        for run in ("all", "direct"):
            base = os.path.join(A.hallmark_dir, f"hallmark_{pl}_{run}")
            t = pd.read_csv(f"{base}/set_enrichment.csv")
            dd = {l.split(":")[0].split()[-1]: float(l.split("=")[1]) for l in open(f"{base}/run_info.txt") if l.startswith("pooled")}
            t["dsign"] = t.metric.map(lambda m: np.sign(dd[m]))
            t["x"] = t.dsign * t.obs_mean_influence * 1e3
            D[(pl, run)] = t
    lim = max(np.abs(t.x).max() for t in D.values()) * 1.1
    pmin = min(t.p_perm.min() for t in D.values())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharey=True)
    for ax, (pl, run) in zip(axes.flat, [("ont", "all"), ("ont", "direct"), ("pacbio", "all"), ("pacbio", "direct")]):
        t = D[(pl, run)]
        ax.axhline(-np.log10(0.05), color=MUTE, lw=0.8, ls="--"); ax.axvline(0, color=LINE, lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, -np.log10(pmin) * 1.15); pts = []
        for m, mk in [("fsr", "o"), ("entropy_short", "s")]:
            d = t[t.metric == m]
            col = [(HCC if x > 0 else HBV) if q < 0.1 else "#B8BEC6" for x, q in zip(d.x, d.q_bh)]
            ax.scatter(d.x, -np.log10(d.p_perm), c=col, marker=mk, s=30, edgecolor="white", lw=1.0, zorder=3)
            pts += [(r.x, -np.log10(r.p_perm), KEY[r.test_set] + (" (fsr)" if m == "fsr" else ""), INK if r.q_bh < 0.1 else MUTE)
                    for _, r in d.iterrows() if r.test_set in KEY and r.p_perm < 0.06]
        place_labels(ax, pts)
        ax.set_title(f"{'ONT' if pl == 'ont' else 'PacBio'}, {run} fragments")
        ax.text(0.02, 0.97, "set pushes: lower in HCC", transform=ax.transAxes, color=HBV, fontsize=8, va="top")
        ax.text(0.98, 0.97, "higher in HCC", transform=ax.transAxes, color=HCC, fontsize=8, va="top", ha="right")
    for ax in axes[1]:
        ax.set_xlabel("Mean leave-one-out influence x 1000 (signed by direction)")
    for ax in axes[:, 0]:
        ax.set_ylabel("-log10 p (3,003 relabellings)")
    h = [plt.Line2D([], [], marker="o", ls="", color=INK, label=METRIC["fsr"]),
         plt.Line2D([], [], marker="s", ls="", color=INK, label=METRIC["entropy_short"]),
         plt.Line2D([], [], marker="o", ls="", color="#B8BEC6", label="q >= 0.1")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Hallmark programmes (50) inside HCC-expressed promoters: direction and significance", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(os.path.join(out, "fig4_hallmark_volcano.png")); plt.close(fig)


def trend_heatmap(T, out):
    order = ["S3_hcc_tertile_low", "S4_hcc_tertile_mid", "S5_hcc_tertile_high", "S2_hcc_blood_expressed", "S1_hcc_blood_silent", "S6_liver_enriched"]
    labels = ["S3 HCC expr. low", "S4 HCC expr. mid", "S5 HCC expr. high", "S2 blood-expressed", "S1 blood-silent", "S6 liver-enriched"]
    cols = [(pl, run, m) for m in ("fsr", "entropy_short") for pl in ("ONT", "PacBio") for run in ("all", "direct")]
    X = np.full((len(order), len(cols)), np.nan); S = np.full_like(X, np.nan)
    for i, s in enumerate(order):
        for j, (pl, run, m) in enumerate(cols):
            r = T[(T.set == s) & (T.platform == pl) & (T.run == run) & (T.metric == m)].iloc[0]
            X[i, j], S[i, j] = r.x, r.sig
    v = np.nanmax(np.abs(X))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(X, cmap=CMAP, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.text(j, i, f"{X[i, j]:+.2f}" + ("*" if S[i, j] < 0.05 else ""), ha="center", va="center", fontsize=7.5,
                    color=INK, fontweight="bold" if S[i, j] < 0.05 else "normal")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{pl}\n{run}" for pl, run, _ in cols], fontsize=7.5)
    ax.axhline(2.5, color="white", lw=3); ax.axvline(3.5, color="white", lw=3)
    for j0, m in [(1.5, "fsr (share <=246 bp)"), (5.5, "entropy (<=600 bp)")]:
        ax.text(j0, -0.85, m, ha="center", fontsize=8.5, fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("change beyond matched random sets\n(blue = lower in HCC, orange = higher)", fontsize=7.5)
    ax.set_title("Promoter pools by expression: HCC vs HBV change beyond random sets (* q < 0.05)", pad=22, fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "geneset_trend_heatmap.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gene-set and Hallmark volcano plots (Figures 3.11 and 3.14).")
    ap.add_argument("--geneset-null-all", required=True,
                    help="ONT gene-set random-null results, all fragments (csv: set, metric, obs_d, random_d_median, emp_p_abs, q_bh_abs)")
    ap.add_argument("--geneset-null-direct", required=True,
                    help="ONT gene-set random-null results, direct fragments (same columns)")
    ap.add_argument("--hallmark-dir", required=True,
                    help="folder with hallmark_<ont|pacbio>_<all|direct>/ subfolders, each with set_enrichment.csv and run_info.txt")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    a = ap.parse_args(); A = a
    os.makedirs(a.out, exist_ok=True)
    T = geneset_table(); T.to_csv(os.path.join(a.out, "geneset_volcano_values.csv"), index=False)
    volcano_geneset(T, a.out); volcano_hallmark(a.out); trend_heatmap(T, a.out)
    print("written to", a.out)
