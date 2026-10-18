#!/usr/bin/env python3
"""Draw the fragment-length profile, size ratio R and cross-platform summary figures.

Figures:
  Figure 3.16  fig4_size_profiles.png          per-sample length profiles (direct fragments, ONT and PacBio, log scale),
                                               HCC/HBV ratio of group medians, KS distance with permutation p
  Figure 3.17  fig4_size_ratio_R.png           R = (256-650 bp) / (>2 kb) per sample, ONT against PacBio and by group
  Figure 3.20  fig4_cross_platform_summary.png effect sizes of the results replicated on both platforms (typed from Table 3.11)

Inputs: 1-bp length histograms per platform and fragment set (length_hist_<ont|pacbio>_<all|direct>.tsv.gz,
one row per sample, columns 1..5000 and >5000), the per-sample periodicity table (sample, group, run, R),
the KS size-profile table and the ONT genome-wide size-bin counts (the last two for annotation and an R check).
Output: PNG files (300 dpi) in the --out folder; an R cross-check table and test results are printed.

Usage:
  python 09_plot_size_profiles_ratio_R_summary.py --hist-dir results/length_hist \
      --periodicity results/length_hist/periodicity_per_sample.tsv --ks results/length_hist/ks_size_profiles.tsv \
      --bin-counts results/size_bins/ont_bin_counts_genomewide.tsv -o results/figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, spearmanr

ap = argparse.ArgumentParser(description="Size profiles, size ratio R and cross-platform summary (Figures 3.16, 3.17, 3.20).")
ap.add_argument("--hist-dir", required=True,
                help="folder with the 1-bp length histograms length_hist_<ont|pacbio>_<all|direct>.tsv.gz")
ap.add_argument("--periodicity", required=True,
                help="per-sample periodicity table (tsv: sample, group, run, R, ...)")
ap.add_argument("--ks", required=True,
                help="KS distance between group length profiles (tsv: platform, fragments, range, D, at_bp, perm_p)")
ap.add_argument("--bin-counts", required=True,
                help="ONT genome-wide fragment counts per size bin (tsv: sample, size_bin, n_direct); used only to cross-check R")
ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
A = ap.parse_args()
OUT = A.out
os.makedirs(OUT, exist_ok=True)
HCC, HBV = "#EB6834", "#2A78D6"
GCOL = {"HCC": HCC, "HBV": HBV}
ONTC, PBC = "#3A3A3A", "#8A5FBF"
SHADE_R, SHADE_L = "#E4E4E4", "#F1F1F1"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8, "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 300,
})

per = pd.read_csv(A.periodicity, sep="\t")
group = per.drop_duplicates("sample").set_index("sample")["group"]

def load(p, k):
    """1-bp length histogram of one platform (ont/pacbio) and fragment set (all/direct)"""
    d = pd.read_csv(os.path.join(A.hist_dir, f"length_hist_{p}_{k}.tsv.gz"), sep="\t", index_col=0)
    return d

H = {p: load(p, "direct") for p in ["ont", "pacbio"]}
cols = lambda a, z: [str(i) for i in range(a, z + 1)]

def ratio_R(d):
    """size ratio R = fragments 256-650 bp / fragments > 2000 bp, per sample"""
    return d[cols(256, 650)].sum(axis=1) / (d[cols(2001, 5000)].sum(axis=1) + d[">5000"])

# ---------------- Figure 3.16: size profiles ----------------
edges = np.logspace(np.log10(50), np.log10(5000), 101)   # 100 log bins, 0.02 decade
lo = np.ceil(edges[:-1]).astype(int); hi = np.ceil(edges[1:]).astype(int) - 1
hi[-1] = 5000
centres = np.sqrt(edges[:-1] * edges[1:])

def binned(d):
    arr = d[cols(1, 5000)].to_numpy(float)
    tot = d.sum(axis=1).to_numpy(float)[:, None]
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), arr.cumsum(axis=1)], axis=1)
    counts = cs[:, hi] - cs[:, lo - 1]
    dlog = np.log10(hi + 0.5) - np.log10(lo - 0.5)
    dens = counts / tot / dlog          # fraction of all fragments per log10 unit
    gt5 = d[">5000"].to_numpy(float) / tot[:, 0]
    return dens, gt5

fig = plt.figure(figsize=(10.5, 6.6))
gs = fig.add_gridspec(2, 5, width_ratios=[6, 0.55, 0.9, 6, 0.55], height_ratios=[2.2, 1.2],
                      wspace=0.22, hspace=0.32)
titles = {"ont": "ONT, direct fragments", "pacbio": "PacBio, direct fragments"}
letters = iter("ABCDEF")
for j, p in enumerate(["ont", "pacbio"]):
    d = H[p]; g = group.loc[d.index]
    dens, gt5 = binned(d)
    c0 = 0 if j == 0 else 3
    ax = fig.add_subplot(gs[0, c0]); axs = fig.add_subplot(gs[0, c0 + 1], sharey=ax)
    axr = fig.add_subplot(gs[1, c0], sharex=ax); axrs = fig.add_subplot(gs[1, c0 + 1], sharey=axr)
    for a in (ax, axr):
        a.axvspan(256, 650, color=SHADE_R, lw=0, zorder=0)
        a.axvspan(2000, 5000, color=SHADE_L, lw=0, zorder=0)
    for grp in ["HBV", "HCC"]:
        m = (g == grp).to_numpy()
        for row in dens[m]:
            ax.plot(centres, row, color=GCOL[grp], lw=0.9, alpha=0.75)
        xs = np.full(m.sum(), 0.0) + (-0.12 if grp == "HBV" else 0.12)
        axs.scatter(xs, gt5[m], s=14, color=GCOL[grp], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(50, 5000)
    ax.set_title(f"{next(letters)}  {titles[p]}")
    ax.set_ylabel("fraction of fragments per log$_{10}$ bp" if j == 0 else "")
    plt.setp(ax.get_xticklabels(), visible=False)
    ymax = np.nanmax(dens) * 3; ax.set_ylim(np.nanmin(dens[dens > 0]) / 2, ymax)
    ax.text(np.sqrt(256 * 650), ymax / 1.3, "256–650 bp", ha="center", va="top", fontsize=7.5, color="#555555")
    ax.text(np.sqrt(2000 * 5000), ymax / 1.3, ">2 kb", ha="center", va="top", fontsize=7.5, color="#555555")
    # >5 kb side panel: share of all fragments (different unit)
    axs.set_xlim(-0.5, 0.5); axs.set_xticks([0]); axs.set_xticklabels([">5 kb"])
    axs.tick_params(axis="y", which="both", left=False, labelleft=False)
    axs.spines["left"].set_visible(False)
    axs.set_title("share of\nall frags", fontsize=7, loc="center", color="#555555")
    # ratio row
    medH = np.median(dens[(g == "HCC").to_numpy()], axis=0)
    medB = np.median(dens[(g == "HBV").to_numpy()], axis=0)
    axr.plot(centres, medH / medB, color="#333333", lw=1.4)
    axr.axhline(1, color="#999999", lw=0.8, ls="--")
    axr.set_yscale("log"); axr.set_xscale("log")
    axr.set_xlabel("fragment length (bp, log scale)")
    axr.set_ylabel("HCC / HBV\n(ratio of group medians)" if j == 0 else "")
    axr.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
    axr.set_xticklabels(["50", "100", "200", "500", "1000", "2000", "5000"])
    axr.set_yticks([0.2, 0.5, 1, 2]); axr.set_yticklabels(["0.2", "0.5", "1", "2"])
    axr.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axr.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    r5 = np.median(gt5[(g == "HCC").to_numpy()]) / np.median(gt5[(g == "HBV").to_numpy()])
    axrs.scatter([0], [r5], s=18, color="#333333", zorder=3)
    axrs.axhline(1, color="#999999", lw=0.8, ls="--")
    axrs.set_xlim(-0.5, 0.5); axrs.set_xticks([0]); axrs.set_xticklabels([">5 kb"]); axrs.tick_params(axis="x", pad=3)
    axrs.tick_params(axis="y", which="both", left=False, labelleft=False); axrs.spines["left"].set_visible(False)
    for a in (ax, axr):
        a.grid(axis="y", color="#EEEEEE", lw=0.6, which="major"); a.set_axisbelow(True)
# relabel lower panels C, D
fig.axes[2].set_title("C  Group median ratio, ONT")
fig.axes[6].set_title("D  Group median ratio, PacBio")
fig.axes[0].set_title("A  ONT, direct fragments"); fig.axes[4].set_title("B  PacBio, direct fragments")
h = [Line2D([], [], color=HCC, lw=1.5, label=f"HCC (n={(group=='HCC').sum()})"),
     Line2D([], [], color=HBV, lw=1.5, label=f"HBV carriers (n={(group=='HBV').sum()})")]
fig.axes[4].legend(handles=h, loc="lower left", fontsize=8)
# KS distance between the group CDFs (mean of per-sample CDFs), exact permutation p; from the KS size-profile table
ks = pd.read_csv(A.ks, sep="\t")
for axi, plat in ((0, "ont"), (4, "pacbio")):
    r = ks[(ks.platform == plat) & (ks.fragments == "direct") & (ks["range"] == "all lengths")].iloc[0]
    fig.axes[axi].text(0.98, 0.04, f"KS D = {r.D:.3f} at {r.at_bp} bp\npermutation p = {r.perm_p:.3f}",
                       transform=fig.axes[axi].transAxes, ha="right", va="bottom", fontsize=7.5, color="#333333",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
fig.savefig(f"{OUT}/fig4_size_profiles.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Figure 3.17: size ratio R ----------------
R = pd.DataFrame({"ont": ratio_R(H["ont"]), "pacbio": ratio_R(H["pacbio"])})
R["group"] = group.loc[R.index]
fig = plt.figure(figsize=(9.5, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.75, 0.75], wspace=0.45)
ax = fig.add_subplot(gs[0])
for grp in ["HBV", "HCC"]:
    s = R[R.group == grp]
    ax.scatter(s.ont, s.pacbio, s=34, color=GCOL[grp], edgecolor="white", linewidth=0.7, zorder=3,
               label=f"{'HBV carriers' if grp=='HBV' else 'HCC'} (n={len(s)})")
for smp, (dx, dy) in {"TBR3013": (-8, 6), "TBR3976": (6, -10)}.items():
    ax.annotate(smp, (R.loc[smp, "ont"], R.loc[smp, "pacbio"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, color="#333333")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("R, ONT direct (256–650 bp / >2 kb)"); ax.set_ylabel("R, PacBio direct")
rho, prho = spearmanr(R.ont, R.pacbio)
rhoH, prhoH = spearmanr(R[R.group == "HCC"].ont, R[R.group == "HCC"].pacbio)
ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f} (all 14)\nwithin HCC ρ = {rhoH:.2f}",
        transform=ax.transAxes, va="top", fontsize=8, color="#333333")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=8)
ax.set_xticks([100, 200, 500]); ax.set_xticklabels(["100", "200", "500"])
ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
for aa in (ax.xaxis, ax.yaxis): aa.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title("A  Same sample order on both platforms")
ax.grid(color="#EEEEEE", lw=0.6, which="major"); ax.set_axisbelow(True)
stats = {}
rng = np.random.default_rng(1)
for k, (p, lab) in enumerate([("ont", "B  ONT direct"), ("pacbio", "C  PacBio direct")]):
    a = fig.add_subplot(gs[k + 1])
    for i, grp in enumerate(["HBV", "HCC"]):
        v = R[R.group == grp][p]
        x = i + rng.uniform(-0.12, 0.12, len(v))
        a.scatter(x, v, s=26, color=GCOL[grp], edgecolor="white", linewidth=0.6, zorder=3)
        a.hlines(np.median(v), i - 0.25, i + 0.25, color="#333333", lw=1.2, zorder=4)
        for smp in ["TBR3013", "TBR3976"]:
            if smp in v.index:
                a.annotate(smp, (x[list(v.index).index(smp)], v[smp]), xytext=(7, 2 if v[smp] == max(v.get("TBR3013"), v.get("TBR3976")) else -9),
                           textcoords="offset points", fontsize=6.5, color="#555555")
    pv = mannwhitneyu(R[R.group == "HCC"][p], R[R.group == "HBV"][p]).pvalue
    stats[p] = pv
    a.set_yscale("log"); a.set_xticks([0, 1]); a.set_xticklabels(["HBV", "HCC"]); a.set_xlim(-0.6, 1.9)
    a.set_ylabel("R (log scale)"); a.set_title(lab)
    a.set_xlabel(f"Mann–Whitney p = {pv:.2g}", fontsize=7.5, color="#333333")
    yt = [70, 100, 200, 400, 700] if p == "ont" else [0.7, 1, 3, 10, 30]
    a.set_yticks(yt); a.set_yticklabels([f"{t:g}" for t in yt]); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.grid(axis="y", color="#EEEEEE", lw=0.6); a.set_axisbelow(True)
fig.savefig(f"{OUT}/fig4_size_ratio_R.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# cross-check R from the histograms against the periodicity table and the size-bin counts
chk = per[per.run == "ont_direct"].set_index("sample")["R"]
cmp = pd.DataFrame({"R_hist_ont_direct": R.ont, "R_column_ont_direct_rows": chk,
                    "R_hist_pacbio_direct": R.pacbio})
pb_all = ratio_R(load("pacbio", "all"))
cmp["R_hist_pacbio_all"] = pb_all
binc = pd.read_csv(A.bin_counts, sep="\t").pivot(index="sample", columns="size_bin", values="n_direct")
cmp["R_bincounts_ont_direct"] = (binc["256-420bp"] + binc["421-650bp"]) / (binc["2001-5000bp"] + binc[">5000bp"])
print(cmp.round(2).to_string())
print("MW p", stats, "rho", rho, prho, "rhoHCC", rhoH, prhoH)
print("ONT direct HCC range", R[R.group=="HCC"].ont.min(), R[R.group=="HCC"].ont.max(),
      "HBV", R[R.group=="HBV"].ont.min(), R[R.group=="HBV"].ont.max())

# ---------------- Figure 3.20: cross-platform summary (values typed from Table 3.11) ----------------
fig = plt.figure(figsize=(10.5, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.75, wspace=0.95)

def mk(ax, y, v, plat, filled=True, size=40):
    c = ONTC if plat == "ONT" else PBC
    m = "o" if plat == "ONT" else "s"
    ax.scatter([v], [y], marker=m, s=size, color=c if filled else "white", edgecolor=c, linewidth=1.2, zorder=3)

def setup(ax, labels, title, xlabel, zero=0.0, log=False):
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.axvline(zero, color="#999999", lw=0.8, ls="--", zorder=1)
    if log: ax.set_xscale("log")
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.6); ax.set_axisbelow(True)

# A: fold change of medians by size bin
ax = fig.add_subplot(gs[0, 0])
rows = [("256–420 bp", 6.2, 6.6), ("2001–5000 bp", 1.4, 1.06)]
for i, (l, o, p) in enumerate(rows):
    ax.plot([o, p], [i, i], color="#CCCCCC", lw=1, zorder=2); mk(ax, i, o, "ONT"); mk(ax, i, p, "PacBio")
setup(ax, [r[0] for r in rows], "A  Fragment counts, HCC / HBV", "ratio of group medians (log)", zero=1, log=True)
ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1", "2", "5", "10"]); ax.set_xlim(0.8, 10)

# B: R ranges per group (different absolute scales per platform)
ax = fig.add_subplot(gs[0, 1:])
rr = [("ONT direct", "HBV", 77, 148), ("ONT direct", "HCC", 250, 696),
      ("PacBio", "HBV", 0.77, 1.40), ("PacBio", "HCC", 1.52, 30.1)]
for i, (pl, grp, a, b) in enumerate(rr):
    ax.plot([a, b], [i, i], color=GCOL[grp], lw=3, solid_capstyle="round", zorder=3)
    ax.text(b * 1.15, i, f"{grp} {a:g}–{b:.2f}" if b < 10 else f"{grp} {a:g}–{b:g}", va="center", fontsize=7.5, color="#333333")
setup(ax, [f"{pl}, {g}" for pl, g, _, _ in rr], "B  Size ratio R (256–650 bp / >2 kb), range per group",
      "R (log scale); absolute values differ by platform", zero=1, log=True)
ax.set_xlim(0.5, 3000)

# C: H1 entropy d
ax = fig.add_subplot(gs[1, 0])
rows = ["genome-wide, direct", "genome-wide, all", "exon / promoter"]
mk(ax, 0, -3.10, "ONT"); mk(ax, 0, -1.80, "PacBio")
mk(ax, 1, -3.27, "ONT"); mk(ax, 1, -1.80, "PacBio")
ax.plot([-3.6, -2.9], [2, 2], color=ONTC, lw=2.5, solid_capstyle="round", zorder=3)
ax.plot([-2.09, -1.66], [2.0, 2.0], color=PBC, lw=2.5, solid_capstyle="round", zorder=3)
setup(ax, rows, "C  H1 entropy, 1001–2000 bp", "Cohen's d (rarefied), HCC − HBV")
ax.set_xlim(-4, 0.3)

# D: motif CLR differences, direct
ax = fig.add_subplot(gs[1, 1])
mrows = [("H3 CCAG, 601–2000", 0.20, 0.31, False), ("H3 CCCA, 2001–5000", 0.27, 0.44, False),
         ("H3 CCAG, 2001–5000", 0.37, 0.49, False), ("H4 CCAG, promoter", 0.26, 0.37, True),
         ("H4 GGAG, promoter", 0.20, 0.29, True)]
for i, (l, o, p, sig) in enumerate(mrows):
    ax.plot([o, p], [i, i], color="#CCCCCC", lw=1, zorder=2); mk(ax, i, o, "ONT", filled=sig); mk(ax, i, p, "PacBio")
setup(ax, [r[0] for r in mrows], "D  End-motif CLR, direct", "CLR difference, HCC − HBV")
ax.set_xlim(-0.05, 0.6)

# E: fsr excess over matched random promoters
ax = fig.add_subplot(gs[1, 2])
frows = [("HCC-expressed, all", 0.17, 0.06), ("HCC-expressed, direct", 0.12, 0.06),
         ("S5 high tertile, all", 0.50, 0.26), ("S5 high tertile, direct", 0.44, 0.24)]
for i, (l, o, p) in enumerate(frows):
    ax.plot([o, p], [i, i], color="#CCCCCC", lw=1, zorder=2); mk(ax, i, o, "ONT"); mk(ax, i, p, "PacBio")
setup(ax, [r[0] for r in frows], "E  Promoter fsr excess", "excess over matched random sets (d)")
ax.set_xlim(-0.05, 0.6)

h = [Line2D([], [], marker="o", ls="", color=ONTC, markersize=6, label="ONT"),
     Line2D([], [], marker="s", ls="", color=PBC, markersize=6, label="PacBio"),
     Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ONTC, markersize=6,
            label="ONT, not significant on direct\n(significant on all fragments)")]
fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
fig.savefig(f"{OUT}/fig4_cross_platform_summary.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
