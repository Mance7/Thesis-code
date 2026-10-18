#!/usr/bin/env python3
"""Draw the analysis workflow and four result figures that come straight from per-sample tables.

Figures:
  Figure 2.1   fig3_workflow.png           analysis workflow (schematic)
  Figure 3.2   fig4_entropy_per_sample.png rarefied length entropy per sample, 1001-2000 bp (H1) and 151-255 bp (H2), ONT and PacBio direct
  Figure 3.5   fig4_end_motifs.png         CLR difference of the literature C-end motifs (CCCA, CCAG, CCTG) per merged length bin
  Figure 3.10  fig4_geneset_null.png       fsr excess at HCC-expressed promoters over matched random promoter sets
  Figure 3.23  fig4_tf_vs_R.png            ichorCNA tumour fraction against the size ratio R

Inputs (see --help): the per-sample periodicity table (used to check the HCC/HBV grouping rule),
the rarefied entropy feature tables (ONT and PacBio, direct fragments), the ONT pre-specified
motif family results, and the per-sample tumour fraction / R table. PacBio values of Figure 3.5 and
all values of Figure 3.10 are typed in from the results tables (Tables 3.4, 3.9, 3.11).
Output: PNG files (300 dpi) in the --out folder; a few check values are printed.

Usage:
  python 02_plot_workflow_entropy_motifs_geneset_null_tf.py --periodicity results/length_hist/periodicity_per_sample.tsv \
      --ont-feats results/features/ont --pacbio-feats results/features/pacbio \
      --ont-motif-families results/fig_inputs/ont_motif_prespecified_families_direct.csv \
      --tf-table results/tumour_fraction/tf_vs_R_per_sample.tsv -o figures
  (put function names first, e.g. fig_tf_vs_R, to draw only those)
"""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr, mannwhitneyu

A = None     # command-line arguments (input tables), set in main()
OUT = None   # output folder, set from --out
HCC, HBV = "#E8703A", "#2F7BD8"   # house style (orange/blue), colour-blind safe pair
plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 300,
})

def group_of(s):
    """HBV carriers are the TBR44xx samples, all others are HCC"""
    return "HBV" if s.startswith("TBR44") else "HCC"

def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp

# ---------------- Figure 3.2: entropy per sample ----------------
def fig_entropy_per_sample():
    files = {"ONT": os.path.join(A.ont_feats, "direct", "F_entropy_rare.csv"),
             "PacBio": os.path.join(A.pacbio_feats, "direct", "F_entropy_rare.csv")}
    # d and q annotations as reported in the thesis (Table 3.2, Section 3.7.3, Table 3.11)
    ann = {("ONT", "ent_1001-2000bp"): "d = −3.10, q = 0.007",
           ("ONT", "ent_151-255bp"): "d = +2.16, q = 0.017",
           ("PacBio", "ent_1001-2000bp"): "d = −1.80, q = 0.013",
           ("PacBio", "ent_151-255bp"): "d = +1.00, p = 0.14 (n.s.)"}
    bins = [("ent_1001-2000bp", "1001–2000 bp"), ("ent_151-255bp", "151–255 bp")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    rng = np.random.default_rng(3)
    letters = iter("ABCD")
    for i, (plat, f) in enumerate(files.items()):
        d = pd.read_csv(f)
        d["group"] = d["sample"].map(group_of)
        for j, (col, lab) in enumerate(bins):
            ax = axes[i, j]
            for k, (g, c) in enumerate([("HBV", HBV), ("HCC", HCC)]):
                v = d.loc[d.group == g, col].values
                s = d.loc[d.group == g, "sample"].values
                x = k + rng.uniform(-0.12, 0.12, len(v))
                ax.scatter(x, v, s=30, color=c, edgecolor="white", linewidth=0.5, zorder=3)
                ax.hlines(np.median(v), k-0.25, k+0.25, color="#333333", lw=1.4, zorder=4)
                for xx, vv, ss in zip(x, v, s):
                    if ss == "TBR3976":
                        ax.annotate("TBR3976", (xx, vv), xytext=(6, 0), textcoords="offset points",
                                    fontsize=7, color="#444444", va="center")
            a = d.loc[d.group == "HCC", col]; b = d.loc[d.group == "HBV", col]
            print("Figure 3.2", plat, col, "d from file", round(cohen_d(a, b), 2), "MWU p", mannwhitneyu(a, b).pvalue)
            ax.set_xticks([0, 1], ["HBV (n=6)", "HCC (n=8)"])
            ax.set_xlim(-0.6, 1.8)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{next(letters)}  {plat} direct: {lab}", fontsize=10)
            lo, hi = d[col].min(), d[col].max(); pad = (hi - lo)
            ax.set_ylim(lo - 0.06*pad, hi + 0.22*pad)
            ax.text(0.98, 0.97, ann[(plat, col)], transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="#222222")
            if j == 0:
                ax.set_ylabel("rarefied normalised entropy")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_entropy_per_sample.png")
    plt.close(fig)

# ---------------- Figure 3.5: literature C-end motifs ----------------
def fig_end_motifs():
    d = pd.read_csv(A.ont_motif_families)
    d = d[(d.family == "F1_literature") & d.motif.isin(["CCCA", "CCAG", "CCTG"])]
    bins = ["<=600bp", "601-2000bp", "2001-5000bp", ">5000bp"]
    blab = ["≤600", "601–\n2000", "2001–\n5000", ">5000"]
    motifs = ["CCCA", "CCAG", "CCTG"]
    ont_v = np.full((3, 4), np.nan); ont_q = np.full((3, 4), np.nan)
    for _, r in d.iterrows():
        i, j = motifs.index(r.motif), bins.index(r.size_bin)
        ont_v[i, j], ont_q[i, j] = r.clr_diff, r.padj_family
    # PacBio direct, typed from the PacBio end-motif results (F1 literature family; same values as Table 3.11)
    pb_v = np.array([[0.16, 0.32, 0.44, 0.32],
                     [0.16, 0.31, 0.49, 0.62],
                     [0.20, 0.31, 0.30, 0.03]])
    pb_q = np.array([[0.001, 0.0003, 0.0002, 0.046],
                     [0.002, 0.0004, 0.0005, 0.0006],
                     [0.0004, 0.0002, 0.0006, 0.89]])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    cmap = plt.get_cmap("PuOr_r")
    vmax = 0.65
    for ax, v, q, title in [(axes[0], ont_v, ont_q, "A  ONT direct"), (axes[1], pb_v, pb_q, "B  PacBio direct")]:
        m = np.ma.masked_invalid(v)
        cm = cmap.copy(); cm.set_bad("#EEEEEE")
        im = ax.imshow(m, cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(3):
            for j in range(4):
                if np.isnan(v[i, j]):
                    ax.text(j, i, "not\ntestable", ha="center", va="center", fontsize=7, color="#777777")
                    continue
                sig = q[i, j] < 0.05
                qs = f"q {q[i,j]:.2f}" if q[i, j] >= 0.1 else f"q {q[i,j]:.3f}" if q[i, j] >= 0.01 else f"q {q[i,j]:.3f}" if q[i, j] >= 0.001 else f"q {q[i,j]:.0e}".replace("e-0", "e−")
                ax.text(j, i - 0.12, f"{v[i,j]:+.2f}".replace("-", "−") + ("*" if sig else ""), ha="center", va="center",
                        fontsize=8.5, fontweight="bold" if sig else "normal",
                        color="white" if abs(v[i, j]) > 0.42 else "#222222")
                ax.text(j, i + 0.25, qs, ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v[i, j]) > 0.42 else "#555555")
        ax.set_xticks(range(4), blab); ax.set_yticks(range(3), motifs)
        ax.set_xlabel("merged length bin (bp)")
        ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("CLR difference, HCC − HBV")
    cb.outline.set_visible(False)
    fig.savefig(f"{OUT}/fig4_end_motifs.png", bbox_inches="tight")
    plt.close(fig)

# ---------------- Figure 3.10: gene-set null ----------------
def fig_geneset_null():
    sets = ["HCC-expressed, full (11,333)", "S1 blood-silent (2,100)", "S2 blood-expressed (9,157)",
            "S3 low tertile (3,767)", "S4 mid tertile (3,780)", "S5 high tertile (3,786)",
            "S6 liver-enriched (398)"]
    # fsr excess (|obs d| − |median random d|) and q (full set: empirical p). Table 3.4 (ONT), Table 3.9 (PacBio)
    ont = {"all": ([0.17, 0.54, 0.06, 0.05, -0.04, 0.50, 0.95], [0.001, 0.002, 0.042, 0.39, 0.65, 0.002, 0.002]),
           "direct": ([0.12, 0.39, 0.03, -0.02, -0.08, 0.44, 0.45], [0.003, 0.007, 0.34, 0.65, 0.73, 0.007, 0.077])}
    pb = {"all": ([0.06, 0.06, 0.05, -0.11, 0.04, 0.26, 0.14], [0.005, 0.37, 0.019, 0.94, 0.37, 0.007, 0.37]),
          "direct": ([0.06, 0.06, 0.06, -0.09, 0.04, 0.24, 0.12], [0.004, 0.35, 0.007, 0.91, 0.35, 0.007, 0.35])}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    y = np.arange(len(sets))[::-1]
    cols = {"all": "#A9A9A9", "direct": "#3C3C3C"}
    for ax, dat, title in [(axes[0], ont, "A  ONT"), (axes[1], pb, "B  PacBio")]:
        for k, run in enumerate(["all", "direct"]):
            ex, q = dat[run]
            yy = y + (0.19 if run == "all" else -0.19)
            ax.barh(yy, ex, height=0.36, color=cols[run], label=f"{run} fragments" if run == "all" else "direct fragments")
            for e, qq, yv, i in zip(ex, q, yy, range(len(ex))):
                lab = ("p " if i == 0 else "q ") + f"{qq:g}"
                sig = qq < 0.05
                xpos = max(e, 0) + 0.02
                ax.text(xpos, yv, lab + (" *" if sig else ""), va="center", fontsize=6.8,
                        color="#111111" if sig else "#888888", fontweight="bold" if sig else "normal")
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlim(-0.2, 1.3)
        ax.set_xlabel("fsr excess over matched random sets\n(|observed d| − |median random d|)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].set_yticks(y, sets)
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_geneset_null.png")
    plt.close(fig)

# ---------------- Figure 3.23: tumour fraction vs R ----------------
def fig_tf_vs_R():
    d = pd.read_csv(A.tf_table, sep="\t")
    assert all(d["sample"].map(group_of) == d["group"])
    label = {"TBR3976", "TBR2967", "TBR3013", "TBR2872", "TBR2776", "TBR2897", "TBR3032"}
    hcc = d[d.group == "HCC"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    offs = {"ONT": {"TBR3976": (8, 0), "TBR2967": (8, 0), "TBR3013": (-52, 0), "TBR2872": (-10, 26),
                    "TBR2776": (-62, 14), "TBR2897": (-6, 40), "TBR3032": (-62, 16)},
            "PacBio": {"TBR3976": (8, 0), "TBR2967": (8, 0), "TBR3013": (8, -4), "TBR2872": (-62, 18),
                       "TBR2776": (-50, 36), "TBR2897": (2, 30), "TBR3032": (-58, 16)}}
    for ax, col, plat, letter in [(axes[0], "R_ont_direct", "ONT", "A"), (axes[1], "R_pacbio", "PacBio", "B")]:
        for g, c in [("HBV", HBV), ("HCC", HCC)]:
            s = d[d.group == g]
            ax.scatter(s[col], s["TF_ont_lowTF"], s=34, color=c, edgecolor="white", linewidth=0.5,
                       label=f"{g} (n={len(s)})", zorder=3)
        for _, r in d[d["sample"].isin(label)].iterrows():
            dx, dy = offs[plat][r["sample"]]
            far = abs(dx) > 12 or abs(dy) > 12
            ax.annotate(r["sample"], (r[col], r["TF_ont_lowTF"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.8, color="#333333", va="center",
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, shrinkA=1, shrinkB=3) if far else None)
        ax.axhline(0.03, color="#888888", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_ylim(-0.0008, 1.3)
        ax.set_yticks([0, 0.005, 0.01, 0.03, 0.1, 0.3], ["0", "0.005", "0.01", "0.03", "0.1", "0.3"])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rho, p = spearmanr(hcc[col], hcc["TF_ont_lowTF"])
        print("Figure 3.23", plat, "within-HCC rho", round(rho, 2), "p", round(p, 3))
        ax.text(0.02, 0.97, f"within HCC: Spearman ρ = {rho:.2f}, p = {p:.3f}".replace("-", "−"), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.set_xlabel(f"size ratio R, {plat}{' direct' if plat=='ONT' else ''} (256–650 bp / >2 kb, log scale)")
        ax.set_title(f"{letter}  ONT tumour fraction vs {plat} R", pad=6)
    axes[0].set_xticks([80, 100, 200, 300, 500, 700])
    axes[1].set_xticks([1, 2, 5, 10, 20, 30])
    axes[0].text(0.02, 0.034, "≈3% detection limit", transform=axes[0].get_yaxis_transform(), fontsize=6.8, color="#777777")
    axes[0].set_ylabel("tumour fraction (ichorCNA, ONT lowTF;\nsymmetric log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fig4_tf_vs_R.png")
    plt.close(fig)
    # check the 'extra cfDNA' values of Table 3.12 against the fragment counts in this file
    med = d.loc[d.group == "HBV", "ont_direct_fragments"].median()
    print("Figure 3.23 extra cfDNA ONT:", {r["sample"]: round(100*(1-med/r.ont_direct_fragments)) for _, r in hcc.iterrows()})
    medp = d.loc[d.group == "HBV", "pacbio_fragments"].median()
    print("Figure 3.23 extra cfDNA PacBio:", {r["sample"]: round(100*(1-medp/r.pacbio_fragments)) for _, r in hcc.iterrows()})

# ---------------- Figure 2.1: workflow ----------------
def fig_workflow():
    fig = plt.figure(figsize=(7.2, 9.0))
    ax = fig.add_axes([0.01, 0.005, 0.98, 0.99]); ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.axis("off")
    def box(x, y, w, h, title, body="", fc="#F3F3F3", ec="#9A9A9A", tsize=8.6, bsize=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        if body:
            ax.text(x + w/2, y + h - 1.4, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color="#1F1F1F")
            ax.text(x + w/2, y + h - 4.2, body, ha="center", va="top", fontsize=bsize, color="#333333", linespacing=1.35)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color="#1F1F1F")
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=9))
    ONTc, PBc, W = "#E7EEF8", "#EAF3E6", "#FFFFFF"
    box(25, 112, 50, 7, "Plasma cfDNA, same libraries", "8 HCC, 6 HBV carriers (Yu et al. 2023)", fc=W)
    arrow(40, 111.6, 26, 107.6); arrow(60, 111.6, 74, 107.6)
    box(3, 79, 46, 28, "ONT", "deposited basecalled FASTQ\nCutadapt (both adapters required)\nminimap2 2.31 → GRCh38\n"
        "primary, canonical chromosomes, MAPQ ≥ 20\nzero-clip rule\nconcatemer recovery (split ligated reads)", fc=ONTc, ec="#7F9CC4")
    box(51, 79, 46, 28, "PacBio", "CCS reads, ≥ 3 full passes (np ≥ 3)\nBWA-MEM → GRCh38\n"
        "duplicate removal (samtools markdup)\nMAPQ ≥ 30\nsame Stage 1: zero-clip rule,\nconcatemer recovery (~1% of fragments)", fc=PBc, ec="#8DB07F")
    box(5, 80.3, 20, 4.6, "direct fragments", fc=W, tsize=7.6)
    box(27, 80.3, 20, 4.6, "recovered fragments", fc=W, tsize=7.6)
    ax.text(26, 86.0, "confirmatory analyses use direct fragments", ha="center", fontsize=6.8, style="italic", color="#3D5A80")
    arrow(26, 78.6, 45, 74.9); arrow(74, 78.6, 55, 74.9)
    box(14, 66, 72, 8.5, "Problem-region filter (by fragment midpoint)",
        "ENCODE blacklist v2 · UCSC gap track · centromeres · satellite arrays", fc=W)
    arrow(50, 65.6, 50, 62.9)
    box(6, 53, 88, 9.5, "Size bins",
        "11 bins: <90, 90–125, 126–135, 136–150, 151–255, 256–420, 421–650, 651–1000, 1001–2000, 2001–5000, >5000 bp\n"
        "4 merged bins (end motifs): ≤600, 601–2000, 2001–5000, >5000 bp", fc=W, bsize=6.9)
    arrow(50, 52.6, 50, 50.2)
    ax.text(50, 48.9, "Three levels of resolution", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#1F1F1F")
    for x, t, b in [(3, "Region-pooled", "genome-wide, gene body, exon,\npromoter, TSS, CpG island"),
                    (35, "Individual promoters", "19,924 GENCODE v49\n(−2,000 to +500 bp)"),
                    (67, "Gene-set pooled", "HCC-expressed (11,333),\nsubsets S1–S6, Hallmark sets")]:
        box(x, 37, 30, 10, t, b, fc=W, bsize=7.0)
        arrow(x + 15, 36.6, x + 15, 33.9)
    box(3, 24, 94, 9.5, "Metrics",
        "rarefied length entropy  ·  4-mer end motifs (CLR)  ·  short-fragment ratio (fsr) and size ratio R (256–650 bp / >2 kb)\n"
        "motif diversity score (MDS; checked for fragment-count dependence)", fc=W, bsize=6.9)
    arrow(50, 23.6, 50, 20.9)
    box(3, 11, 94, 9.5, "Statistics",
        "Mann–Whitney U, Cohen's d, moderated t (motifs)  ·  Benjamini–Hochberg FDR within pre-specified families\n"
        "permutation nulls: 3,003 sample relabellings; 1,000 matched random promoter sets", fc=W, bsize=6.9)
    for x0 in (17, 50, 83):
        arrow(50, 10.6, x0, 8.9)
    for x, t, b in [(3, "Cross-platform replication", "same scripts on PacBio"),
                    (36, "Tumour fraction", "ichorCNA, 1 Mb bins,\ndirect fragments"),
                    (69, "Classifiers", "L2 logistic, LOO-CV,\n1,000-permutation tests")]:
        box(x, 0.5, 28, 8, t, b, fc="#F6EFE4", ec="#C8A777", tsize=8.0, bsize=6.8)
    fig.savefig(f"{OUT}/fig3_workflow.png")
    plt.close(fig)

FIGURES = (fig_workflow, fig_entropy_per_sample, fig_end_motifs, fig_geneset_null, fig_tf_vs_R)


def main():
    global A, OUT
    ap = argparse.ArgumentParser(description="Workflow (Figure 2.1) and Figures 3.2, 3.5, 3.10, 3.23.")
    ap.add_argument("figures", nargs="*", help="function names to run, e.g. fig_tf_vs_R (default: all)")
    ap.add_argument("--periodicity", required=True,
                    help="per-sample periodicity table with columns sample, group (tsv); used to check the grouping rule")
    ap.add_argument("--ont-feats", required=True,
                    help="ONT per-sample feature folder with all/ and direct/ subfolders (F_entropy_rare.csv)")
    ap.add_argument("--pacbio-feats", required=True,
                    help="PacBio per-sample feature folder with all/ and direct/ subfolders (F_entropy_rare.csv)")
    ap.add_argument("--ont-motif-families", required=True,
                    help="ONT direct results of the pre-specified motif families (csv: family, motif, size_bin, clr_diff, padj_family)")
    ap.add_argument("--tf-table", required=True,
                    help="per-sample tumour fraction, R and fragment counts (tsv: sample, group, R_ont_direct, R_pacbio, TF_ont_lowTF, ...)")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder for the PNG files (created if missing)")
    A = ap.parse_args()
    OUT = A.out
    os.makedirs(OUT, exist_ok=True)
    # check the grouping rule against the periodicity file
    per = pd.read_csv(A.periodicity, sep="\t")
    assert all(per["sample"].map(group_of) == per["group"]), "group rule mismatch"
    for f in FIGURES:
        if not A.figures or f.__name__ in A.figures:
            f()
    print("done")


if __name__ == "__main__":
    main()
