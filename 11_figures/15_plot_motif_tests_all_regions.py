#!/usr/bin/env python3
"""
Full views of the end-motif test output of 06_end_motifs/05_motif_reduced_family_stats.py (motif_tests_all.csv):
every motif, every size bin, every region. Supplementary views behind Figures 3.6 and 3.7, which show only the
genome-wide cell; these plots are not numbered thesis figures.
  volcano_<run>.png       : rows = regions, columns = size bins; x = CLR diff (HCC - HBV), y = -log10 p (moderated t)
                            colour = global BH q < 0.05 (padj_global_modt); ring = hierarchical-significant (hier_sig)
  grid_<run>_<region>.png : 256 motifs as a 16 x 16 grid per bin (row = bases 1-2, column = bases 3-4)
One or more runs:  --run LABEL=path/to/motif_tests_all.csv  (repeatable). Output to a new -o folder.

  python 15_plot_motif_tests_all_regions.py --run "ONT_direct=results/motifs/ont_direct/motif_tests_all.csv" \
      --run "PacBio_direct=results/motifs/pacbio_direct/motif_tests_all.csv" \
      -o "results/figures/motif_tests" --threads 4
"""
import argparse, itertools, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HCC, HBV, GREY = "#E8703A", "#2F7BD8", "#B8B8B8"
LIT = ["CCCA", "CCAG", "CCTG", "AAAA", "TAAA", "TTTT"]
REG_ORDER = ["genome_wide", "is_promoter", "is_tss", "is_gene_body", "is_exon", "is_cpg_island"]
DI = ["".join(p) for p in itertools.product("ACGT", repeat=2)]


def bin_key(b):
    """sort key for size-bin labels such as <=600bp, 601-2000bp, >5000bp"""
    s = str(b).replace("bp", "").replace("<=", "").replace(">", "").replace("<", "")
    lo = s.split("-")[0]
    return (1 if str(b).startswith(">") else 0, float(lo) if lo.replace(".", "").isdigit() else 0)


def plot_run(label, path, out, q):
    """volcano grid and per-region motif grids for one run"""
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "axes.axisbelow": True,
                         "axes.titlelocation": "left", "savefig.dpi": 220})
    r = pd.read_csv(path)
    bins = sorted(r.size_bin.unique(), key=bin_key)
    regs = [g for g in REG_ORDER if g in set(r.region)] + sorted(set(r.region) - set(REG_ORDER))
    xl = np.nanmax(np.abs(r.clr_diff)) * 1.05; yl = np.nanmax(-np.log10(r.p_modt)) * 1.08
    fig, axes = plt.subplots(len(regs), len(bins), figsize=(2.5 * len(bins), 2.2 * len(regs) + 0.6), squeeze=False)
    for i, g in enumerate(regs):
        for j, b in enumerate(bins):
            ax, s = axes[i, j], r[(r.region == g) & (r.size_bin == b)]
            c = np.where(s.padj_global_modt < q, np.where(s.clr_diff > 0, HCC, HBV), GREY)
            ax.scatter(s.clr_diff, -np.log10(s.p_modt), s=7, c=c, linewidths=0)
            h = s[s.get("hier_sig", False) == True]
            ax.scatter(h.clr_diff, -np.log10(h.p_modt), s=16, facecolors="none", edgecolors="#222", linewidths=0.5)
            for _, x in s[s.motif.isin(LIT)].iterrows():
                ax.annotate(x.motif, (x.clr_diff, -np.log10(x.p_modt)), fontsize=5.5, xytext=(2, 1), textcoords="offset points")
            ax.axvline(0, color="#999", lw=0.5); ax.set_xlim(-xl, xl); ax.set_ylim(0, yl)
            up = ((s.padj_global_modt < q) & (s.clr_diff > 0)).sum(); dn = ((s.padj_global_modt < q) & (s.clr_diff < 0)).sum()
            ax.set_title(f"{g.replace('is_', '')} · {b}  ↑{up} ↓{dn}", fontsize=7.5)
            if j == 0: ax.set_ylabel("−log10 p")
            if i == len(regs) - 1: ax.set_xlabel("CLR diff, HCC − HBV")
    fig.suptitle(f"{label}: all motifs, all bins, all regions  (orange/blue = global q < {q}; ring = hierarchical-significant)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(f"{out}/volcano_{label}.png", bbox_inches="tight"); plt.close(fig)
    v = np.nanmax(np.abs(r.clr_diff)); norm = TwoSlopeNorm(0, -v, v)
    for g in regs:
        fig, axes = plt.subplots(1, len(bins), figsize=(2.9 * len(bins), 3.2), squeeze=False)
        t = r[r.region == g].set_index(["size_bin", "motif"])
        for j, b in enumerate(bins):
            ax = axes[0, j]
            M = np.array([[t.clr_diff.get((b, x + y), np.nan) for y in DI] for x in DI])
            S = np.array([[t.padj_global_modt.get((b, x + y), 1) < q for y in DI] for x in DI])
            im = ax.imshow(M, cmap="PuOr_r", norm=norm)
            yy, xx = np.where(S); ax.scatter(xx, yy, s=3, c="#111", linewidths=0)
            ax.set_xticks(range(16), DI, fontsize=5, rotation=90); ax.set_yticks(range(16), DI, fontsize=5); ax.grid(False)
            for k in (3.5, 7.5, 11.5): ax.axhline(k, color="white", lw=1); ax.axvline(k, color="white", lw=1)
            ax.set_title(b, fontsize=8)
        cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label("CLR diff (orange = HCC higher)")
        fig.suptitle(f"{label} · {g}: 256 motifs (rows bases 1–2, columns 3–4; dot = q < {q}; grey = filtered out)", x=0.01, ha="left")
        fig.savefig(f"{out}/grid_{label}_{g}.png", bbox_inches="tight"); plt.close(fig)
    return label


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Volcano and grid views of every motif test, all regions and bins.")
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=path to motif_tests_all.csv from the motif statistics script (repeat for each run)")
    ap.add_argument("-o", "--out", required=True, help="output folder for the PNG files (created if missing)")
    ap.add_argument("--q", type=float, default=0.05, help="BH q threshold for colouring (default 0.05)")
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel processes, one run each")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = [x.split("=", 1) for x in a.run]
    with ProcessPoolExecutor(a.threads) as ex:
        for done in ex.map(plot_run, [l for l, _ in runs], [p for _, p in runs], [a.out] * len(runs), [a.q] * len(runs)):
            print("plotted", done)
