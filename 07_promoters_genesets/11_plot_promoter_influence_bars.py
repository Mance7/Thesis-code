#!/usr/bin/env python3
"""
11_plot_promoter_influence_bars.py

Promoter influence (08_promoter_contribution_rank.py output, ONT) as bar plots, one figure per metric (fsr, entropy).
Bar = signed leave-one-out contribution = d_full - d_without_promoter (Cohen's d units, x1000),
so bars above zero push HCC higher, bars below zero push HCC lower.
Shown per run (all / direct): the --top promoters that carry the pooled difference and the --against promoters
that push the opposite way. Gene symbols from the annotated top-100 lists (results/promoter_symbols).

  python 11_plot_promoter_influence_bars.py --runs "all=promoter_rank_all,direct=promoter_rank_direct" \
      --symbols "results/promoter_symbols" -o promoter_influence --top 20 --against 10 --threads 1
"""
import argparse, glob, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HCC, HBV, INK, MUTE = "#E8703A", "#2F7BD8", "#222222", "#6B6B6B"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 250, "axes.titlelocation": "left"})
MET = {"fsr": ("FSR (share ≤246 bp among ≤600 bp)", "influence_fsr"),
       "entropy_short": ("Length entropy (≤600 bp)", "influence_entropy_short")}
SET_LAB = {"S1": "blood-silent", "S5": "high HCC tertile", "S6": "liver-enriched", "X1": "blood-silent, not liver"}


def pooled_d(run_dir):
    return {l.split(":")[0].split()[-1]: float(l.split("=")[1]) for l in open(os.path.join(run_dir, "run_info.txt")) if l.startswith("pooled")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="NAME=promoter_rank folder,NAME=folder")
    ap.add_argument("--symbols", required=True, help="folder with top_promoters_*_annotated.tsv (gene -> symbol)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--top", type=int, default=20); ap.add_argument("--against", type=int, default=10)
    ap.add_argument("-t", "--threads", type=int, default=1, help="kept for the project convention; plotting only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sym = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(a.symbols, "top_promoters_*_annotated.tsv"))])
    sym = sym.drop_duplicates("gene").set_index("gene").symbol
    runs = [r.split("=", 1) for r in a.runs.split(",")]
    tabs = []
    for metric, (mlab, col) in MET.items():
        fig, axes = plt.subplots(1, len(runs), figsize=(5.0 * len(runs), 0.2 * (a.top + a.against) + 2.0), gridspec_kw=dict(wspace=0.55))
        for ax, (name, d) in zip(np.atleast_1d(axes), runs):
            t = pd.read_csv(os.path.join(d, "promoter_influence.csv"))
            d0 = pooled_d(d)[metric]
            t["contribution_x1000"] = np.sign(d0) * t[col] * 1e3      # d_full - d_without, signed HCC - HBV
            t["symbol"] = t.gene.map(sym).fillna(t.gene.str.replace(r"\.\d+$", "", regex=True))
            pick = pd.concat([t.nlargest(a.top, col), t.nsmallest(a.against, col).iloc[::-1]])
            tabs.append(pick.assign(metric=metric, run=name, pooled_d=d0))
            y = np.arange(len(pick))[::-1]
            c = [HCC if v > 0 else HBV for v in pick.contribution_x1000]
            ax.barh(y, pick.contribution_x1000, color=c, height=0.72)
            ax.axvline(0, color=INK, lw=0.8)
            ax.axhline(y[a.top - 1] - 0.5, color=MUTE, lw=0.6, ls="--")
            lab = [f"{s}{' (S1)' if r.in_S1 else ''}{' (S6)' if r.in_S6 else ''}" for s, (_, r) in zip(pick.symbol, pick.iterrows())]
            ax.set_yticks(y, lab, fontsize=7)
            ax.set_xlabel("contribution to HCC − HBV difference (Cohen's d × 1000)")
            ax.set_title(f"ONT {name} fragments · pooled d = {d0:+.2f}", fontsize=9, pad=16)
            lim = np.abs(pick.contribution_x1000).max() * 1.1; ax.set_xlim(-lim, lim)
            ax.text(0.0, 1.005, "← lower in HCC", transform=ax.transAxes, color=HBV, fontsize=7.5, va="bottom")
            ax.text(1.0, 1.005, "higher in HCC →", transform=ax.transAxes, color=HCC, fontsize=7.5, va="bottom", ha="right")
            e = pd.read_csv(os.path.join(d, "set_enrichment.csv")); e = e[e.metric == metric]
            ax.text(-0.45, -0.16, "Gene sets, mean influence vs 3,003 relabellings:\n" +
                    "\n".join(f"  {r.test_set} {SET_LAB.get(r.test_set, '')}: q {r.q_bh:.3f}" for _, r in e.iterrows()),
                    transform=ax.transAxes, fontsize=6.8, color=MUTE, va="top")
        fig.suptitle(f"Promoter influence, {mlab}, HCC-expressed promoters (ONT). Top {a.top} carry the pooled difference; "
                     f"below the\ndashed line, {a.against} push the other way. (S1) blood-silent, (S6) liver-enriched; Ensembl ID where no symbol is in the local tables", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
        fig.savefig(os.path.join(a.out, f"promoter_influence_bars_{'fsr' if metric == 'fsr' else 'entropy'}.png"), bbox_inches="tight")
        plt.close(fig)
    pd.concat(tabs).to_csv(os.path.join(a.out, "promoter_influence_bars_values.csv"), index=False)


if __name__ == "__main__":
    main()
