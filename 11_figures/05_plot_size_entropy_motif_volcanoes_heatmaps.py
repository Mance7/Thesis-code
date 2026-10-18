#!/usr/bin/env python3
"""
Full (non-selected) views of the region-pooled genome-wide results, HCC vs HBV, both platforms:
  end motifs : all 256 motifs x 4 merged bins (CLR, moderated t as in 06_end_motifs/05_motif_reduced_family_stats.py)
  size       : share of fragments in each of the 11 length bins
  entropy    : rarefied length entropy in each of the 11 length bins

Figures:
  Figure 3.3   fig4_size_entropy_volcano.png   size shares and length entropy, 11 bins, ONT and PacBio direct (exact MWU, BH over 11)
  Figure 3.4   fig4_size_entropy_heatmap.png   per-sample z-scores of the size shares and entropy, 11 bins
  Figure 3.6   fig4_motif_volcano.png          volcano of all 256 motifs x 4 bins, ONT and PacBio direct (BH over 1024 tests per run)
  Figure 3.7   fig4_motif_grid.png             HCC - HBV difference of all 256 motifs as a 16 x 16 grid per bin
  Figure 3.8   fig4_motif_per_sample.png       per-sample z-scores of all 1024 motif x bin values
  Figure 3.19  fig4_motif_ont_vs_pacbio.png    ONT against PacBio difference, every motif, per bin
  (not numbered) fig4_motif_volcano_ont_all.png, fig4_motif_grid_ont_all.png: the same views for ONT all fragments

Input: per-sample feature tables at full depth (the same tables the classifiers use), one folder per
platform with all/ and direct/ subfolders holding F_motif.csv, F_size.csv and F_entropy_rare.csv.
Output (-o, new folder): the PNGs above and all_tests_genome_wide.csv (every test, positive = HCC higher).

  python 05_plot_size_entropy_motif_volcanoes_heatmaps.py --ont-feats "results/features/ont" \
      --pacbio-feats "results/features/pacbio" -o "results/figures" --threads 4
"""
import argparse, importlib.util, os, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import mannwhitneyu

# moderated t, Cliff's delta and BH come from the motif statistics script (loaded in main() from --motif-stats)
HERE = os.path.dirname(os.path.abspath(__file__))
MOTIF_STATS = os.path.join(HERE, "..", "06_end_motifs", "05_motif_reduced_family_stats.py")
mrf = None


def load_motif_stats(path):
    """import the motif statistics script by path (its file name starts with a digit)"""
    spec = importlib.util.spec_from_file_location("mrf", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

HCC, HBV, GREY = "#E8703A", "#2F7BD8", "#B8B8B8"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 250,
})
BINS4 = ["<=600bp", "601-2000bp", "2001-5000bp", ">5000bp"]
BLAB4 = {"<=600bp": "≤600 bp", "601-2000bp": "601–2000 bp", "2001-5000bp": "2001–5000 bp", ">5000bp": ">5000 bp"}
BINS11 = ["<90bp", "90-125bp", "126-135bp", "136-150bp", "151-255bp", "256-420bp", "421-650bp",
          "651-1000bp", "1001-2000bp", "2001-5000bp", ">5000bp"]
LIT = ["CCCA", "CCAG", "CCTG", "AAAA", "TAAA", "TTTT"]
NT = "ACGT"
# (platform, fragment set, feature folder argument)
RUNS = [("ONT", "direct", "ont_feats"), ("ONT", "all", "ont_feats"), ("PacBio", "direct", "pacbio_feats")]
Q = 0.05


def is_hcc(idx):
    """True for HCC samples; HBV carriers are the TBR44xx samples"""
    return ~pd.Index(idx).str.startswith("TBR44")


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def motif_tests(path, plat, src):
    d = pd.read_csv(path).set_index("sample")
    y = is_hcc(d.index)
    out = []
    for b in BINS4:
        X = d[[c for c in d.columns if c.startswith(f"clr_{b}_")]]
        t, df, p, diff = mrf.moderated_t(X.values, y)
        mw = [mannwhitneyu(X.values[y, j], X.values[~y, j]).pvalue for j in range(X.shape[1])]
        cd = [mrf.cliffs_delta(X.values[y, j], X.values[~y, j]) for j in range(X.shape[1])]
        out.append(pd.DataFrame(dict(platform=plat, source=src, feature="end_motif", size_bin=b,
                                     motif=[c.rsplit("_", 1)[1] for c in X.columns], effect=diff,
                                     effect_type="CLR diff", mod_t=t, p=p, p_mwu=mw, cliffs_delta=cd)))
    r = pd.concat(out, ignore_index=True)
    r["q"] = mrf.bh(r.p.values)          # BH over all 1024 genome-wide motif x bin tests of this run
    return r


def bin_tests(path, plat, src, kind):
    d = pd.read_csv(path).set_index("sample")
    y = is_hcc(d.index)
    rows = []
    for b in BINS11:
        col = ("frac_" if kind == "size" else "ent_") + b
        if col not in d:
            continue
        a, c = d.loc[y, col].dropna().values, d.loc[~y, col].dropna().values
        if len(a) < 3 or len(c) < 3:
            continue
        eff = np.log2(a.mean() / c.mean()) if kind == "size" else cohen_d(a, c)
        rows.append(dict(platform=plat, source=src, feature=kind, size_bin=b, motif="", effect=eff,
                         effect_type="log2 FC of fraction" if kind == "size" else "Cohen's d",
                         p=mannwhitneyu(a, c, method="exact").pvalue, cliffs_delta=mrf.cliffs_delta(a, c)))
    r = pd.DataFrame(rows)
    r["q"] = mrf.bh(r.p.values)          # BH over the 11 bins
    return r, d


def colour(r):
    return np.where(r.q < Q, np.where(r.effect > 0, HCC, HBV), GREY)


def legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=HCC, label=f"HCC higher, q < {Q}"),
         Line2D([], [], marker="o", ls="", color=HBV, label=f"HBV higher, q < {Q}"),
         Line2D([], [], marker="o", ls="", color=GREY, label="not significant")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))


def volcano_motifs(res, runs, fn, title):
    fig, axes = plt.subplots(len(runs), 4, figsize=(11, 2.75 * len(runs) + 0.6), squeeze=False)
    xl = np.nanmax(np.abs(res.effect)) * 1.05
    for i, (plat, src) in enumerate(runs):
        rr = res[(res.platform == plat) & (res.source == src)]
        yl = np.nanmax(-np.log10(rr.p)) * 1.1
        for j, b in enumerate(BINS4):
            ax, r = axes[i, j], rr[rr.size_bin == b]
            lp = -np.log10(r.p)
            ax.scatter(r.effect, lp, s=9, c=colour(r), linewidths=0, alpha=0.85)
            ax.axvline(0, color="#999999", lw=0.6)
            # dashed line = p at which q crosses 0.05 in this run (if any test passes)
            sig = rr[rr.q < Q]
            if len(sig):
                ax.axhline(-np.log10(sig.p.max()), color="#888888", lw=0.6, ls="--")
            # label the six literature motifs (pre-specified) plus the 3 smallest p in this panel
            lab = set(LIT) | set(r.nsmallest(3, "p").motif)
            for _, x in r[r.motif.isin(lab)].iterrows():
                ax.annotate(x.motif, (x.effect, -np.log10(x.p)), fontsize=6.5, xytext=(3, 2),
                            textcoords="offset points", color="#222222" if x.motif in LIT else "#666666",
                            fontweight="bold" if x.motif in LIT else "normal")
            ax.set_xlim(-xl, xl); ax.set_ylim(0, yl)
            n_up, n_dn = ((r.q < Q) & (r.effect > 0)).sum(), ((r.q < Q) & (r.effect < 0)).sum()
            ax.set_title(f"{plat} {src} · {BLAB4[b]}   ↑{n_up} ↓{n_dn}")
            if i == len(runs) - 1: ax.set_xlabel("CLR difference, HCC − HBV")
            if j == 0: ax.set_ylabel("−log10 p (moderated t)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    legend(fig); fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def grid_heatmap(res, runs, fn, title):
    """Each panel = all 256 motifs as a 16 x 16 grid: row = first two bases, column = bases 3-4."""
    fig, axes = plt.subplots(len(runs), 4, figsize=(11.5, 3.0 * len(runs) + 0.5), squeeze=False)
    v = np.nanmax(np.abs(res.effect)); norm = TwoSlopeNorm(0, -v, v)
    di = ["".join(p) for p in itertools.product(NT, repeat=2)]
    for i, (plat, src) in enumerate(runs):
        rr = res[(res.platform == plat) & (res.source == src)].set_index(["size_bin", "motif"])
        for j, b in enumerate(BINS4):
            ax = axes[i, j]
            M = np.array([[rr.effect.get((b, r + c), np.nan) for c in di] for r in di])
            S = np.array([[rr.q.get((b, r + c), 1.0) < Q for c in di] for r in di])
            im = ax.imshow(M, cmap="PuOr_r", norm=norm)
            yy, xx = np.where(S)
            ax.scatter(xx, yy, s=3, c="#111111", linewidths=0)
            ax.set_xticks(range(16), di, fontsize=5.5, rotation=90); ax.set_yticks(range(16), di, fontsize=5.5)
            ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
            for k in range(4, 16, 4):
                ax.axhline(k - 0.5, color="white", lw=1.2); ax.axvline(k - 0.5, color="white", lw=1.2)
            ax.set_title(f"{plat} {src} · {BLAB4[b]}")
            if j == 0: ax.set_ylabel("bases 1–2")
            if i == len(runs) - 1: ax.set_xlabel("bases 3–4")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("CLR difference, HCC − HBV (orange = HCC higher)")
    cb.outline.set_visible(False)
    fig.suptitle(title + "   (dot = q < 0.05)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def sample_heatmap(feats, runs, fn):
    """Per-sample view: z-scored CLR of all 1024 motif x bin features; columns sorted by HCC − HBV difference within each bin."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs) + 0.4), squeeze=False, gridspec_kw=dict(hspace=0.45))
    for i, (plat, src) in enumerate(runs):
        d = feats[(plat, src)]
        y = is_hcc(d.index)
        d = pd.concat([d[y], d[~y]])
        cols, edges = [], []
        for b in BINS4:
            c = [x for x in d.columns if x.startswith(f"clr_{b}_")]
            diff = d.loc[d.index[:y.sum()], c].mean() - d.loc[d.index[y.sum():], c].mean()
            cols += list(diff.sort_values(ascending=False).index); edges.append(len(cols))
        Z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=1)
        ax = axes[i, 0]
        im = ax.imshow(Z.values, aspect="auto", cmap="PuOr_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_yticks(range(len(d)), [f"{s} ({'HCC' if not s.startswith('TBR44') else 'HBV'})" for s in d.index], fontsize=6)
        ax.axhline(y.sum() - 0.5, color="black", lw=1)
        for e in edges[:-1]: ax.axvline(e - 0.5, color="black", lw=0.8)
        ax.set_xticks([e - 128 for e in edges], [BLAB4[b] for b in BINS4]); ax.grid(False)
        ax.set_title(f"{plat} {src}: every motif in every bin, per sample (columns sorted by HCC − HBV difference within bin)")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("z-score of CLR across the 14 samples")
    cb.outline.set_visible(False)
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def cross_platform(res, fn):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    o = res[(res.platform == "ONT") & (res.source == "direct")].set_index(["size_bin", "motif"])
    p = res[(res.platform == "PacBio") & (res.source == "direct")].set_index(["size_bin", "motif"])
    j = o[["effect", "q"]].join(p[["effect", "q"]], lsuffix="_ont", rsuffix="_pb")
    lim = np.nanmax(np.abs(j[["effect_ont", "effect_pb"]].values)) * 1.05
    for ax, b in zip(axes, BINS4):
        r = j.loc[b]
        both = (r.q_ont < Q) & (r.q_pb < Q)
        c = np.where(both, "#6B3FA0", np.where(r.q_pb < Q, "#8FB3E0", np.where(r.q_ont < Q, "#F2B48F", GREY)))
        ax.scatter(r.effect_ont, r.effect_pb, s=8, c=c, linewidths=0)
        ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], color="#CCC", lw=0.6, ls="--")
        for m in LIT:
            if m in r.index:
                ax.annotate(m, (r.loc[m, "effect_ont"], r.loc[m, "effect_pb"]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        rho = r[["effect_ont", "effect_pb"]].corr(method="spearman").iloc[0, 1]
        agree = (np.sign(r.effect_ont[both]) == np.sign(r.effect_pb[both])).sum()
        ax.set_title(f"{BLAB4[b]}\nρ {rho:.2f} · sig on both {both.sum()} ({agree} same sign)", fontsize=8.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("ONT direct, CLR diff")
    axes[0].set_ylabel("PacBio direct, CLR diff")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
         [("#6B3FA0", "q < 0.05 on both"), ("#8FB3E0", "PacBio only"), ("#F2B48F", "ONT only"), (GREY, "neither")]]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("End motifs, ONT vs PacBio: HCC − HBV difference for all 256 motifs", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def volcano_bins(res, fn):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, kind, xl in [(axes[0], "size", "log2 fold change of fraction, HCC / HBV"), (axes[1], "entropy", "Cohen's d (rarefied entropy), HCC − HBV")]:
        for plat, mk in [("ONT", "o"), ("PacBio", "s")]:
            r = res[(res.feature == kind) & (res.platform == plat) & (res.source == "direct")]
            ax.scatter(r.effect, -np.log10(r.p), s=34, marker=mk, c=colour(r), edgecolors="white", linewidths=0.6, label=plat)
            for _, x in r.iterrows():
                ax.annotate(x.size_bin.replace("bp", ""), (x.effect, -np.log10(x.p)), fontsize=6.3, xytext=(4, 1), textcoords="offset points", color="#444")
        ax.axvline(0, color="#999", lw=0.6)
        ax.axhline(-np.log10(2 / 3003), color="#BBB", lw=0.6, ls=":")
        ax.text(ax.get_xlim()[1], -np.log10(2 / 3003), "smallest possible exact p (8 vs 6) ", fontsize=6, va="bottom", ha="right", color="#888")
        ax.set_xlabel(xl); ax.set_ylabel("−log10 p (exact Mann–Whitney)")
        ax.set_title({"size": "A  Share of fragments per length bin", "entropy": "B  Length entropy per bin (rarefied)"}[kind])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color="#666", label="ONT direct"), Line2D([], [], marker="s", ls="", color="#666", label="PacBio direct"),
         Line2D([], [], marker="o", ls="", color=HCC, label="HCC higher, q < 0.05"), Line2D([], [], marker="o", ls="", color=HBV, label="HBV higher, q < 0.05"),
         Line2D([], [], marker="o", ls="", color=GREY, label="not significant")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def heat_bins(tabs, fn):
    """Per-sample heatmap: rows = samples, columns = 11 bins; z-score within bin. One panel per platform x feature."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), gridspec_kw=dict(hspace=0.42, wspace=0.32))
    for i, kind in enumerate(["size", "entropy"]):
        for j, plat in enumerate(["ONT", "PacBio"]):
            d = tabs[(plat, kind)]; y = is_hcc(d.index); d = pd.concat([d[y], d[~y]])
            cols = [("frac_" if kind == "size" else "ent_") + b for b in BINS11]
            Z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=1)
            ax = axes[i, j]
            im = ax.imshow(Z.values, aspect="auto", cmap="PuOr_r", vmin=-2.5, vmax=2.5)
            ax.set_yticks(range(len(d)), [f"{s} {'HCC' if not s.startswith('TBR44') else 'HBV'}" for s in d.index], fontsize=6)
            ax.set_xticks(range(11), [b.replace("bp", "") for b in BINS11], rotation=45, fontsize=6.5, ha="right")
            ax.axhline(y.sum() - 0.5, color="black", lw=1); ax.grid(False)
            ax.set_title(f"{plat} direct · {'share of fragments' if kind == 'size' else 'rarefied length entropy'}")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("z-score across the 14 samples (orange = higher)")
    cb.outline.set_visible(False)
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def main():
    global mrf
    ap = argparse.ArgumentParser(description="Volcano and heatmap views of all genome-wide size, entropy and end-motif tests (Figures 3.3, 3.4, 3.6-3.8, 3.19).")
    ap.add_argument("--ont-feats", required=True,
                    help="ONT per-sample feature folder with all/ and direct/ subfolders (F_motif.csv, F_size.csv, F_entropy_rare.csv)")
    ap.add_argument("--pacbio-feats", required=True,
                    help="PacBio per-sample feature folder with a direct/ subfolder (F_motif.csv, F_size.csv, F_entropy_rare.csv)")
    ap.add_argument("--motif-stats", default=MOTIF_STATS,
                    help="path to 05_motif_reduced_family_stats.py (moderated t, Cliff's delta and BH are reused from it)")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    ap.add_argument("-t", "--threads", type=int, default=4, help="parallel threads for the motif tests")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    mrf = load_motif_stats(a.motif_stats)
    jobs = [(f"{getattr(a, f)}/{s}/F_motif.csv", p, s) for p, s, f in RUNS]
    with ThreadPoolExecutor(a.threads) as ex:
        mot = pd.concat(list(ex.map(lambda j: motif_tests(*j), jobs)), ignore_index=True)
    feats = {(p, s): pd.read_csv(f"{getattr(a, f)}/{s}/F_motif.csv").set_index("sample") for p, s, f in RUNS}
    bres, tabs = [], {}
    for plat, f in [("ONT", "ont_feats"), ("PacBio", "pacbio_feats")]:
        for kind, fn in [("size", "F_size.csv"), ("entropy", "F_entropy_rare.csv")]:
            r, d = bin_tests(f"{getattr(a, f)}/direct/{fn}", plat, "direct", kind); bres.append(r); tabs[(plat, kind)] = d
    allres = pd.concat([mot] + bres, ignore_index=True)
    allres.to_csv(f"{a.out}/all_tests_genome_wide.csv", index=False)
    two = [("ONT", "direct"), ("PacBio", "direct")]
    # Figure 3.6 (and ONT all), Figure 3.7 (and ONT all), Figure 3.8, Figure 3.19, Figure 3.3, Figure 3.4
    volcano_motifs(mot, two, f"{a.out}/fig4_motif_volcano.png", "End motifs: all 256 motifs × 4 bins, genome-wide, direct fragments")
    volcano_motifs(mot, [("ONT", "all")], f"{a.out}/fig4_motif_volcano_ont_all.png", "End motifs: ONT all fragments")
    grid_heatmap(mot, two, f"{a.out}/fig4_motif_grid.png", "End motifs: HCC − HBV for all 256 motifs")
    grid_heatmap(mot, [("ONT", "all")], f"{a.out}/fig4_motif_grid_ont_all.png", "End motifs, ONT all fragments")
    sample_heatmap(feats, two, f"{a.out}/fig4_motif_per_sample.png")
    cross_platform(mot, f"{a.out}/fig4_motif_ont_vs_pacbio.png")
    volcano_bins(allres, f"{a.out}/fig4_size_entropy_volcano.png")
    heat_bins(tabs, f"{a.out}/fig4_size_entropy_heatmap.png")
    print(allres.groupby(["feature", "platform", "source"]).apply(lambda g: pd.Series(dict(n=len(g), sig=(g.q < Q).sum(),
          up=((g.q < Q) & (g.effect > 0)).sum(), down=((g.q < Q) & (g.effect < 0)).sum()))).to_string())


if __name__ == "__main__":
    main()
