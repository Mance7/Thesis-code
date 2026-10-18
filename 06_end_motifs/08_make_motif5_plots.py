#!/usr/bin/env python3
"""
08_make_motif5_plots.py

End motifs (5' end, one end per fragment) in 5 size bins, genome-wide, HCC vs HBV, both platforms.
Same views as the 4-bin version (the four-bin version): volcano, 16 x 16 grid heatmap,
per-sample heatmap, ONT vs PacBio scatter. Motif names are never written inside a panel:
points carry a rank number and the ranked names are listed beside the panel.

Bins: <=255 | 256-650 | 651-2000 | 2001-5000 | >5000 bp.
Per sample and bin: CLR of the 256 motif counts (pseudocount 0.5), moderated t (limma-style), BH over the
1,280 motif x bin tests of a run. Literature family F1 (CCCA CCAG CCTG AAAA TAAA TTTT x 5 bins = 30 tests,
pre-specified in plan/motif_families_5bin.tsv) gets its own BH.
ONT = discovery. PacBio replication = motif significant on ONT (q < 0.05) and on PacBio p < 0.05 in the same direction.

  python 08_make_motif5_plots.py --counts motif_5bin_counts_genome_wide.csv.gz --stats-lib scripts/ -o motif5bin --threads 4
"""
import argparse, importlib.util, os, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HCC, HBV, GREY, INK, MUTE = "#E8703A", "#2F7BD8", "#B8B8B8", "#222222", "#6B6B6B"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 250,
})
BINS = ["<=255bp", "256-650bp", "651-2000bp", "2001-5000bp", ">5000bp"]
BLAB = {"<=255bp": "≤255 bp", "256-650bp": "256–650 bp", "651-2000bp": "651–2000 bp",
        "2001-5000bp": "2001–5000 bp", ">5000bp": ">5000 bp"}
LIT = ["CCCA", "CCAG", "CCTG", "AAAA", "TAAA", "TTTT"]
NT = "ACGT"
KM = ["".join(p) for p in itertools.product(NT, repeat=4)]
Q, TOPN = 0.05, 5


def is_hcc(idx):
    return ~pd.Index(idx).str.startswith("TBR44")


def clr_table(cnt, plat, src):
    """wide table: sample x (bin, motif) CLR"""
    c = cnt[(cnt.platform == plat) & (cnt.source == src)]
    out = {}
    for b in BINS:
        w = c[c.bin == b].pivot_table(index="sample", columns="motif", values="count", aggfunc="sum").reindex(columns=KM).fillna(0)
        lc = np.log(w.values + 0.5)
        out[b] = pd.DataFrame(lc - lc.mean(1, keepdims=True), index=w.index, columns=KM)
    return out


def motif_tests(cl, plat, src, mrf):
    out = []
    for b in BINS:
        X = cl[b]; y = is_hcc(X.index)
        t, df, p, diff = mrf.moderated_t(X.values, y)
        cd = [mrf.cliffs_delta(X.values[y, j], X.values[~y, j]) for j in range(X.shape[1])]
        out.append(pd.DataFrame(dict(platform=plat, source=src, size_bin=b, motif=KM, clr_diff=diff,
                                     mod_t=t, p=p, cliffs_delta=cd, n_hcc=y.sum(), n_hbv=(~y).sum())))
    r = pd.concat(out, ignore_index=True)
    r["q"] = mrf.bh(r.p.values)                       # BH over 1,280 motif x bin tests of this run
    lit = r.motif.isin(LIT)
    r["q_F1_literature"] = np.nan
    r.loc[lit, "q_F1_literature"] = mrf.bh(r.loc[lit, "p"].values)   # pre-specified family, 30 tests
    return r


def colour(r):
    return np.where(r.q < Q, np.where(r.clr_diff > 0, HCC, HBV), GREY)


def legend(fig, y=-0.01):
    h = [Line2D([], [], marker="o", ls="", color=HCC, label=f"HCC higher, q < {Q}"),
         Line2D([], [], marker="o", ls="", color=HBV, label=f"HBV higher, q < {Q}"),
         Line2D([], [], marker="o", ls="", color=GREY, label="not significant"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="literature motif (CCCA CCAG CCTG AAAA TAAA TTTT)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, y))


def ranked(r, n=TOPN):
    """top n significant motifs of one panel, by p"""
    return r[r.q < Q].nsmallest(n, "p")


def volcano(res, runs, fn, title):
    fig = plt.figure(figsize=(14.2, 2.9 * len(runs) + 0.8))
    gs = fig.add_gridspec(len(runs), 6, width_ratios=[1, 1, 1, 1, 1, 0.95], wspace=0.28, hspace=0.42)
    xl = np.nanmax(np.abs(res.clr_diff)) * 1.08
    for i, (plat, src) in enumerate(runs):
        rr = res[(res.platform == plat) & (res.source == src)]
        yl = np.nanmax(-np.log10(rr.p)) * 1.12
        lines = []
        for j, b in enumerate(BINS):
            ax, r = fig.add_subplot(gs[i, j]), rr[rr.size_bin == b]
            lp = -np.log10(r.p)
            ax.scatter(r.clr_diff, lp, s=9, c=colour(r), linewidths=0, alpha=0.85)
            l = r[r.motif.isin(LIT)]
            ax.scatter(l.clr_diff, -np.log10(l.p), s=26, facecolors="none", edgecolors=INK, linewidths=0.7)
            ax.axvline(0, color="#999999", lw=0.6)
            sig = rr[rr.q < Q]
            if len(sig):     # dashed line = p where q crosses 0.05 in this run
                ax.axhline(-np.log10(sig.p.max()), color="#888888", lw=0.6, ls="--")
            top = ranked(r)
            for k, (_, x) in enumerate(top.iterrows(), 1):
                ax.annotate(str(k), (x.clr_diff, -np.log10(x.p)), fontsize=7, xytext=(4 if k % 2 else -9, 1 if k % 2 else -3), textcoords="offset points",
                            color=INK, fontweight="bold")
            lines.append((BLAB[b], [(k, x.motif, x.clr_diff, x.q) for k, (_, x) in enumerate(top.iterrows(), 1)]))
            ax.set_xlim(-xl, xl); ax.set_ylim(0, yl)
            n_up, n_dn = ((r.q < Q) & (r.clr_diff > 0)).sum(), ((r.q < Q) & (r.clr_diff < 0)).sum()
            ax.set_title(f"{plat} {src} · {BLAB[b]}\n↑HCC {n_up}  ↓HCC {n_dn}", fontsize=8.5)
            if i == len(runs) - 1: ax.set_xlabel("CLR difference, HCC − HBV")
            if j == 0: ax.set_ylabel("−log10 p (moderated t)")
        tx = fig.add_subplot(gs[i, 5]); tx.axis("off")
        y0 = 1.0
        tx.text(0, y0, f"{plat} {src}: top {TOPN} per bin", fontsize=8, fontweight="bold", va="top", transform=tx.transAxes)
        y0 -= 0.085
        for lab, items in lines:
            tx.text(0, y0, lab, fontsize=7.2, fontweight="bold", color=MUTE, va="top", transform=tx.transAxes)
            s = "   ".join(f"{k} {m}{'↑' if d > 0 else '↓'}" for k, m, d, q in items) or "none with q < 0.05"
            tx.text(0, y0 - 0.075, s, fontsize=7.2, va="top", transform=tx.transAxes, family="DejaVu Sans Mono")
            y0 -= 0.19
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.035, "Numbers = rank by p among q < 0.05 in that panel; names listed on the right (↑ = higher in HCC, ↓ = lower in HCC).",
             fontsize=7.2, color=MUTE)
    legend(fig, y=-0.02)
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def grid_heatmap(res, runs, fn, title):
    """all 256 motifs as a 16 x 16 grid: row = bases 1-2, column = bases 3-4"""
    fig, axes = plt.subplots(len(runs), 5, figsize=(14, 3.0 * len(runs) + 0.5), squeeze=False)
    v = np.nanmax(np.abs(res.clr_diff)); norm = TwoSlopeNorm(0, -v, v)
    di = ["".join(p) for p in itertools.product(NT, repeat=2)]
    for i, (plat, src) in enumerate(runs):
        rr = res[(res.platform == plat) & (res.source == src)].set_index(["size_bin", "motif"])
        for j, b in enumerate(BINS):
            ax = axes[i, j]
            M = np.array([[rr.clr_diff.get((b, r + c), np.nan) for c in di] for r in di])
            S = np.array([[rr.q.get((b, r + c), 1.0) < Q for c in di] for r in di])
            im = ax.imshow(M, cmap="PuOr_r", norm=norm)
            yy, xx = np.where(S)
            ax.scatter(xx, yy, s=3, c="#111111", linewidths=0)
            ax.set_xticks(range(16), di, fontsize=5.5, rotation=90); ax.set_yticks(range(16), di, fontsize=5.5)
            ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
            for k in range(4, 16, 4):
                ax.axhline(k - 0.5, color="white", lw=1.2); ax.axvline(k - 0.5, color="white", lw=1.2)
            ax.set_title(f"{plat} {src} · {BLAB[b]}", fontsize=8.5)
            if j == 0: ax.set_ylabel("bases 1–2")
            if i == len(runs) - 1: ax.set_xlabel("bases 3–4")
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label("CLR difference, HCC − HBV (orange = HCC higher)")
    cb.outline.set_visible(False)
    fig.suptitle(title + "   (dot = q < 0.05)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def sample_heatmap(cls, res, runs, fn):
    """z-scored CLR of all 1,280 motif x bin features per sample; columns sorted by HCC − HBV difference within bin"""
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 2.6 * len(runs) + 0.4), squeeze=False, gridspec_kw=dict(hspace=0.5))
    for i, (plat, src) in enumerate(runs):
        cl = cls[(plat, src)]
        idx = cl[BINS[0]].index; y = is_hcc(idx); order = list(idx[y]) + list(idx[~y])
        blocks, edges = [], []
        for b in BINS:
            d = cl[b].loc[order]
            diff = res[(res.platform == plat) & (res.source == src) & (res.size_bin == b)].set_index("motif").clr_diff
            d = d[diff.sort_values(ascending=False).index]
            blocks.append(((d - d.mean()) / d.std(ddof=1)).values); edges.append(sum(x.shape[1] for x in blocks))
        Z = np.hstack(blocks)
        ax = axes[i, 0]
        im = ax.imshow(Z, aspect="auto", cmap="PuOr_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_yticks(range(len(order)), [f"{s} ({'HCC' if not s.startswith('TBR44') else 'HBV'})" for s in order], fontsize=6)
        ax.axhline(y.sum() - 0.5, color="black", lw=1)
        for e in edges[:-1]: ax.axvline(e - 0.5, color="black", lw=0.8)
        ax.set_xticks([e - 128 for e in edges], [BLAB[b] for b in BINS]); ax.grid(False)
        ax.set_title(f"{plat} {src}: every motif in every bin, per sample (columns sorted by HCC − HBV difference within bin)")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("z-score of CLR across the 14 samples")
    cb.outline.set_visible(False)
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def cross_platform(res, src, fn):
    """ONT (discovery) vs PacBio difference per motif; ranked shared hits listed under each panel"""
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.3], hspace=0.42, wspace=0.3)
    o = res[(res.platform == "ONT") & (res.source == src)].set_index(["size_bin", "motif"])
    p = res[(res.platform == "PacBio") & (res.source == src)].set_index(["size_bin", "motif"])
    j = o[["clr_diff", "q", "p"]].join(p[["clr_diff", "q", "p"]], lsuffix="_ont", rsuffix="_pb")
    lim = np.nanmax(np.abs(j[["clr_diff_ont", "clr_diff_pb"]].values)) * 1.05
    rows = []
    for k, b in enumerate(BINS):
        ax = fig.add_subplot(gs[0, k]); r = j.loc[b]
        rep = (r.q_ont < Q) & (r.p_pb < 0.05) & (np.sign(r.clr_diff_ont) == np.sign(r.clr_diff_pb))
        both = (r.q_ont < Q) & (r.q_pb < Q)
        c = np.where(rep, "#6B3FA0", np.where(r.q_ont < Q, "#F2B48F", GREY))
        ax.scatter(r.clr_diff_ont, r.clr_diff_pb, s=8, c=c, linewidths=0)
        ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], color="#CCC", lw=0.6, ls="--")
        top = r[rep].nsmallest(TOPN, "p_ont")
        for n, (m, x) in enumerate(top.iterrows(), 1):
            ax.annotate(str(n), (x.clr_diff_ont, x.clr_diff_pb), fontsize=7, xytext=(4 if n % 2 else -9, 1 if n % 2 else -4), textcoords="offset points", fontweight="bold")
        rho = r[["clr_diff_ont", "clr_diff_pb"]].corr(method="spearman").iloc[0, 1]
        n_ont = (r.q_ont < Q).sum()
        ax.set_title(f"{BLAB[b]}\nρ {rho:.2f} · ONT hits {n_ont} · replicated {rep.sum()}", fontsize=8.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel(f"ONT {src}, CLR diff (discovery)")
        if k == 0: ax.set_ylabel(f"PacBio {src}, CLR diff")
        tx = fig.add_subplot(gs[1, k]); tx.axis("off")
        s = "\n".join(f"{n} {m} {'↑' if x.clr_diff_ont > 0 else '↓'}  q_ONT {x.q_ont:.1e}  p_PB {x.p_pb:.1e}" for n, (m, x) in enumerate(top.iterrows(), 1))
        tx.text(0, 1, s or "none replicated", fontsize=6.8, va="top", family="DejaVu Sans Mono", transform=tx.transAxes)
        rows.append(dict(source=src, size_bin=b, spearman_rho=rho, ont_sig=n_ont, pacbio_sig=(r.q_pb < Q).sum(),
                         sig_both_q=both.sum(), both_same_sign=(both & (np.sign(r.clr_diff_ont) == np.sign(r.clr_diff_pb))).sum(),
                         replicated_pb_p05_same_sign=rep.sum(),
                         ont_hits_same_sign_on_pb=((r.q_ont < Q) & (np.sign(r.clr_diff_ont) == np.sign(r.clr_diff_pb))).sum()))
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
         [("#6B3FA0", "ONT q < 0.05 and PacBio p < 0.05, same direction"), ("#F2B48F", "ONT q < 0.05 only"), (GREY, "not significant on ONT")]]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"End motifs, ONT (discovery) vs PacBio ({src} fragments): HCC − HBV for all 256 motifs. "
                 "Numbers = top replicated motifs, listed below each panel", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(rows)


def lit_plot(res, fn):
    """pre-specified literature motifs: rows = motifs, columns = bins; colour = CLR difference, * = family q < 0.05"""
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    sub = res[res.motif.isin(LIT)]; v = np.nanmax(np.abs(sub.clr_diff)); norm = TwoSlopeNorm(0, -v, v)
    for ax, (plat, src) in zip(axes, [(p, s) for s in ("direct", "all") for p in ("ONT", "PacBio")]):
        r = sub[(sub.platform == plat) & (sub.source == src)].set_index(["motif", "size_bin"])
        M = np.array([[r.clr_diff[(m, b)] for b in BINS] for m in LIT]); S = np.array([[r.q_F1_literature[(m, b)] for b in BINS] for m in LIT])
        im = ax.imshow(M, cmap="PuOr_r", norm=norm, aspect="auto")
        for i in range(len(LIT)):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if S[i, j] < Q else ""), ha="center", va="center", fontsize=7,
                        fontweight="bold" if S[i, j] < Q else "normal", color=INK)
        ax.set_xticks(range(5), [BLAB[b].replace(" bp", "") for b in BINS], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(len(LIT)), LIT, fontsize=7.5); ax.grid(False)
        ax.set_title(f"{plat} {src}{'  (discovery)' if plat == 'ONT' else '  (prediction only)'}", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label("CLR diff, HCC − HBV"); cb.outline.set_visible(False)
    fig.suptitle("Literature end motifs (pre-specified, 30 tests per run). * = q < 0.05 within the family; orange = higher in HCC",
                 x=0.01, y=1.08, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="motif_5bin_counts_genome_wide.csv.gz (motif_5bin_counts.py)")
    ap.add_argument("--stats-lib", required=True, help="folder with motif_reduced_family_stats.py (moderated t, BH)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-t", "--threads", type=int, default=4)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "motif5_tests_genome_wide.csv")):
        raise SystemExit(f"{a.out} already has results; pick a new -o folder")
    os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("mrf", os.path.join(a.stats_lib, "motif_reduced_family_stats.py"))
    mrf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mrf)
    cnt = pd.read_csv(a.counts)
    runs = [(p, s) for p in ("ONT", "PacBio") for s in ("direct", "all")]
    with ThreadPoolExecutor(a.threads) as ex:
        cls = dict(zip(runs, ex.map(lambda r: clr_table(cnt, *r), runs)))
    res = pd.concat([motif_tests(cls[r], *r, mrf) for r in runs], ignore_index=True)
    res.to_csv(f"{a.out}/motif5_tests_genome_wide.csv", index=False)
    dr, al = [("ONT", "direct"), ("PacBio", "direct")], [("ONT", "all"), ("PacBio", "all")]
    volcano(res, dr, f"{a.out}/V1_motif5_volcano_direct.png", "End motifs: all 256 motifs × 5 bins, genome-wide, direct fragments")
    volcano(res, al, f"{a.out}/V1b_motif5_volcano_all.png", "End motifs: all 256 motifs × 5 bins, genome-wide, all fragments")
    grid_heatmap(res, dr, f"{a.out}/H1_motif5_grid_heatmap_direct.png", "End motifs: HCC − HBV for all 256 motifs, direct")
    grid_heatmap(res, al, f"{a.out}/H1b_motif5_grid_heatmap_all.png", "End motifs: HCC − HBV for all 256 motifs, all fragments")
    sample_heatmap(cls, res, dr, f"{a.out}/H2_motif5_per_sample_heatmap_direct.png")
    x = pd.concat([cross_platform(res, s, f"{a.out}/X1_motif5_ont_vs_pacbio_{s}.png") for s in ("direct", "all")])
    x.to_csv(f"{a.out}/motif5_cross_platform_summary.csv", index=False)
    lit_plot(res, f"{a.out}/L1_motif5_literature_motifs.png")
    # ranked tables: every significant motif, and the ONT hits that replicate on PacBio
    sig = res[res.q < Q].sort_values(["platform", "source", "size_bin", "p"])
    sig["rank_in_bin"] = sig.groupby(["platform", "source", "size_bin"]).cumcount() + 1
    sig.to_csv(f"{a.out}/motif5_significant_ranked.csv", index=False)
    o = res[res.platform == "ONT"].set_index(["source", "size_bin", "motif"])
    p = res[res.platform == "PacBio"].set_index(["source", "size_bin", "motif"])
    j = o[["clr_diff", "p", "q"]].join(p[["clr_diff", "p", "q"]], lsuffix="_ont", rsuffix="_pacbio").reset_index()
    rep = j[(j.q_ont < Q) & (j.p_pacbio < 0.05) & (np.sign(j.clr_diff_ont) == np.sign(j.clr_diff_pacbio))]
    rep.sort_values(["source", "size_bin", "p_ont"]).to_csv(f"{a.out}/motif5_ont_hits_replicated_on_pacbio.csv", index=False)
    res[res.motif.isin(LIT)].sort_values(["platform", "source", "size_bin", "motif"]).to_csv(f"{a.out}/motif5_literature_family.csv", index=False)
    print(res.groupby(["platform", "source", "size_bin"], sort=False).apply(lambda g: pd.Series(dict(
        sig=(g.q < Q).sum(), up=((g.q < Q) & (g.clr_diff > 0)).sum(), down=((g.q < Q) & (g.clr_diff < 0)).sum()))).to_string())
    print(x.to_string())


if __name__ == "__main__":
    main()
