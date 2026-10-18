#!/usr/bin/env python3
"""Per-size-bin HCC/HBV fold difference and an out-of-sample check of the size-ratio bins.

Produces Figures 4.3 and 4.4 (Discussion 4.3) and their tables.

1. Fold per bin: each sample's count in a bin (20 bp bins up to 1,000 bp,
   100 bp bins from 1,000 to 2,000 bp) divided by its own >2 kb count (the
   range where HCC and HBV do not differ), then divided by the HBV median.
   Effect size per bin = Hedges g of the log ratio, with AUC and exact
   Mann-Whitney p.
2. Window search: every numerator window [a, b] on a 10 bp grid (60-2000 bp,
   width >= 50 bp) is ranked on ONT only (largest standardised difference of
   log(window / >2 kb)), and the chosen window is then tested on PacBio.
   The size ratio R (256-650 / >2000 bp) stays the headline; its bins were
   picked after seeing the data, so only external cohorts test it.
Also compares R with alternative numerators and with 651-2000 bp as denominator,
and plots the rarefied per-bin length entropy of both platforms.

Input : 1 bp length histograms <hist-dir>/length_hist_{ont,pacbio}_direct.tsv.gz
        (04_length_histograms.py); rarefied entropy feature tables
        F_entropy_rare.csv of ONT and PacBio direct fragments
        (08_classifiers/02_build_classifier_features.py); metadata csv (sample, group)
Output: <out>/fold_per_bin_per_sample.tsv, effect_per_bin.tsv, ont_window_grid.tsv.gz,
        ratio_comparison.tsv, ont_top1pct_windows_on_pacbio.tsv,
        size_bin_fold_and_entropy.png, ont_window_landscape.png

Runs in seconds:
  python 06_size_bin_fold_and_window.py --hist-dir length_hist \\
      --ent-ont feats_ont/direct/F_entropy_rare.csv --ent-pacbio feats_pacbio/direct/F_entropy_rare.csv \\
      --metadata metadata/ont_samples.csv -o size_bins/ --threads 4
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HCC_C, HBV_C = "#d9822b", "#3b75af"


def load(path):
    d = pd.read_csv(path, sep="\t", index_col=0)
    gt = d.pop(">5000").to_numpy(float)
    h = d.to_numpy(float)                      # columns 1..5000 bp
    return d.index.to_numpy(), h, gt


GROUP = {}   # sample -> group, filled from --metadata in main()


def group(samples):
    """'HCC' for HCC samples, 'HBV' for HBV carriers."""
    return np.array(["HCC" if GROUP[s] == "HCC" else "HBV" for s in samples])


def cum(h, gt):
    c = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, 1)], 1)
    return c, gt                                # count in [a,b] = c[:,b]-c[:,a-1]


def win(c, a, b):
    return c[:, b] - c[:, a - 1]


def long_count(c, gt, lo=2001):
    return c[:, 5000] - c[:, lo - 1] + gt


def hedges(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp * (1 - 3 / (4 * (nx + ny) - 9))


def stats(lr, g):
    x, y = lr[g == "HCC"], lr[g == "HBV"]
    auc = mannwhitneyu(x, y).statistic / (len(x) * len(y))
    p = mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue
    return dict(g=hedges(x, y), auc=auc, p=p, margin=x.min() - y.max())


def score_chunk(args):
    c, den, g, starts, step, bmax = args
    hcc, hbv = g == "HCC", g == "HBV"
    out = []
    for a in starts:
        for b in range(a + 49, bmax + 1, step):
            lr = np.log(win(c, a, b) / den)
            x, y = lr[hcc], lr[hbv]
            nx, ny = len(x), len(y)
            sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
            out.append((a, b, (x.mean() - y.mean()) / sp, x.min() - y.max()))
    return out


TICKS = [60, 100, 150, 200, 250, 300, 400, 500, 650, 800, 1000, 1500, 2000]
RUNGS = {"mono\n~167": 167, "di\n~350": 350, "tri\n~545": 545, "4\n~740": 742, "5\n~940": 937}


def size_axis(A, xlabel=True):
    """Log length axis with plain bp ticks, nucleosome rungs labelled on top."""
    A.set_xscale("log"); A.set_xlim(60, 2000)
    A.set_xticks(TICKS); A.set_xticklabels([str(x) for x in TICKS], fontsize=8)
    A.minorticks_off()
    for n in [167, 350, 545, 742, 937, 1132, 1325, 1520, 1715, 1910]:
        A.axvline(n, color="grey", lw=.4, ls=":")
    sec = A.secondary_xaxis("top")
    sec.set_xticks(list(RUNGS.values())); sec.set_xticklabels(list(RUNGS.keys()), fontsize=7)
    sec.tick_params(length=0)
    if xlabel:
        A.set_xlabel("fragment length (bp); bins are 20 bp wide up to 1,000 bp and 100 bp wide from 1,000 to 2,000 bp\n"
                     "(dotted lines = nucleosome peaks; green = R numerator 256-650 bp)", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hist-dir", required=True,
                    help="folder with length_hist_{ont,pacbio}_direct.tsv.gz")
    ap.add_argument("--ent-ont", required=True,
                    help="rarefied entropy feature table, ONT direct fragments (csv)")
    ap.add_argument("--ent-pacbio",
                    required=True,
                    help="rarefied entropy feature table, PacBio direct fragments (csv)")
    ap.add_argument("--metadata", required=True,
                    help="csv with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--threads", type=int, default=4, help="parallel workers for the window search")
    ap.add_argument("--step", type=int, default=10, help="window grid step, bp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.metadata, sep=None, engine="python", encoding_errors="replace")
    meta.columns = [c.strip().lower() for c in meta.columns]
    GROUP.update(zip(meta["sample"].astype(str).str.strip(), meta["group"].astype(str).str.strip()))

    data = {}
    for plat in ["ont", "pacbio"]:
        s, h, gt = load(os.path.join(a.hist_dir, f"length_hist_{plat}_direct.tsv.gz"))
        c, gt = cum(h, gt)
        data[plat] = dict(s=s, g=group(s), c=c, gt=gt, den=long_count(c, gt))

    # ---- 1. fold per bin (normalised to own >2 kb count) ----
    edges = list(range(60, 1000, 20)) + list(range(1000, 2001, 100))
    bins = [(lo, hi - 1) for lo, hi in zip(edges[:-1], edges[1:])]
    rows = []
    for plat, d in data.items():
        for lo, hi in bins:
            v = win(d["c"], lo, hi) / d["den"]
            fold = v / np.median(v[d["g"] == "HBV"])
            st = stats(np.log(v), d["g"])
            for smp, grp, f in zip(d["s"], d["g"], fold):
                rows.append(dict(platform=plat, lo=lo, hi=hi, sample=smp, group=grp, fold=f,
                                 g=st["g"], auc=st["auc"], p=st["p"]))
    fold = pd.DataFrame(rows)
    fold.to_csv(os.path.join(a.out, "fold_per_bin_per_sample.tsv"), sep="\t", index=False)
    eff = fold.drop_duplicates(["platform", "lo"])[["platform", "lo", "hi", "g", "auc", "p"]]
    eff.to_csv(os.path.join(a.out, "effect_per_bin.tsv"), sep="\t", index=False)

    # ---- 2. window search on ONT, test on PacBio ----
    d = data["ont"]
    starts = list(range(60, 1951, a.step))
    chunks = [starts[i::a.threads] for i in range(a.threads)]
    with ProcessPoolExecutor(a.threads) as ex:
        res = ex.map(score_chunk, [(d["c"], d["den"], d["g"], ch, a.step, 2000) for ch in chunks])
    grid = pd.DataFrame([r for part in res for r in part], columns=["a", "b", "d_ont", "margin_ont"])
    grid.to_csv(os.path.join(a.out, "ont_window_grid.tsv.gz"), sep="\t", index=False)
    best = grid.sort_values("d_ont", ascending=False).iloc[0]
    top = grid[grid.d_ont >= grid.d_ont.quantile(0.99)]

    summ = []
    for name, (lo, hi) in {"original R (256-650)": (256, 650),
                           f"ONT-chosen ({int(best.a)}-{int(best.b)})": (int(best.a), int(best.b)),
                           "mono only (151-255)": (151, 255),
                           "150-650": (150, 650),
                           "di only (256-420)": (256, 420),
                           "tri only (421-650)": (421, 650)}.items():
        for plat, dd in data.items():
            st = stats(np.log(win(dd["c"], lo, hi) / dd["den"]), dd["g"])
            summ.append(dict(ratio=name, platform=plat, **st))
    # 650-2000 as denominator (supervisor's alternative)
    for plat, dd in data.items():
        st = stats(np.log(win(dd["c"], 150, 650) / win(dd["c"], 651, 2000)), dd["g"])
        summ.append(dict(ratio="150-650 / 651-2000", platform=plat, **st))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.out, "ratio_comparison.tsv"), sep="\t", index=False)

    # PacBio performance of every ONT window, to show how the ONT ranking transfers
    dp = data["pacbio"]
    lr_p = [stats(np.log(win(dp["c"], int(r.a), int(r.b)) / dp["den"]), dp["g"])["g"]
            for r in top.itertuples()]
    top = top.assign(d_pacbio=lr_p)
    top.to_csv(os.path.join(a.out, "ont_top1pct_windows_on_pacbio.tsv"), sep="\t", index=False)

    # ---- figure ----
    nuc = [167, 350, 545, 742, 937, 1132, 1325, 1520, 1715, 1910]
    fig, ax = plt.subplots(3, 2, figsize=(14, 13), gridspec_kw=dict(height_ratios=[1.3, 1.1, 1.1]))
    for j, plat in enumerate(["ont", "pacbio"]):
        f = fold[fold.platform == plat]
        A = ax[0, j]
        for smp, sub in f.groupby("sample"):
            col = HCC_C if sub.group.iloc[0] == "HCC" else HBV_C
            mid = (sub.lo + sub.hi) / 2
            A.plot(mid, np.log2(sub.fold), color=col, lw=1, alpha=.8)
        A.axhline(0, color="grey", lw=.8)
        A.axvspan(256, 650, color="#9bc49b", alpha=.25, lw=0)
        size_axis(A)
        A.set_title(f"{'ONT' if plat == 'ont' else 'PacBio'} direct: fold vs HBV median, per sample", pad=22)
        A.set_ylabel("log2 fold (bin count / own >2 kb count,\nrelative to HBV median)")
        B = ax[1, j]
        e = eff[eff.platform == plat]
        B.bar((e.lo + e.hi) / 2, e.g, width=(e.hi - e.lo) * .9, color="#777")
        B.axvspan(256, 650, color="#9bc49b", alpha=.25, lw=0)
        size_axis(B)
        B.set_title(f"{'ONT' if plat == 'ont' else 'PacBio'} direct: how well each size bin separates HCC from HBV", pad=22)
        B.set_ylabel("effect size (Hedges g,\nlog ratio HCC vs HBV)")
    for A in ax[0]:
        A.plot([], [], color=HCC_C, label="HCC (n=8)"); A.plot([], [], color=HBV_C, label="HBV (n=6)")
        A.plot([], [], color="#9bc49b", lw=6, alpha=.5, label="R numerator 256-650 bp")
        A.legend(fontsize=8, loc="upper right")
    # entropy boxplots
    for j, (plat, path) in enumerate([("ONT", a.ent_ont), ("PacBio", a.ent_pacbio)]):
        E = pd.read_csv(path, index_col=0)
        g = group(E.index.to_numpy())
        C = ax[2, j]
        pos = np.arange(E.shape[1])
        for k, grp in enumerate(["HBV", "HCC"]):
            bp = C.boxplot([E.loc[g == grp, col].values for col in E.columns], positions=pos + (k - .5) * .35,
                           widths=.3, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(HBV_C if grp == "HBV" else HCC_C); b.set_alpha(.6)
        C.set_xticks(pos); C.set_xticklabels([c.replace("ent_", "").replace("bp", "") for c in E.columns],
                                             rotation=45, fontsize=8)
        C.set_ylabel("rarefied length entropy"); C.set_title(f"{plat} direct: length entropy per bin")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "size_bin_fold_and_entropy.png"), dpi=160)

    # window landscape
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    piv = grid.pivot(index="b", columns="a", values="d_ont")
    im = ax.imshow(piv.values, origin="lower", aspect="auto",
                   extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()], cmap="viridis")
    ax.plot(256, 650, "w*", ms=12, label="original R")
    ax.plot(best.a, best.b, "r+", ms=12, mew=2, label="ONT best")
    ax.set_xlabel("window start a (bp)"); ax.set_ylabel("window end b (bp)")
    ax.set_title("ONT: effect size of (a-b)/(>2 kb), HCC vs HBV")
    fig.colorbar(im, label="standardised difference"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "ont_window_landscape.png"), dpi=160)

    print(summ.to_string())
    print("best ONT window", best.to_dict())
    print("top1% ONT windows on PacBio: d range", top.d_pacbio.min(), top.d_pacbio.max(), "n", len(top))


if __name__ == "__main__":
    main()
