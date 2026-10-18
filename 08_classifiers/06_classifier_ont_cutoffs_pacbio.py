#!/usr/bin/env python3
"""
06_classifier_ont_cutoffs_pacbio.py

No train/test split (8 HCC vs 6 HBV). Everything is set on ONT, then applied to PacBio unchanged.
  1. ONT: leave-one-out. In each fold the model (median impute, z-score, logistic regression C=1, every feature
     of the set, same as 03_nested_cv_classifier.py) is fit on 13 samples and scores the left-out one -> ONT AUC.
  2. ONT cut-off: the probability threshold that maximises Youden J (sens + spec - 1) on the ONT leave-one-out scores.
     For R (fixed rule, nothing fitted) the cut-off is the Youden-best value of log R on ONT.
  3. Lock: final model fit on all 14 ONT samples + the ONT cut-off. PacBio is only scored: AUC, and sens/spec/
     accuracy at the locked ONT cut-off. No PacBio refit, no PacBio folds, no PacBio threshold.
  4. Second version, label-free platform scaling: each platform's features are z-scored with that platform's own
     mean and SD over its 14 samples (no labels used), then the same steps. This removes the platform scale shift
     but uses the PacBio cohort's mean, so it is reported as a sensitivity check, not the main result.
Input: per-sample feature tables (batch13 send/: feats_step4 = ONT, feats_pacbio = PacBio) and R_per_sample.csv.

  python 06_classifier_ont_cutoffs_pacbio.py --feats "results/classifier_features" \
      --r-table "results/size_ratio/R_per_sample.csv" -o classifier_cutoffs --threads 4
"""
import argparse, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SETS = [("R", "Size ratio R"), ("F_size", "F-size"), ("F_entropy_rare", "F-entropy"),
        ("F_motif_lit", "F-motif-lit"), ("F_motif", "F-motif (256)"), ("F_combined_rare", "F-combined")]
ONT, PB, INK, MUTE, HCC, HBV = "#333333", "#8B5FBF", "#222222", "#6B6B6B", "#E8703A", "#2F7BD8"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 250, "axes.titlelocation": "left"})


def is_hcc(idx):
    return ~pd.Index(idx).str.startswith("TBR44")


def fit(X, y):
    med = np.nanmedian(X, 0); med = np.where(np.isnan(med), 0, med)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0); sd[sd == 0] = 1
    clf = LogisticRegression(C=1.0, max_iter=5000).fit((Xi - mu) / sd, y)
    return med, mu, sd, clf


def predict(m, X):
    med, mu, sd, clf = m
    return clf.predict_proba((np.where(np.isnan(X), med, X) - mu) / sd)[:, 1]


def youden(y, s):
    """cut-off with the largest sens + spec - 1; ties -> midpoint, so it sits between samples"""
    u = np.unique(s); cand = np.r_[u[0] - 1e-9, (u[:-1] + u[1:]) / 2, u[-1] + 1e-9]
    j = [((s >= c) & y).sum() / y.sum() + ((s < c) & ~y).sum() / (~y).sum() - 1 for c in cand]
    return cand[int(np.argmax(j))]


def at_cut(y, s, c):
    p = s >= c
    return dict(sens=(p & y).sum() / y.sum(), spec=(~p & ~y).sum() / (~y).sum(), acc=(p == y).mean(),
                n_called_hcc=int(p.sum()))


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.where(y)[0], np.where(~y)[0]; v = []
    for _ in range(n):
        i = np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))]
        v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def run(name, Xo, yo, Xp, yp, mode):
    if name == "R":                         # fixed rule: score = log R itself
        so, sp = Xo[:, 0], Xp[:, 0]
        cut = youden(yo, so)
        loo_cut = []                        # honest ONT accuracy of the cut-off: choose it without the left-out sample
        for i in range(len(yo)):
            k = np.arange(len(yo)) != i
            loo_cut.append(so[i] >= youden(yo[k], so[k]))
        loo_acc = (np.array(loo_cut) == yo).mean()
    else:
        so = np.zeros(len(yo))
        for i in range(len(yo)):
            k = np.arange(len(yo)) != i
            so[i] = predict(fit(Xo[k], yo[k]), Xo[[i]])[0]
        cut = youden(yo, so)
        sp = predict(fit(Xo, yo), Xp)       # locked ONT model, PacBio scored only
        loo_acc = np.nan
    r = []
    for plat, y, s in [("ONT", yo, so), ("PacBio", yp, sp)]:
        lo, hi = boot_ci(y, s)
        r.append(dict(set=name, mode=mode, platform=plat, role="leave-one-out (cut-off set here)" if plat == "ONT" else "prediction only",
                      auc=roc_auc_score(y, s), lo=lo, hi=hi, cutoff=cut, **at_cut(y, s, cut),
                      loo_cutoff_acc=loo_acc if plat == "ONT" else np.nan))
    sc = pd.DataFrame(dict(set=name, mode=mode, platform=["ONT"] * len(yo) + ["PacBio"] * len(yp),
                           hcc=np.r_[yo, yp], score=np.r_[so, sp], cutoff=cut))
    return r, sc


def load(feats, sub, name, src, rtab):
    if name == "R":
        t = rtab[(rtab.platform == ("ONT" if sub == "feats_step4" else "PacBio")) & (rtab.fragments == src)]
        return t.set_index("sample")[["logR_full"]]
    return pd.read_csv(os.path.join(feats, sub, src, f"{name}.csv")).set_index("sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="folder with feats_step4/ (ONT) and feats_pacbio/ (PacBio)")
    ap.add_argument("--r-table", required=True, help="R_per_sample.csv (log R per sample, both platforms)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-t", "--threads", type=int, default=4)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "cutoff_results.csv")):
        raise SystemExit(f"{a.out} already has results; pick a new -o folder")
    os.makedirs(a.out, exist_ok=True)
    rtab = pd.read_csv(a.r_table)
    jobs = []
    for src in ("direct", "all"):
        for name, _ in SETS:
            o, p = load(a.feats, "feats_step4", name, src, rtab), load(a.feats, "feats_pacbio", name, src, rtab)
            cols = [c for c in o.columns if c in p.columns]
            o, p = o[cols], p[cols]
            for mode in ("locked", "platform_scaled"):
                Xo, Xp = o.values.astype(float), p.values.astype(float)
                if mode == "platform_scaled":   # label-free: each platform z-scored with its own mean/SD
                    Xo = (Xo - np.nanmean(Xo, 0)) / np.where(np.nanstd(Xo, 0) > 0, np.nanstd(Xo, 0), 1)
                    Xp = (Xp - np.nanmean(Xp, 0)) / np.where(np.nanstd(Xp, 0) > 0, np.nanstd(Xp, 0), 1)
                jobs.append((src, name, Xo, np.asarray(is_hcc(o.index)), Xp, np.asarray(is_hcc(p.index)), mode))
    with ThreadPoolExecutor(a.threads) as ex:
        outs = list(ex.map(lambda j: run(j[1], *j[2:6], j[6]), jobs))
    res = pd.concat([pd.DataFrame(r).assign(fragments=j[0]) for (r, _), j in zip(outs, jobs)], ignore_index=True)
    sco = pd.concat([s.assign(fragments=j[0]) for (_, s), j in zip(outs, jobs)], ignore_index=True)
    res.to_csv(os.path.join(a.out, "cutoff_results.csv"), index=False)
    sco.to_csv(os.path.join(a.out, "cutoff_scores_per_sample.csv"), index=False)
    plot(res, sco, a.out)
    print(res[["fragments", "mode", "set", "platform", "auc", "cutoff", "sens", "spec", "acc"]].round(3).to_string())


def plot(res, sco, out):
    for src in ("direct", "all"):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), gridspec_kw=dict(width_ratios=[1, 1.25], hspace=0.38, wspace=0.85))
        for i, mode in enumerate(("locked", "platform_scaled")):
            ax = axes[i, 0]; sub = res[(res.fragments == src) & (res["mode"] == mode)]
            for k, (s, lab) in enumerate(SETS):
                y0 = len(SETS) - 1 - k
                for off, pl, col in [(0.17, "ONT", ONT), (-0.17, "PacBio", PB)]:
                    r = sub[(sub.set == s) & (sub.platform == pl)].iloc[0]
                    ax.plot([r.lo, r.hi], [y0 + off, y0 + off], color=col, lw=1.6)
                    ax.scatter(r.auc, y0 + off, s=40, marker="o" if pl == "ONT" else "s", color=col, zorder=3)
                    ax.text(1.03, y0 + off, f"AUC {r.auc:.2f} · sens {r.sens:.2f} spec {r.spec:.2f}", va="center", fontsize=6.8, color=col)
            ax.axvline(0.5, color=MUTE, lw=0.8, ls=(0, (3, 3)))
            ax.set_yticks(range(len(SETS)), [l for _, l in SETS][::-1]); ax.set_xlim(0, 1.0); ax.set_xlabel("AUC with 95% CI")
            ax.set_title(("A  Locked ONT model and ONT cut-off, raw features" if mode == "locked" else
                          "C  Same, after label-free z-scoring within each platform"), fontsize=9.5, fontweight="bold")
            # per-sample scores for R and F-combined against the ONT cut-off
            ax = axes[i, 1]
            for j, s in enumerate(["R", "F_size", "F_combined_rare"]):
                for pl, dx, mk in [("ONT", -0.15, "o"), ("PacBio", 0.15, "s")]:
                    d = sco[(sco.fragments == src) & (sco["mode"] == mode) & (sco.set == s) & (sco.platform == pl)]
                    v = d.score.values; cut = d.cutoff.iloc[0]
                    if s == "R":            # put log R on a 0-1 scale for display only (shared min-max over both platforms)
                        allv = sco[(sco.fragments == src) & (sco["mode"] == mode) & (sco.set == "R")].score
                        v, cut = (v - allv.min()) / (allv.max() - allv.min()), (cut - allv.min()) / (allv.max() - allv.min())
                    x = j * 1.0 + dx + np.random.default_rng(j).uniform(-0.05, 0.05, len(v))
                    ax.scatter(x, v, marker=mk, s=22, c=np.where(d.hcc, HCC, HBV), edgecolors="white", linewidths=0.4, zorder=3)
                ax.plot([j - 0.35, j + 0.35], [cut, cut], color=INK, lw=1.2)
            ax.set_xticks([0, 1, 2], ["R (log, rescaled)", "F-size", "F-combined"])
            ax.set_ylabel("score (probability of HCC)"); ax.set_ylim(-0.05, 1.05)
            ax.set_title(("B" if mode == "locked" else "D") + "  Per sample: ONT (left, circles) and PacBio (right, squares); line = ONT cut-off",
                         fontsize=9.5, fontweight="bold")
        h = [Line2D([], [], marker="o", color=ONT, lw=1.6, label="ONT: leave-one-out AUC; cut-off chosen here"),
             Line2D([], [], marker="s", color=PB, lw=1.6, label="PacBio: locked ONT model and cut-off, prediction only"),
             Line2D([], [], marker="o", ls="", color=HCC, label="HCC"), Line2D([], [], marker="o", ls="", color=HBV, label="HBV")]
        fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Classifiers with ONT-set cut-offs applied to PacBio ({src} fragments)", x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.savefig(os.path.join(out, f"classifier_ont_cutoffs_pacbio_{src}.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
