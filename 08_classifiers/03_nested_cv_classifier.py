#!/usr/bin/env python3
"""
03_nested_cv_classifier.py
--------------------------
HCC vs HBV carrier classifier on a wide feature table (Methods 2.10, H6/H7):
L2-regularised logistic regression with nested cross-validation, a
full-pipeline permutation test, and locked models for PacBio.

 1. --selection none (default): the model uses every feature of a feature set
    fixed in advance (02_build_classifier_features.py), chosen from biology or the
    literature, never from this cohort's results. --pca N compresses the set
    inside each training fold (label-free, as in DELFI; Cristiano et al. 2019).
    --selection welch (top-k by |Welch t| inside each fold) belongs to the earlier
    exploratory classifiers and is not used for the reported models.
 2. Everything is fitted inside the training fold only: median imputation,
    standardisation, PCA, depth adjustment and (with welch) the feature ranking
    and the inner-LOOCV choice of k from --k-grid.
 3. The permutation test re-runs the whole pipeline on --n-perm (1,000) label
    permutations; p = (b + 1) / (N + 1).
 4. Two outer schemes: leave-one-out and repeated stratified K-fold (7-fold x 50).
 5. Reported: LOOCV AUC with a stratified-bootstrap 95% CI, balanced accuracy,
    sensitivity and specificity at 0.5, out-of-fold probabilities per sample.
 6. Baseline (--covariates): the same pipeline on the processing covariates only
    (eligible fragment count and concatemer-recovery %). A feature set counts as
    carrying biological signal only if it beats this baseline or stays predictive
    on direct fragments.
 7. --adjust COVFILE: every feature is regressed (OLS) on the covariates (default
    log10 n_fragments) using the training samples only, and the stored
    coefficients are applied to the held-out sample. Recovery % is not used for
    adjustment: it separates the groups completely, so adjusting for it would
    remove any group difference, genuine or not (a pure group-shift feature drops
    from AUC 0.79 to 0.29 in a test). Recovery is handled by the direct-only
    feature sets and by 04_within_group_confound_check.py.
 8. --lock writes the final model (fitted on all samples) to a .joblib file;
    --apply runs a locked model unchanged on new data (PacBio, external cohorts),
    with no retraining.

Feature table: one row per sample, first column = sample id (or a 'sample' column),
other columns numeric. A long table (sample, feature, value) can be read with --long.
Features missing in more than half of the samples are dropped (label-blind).

Outputs in --out: summary.json, <block>_oof_predictions.csv, <block>_null_auc.txt,
<block>_feature_stability.csv; with --apply: locked_model_predictions.csv and
locked_model_summary.json.

Usage:
  python 03_nested_cv_classifier.py --features feats/all/F_size.csv --metadata ont_samples.csv \
      --covariates feats/all/covariates.csv --n-perm 1000 -t 8 \
      --lock clf_F_size_all.joblib --out clf_F_size_all/
  python 03_nested_cv_classifier.py --features feats/all/F_motif.csv --metadata ont_samples.csv \
      --pca 5 --n-perm 1000 -t 8 --lock clf_F_motif_all.joblib --out clf_F_motif_all/
  python 03_nested_cv_classifier.py --features feats/all/F_size.csv --metadata ont_samples.csv \
      --adjust feats/all/covariates.csv --n-perm 1000 -t 20 --out clf_adj/F_size_all/
  python 03_nested_cv_classifier.py --apply clf_F_size_all.joblib \
      --features pacbio_feats/all/F_size.csv --metadata ont_samples.csv --out clf_locked/F_size_all/
"""
import argparse, json, os, re, sys, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")


# ---------- pipeline pieces ----------
def welch_rank(X, y):
    a, b = X[y == 1], X[y == 0]
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    t = (a.mean(0) - b.mean(0)) / np.sqrt(va + vb + 1e-12)
    t = np.nan_to_num(np.abs(t), nan=-1)
    return np.argsort(-t, kind="stable")


SELECTION = "welch"   # set from --selection in main()
N_PCA = 0             # set from --pca in main()
N_ADJ = 0             # number of trailing covariate columns in X used only for in-fold adjustment (--adjust)


def _design(Cc, cmu, csd):
    return np.c_[np.ones(len(Cc)), (Cc - cmu) / csd]


def fit(X, y, k, C):
    adj = None
    if N_ADJ > 0:                               # split off covariates; they never enter the classifier
        X, Cc = X[:, :-N_ADJ], X[:, -N_ADJ:]
    med = np.nanmedian(X, 0)
    med = np.where(np.isnan(med), 0, med)
    Xi = np.where(np.isnan(X), med, X)
    if N_ADJ > 0:                               # in-fold OLS residualisation on the covariates
        cmu, csd = Cc.mean(0), Cc.std(0)
        csd[csd == 0] = 1
        B = np.linalg.lstsq(_design(Cc, cmu, csd), Xi, rcond=None)[0]
        Xi = Xi - _design(Cc, cmu, csd) @ B
        adj = dict(cmu=cmu, csd=csd, B=B)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1
    Z = (Xi - mu) / sd
    if SELECTION == "none":
        sel = np.arange(Z.shape[1])            # every pre-specified feature, no label-based choice
    else:
        sel = welch_rank(Z, y)[:k]
    Zs, pca = Z[:, sel], None
    if N_PCA > 0:                               # label-free compression, fit on the training fold only
        pca = PCA(n_components=min(N_PCA, Zs.shape[0] - 1, Zs.shape[1])).fit(Zs)
        Zs = pca.transform(Zs)
    clf = LogisticRegression(C=C, max_iter=5000).fit(Zs, y)
    return dict(med=med, mu=mu, sd=sd, sel=sel, pca=pca, clf=clf, adj=adj)


def predict(m, X):
    adj = m.get("adj")
    if adj is not None:
        n_adj = adj["B"].shape[0] - 1
        X, Cc = X[:, :-n_adj], X[:, -n_adj:]
    Xi = np.where(np.isnan(X), m["med"], X)
    if adj is not None:
        Xi = Xi - _design(Cc, adj["cmu"], adj["csd"]) @ adj["B"]
    Z = (Xi - m["mu"]) / m["sd"]
    Zs = Z[:, m["sel"]]
    if m.get("pca") is not None:
        Zs = m["pca"].transform(Zs)
    return m["clf"].predict_proba(Zs)[:, 1]


def choose_k(X, y, k_grid, C):
    """Inner LOOCV over the training fold; pick k with best AUC (ties -> smaller k)."""
    if len(k_grid) == 1 or SELECTION == "none":
        return k_grid[0]
    best, best_auc = k_grid[0], -1
    for k in sorted(k_grid):
        p = np.empty(len(y))
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            if len(np.unique(y[tr])) < 2:
                return k_grid[0]
            p[i] = predict(fit(X[tr], y[tr], min(k, X.shape[1]), C), X[i:i + 1])[0]
        auc = roc_auc_score(y, p)
        if auc > best_auc + 1e-9:
            best, best_auc = k, auc
    return best


def outer_loocv(X, y, k_grid, C, record=False):
    n = len(y); p = np.empty(n); sels, ks = [], []
    for i in range(n):
        tr = np.arange(n) != i
        k = choose_k(X[tr], y[tr], k_grid, C)
        m = fit(X[tr], y[tr], min(k, X.shape[1]), C)
        p[i] = predict(m, X[i:i + 1])[0]
        if record:
            sels.append(m["sel"]); ks.append(k)
    return p, sels, ks


def outer_repeated_kfold(X, y, k_grid, C, n_splits, n_repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, probs = [], np.zeros((n_repeats, len(y)))
    for r_i, (tr, te) in enumerate(rskf.split(X, y)):
        rep = r_i // n_splits
        k = choose_k(X[tr], y[tr], k_grid, C)
        probs[rep, te] = predict(fit(X[tr], y[tr], min(k, X.shape[1]), C), X[te])
    for rep in range(n_repeats):
        aucs.append(roc_auc_score(y, probs[rep]))
    return np.array(aucs), probs.mean(0)


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(n):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        v.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(v, [2.5, 97.5])


def summarize(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    sens = (pred[y == 1] == 1).mean(); spec = (pred[y == 0] == 0).mean()
    return dict(auc=roc_auc_score(y, p), balanced_accuracy=(sens + spec) / 2, accuracy=(pred == y).mean(),
                sensitivity=sens, specificity=spec)


def jaccard_mean(sels):
    s = [set(x) for x in sels]; v = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            v.append(len(s[i] & s[j]) / len(s[i] | s[j]))
    return float(np.mean(v)) if v else np.nan


# ---------- io ----------
def load_features(path, long=False):
    df = pd.read_csv(path, sep=None, engine="python")
    if long:
        df = df.pivot_table(index=df.columns[0], columns=df.columns[1], values=df.columns[2])
    else:
        idx = "sample" if "sample" in df.columns else df.columns[0]
        df = df.set_index(idx)
    df.index = df.index.astype(str).str.strip()
    return df.apply(pd.to_numeric, errors="coerce")


def load_meta(path):
    m = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    m.columns = [c.strip().lower() for c in m.columns]
    return dict(zip(m["sample"].astype(str).str.strip(), m["group"].astype(str).str.strip()))


def load_adjust(path, cols, log_cols):
    A = load_features(path)
    miss = [c for c in cols if c not in A.columns]
    if miss:
        sys.exit(f"--adjust file {path} lacks columns {miss}; has {list(A.columns)}")
    A = A[cols].copy()
    for c in log_cols:
        if c in A.columns:
            A[c] = np.log10(A[c].clip(lower=1))
    return A


def run_block(name, F, y, samples, a, out, adj=None):
    """adj: optional DataFrame of covariates (same rows as F) for in-fold residualisation."""
    global N_ADJ
    N_ADJ = 0 if adj is None else adj.shape[1]
    n_feat = F.shape[1]
    X = F.to_numpy(float) if adj is None else np.c_[F.to_numpy(float), adj.to_numpy(float)]
    kg = [k for k in a.k_grid if k <= n_feat] or [n_feat]
    p_loo, sels, ks = outer_loocv(X, y, kg, a.C, record=True)
    s = summarize(y, p_loo); ci = boot_ci(y, p_loo)
    aucs_rk, p_rk = outer_repeated_kfold(X, y, kg, a.C, a.kfold, a.repeats, a.seed)

    cfg = (SELECTION, N_PCA, N_ADJ)

    def perm_auc(seed):
        global SELECTION, N_PCA, N_ADJ
        SELECTION, N_PCA, N_ADJ = cfg          # make worker processes use the same pipeline settings
        yp = np.random.default_rng(seed).permutation(y)
        return roc_auc_score(yp, outer_loocv(X, yp, kg, a.C)[0])
    null = np.array(Parallel(n_jobs=a.n_jobs)(delayed(perm_auc)(a.seed + 1000 + i) for i in range(a.n_perm))) if a.n_perm else np.array([])
    p_perm = (np.sum(null >= s["auc"] - 1e-12) + 1) / (len(null) + 1) if len(null) else np.nan

    freq = pd.Series(np.concatenate(sels)).value_counts()
    stab = pd.DataFrame({"feature": F.columns[freq.index], "folds_selected": freq.values, "of": len(y)})
    stab.to_csv(os.path.join(out, f"{name}_feature_stability.csv"), index=False)
    pd.DataFrame({"sample": samples, "label": y, "oof_prob_loocv": p_loo, "oof_prob_repkfold_mean": p_rk,
                  "k_chosen": ks}).to_csv(os.path.join(out, f"{name}_oof_predictions.csv"), index=False)
    if len(null):
        np.savetxt(os.path.join(out, f"{name}_null_auc.txt"), null, fmt="%.4f")
    res = dict(block=name, n_features=n_feat, adjusted_for=list(adj.columns) if adj is not None else [], n=len(y), n_case=int(y.sum()), k_grid=kg,
               k_chosen_counts=pd.Series(ks).value_counts().to_dict(), **{f"loocv_{k}": v for k, v in s.items()},
               loocv_auc_ci95=list(ci), repkfold_auc_mean=float(aucs_rk.mean()),
               repkfold_auc_2p5_97p5=list(np.percentile(aucs_rk, [2.5, 97.5])),
               perm_p=p_perm, n_perm=len(null), null_auc_95th=float(np.percentile(null, 95)) if len(null) else None,
               null_auc_max=float(null.max()) if len(null) else None, jaccard_mean=jaccard_mean(sels),
               misclassified=[samples[i] for i in np.where((p_loo >= 0.5).astype(int) != y)[0]])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="path to a wide feature table (csv, one row per sample)")
    ap.add_argument("--metadata", required=True, help="path to the sample sheet with columns sample, group")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--long", action="store_true", help="--features is a long table (sample, feature, value)")
    ap.add_argument("--feature-regex", help="keep only features whose name matches this regular expression")
    ap.add_argument("--covariates", help="path to the covariate table (covariates.csv) for the baseline model")
    ap.add_argument("--adjust", help="path to the covariate table (covariates.csv); every feature is residualised on "
                                     "--adjust-cols inside each training fold")
    ap.add_argument("--adjust-cols", default="n_fragments", help="covariate columns used by --adjust")
    ap.add_argument("--adjust-log", default="n_fragments", help="of those, columns to log10-transform first")
    ap.add_argument("--exclude-samples", default="", help="comma-separated samples to leave out (sensitivity run: TBR3976)")
    ap.add_argument("--k-grid", default="5", type=lambda s: [int(x) for x in s.split(",")], help="k values for --selection welch")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 penalty of the logistic regression")
    ap.add_argument("--selection", choices=["none", "welch"], default="none",
                    help="none (default): use every feature in the pre-specified set, no label-based selection; "
                         "welch: top-k by |Welch t| inside each fold (exploratory only)")
    ap.add_argument("--pca", type=int, default=0, help="compress the feature set to this many PCs inside each fold (DELFI-style); 0 = off")
    ap.add_argument("--kfold", type=int, default=7, help="folds of the repeated stratified K-fold")
    ap.add_argument("--repeats", type=int, default=50, help="repeats of the repeated stratified K-fold")
    ap.add_argument("--n-perm", type=int, default=1000, help="label permutations for the permutation test")
    ap.add_argument("-t", "--threads", "--n-jobs", dest="n_jobs", type=int, default=4, help="parallel workers for the permutation test")
    ap.add_argument("--case", default="HCC"); ap.add_argument("--control", default="HBV_carrier")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--lock", help="write the final model, fitted on all samples, to this .joblib file")
    ap.add_argument("--apply", help="path to a locked .joblib model to apply to --features (no training)")
    a = ap.parse_args()
    global SELECTION, N_PCA, N_ADJ
    SELECTION, N_PCA = a.selection, a.pca
    os.makedirs(a.out, exist_ok=True)
    grp = load_meta(a.metadata)
    F = load_features(a.features, a.long)

    if a.apply:
        m = load(a.apply)
        SELECTION, N_PCA = m.get("selection", "none"), m.get("pca", 0)
        missing = [c for c in m["features"] if c not in F.columns]
        if missing:
            sys.exit(f"{len(missing)} model features missing in new data, e.g. {missing[:5]}")
        F = F[[s in grp for s in F.index]]
        X = F[m["features"]].to_numpy(float)
        if m.get("adjust_cols"):
            if not a.adjust:
                sys.exit(f"model was trained with --adjust {m['adjust_cols']}; give --adjust COVFILE for the new data")
            A = load_adjust(a.adjust, m["adjust_cols"], m["adjust_log"]).reindex(F.index)
            X = np.c_[X, A.to_numpy(float)]
        p = predict(m["model"], X)
        y = np.array([grp[s] == a.case for s in F.index], int)
        out = pd.DataFrame({"sample": F.index, "label": y, "prob": p})
        out.to_csv(os.path.join(a.out, "locked_model_predictions.csv"), index=False)
        res = summarize(y, p) if len(np.unique(y)) == 2 else {}
        if res:
            res["auc_ci95"] = list(boot_ci(y, p))
        json.dump(res, open(os.path.join(a.out, "locked_model_summary.json"), "w"), indent=2, default=float)
        print(json.dumps(res, indent=2, default=float)); return

    excl = {s.strip() for s in a.exclude_samples.split(",") if s.strip()}
    F = F[[s in grp and grp[s] in (a.case, a.control) and s not in excl for s in F.index]]
    if a.feature_regex:
        F = F.loc[:, [c for c in F.columns if re.search(a.feature_regex, c)]]
    F = F.loc[:, F.notna().mean() >= 0.5]          # label-blind: drop features missing in >half the samples
    y = np.array([grp[s] == a.case for s in F.index], int)
    samples = list(F.index)
    print(f"{len(y)} samples ({y.sum()} {a.case}), {F.shape[1]} features, k grid {a.k_grid}")

    adj_cols = [c.strip() for c in a.adjust_cols.split(",") if c.strip()] if a.adjust else []
    adj_log = [c.strip() for c in a.adjust_log.split(",") if c.strip()]
    A = None
    if a.adjust:
        A = load_adjust(a.adjust, adj_cols, adj_log).reindex(F.index)
        if A.isna().any().any():
            sys.exit(f"--adjust: missing covariate values for {list(A.index[A.isna().any(axis=1)])}")
        print(f"adjusting every feature for {adj_cols} (log10: {[c for c in adj_log if c in adj_cols]}) inside each training fold")
    results = [run_block("features_adjusted" if A is not None else "features", F, y, samples, a, a.out, adj=A)]
    if a.covariates:
        Cv = load_features(a.covariates).reindex(F.index)
        results.append(run_block("baseline_covariates", Cv, y, samples, a, a.out))

    if a.lock:
        N_ADJ = 0 if A is None else A.shape[1]
        X = F.to_numpy(float) if A is None else np.c_[F.to_numpy(float), A.to_numpy(float)]
        k = choose_k(X, y, [k for k in a.k_grid if k <= F.shape[1]] or [F.shape[1]], a.C)
        m = fit(X, y, min(k, F.shape[1]), a.C)
        dump({"model": m, "features": list(F.columns), "k": k, "C": a.C, "selection": SELECTION, "pca": N_PCA,
              "adjust_cols": adj_cols, "adjust_log": adj_log, "trained_on": samples}, a.lock)
        sel = [F.columns[i] for i in m["sel"]]
        json.dump({"selection": SELECTION, "pca_components": N_PCA, "k": k, "features_used": sel,
                   "coefficients": m["clf"].coef_[0].tolist()},
                  open(os.path.splitext(a.lock)[0] + "_card.json", "w"), indent=2)
        print(f"locked model: k={k}, features={sel}")

    json.dump(results, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    for r in results:
        print(f"\n[{r['block']}] AUC={r['loocv_auc']:.3f} (95% CI {r['loocv_auc_ci95'][0]:.2f}-{r['loocv_auc_ci95'][1]:.2f}), "
              f"bal.acc={r['loocv_balanced_accuracy']:.3f}, perm p={r['perm_p']:.4f} ({r['n_perm']} perms), "
              f"repeated {a.kfold}-fold AUC={r['repkfold_auc_mean']:.3f}, Jaccard={r['jaccard_mean']:.3f}, "
              f"misclassified={r['misclassified']}")


if __name__ == "__main__":
    main()
