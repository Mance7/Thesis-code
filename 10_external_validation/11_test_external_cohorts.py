#!/usr/bin/env python3
"""
11_test_external_cohorts.py

Tests on the outputs of 07_ or 09_extract_external_cohorts*.py (Methods 2.13), case group
against control group(s). Size tests are one-sided exact Mann-Whitney tests in the
discovery direction (both directions are reported).
  1. size: R_sr = n(256-600) / n(100-250) and the shares <=150, 151-255 and 256-650 bp
  2. promoter sets (discovery definitions): HCC-expressed = S3+S4+S5, S5 high tertile, S4, S3,
     S1 blood-silent, S2 blood-expressed, S6 liver-enriched.
     Metric fsr = n(<=246) / n(<=600) (discovery definition; HCC LOWER in the discovery cohort),
     pooled over the promoters of the set, per sample; group difference = Cohen's d;
     null = B random promoter sets of the same size drawn from all promoters seen in this data set
     (not matched on blood expression or CpG: a weaker null than the discovery one);
     p = (b+1)/(B+1), b = random sets with |d| >= |d_obs|
  3. end motifs (wgs data only): CLR of the 256 5' 4-mers per size class; the six motifs named in
     Results 3.7.4 (CCCA, CCAG, CCTG, GGAG higher; AGTT, ATTT lower in HCC) are tested in their
     expected direction, and the whole 256-motif pattern is compared with the discovery cohort
     (<=600 bp bin, HCC - HBV CLR differences on ONT and PacBio): Spearman correlation and sign
     agreement of the 30 motifs with the largest discovery differences.

Uses:
  GSE71378: --dir out_gse --sheet runs_gse71378_short_batch.tsv --case liver_cancer --ctrl healthy,noncancer_disease
  CRA001537 cohort: --dir out_zhang --sheet out_zhang/ev_summary.tsv --case HCC --ctrl hepatitis
              (secondary: --ctrl cirrhosis, --ctrl HCC_post_surgery)
Sheet: tab-separated with a header row; first column run, second column group (extra columns ignored).

Output: OUT/PREFIX_size.csv, PREFIX_promoter_sets.csv, PREFIX_motifs_named.csv, PREFIX_motifs_all.csv
        and a printed summary (including the motif-pattern agreement)
"""
import argparse, os, re, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ap = argparse.ArgumentParser(description="External cohorts: size, promoter-set and end-motif tests")
ap.add_argument("--dir", required=True, help="output folder of 07_ or 09_extract_external_cohorts*.py (ev_summary.tsv, ...)")
ap.add_argument("--sheet", required=True, help="TSV with header: run, group (only these runs are used)")
ap.add_argument("-o", "--out", required=True, help="output folder")
ap.add_argument("--prefix", default="ev", help="prefix of the output file names")
ap.add_argument("--case", required=True, help="case group(s), comma-separated")
ap.add_argument("--ctrl", required=True, help="control group(s), comma-separated")
ap.add_argument("--subsets", required=True,
                help="folder with the discovery promoter gene sets (S1_hcc_blood_silent.txt ... S6_liver_enriched.txt)")
ap.add_argument("--disc-dir", required=True,
                help="discovery feature folder with feats_step4/direct/F_motif.csv (ONT) and feats_pacbio/direct/F_motif.csv (PacBio)")
ap.add_argument("--B", type=int, default=1000, help="random promoter sets for the null")
ap.add_argument("--seed", type=int, default=16, help="random seed")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
a.out = os.path.join(a.out, a.prefix)   # all output files start with OUT/PREFIX
case, ctrl = a.case.split(","), a.ctrl.split(",")
sh = pd.read_csv(a.sheet, sep=None, engine="python"); sh = sh.set_index(sh.columns[0])
grp = sh.iloc[:, 0]
cases = [r for r in grp.index if grp[r] in case]; ctrls = [r for r in grp.index if grp[r] in ctrl]
summ = pd.read_csv(f"{a.dir}/ev_summary.tsv", sep="\t").set_index("run")
cases = [r for r in cases if r in summ.index]; ctrls = [r for r in ctrls if r in summ.index]
print(f"case {len(cases)}: {cases}\nctrl {len(ctrls)}: {ctrls}")
rng = np.random.default_rng(a.seed)

def d_cohen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan

def mw(x, y, alt="greater"):
    if len(x) < 2 or len(y) < 1:
        return np.nan
    return mannwhitneyu(x, y, alternative=alt, method="exact").pvalue

# ---------- 1. size ----------
rows = []
for m in ("R_sr", "share_le150", "share_151_255", "share_256_650"):
    x, y = summ.loc[cases, m], summ.loc[ctrls, m]
    rows.append(dict(metric=m, case_values=",".join(f"{v:.4g}" for v in x), case_median=x.median(), ctrl_median=y.median(),
                     ctrl_range=f"{y.min():.4g}-{y.max():.4g}", fold=x.median() / max(y.median(), 1e-12),
                     p_greater=mw(x, y), p_less=mw(x, y, "less")))
size = pd.DataFrame(rows); size.to_csv(f"{a.out}_size.csv", index=False)
print("\nSIZE\n", size.to_string(index=False))

# ---------- 2. promoter sets ----------
pf = f"{a.dir}/ev_promoters.tsv.gz"
if os.path.exists(pf):
    P = pd.read_csv(pf, sep="\t")
    P = P[P.run.isin(cases + ctrls)]
    idx = pd.read_csv(f"{a.dir}/ev_promoter_index.tsv", sep="\t")
    ids = idx.promoter.tolist(); base = [re.sub(r"\.\d+$", "", i) for i in ids]
    pos = {b: i for i, b in enumerate(base)}
    runs = cases + ctrls
    S = np.zeros((len(runs), len(ids)), dtype=np.int64); Lc = np.zeros_like(S)
    ri = {r: i for i, r in enumerate(runs)}; pi_ = {p: i for i, p in enumerate(ids)}
    short = (P["n_1_150"] + P["n_151_246"]).to_numpy(); le220 = (P["n_1_150"] + P["n_151_246"] + P["n_247_600"]).to_numpy()
    r_ix = P.run.map(ri).to_numpy(); p_ix = P.promoter.map(pi_).to_numpy()
    np.add.at(S, (r_ix, p_ix), short); np.add.at(Lc, (r_ix, p_ix), le220)
    def load(f):
        return [l.strip() for l in open(os.path.join(a.subsets, f)) if l.strip()]
    sets = {"HCC-expressed (S3+S4+S5)": load("S3_hcc_tertile_low.txt") + load("S4_hcc_tertile_mid.txt") + load("S5_hcc_tertile_high.txt"),
            "S5 high tertile": load("S5_hcc_tertile_high.txt"), "S4 mid tertile": load("S4_hcc_tertile_mid.txt"),
            "S3 low tertile": load("S3_hcc_tertile_low.txt"), "S1 blood-silent": load("S1_hcc_blood_silent.txt"),
            "S2 blood-expressed": load("S2_hcc_blood_expressed.txt"), "S6 liver-enriched": load("S6_liver_enriched.txt")}
    ncase = len(cases)
    def stat(cols):
        sh_ = S[:, cols].sum(1); tot = Lc[:, cols].sum(1)
        f = sh_ / np.maximum(tot, 1)
        return f, d_cohen(f[:ncase], f[ncase:])
    allp = np.arange(len(ids)); rows = []
    for name, genes in sets.items():
        cols = np.array([pos[g] for g in genes if g in pos])
        if len(cols) < 50:
            rows.append(dict(set=name, n_promoters=len(cols), note="too few promoters on the indexed chromosomes")); continue
        f, d = stat(cols)
        null = []
        if not np.isnan(d):
            for _ in range(a.B):
                null.append(stat(rng.choice(allp, len(cols), replace=False))[1])
        null = np.array([x for x in null if not np.isnan(x)])
        b = int((np.abs(null) >= abs(d)).sum()) if len(null) else np.nan
        rows.append(dict(set=name, n_promoters=len(cols), fsr_case_median=np.median(f[:ncase]), fsr_ctrl_median=np.median(f[ncase:]),
                         d=d, p_mw_less=mw(f[:ncase], f[ncase:], "less"), null_median_d=np.median(null) if len(null) else np.nan,
                         p_random=(b + 1) / (len(null) + 1) if len(null) else np.nan, B=len(null)))
    ps = pd.DataFrame(rows); ps.to_csv(f"{a.out}_promoter_sets.csv", index=False)
    print("\nPROMOTER SETS (fsr = n<=246 / n<=600; discovery direction: HCC lower)\n", ps.to_string(index=False))

# ---------- 3. motifs ----------
mf = f"{a.dir}/ev_motifs.tsv.gz"
if os.path.exists(mf):
    M = pd.read_csv(mf, sep="\t")
    named = {"CCCA": 1, "CCAG": 1, "CCTG": 1, "GGAG": 1, "AGTT": -1, "ATTT": -1}   # Results 3.7.4: + higher in HCC, - lower
    rows, allrows = [], []
    for cl, d in M.groupby("class"):
        d = d.set_index("run").drop(columns="class").reindex(cases + ctrls).dropna()
        Pm = d + 0.5; clr = np.log(Pm.div(Pm.sum(1), axis=0)); clr = clr.sub(clr.mean(1), axis=0)
        cc = [r for r in cases if r in clr.index]; kk = [r for r in ctrls if r in clr.index]
        diff = clr.loc[cc].mean() - clr.loc[kk].mean()
        for m, sgn in named.items():
            rows.append(dict(cls=cl, motif=m, expected=("higher" if sgn > 0 else "lower"), clr_diff=diff[m],
                             p_expected_dir=mw(clr.loc[cc, m], clr.loc[kk, m], "greater" if sgn > 0 else "less"),
                             sign_agrees=bool(np.sign(diff[m]) == sgn)))
        allrows.append(pd.DataFrame({"cls": cl, "motif": diff.index, "diff": diff.values}))
    mn = pd.DataFrame(rows); mn.to_csv(f"{a.out}_motifs_named.csv", index=False)
    print("\nNAMED MOTIFS\n", mn.to_string(index=False))
    AM = pd.concat(allrows); AM.to_csv(f"{a.out}_motifs_all.csv", index=False)
    # agreement of the whole 256-motif pattern with the discovery cohort (<=600 bp bin; CLR differences, HCC - HBV)
    disc = {}
    for plat, sub in (("ONT", "feats_step4"), ("PacBio", "feats_pacbio")):
        fp = f"{a.disc_dir}/{sub}/direct/F_motif.csv"
        if os.path.exists(fp):
            dd = pd.read_csv(fp, index_col=0); dd = dd[[c for c in dd.columns if c.startswith("clr_<=600bp_")]]; dd.columns = [c[-4:] for c in dd.columns]
            hbv = [s for s in dd.index if s.startswith("TBR44")]; hcc = [s for s in dd.index if s not in hbv]
            disc[plat] = dd.loc[hcc].mean() - dd.loc[hbv].mean()
    if disc:
        D = pd.DataFrame(disc)
        for cl in AM.cls.unique():
            g = AM[AM.cls == cl].set_index("motif")["diff"].reindex(D.index)
            out = {plat: spearmanr(D[plat], g)[0] for plat in D}
            top = D.mean(1).abs().sort_values(ascending=False).index[:30]
            ag = int((np.sign(g[top]) == np.sign(D.loc[top].mean(1))).sum())
            print(f"class {cl}: Spearman with discovery (all 256 motifs) {out}; top-30 discovery motifs sign agreement {ag}/30")

print("\nsee the CSV files with prefix", a.out)
