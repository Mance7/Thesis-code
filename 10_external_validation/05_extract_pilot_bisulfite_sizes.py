#!/usr/bin/env python3
"""
05_extract_pilot_bisulfite_sizes.py

External validation, CRA001537 pilot (Zhang et al. 2020): cfDNA fragment sizes from
paired-end whole-genome bisulfite reads of one HCC, one hepatitis and one healthy
plasma sample. Bisulfite reads cannot be aligned with plain bwa, so we follow the
bwa-meth scheme, only as far as needed for SIZES (methylation is not called):
  1. reference: chosen chromosomes (chr1-chr8), two converted copies (C->T as f_chrN,
     G->A as r_chrN), bwa index built once
  2. reads: read 1 fully C->T, read 2 fully G->A, interleaved into bwa mem -p
  3. fragment length = |TLEN| of proper pairs (read 1 only, primary, MAPQ >= 20, both
     reads on one contig); pairs whose read 1 lies on the strand that disagrees with
     its contig are discarded
Short-read stand-in for the long-read ratio R, fixed before any external data were seen:
  R_sr = fragments 256-600 bp / fragments 100-250 bp   (predicted higher in HCC)
Only sizes up to --max-len are counted. No trimming is applied (raw reads).

Input : folder with paired FASTQ (*_f1/_r2, _R1/_R2 or _1/_2 .fq.gz), GRCh38 FASTA with .fai,
        sheet with one line per sample: run<TAB>group (runs_pilot.tsv)
Output: OUT/ref_bs/ref_bs.fa (+ bwa index), OUT/<run>.tlen_hist.tsv.gz,
        OUT/ev_pilot_summary.tsv, OUT/ev_pilot_hist.tsv.gz, OUT/ev_pilot_groups.tsv

Requires: bwa 0.7.19, samtools 1.23.1
"""
import argparse, shutil, gzip, os, re, subprocess, sys, glob, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description="CRA001537 pilot: fragment sizes from bisulfite read pairs")
p.add_argument("--fastq-dir", required=True, help="folder with the paired pilot FASTQ files")
p.add_argument("--ref", required=True,
               help="GRCh38 FASTA (with .fai)")
p.add_argument("-o", "--out", required=True, help="output folder (also holds the converted reference)")
p.add_argument("--chroms", default=",".join(f"chr{i}" for i in range(1, 9)), help="chromosomes in the converted reference")
p.add_argument("--max-pairs", type=int, default=5_000_000, help="read pairs used per sample")
p.add_argument("--max-len", type=int, default=1000, help="longest fragment counted (bp)")
p.add_argument("--min-mapq", type=int, default=20, help="minimum mapping quality")
p.add_argument("-t", "--threads", type=int, default=16, help="total bwa threads (split over --jobs)")
p.add_argument("--jobs", type=int, default=3, help="samples aligned in parallel")
p.add_argument("--sheet", default=str(HERE / "runs_pilot.tsv"), help="lines 'run group'; only these samples are used")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)

for tool in ("bwa", "samtools"):
    if not shutil.which(tool):
        sys.exit(f"'{tool}' not found on PATH: run 'conda activate thesis' first")
# ---------- 1. find read pairs (checked first, before the slow index step) ----------
fq = sorted(glob.glob(os.path.join(a.fastq_dir, "**", "*.f*q.gz"), recursive=True))
pat = re.compile(r"^(.*?)[._](f1|r1|R1|1)(\.f(ast)?q\.gz)$")
pairs = {}
for f in fq:
    m = pat.match(os.path.basename(f))
    if not m:
        continue
    stem, tag, ext = m.group(1), m.group(2), m.group(3)
    sep = os.path.basename(f)[len(stem)]
    mate = {"f1": "r2", "r1": "r2", "R1": "R2", "1": "2"}[tag]
    f2 = os.path.join(os.path.dirname(f), f"{stem}{sep}{mate}{ext}")
    if os.path.exists(f2):
        pairs[stem] = (f, f2)
if not pairs:
    sys.exit(f"no paired FASTQ found under {a.fastq_dir} (looked for *_f1/_r2, _R1/_R2, _1/_2 .fq.gz). "
             f"Files seen: {[os.path.basename(x) for x in fq[:10]]}")
log(f"{len(pairs)} paired samples:", ", ".join(pairs))
groups = {}
if a.sheet:
    for l in open(a.sheet):
        if l.strip():
            k, g = l.split()[:2]; groups[k] = g
if groups:
    missing = [k for k in groups if k not in pairs]
    if missing:
        sys.exit(f"sheet names not found among the FASTQ pairs: {missing}\nPairs found: {list(pairs)}")
    unused = [k for k in pairs if k not in groups]
    if unused:
        log("note: pairs not listed in the sheet are skipped:", unused)
        pairs = {k: v for k, v in pairs.items() if k in groups}

# ---------- 2. converted reference ----------
cref = os.path.join(a.out, "ref_bs", "ref_bs.fa")
os.makedirs(os.path.dirname(cref), exist_ok=True)
if not os.path.exists(cref + ".bwt"):
    chroms = a.chroms.split(",")
    log("extracting", len(chroms), "chromosomes and writing C->T / G->A copies")
    ct, ga = str.maketrans("Cc", "Tt"), str.maketrans("Gg", "Aa")
    seq = subprocess.run(["samtools", "faidx", a.ref] + chroms, capture_output=True, text=True, check=True).stdout
    with open(cref, "w") as fo:
        for tag, tr in (("f_", ct), ("r_", ga)):
            for line in seq.splitlines():
                fo.write(">" + tag + line[1:].split()[0] + "\n" if line.startswith(">") else line.translate(tr) + "\n")
    del seq
    log("bwa index (one-off; about 1 h for chr1-chr8)")
    r = subprocess.run(["bwa", "index", cref], capture_output=True, text=True)
    if r.returncode:
        sys.exit("bwa index failed (exit %d). Last lines of its message:\n%s\nFree space here: %s"
                 % (r.returncode, "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:]),
                    subprocess.run(["df", "-h", os.path.dirname(cref)], capture_output=True, text=True).stdout))
log("reference ready:", cref)

# ---------- 3. align and count ----------
def convert(f1, f2, out):
    """write the first --max-pairs pairs, read 1 C->T and read 2 G->A, interleaved into bwa"""
    ct, ga = str.maketrans("C", "T"), str.maketrans("G", "A")
    with gzip.open(f1, "rt") as a1, gzip.open(f2, "rt") as a2:
        n = 0
        while n < a.max_pairs:
            r1 = [a1.readline() for _ in range(4)]; r2 = [a2.readline() for _ in range(4)]
            if not r1[0] or not r2[0]:
                break
            n1, n2 = r1[0].split()[0], r2[0].split()[0]
            out.write(f"{n1}\n{r1[1].strip().translate(ct)}\n+\n{r1[3].strip()}\n"
                      f"{n1}\n{r2[1].strip().translate(ga)}\n+\n{r2[3].strip()}\n")
            n += 1
    out.close()
    return n

def run(stem):
    """align one sample and count proper-pair fragment lengths"""
    f1, f2 = pairs[stem]
    th = max(1, a.threads // a.jobs)
    hist = [0] * (a.max_len + 1)
    bwa = subprocess.Popen(["bwa", "mem", "-p", "-t", str(th), cref, "-"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    import threading
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("n", convert(f1, f2, bwa.stdin))); t.start()
    n_pp = n_wrong = 0
    for line in bwa.stdout:
        if line[0] == "@":
            continue
        c = line.split("\t", 10)
        fl = int(c[1])
        if fl & 0x904 or not fl & 0x40 or not fl & 0x2 or int(c[4]) < a.min_mapq or c[6] != "=":
            continue
        # bisulfite strand logic: read 1 forward must sit on f_ (top) contig, reverse on r_ (bottom)
        if (not fl & 0x10) != c[2].startswith("f_"):
            n_wrong += 1
            continue
        L = abs(int(c[8]))
        n_pp += 1
        if 0 < L <= a.max_len:
            hist[L] += 1
    t.join(); bwa.wait()
    with gzip.open(os.path.join(a.out, f"{stem}.tlen_hist.tsv.gz"), "wt") as fo:
        fo.write("length\tcount\n" + "".join(f"{i}\t{c}\n" for i, c in enumerate(hist) if c))
    s = lambda lo, hi: sum(hist[lo:hi + 1])
    tot = s(1, a.max_len)
    med = next(i for i in range(len(hist)) if sum(hist[:i + 1]) >= tot / 2) if tot else 0
    row = dict(run=stem, group=groups.get(stem, "?"), pairs_read=res.get("n", 0), proper_pairs_kept=n_pp,
               strand_inconsistent=n_wrong, frags_1_1000=tot, median_len=med,
               n_100_250=s(100, 250), n_256_600=s(256, 600), n_256_650=s(256, 650),
               R_sr=round(s(256, 600) / max(1, s(100, 250)), 5),
               share_le150=round(s(1, 150) / max(1, tot), 5), share_151_255=round(s(151, 255) / max(1, tot), 5),
               share_256_650=round(s(256, 650) / max(1, tot), 5))
    log(stem, row)
    return row, hist

with ThreadPoolExecutor(a.jobs) as ex:
    rows = list(ex.map(run, sorted(pairs)))
cols = list(rows[0][0])
with open(os.path.join(a.out, "ev_pilot_summary.tsv"), "w") as fo:
    fo.write("\t".join(cols) + "\n" + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r, _ in rows))
with gzip.open(os.path.join(a.out, "ev_pilot_hist.tsv.gz"), "wt") as fo:
    fo.write("run\t" + "\t".join(map(str, range(1, a.max_len + 1))) + "\n")
    for r, h in rows:
        fo.write(r["run"] + "\t" + "\t".join(map(str, h[1:])) + "\n")
if groups:
    from statistics import median
    by = {}
    for r, _ in rows:
        by.setdefault(r["group"], []).append(r)
    with open(os.path.join(a.out, "ev_pilot_groups.tsv"), "w") as fo:
        fo.write("group\tn\tR_sr_values\tR_sr_median\tshare_256_650_median\tmedian_len_median\n")
        for g, rs in sorted(by.items()):
            fo.write(f"{g}\t{len(rs)}\t{','.join(str(r['R_sr']) for r in rs)}\t{median(r['R_sr'] for r in rs)}\t"
                     f"{median(r['share_256_650'] for r in rs)}\t{median(r['median_len'] for r in rs)}\n")
    print(open(os.path.join(a.out, "ev_pilot_groups.tsv")).read())
log("done:", os.path.join(a.out, "ev_pilot_summary.tsv"), "and ev_pilot_hist.tsv.gz")
