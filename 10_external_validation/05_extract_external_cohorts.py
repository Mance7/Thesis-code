#!/usr/bin/env python3
"""
05_extract_external_cohorts.py

External validation, extraction for the short-read cohorts (Methods 2.13):
  GSE71378 (plain WGS, FASTQ from ENA)                       --mode wgs --runs runs_gse71378.tsv
  CRA001537 low-pass cohort (Zhang et al. 2020, WGBS, GSA)   --mode bs  --gsa-base URL --sheet runs_zhang.tsv
  (local paired FASTQ can be used instead with --fastq-dir)
Per sample it keeps what the discovery analysis used, as far as short reads allow:
  1. fragment-length histogram, 1-1000 bp                              (ev_hist.tsv.gz)
  2. 5' end 4-mers by size class <=150 / 151-255 / 256-600 (wgs only)  (ev_motifs.tsv.gz)
  3. per-promoter fragment counts by size class (<=150, 151-246, 247-600, 601-1000), promoter = gene-level
     -2000..+500 BED, a fragment counts if it overlaps >=1 bp (as bedtools intersect -u)   (ev_promoters.tsv.gz)
  4. 1 Mb fragment-midpoint counts as ichorCNA WIG (wgs only)          (wig/<run>.1000kb.wig)
  5. summary row with R_sr = n(256-600) / n(100-250) and size shares   (ev_summary.tsv)
Fragments: proper, primary pairs with MAPQ >= 20, length = |TLEN| (1-1000 bp), counted once from read 1.
Bisulfite (bs) mode: reference = C->T / G->A converted copy (f_chrN / r_chrN) as in bwa-meth; read 1 converted C->T,
read 2 G->A; only read 1 of proper pairs is used and pairs whose read 1 disagrees with its contig strand are dropped;
motifs and wig are skipped (conversion changes the sequence; only chromosomes in --chroms are indexed, so promoters
outside them are left out).
Remote runs: the head of each FASTQ (enough bytes for --pairs pairs) is first downloaded to OUT/fastq_cache with
resumable range requests, the next run downloads while the current one aligns, and the cache is deleted afterwards.
Each finished sample is saved in OUT/samples/<run>.npz, so the same command can be rerun to finish missing samples.

Examples (env "thesis"; PROM = promoters_gene_level.bed from 01_annotation):
  python 05_extract_external_cohorts.py --mode wgs --runs runs_gse71378.tsv --only-groups liver_cancer,healthy,noncancer_disease \
     --ref GRCh38.fna --promoters PROM --out OUT_GSE -t 8
  python 05_extract_external_cohorts.py --mode bs --gsa-base ftp://download.big.ac.cn/gsa/CRA001537 --sheet runs_zhang.tsv \
     --ref ref_bs/ref_bs.fa --promoters PROM --out OUT_ZHANG --pairs 4000000 -t 8
sheet (bs mode): lines "run group [label]", no header. runs file (wgs mode): TSV with columns run, group, batch.

Requires: bwa 0.7.19, curl
"""
import argparse, glob, gzip, io, json, os, re, shutil, subprocess, sys, tempfile, threading, time, csv
from array import array
from concurrent.futures import ThreadPoolExecutor
import numpy as np

p = argparse.ArgumentParser(description="External cohorts: align short reads and count fragment sizes, end motifs, promoter and 1 Mb counts")
p.add_argument("--mode", choices=["wgs", "bs"], required=True, help="wgs = plain WGS (GSE71378), bs = bisulfite WGBS (CRA001537)")
p.add_argument("--ref", required=True, help="wgs: GRCh38 FASTA with bwa index and .fai; bs: converted reference ref_bs.fa (bwa index)")
p.add_argument("--promoters", required=True, help="gene-level promoter BED: chrom start end strand gene_id [gene_name]")
p.add_argument("-o", "--out", required=True, help="output folder")
p.add_argument("--runs", default="", help="wgs: TSV with columns run, group, batch (ENA streaming)")
p.add_argument("--only-groups", default="", help="wgs: comma-separated groups to keep (default: all runs)")
p.add_argument("--fastq-dir", default="", help="local paired FASTQ (*_f1/_r2, _R1/_R2, _1/_2)")
p.add_argument("--sheet", default="", help="lines 'run group'; only these runs are used")
p.add_argument("--gsa-base", default="", help="bs mode, stream instead of local FASTQ: base URL, e.g. ftp://download.big.ac.cn/gsa/CRA001537 (files RUN/RUN_f1.fastq.gz, RUN_r2.fastq.gz)")
p.add_argument("--pairs", type=int, default=3_000_000, help="read pairs per run (GSE71378 3,000,000; CRA001537 cohort 4,000,000)")
p.add_argument("--min-mapq", type=int, default=20, help="minimum mapping quality")
p.add_argument("--chroms", default=",".join(f"chr{i}" for i in range(1, 9)), help="bs mode: chromosomes in the converted reference")
p.add_argument("-t", "--threads", type=int, default=8, help="total bwa threads (split over --jobs)")
p.add_argument("--jobs", type=int, default=1, help="samples aligned in parallel")
p.add_argument("--max-wait-h", type=float, default=6, help="keep retrying a sample this long (hours) before giving up on it")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
lock = threading.Lock()
def log(*m):
    with lock:
        print(time.strftime("%H:%M:%S"), *m, flush=True)

need = ["bwa"] + (["curl"] if (a.mode == "wgs" and not a.fastq_dir) or a.gsa_base else [])
for tool in need:
    if not shutil.which(tool):
        sys.exit(f"'{tool}' not found on PATH: run 'conda activate thesis' first")
if not os.path.exists(a.ref + ".bwt"):
    sys.exit(f"no bwa index at {a.ref}.bwt")
MAXL = 1000
CLS_LEN = [(1, 150), (151, 246), (247, 600), (601, 1000)]            # promoter classes
MOT_CLS = [("le150", 1, 150), ("151_255", 151, 255), ("256_600", 256, 600)]
CHROMS_STD = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]

# ---------- promoters ----------
bs_chroms = set(a.chroms.split(","))
prom = []
for l in open(a.promoters):
    f = l.rstrip("\n").split("\t")
    if len(f) < 5:
        continue
    if a.mode == "bs" and f[0] not in bs_chroms:
        continue
    prom.append((f[0], int(f[1]), int(f[2]), f[4]))
if not prom:
    sys.exit("no promoters read (check --promoters and --chroms)")
prom_chrom = np.array([x[0] for x in prom]); prom_s = np.array([x[1] for x in prom]); prom_e = np.array([x[2] for x in prom])
prom_id = [x[3] for x in prom]
log(f"{len(prom)} promoters ({len(set(prom_chrom))} chromosomes)")
fai = {}
if a.mode == "wgs":
    fa = a.ref + ".fai"
    if not os.path.exists(fa):
        sys.exit(f"missing {fa} (samtools faidx)")
    for l in open(fa):
        f = l.split("\t"); fai[f[0]] = int(f[1])

# ---------- samples ----------
samples = []   # dicts: run, group, kind ('ena'|'local'), f1, f2
if a.mode == "wgs" and not a.fastq_dir:
    if not a.runs:
        sys.exit("--runs needed in wgs mode")
    rows = list(csv.DictReader(open(a.runs), delimiter="\t"))
    if a.only_groups:
        keep = set(a.only_groups.split(","))
        rows = [r for r in rows if r["group"] in keep]
    for r in rows:
        samples.append(dict(run=r["run"], group=r["group"], batch=r.get("batch", ""), kind="ena"))
elif a.gsa_base:
    if not a.sheet:
        sys.exit("--sheet needed with --gsa-base")
    for l in open(a.sheet):
        if l.strip():
            k, g = l.split()[:2]
            samples.append(dict(run=k, group=g, batch="", kind="gsa"))
else:
    if not a.fastq_dir:
        sys.exit("--fastq-dir or --gsa-base needed in bs mode")
    fq = sorted(glob.glob(os.path.join(a.fastq_dir, "**", "*.f*q.gz"), recursive=True))
    pat = re.compile(r"^(.*?)[._](f1|r1|R1|1)(\.f(ast)?q\.gz)$")
    pairs = {}
    for f in fq:
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        stem, tag, ext = m.groups()[:3]
        sep = os.path.basename(f)[len(stem)]
        mate = {"f1": "r2", "r1": "r2", "R1": "R2", "1": "2"}[tag]
        f2 = os.path.join(os.path.dirname(f), f"{stem}{sep}{mate}{ext}")
        if os.path.exists(f2):
            pairs[stem] = (f, f2)
    groups = {}
    if a.sheet:
        for l in open(a.sheet):
            if l.strip():
                k, g = l.split()[:2]; groups[k] = g
        miss = [k for k in groups if k not in pairs]
        if miss:
            sys.exit(f"sheet names not found among FASTQ pairs: {miss}\nPairs found: {list(pairs)[:20]}")
        pairs = {k: v for k, v in pairs.items() if k in groups}
    if not pairs:
        sys.exit(f"no paired FASTQ under {a.fastq_dir}")
    for k, (f1, f2) in sorted(pairs.items()):
        samples.append(dict(run=k, group=groups.get(k, "?"), batch="", kind="local", f1=f1, f2=f2))
log(f"{len(samples)} samples, mode {a.mode}, {a.pairs:,} pairs each")

# ---------- remote FASTQ locations (ENA / GSA) ----------
def ena_urls(run):
    try:
        q = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run}&result=read_run&fields=fastq_ftp&format=tsv"
        txt = subprocess.run(["curl", "-sSL", "-m", "10", q], capture_output=True, text=True).stdout   # short timeout: fall back to the standard path quickly
        files = txt.strip().splitlines()[-1].split("\t")[-1].split(";")
        r1 = [f for f in files if f.endswith("_1.fastq.gz")]; r2 = [f for f in files if f.endswith("_2.fastq.gz")]
        if r1 and r2:
            return "https://" + r1[0], "https://" + r2[0]
    except Exception:
        pass
    sub = f"00{run[-1]}" if len(run) == 10 else (f"0{run[-2:]}" if len(run) == 11 else "")
    base = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{run[:6]}/{sub + '/' if sub else ''}{run}/{run}"
    return base + "_1.fastq.gz", base + "_2.fastq.gz"

# ---------- download first, then align ----------
# Streaming straight into bwa stalls the connections whenever bwa is busy, and the server then drops them.
# So each run's needed head (enough bytes for --pairs pairs) is downloaded to a local cache with resumable
# range requests, the next run downloads while the current one aligns, and the cache is deleted afterwards.
CACHE = os.path.join(a.out, "fastq_cache")
BPP = {}   # observed compressed bytes per read, per mate (updated as runs finish)

def remote_urls(s):
    if s["kind"] == "gsa":
        return f"{a.gsa_base}/{s['run']}/{s['run']}_f1.fastq.gz", f"{a.gsa_base}/{s['run']}/{s['run']}_r2.fastq.gz"
    return ena_urls(s["run"])

def remote_size(url):
    for _ in range(5):
        r = subprocess.run(["curl", "-sSIL", "-m", "60", url], capture_output=True, text=True)
        m = re.findall(r"(?im)^content-length:\s*(\d+)", r.stdout)
        if m:
            return int(m[-1])
        time.sleep(30)
    return None

def fetch_range(url, path, upto):
    """append bytes until the file holds `upto` bytes; resumes after drops; False if gave up"""
    t0, k = time.time(), 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= upto:
            return True
        with open(path, "ab") as fo:
            rc = subprocess.run(["curl", "-sSL", "--fail", "--connect-timeout", "30", "--speed-limit", "1000",
                                 "--speed-time", "120", "-r", f"{have}-{upto - 1}", url], stdout=fo,
                                stderr=subprocess.DEVNULL).returncode
        if rc == 0:
            continue
        if have == 0 and rc in (22, 78) and (os.path.getsize(path) if os.path.exists(path) else 0) == 0:
            log(f"FAILED {url}: not found (curl {rc})"); return False
        if time.time() - t0 > a.max_wait_h * 3600:
            log(f"GAVE UP downloading {url} at {have:,} bytes"); return False
        k += 1
        log(f"download interrupted ({url.rsplit('/', 1)[-1]} at {have:,} bytes, curl {rc}); retry {k}")
        time.sleep(min(300, 15 * k))
        while not net_up():
            time.sleep(60)

def count_reads(path, need):
    """complete FASTQ records decodable from a (possibly cut) gzip, stopping at `need`"""
    n = 0
    try:
        with gzip.open(path, "rt") as fh:
            for i, _ in enumerate(fh):
                if i % 4 == 3:
                    n += 1
                    if n >= need:
                        break
    except (EOFError, OSError, gzip.BadGzipFile):
        pass
    return n

def prefetch(s):
    """make sure both mates' heads hold >= --pairs complete reads (or the whole file); True when ready"""
    os.makedirs(CACHE, exist_ok=True)
    us = remote_urls(s)
    for mate, url in zip((1, 2), us):
        path = os.path.join(CACHE, f"{s['run']}_{mate}.fq.gz")
        if os.path.exists(path + ".ok"):
            continue
        size = remote_size(url)
        bpp = BPP.get(mate, 75)
        target = int(a.pairs * bpp * 1.15)
        while True:
            upto = min(target, size) if size else target
            if not fetch_range(url, path, upto):
                return False
            got = count_reads(path, a.pairs + 1)   # +1: the last record of a cut file may be partial
            if got > a.pairs or (size and upto >= size):
                break
            target = int(upto * a.pairs / max(got, 1) * 1.05) + 1_000_000
        BPP[mate] = max(40, os.path.getsize(path) / max(got, 1))
        open(path + ".ok", "w").close()
    s["f1"], s["f2"] = (os.path.join(CACHE, f"{s['run']}_{m}.fq.gz") for m in (1, 2))
    return True

def drop_cache(s):
    for m in (1, 2):
        for ext in ("", ".ok"):
            try: os.remove(os.path.join(CACHE, f"{s['run']}_{m}.fq.gz{ext}"))
            except FileNotFoundError: pass

def open_source(s):
    """returns (iter1, iter2, cleanup); remote runs are read from the prefetched cache"""
    if s["kind"] != "local" and not prefetch(s):
        raise RuntimeError(f"{s['run']}: download failed")
    h1, h2 = gzip.open(s["f1"], "rt"), gzip.open(s["f2"], "rt")
    return h1, h2, lambda: (h1.close(), h2.close())

RC = str.maketrans("ACGTN", "TGCAN")
CT, GA = str.maketrans("C", "T"), str.maketrans("G", "A")
KM = {}
def kmer_index(m):
    v = 0
    for ch in m:
        v = v * 4 + "ACGT".find(ch)
    return v

def count_ge(sorted_arr, x):
    return np.searchsorted(sorted_arr, x, side="left")

def run_once(s):
    run = s["run"]
    th = max(1, a.threads // a.jobs)
    try:
        it1, it2, cleanup = open_source(s)
    except Exception as e:
        log("FAILED", run, e); return None
    ref = a.ref
    os.makedirs(os.path.join(a.out, "bwa_logs"), exist_ok=True)   # keep bwa messages per run
    blog = open(os.path.join(a.out, "bwa_logs", f"{run}.log"), "a")
    # -K fixes the batch at 10 Mbp (default is 10 Mbp x threads); bisulfite batches of 80 Mbp pushed bwa past 10 GB
    bwa = subprocess.Popen(["bwa", "mem", "-v", "2", "-K", "10000000", "-p", "-t", str(th), ref, "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=blog, text=True, bufsize=1 << 20)
    n = {"pairs": 0, "stop": "complete"}
    bs = a.mode == "bs"
    def feed():
        try:
            while n["pairs"] < a.pairs:
                r1 = [next(it1) for _ in range(4)]; r2 = [next(it2) for _ in range(4)]
                nm = r1[0].split()[0]
                s1, s2 = r1[1].strip(), r2[1].strip()
                if bs:
                    s1, s2 = s1.translate(CT), s2.translate(GA)
                bwa.stdin.write(f"{nm}\n{s1}\n+\n{r1[3].strip()}\n{nm}\n{s2}\n+\n{r2[3].strip()}\n")
                n["pairs"] += 1
        except StopIteration:
            n["stop"] = "end_of_file"
            # a real end of file only if both downloads finished (else it was a cut connection)
            for pr in s.get("_procs", ()):
                if pr.wait() != 0:
                    n["stop"] = "stream_cut"
        except Exception as e:
            n["stop"] = type(e).__name__
        finally:
            try: bwa.stdin.close()
            except Exception: pass
            cleanup()
    t = threading.Thread(target=feed); t.start()
    hist = np.zeros(MAXL + 1, dtype=np.int64)
    mot = np.zeros((len(MOT_CLS), 256), dtype=np.int64)
    fs, fe, fl, fc = array("i"), array("i"), array("i"), []
    cidx = {}
    wrong = 0
    for line in bwa.stdout:
        if line[0] == "@":
            continue
        c = line.split("\t", 11)
        fl_ = int(c[1])
        if fl_ & 0x904 or not fl_ & 0x2 or int(c[4]) < a.min_mapq or c[6] != "=":
            continue
        L = abs(int(c[8]))
        if not 0 < L <= MAXL:
            continue
        r1 = bool(fl_ & 0x40)
        if bs:
            if not r1:
                continue
            if (not fl_ & 0x10) != c[2].startswith("f_"):
                wrong += 1; continue
        if r1:
            chrom = c[2][2:] if bs else c[2]
            pos, mpos = int(c[3]), int(c[7])
            st = min(pos, mpos) - 1
            hist[L] += 1
            fs.append(st); fe.append(st + L); fl.append(L)
            fc.append(cidx.setdefault(chrom, len(cidx)))
        if not bs and L <= 600:
            seq = c[9]
            m = seq[:4] if not fl_ & 0x10 else seq[-4:].translate(RC)[::-1]
            if len(m) == 4 and "N" not in m:
                k = kmer_index(m)
                for ci, (_, lo, hi) in enumerate(MOT_CLS):
                    if lo <= L <= hi:
                        mot[ci, k] += 1; break
    t.join(); bwa.wait()
    return dict(run=run, group=s["group"], batch=s["batch"], pairs=n["pairs"], stop=n["stop"], wrong=wrong, hist=hist, mot=mot,
                fs=np.frombuffer(fs, dtype=np.int32), fe=np.frombuffer(fe, dtype=np.int32), fl=np.frombuffer(fl, dtype=np.int32),
                fc=np.array(fc, dtype=np.int32), chroms=[k for k, _ in sorted(cidx.items(), key=lambda x: x[1])])

def net_up():
    # check the file server itself (the ENA portal API is not always reachable)
    host = (a.gsa_base + "/") if a.gsa_base else "https://ftp.sra.ebi.ac.uk/vol1/fastq/"
    return subprocess.run(["curl", "-sS", "-I", "-m", "30", "-o", "/dev/null", host],
                          stderr=subprocess.DEVNULL).returncode == 0

def run_one(s):
    # retry until the stream finishes, waiting for the network with growing pauses (up to --max-wait-h);
    # a sample that never completes is NOT kept, so a later restart redoes it
    if s["run"] in done:
        return done[s["run"]]
    res, attempt, t0 = None, 0, time.time()
    while True:
        attempt += 1
        res = run_once(s)
        if res is not None and (res["stop"] in ("complete", "end_of_file") or s["kind"] == "local"):
            break
        why = "source not reachable" if res is None else f"stream stopped early ({res['stop']}) after {res['pairs']:,} pairs"
        if time.time() - t0 > a.max_wait_h * 3600:
            log(f"GAVE UP {s['run']} after {attempt} attempts: {why}; rerun the script later to retry it")
            return None
        w = min(600, 30 * attempt)
        log(f"RETRY {s['run']} attempt {attempt}: {why}; waiting {w}s")
        time.sleep(w)
        while not net_up():
            log(f"network down, waiting ({s['run']})"); time.sleep(60)
    # promoter counts (overlap >= 1 bp: #(start < pe) - #(end <= ps), per size class)
    P = np.zeros((len(prom), len(CLS_LEN)), dtype=np.int32)
    chrom_names = np.array(res["chroms"])
    for ci, name in enumerate(chrom_names):
        pm = np.where(prom_chrom == name)[0]
        if len(pm) == 0:
            continue
        sel_c = res["fc"] == ci
        for k, (lo, hi) in enumerate(CLS_LEN):
            sel = sel_c & (res["fl"] >= lo) & (res["fl"] <= hi)
            st = np.sort(res["fs"][sel]); en = np.sort(res["fe"][sel])
            P[pm, k] = np.searchsorted(st, prom_e[pm], side="left") - np.searchsorted(en, prom_s[pm], side="right")
    res["P"] = P
    # 1 Mb counts (midpoint) for ichorCNA, wgs only
    if a.mode == "wgs":
        wd = os.path.join(a.out, "wig"); os.makedirs(wd, exist_ok=True)
        mid = (res["fs"].astype(np.int64) + res["fe"].astype(np.int64)) // 2
        with open(os.path.join(wd, f"{res['run']}.1000kb.wig"), "w") as fo:
            for chrom in CHROMS_STD:
                if chrom not in fai:
                    continue
                nb = -(-fai[chrom] // 1_000_000)
                cnt = np.zeros(nb, dtype=np.int64)
                if chrom in chrom_names:
                    ci = int(np.where(chrom_names == chrom)[0][0])
                    b = mid[res["fc"] == ci] // 1_000_000
                    b = b[b < nb]
                    cnt = np.bincount(b, minlength=nb)
                fo.write(f"fixedStep chrom={chrom} start=1 step=1000000 span=1000000\n" + "\n".join(map(str, cnt)) + "\n")
    for k in ("fs", "fe", "fl", "fc"):
        del res[k]
    h = res["hist"]; sm = lambda lo, hi: int(h[lo:hi + 1].sum())
    tot = sm(1, MAXL)
    res["row"] = dict(run=res["run"], group=res["group"], batch=res["batch"], pairs_read=res["pairs"], stop=res["stop"],
                      strand_inconsistent=res["wrong"], fragments=tot, n_100_250=sm(100, 250), n_256_600=sm(256, 600),
                      R_sr=round(sm(256, 600) / max(1, sm(100, 250)), 6), share_le150=round(sm(1, 150) / max(1, tot), 5),
                      share_151_255=round(sm(151, 255) / max(1, tot), 5), share_256_650=round(sm(256, 650) / max(1, tot), 5))
    log(res["run"], res["row"])
    # checkpoint, written atomically, so a restart skips this sample
    tmp = os.path.join(ckdir, f".{res['run']}.tmp.npz")
    np.savez_compressed(tmp, hist=res["hist"], mot=res["mot"], P=res["P"], row=json.dumps(res["row"]))
    os.replace(tmp, os.path.join(ckdir, f"{res['run']}.npz"))
    if s["kind"] != "local":
        drop_cache(s)
    return res

ckdir = os.path.join(a.out, "samples"); os.makedirs(ckdir, exist_ok=True)
done = {}
for s in samples:
    f = os.path.join(ckdir, f"{s['run']}.npz")
    if os.path.exists(f):
        z = np.load(f)
        if z["P"].shape[0] == len(prom):
            done[s["run"]] = dict(run=s["run"], hist=z["hist"], mot=z["mot"], P=z["P"], row=json.loads(str(z["row"])))
log(f"{len(done)} of {len(samples)} samples already finished (checkpoints in {ckdir}); {len(samples) - len(done)} to do")

# one sample aligns while the next one downloads
to_run = [s for s in samples if s["run"] not in done]
res = [done[s["run"]] for s in samples if s["run"] in done]
with ThreadPoolExecutor(1) as dl:
    nxt = dl.submit(prefetch, to_run[0]) if to_run and to_run[0]["kind"] != "local" else None
    for i, s in enumerate(to_run):
        if nxt is not None:
            try: nxt.result()
            except Exception as e: log(f"prefetch error for {s['run']}: {e}")
        nxt = dl.submit(prefetch, to_run[i + 1]) if i + 1 < len(to_run) and to_run[i + 1]["kind"] != "local" else None
        x = run_one(s)
        if x:
            res.append(x)
if len(res) < len(samples):
    log(f"NOT COMPLETE: {len(samples) - len(res)} samples missing; rerun the same command to finish them")
if not res:
    sys.exit("no sample succeeded")
cols = list(res[0]["row"])
with open(os.path.join(a.out, "ev_summary.tsv"), "w") as fo:
    fo.write("\t".join(cols) + "\n" + "".join("\t".join(str(r["row"][c]) for c in cols) + "\n" for r in res))
with gzip.open(os.path.join(a.out, "ev_hist.tsv.gz"), "wt") as fo:
    fo.write("run\t" + "\t".join(map(str, range(1, MAXL + 1))) + "\n")
    for r in res:
        fo.write(r["run"] + "\t" + "\t".join(map(str, r["hist"][1:])) + "\n")
if a.mode == "wgs":
    kmers = ["".join("ACGT"[(k // 4 ** (3 - i)) % 4] for i in range(4)) for k in range(256)]
    with gzip.open(os.path.join(a.out, "ev_motifs.tsv.gz"), "wt") as fo:
        fo.write("run\tclass\t" + "\t".join(kmers) + "\n")
        for r in res:
            for ci, (nm, _, _) in enumerate(MOT_CLS):
                fo.write(f"{r['run']}\t{nm}\t" + "\t".join(map(str, r["mot"][ci])) + "\n")
with gzip.open(os.path.join(a.out, "ev_promoters.tsv.gz"), "wt") as fo:
    fo.write("run\tpromoter\t" + "\t".join(f"n_{lo}_{hi}" for lo, hi in CLS_LEN) + "\n")
    for r in res:
        P = r["P"]
        for i in np.nonzero(P.sum(axis=1))[0]:
            fo.write(f"{r['run']}\t{prom_id[i]}\t" + "\t".join(map(str, P[i])) + "\n")
with open(os.path.join(a.out, "ev_promoter_index.tsv"), "w") as fo:
    fo.write("promoter\tchrom\n" + "".join(f"{prom_id[i]}\t{prom_chrom[i]}\n" for i in range(len(prom))))
log(f"done: {len(res)} of {len(samples)} samples; results in --out (ev_summary, ev_hist, ev_promoters, ev_promoter_index; plus ev_motifs and wig/ in wgs mode)")
