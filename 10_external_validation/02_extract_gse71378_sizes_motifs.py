#!/usr/bin/env python3
"""
02_extract_gse71378_sizes_motifs.py

External validation, GSE71378 / PRJNA291063 (Snyder et al. 2016): plasma cfDNA, plain
paired-end short-read WGS (no bisulfite), open access. 3 liver cancer, 4 healthy,
8 non-cancer inflammatory disease (Crohn's disease, ulcerative colitis, lupus) and
45 other cancer runs.

For each run we stream only the first --pairs read pairs from ENA (no full download),
align them with bwa mem to GRCh38 and keep proper, primary pairs with MAPQ >= 20.
A stream that stops early (dropped connection) is retried up to three times and,
if it still stops early, kept with the pairs read. Per run we record
  - fragment length = |TLEN| of read 1 (histogram 1-1000 bp)
  - 5' end 4-mer of each read (fragment end motif as sequenced), fragments <= 600 bp
  - summary row with R_sr = n(256-600 bp) / n(100-250 bp) and size shares
The tests are run afterwards on these small files (04_test_gse71378.py).

Input : runs TSV (columns run, group, batch), GRCh38 FASTA with bwa index
Output: OUT/gse_summary.tsv, OUT/gse_hist.tsv.gz, OUT/gse_motifs.tsv.gz

Requires: bwa 0.7.19, curl (fastq-dump only if ENA is unreachable)
"""
import argparse, gzip, os, subprocess, sys, time, csv, io, shutil, tempfile
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description="GSE71378: stream, align and count fragment sizes and 5' end motifs per run")
p.add_argument("--runs", default=str(HERE / "runs_gse71378.tsv"), help="runs TSV with columns run, group, batch")
p.add_argument("--ref", required=True, help="GRCh38 no-alt FASTA with a bwa index next to it")
p.add_argument("-o", "--out", required=True, help="output folder")
p.add_argument("--pairs", type=int, default=3_000_000, help="read pairs streamed per run")
p.add_argument("--min-mapq", type=int, default=20, help="minimum mapping quality")
p.add_argument("--max-len", type=int, default=1000, help="longest fragment counted (bp)")
p.add_argument("-t", "--threads", type=int, default=16, help="total bwa threads (split over --jobs)")
p.add_argument("--jobs", type=int, default=3, help="runs processed in parallel")
p.add_argument("--only-groups", default="", help="comma-separated groups to keep (default: all 60 runs)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
lock = threading.Lock()
def log(*m):
    with lock:
        print(time.strftime("%H:%M:%S"), *m, flush=True)

for tool in ("bwa", "curl"):
    if not shutil.which(tool):
        sys.exit(f"'{tool}' not found on PATH: run 'conda activate thesis' first")
if not os.path.exists(a.ref + ".bwt"):
    sys.exit(f"no bwa index at {a.ref}.bwt (build once with: bwa index {a.ref})")
runs = list(csv.DictReader(open(a.runs), delimiter="\t"))
if a.only_groups:
    keep = set(a.only_groups.split(","))
    runs = [r for r in runs if r["group"] in keep]
log(f"{len(runs)} runs; {a.pairs:,} pairs each")

def ena_urls(run):
    """FASTQ URLs of a run from the ENA file report, or the standard ENA path if the API does not answer"""
    try:
        q = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run}&result=read_run&fields=fastq_ftp&format=tsv"
        txt = subprocess.run(["curl", "-sSL", "-m", "60", q], capture_output=True, text=True).stdout
        files = txt.strip().splitlines()[-1].split("\t")[-1].split(";")
        r1 = [f for f in files if f.endswith("_1.fastq.gz")]; r2 = [f for f in files if f.endswith("_2.fastq.gz")]
        if r1 and r2:
            return "https://" + r1[0], "https://" + r2[0]
    except Exception:
        pass
    sub = f"00{run[-1]}" if len(run) == 10 else (f"0{run[-2:]}" if len(run) == 11 else "")
    base = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{run[:6]}/{sub + '/' if sub else ''}{run}/{run}"
    return base + "_1.fastq.gz", base + "_2.fastq.gz"

def stream_fastq(url):
    """curl a gzipped FASTQ and decompress it on the fly"""
    pr = subprocess.Popen(["curl", "-sSL", "--fail", "--connect-timeout", "30", url], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return pr, io.TextIOWrapper(gzip.GzipFile(fileobj=pr.stdout))

def source(run, tmp):
    """returns (fh1, fh2, cleanup, how)"""
    u1, u2 = ena_urls(run)
    p1, f1 = stream_fastq(u1); p2, f2 = stream_fastq(u2)
    try:
        h1, h2 = f1.readline(), f2.readline()
        if h1.startswith("@") and h2.startswith("@"):
            def gen(first, fh):
                yield first
                for l in fh:
                    yield l
            return gen(h1, f1), gen(h2, f2), lambda: (p1.kill(), p2.kill()), "ENA"
    except Exception:
        pass
    p1.kill(); p2.kill()
    if shutil.which("fastq-dump"):
        subprocess.run(["fastq-dump", "-X", str(a.pairs), "--split-3", "-O", tmp, run], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return open(f"{tmp}/{run}_1.fastq"), open(f"{tmp}/{run}_2.fastq"), lambda: None, "fastq-dump"
    raise RuntimeError(f"{run}: ENA not reachable ({u1}) and fastq-dump not installed")

# complement table: the 5' end of a reverse-strand read is the reverse complement of its last 4 bases
RC = str.maketrans("ACGTN", "TGCAN")
def run_once(r):
    """stream, align and count one run; returns (summary row, length histogram, motif counts)"""
    run = r["run"]
    tmp = tempfile.mkdtemp(dir=a.out)
    th = max(1, a.threads // a.jobs)
    hist = [0] * (a.max_len + 1)
    mot = {}
    try:
        f1, f2, cleanup, how = source(run, tmp)
    except Exception as e:
        log("FAILED", run, e); return None
    bwa = subprocess.Popen(["bwa", "mem", "-p", "-t", str(th), a.ref, "-"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    n = {"pairs": 0, "stop": "complete"}
    # feed interleaved pairs to bwa in a separate thread while we read its output
    def feed():
        try:
            while n["pairs"] < a.pairs:
                r1 = [next(f1) for _ in range(4)]; r2 = [next(f2) for _ in range(4)]
                nm = r1[0].split()[0]
                bwa.stdin.write(f"{nm}\n{r1[1].strip()}\n+\n{r1[3].strip()}\n{nm}\n{r2[1].strip()}\n+\n{r2[3].strip()}\n")
                n["pairs"] += 1
        except StopIteration:
            n["stop"] = "end_of_file"
        except Exception as e:            # dropped connection, truncated gzip, ...
            n["stop"] = type(e).__name__
        finally:
            try: bwa.stdin.close()
            except Exception: pass
            cleanup()
    t = threading.Thread(target=feed); t.start()
    kept = 0
    for line in bwa.stdout:
        if line[0] == "@":
            continue
        c = line.split("\t", 11)
        fl = int(c[1])
        # keep proper, primary, mapped pairs with MAPQ >= min-mapq on one chromosome
        if fl & 0x904 or not fl & 0x2 or int(c[4]) < a.min_mapq or c[6] != "=":
            continue
        L = abs(int(c[8]))
        if not 0 < L <= a.max_len:
            continue
        # length histogram from read 1 only (one count per fragment)
        if fl & 0x40:
            hist[L] += 1; kept += 1
        # 5' end motif of both reads, fragments <= 600 bp
        if L <= 600:
            s = c[9]
            m = s[:4] if not fl & 0x10 else s[-4:].translate(RC)[::-1]
            if len(m) == 4 and "N" not in m:
                mot[m] = mot.get(m, 0) + 1
    t.join(); bwa.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    s = lambda lo, hi: sum(hist[lo:hi + 1])
    tot = s(1, a.max_len)
    row = dict(run=run, group=r["group"], batch=r["batch"], source=how, pairs_read=n["pairs"], stop=n["stop"], fragments=tot,
               n_100_250=s(100, 250), n_256_600=s(256, 600),
               R_sr=round(s(256, 600) / max(1, s(100, 250)), 6),
               share_151_255=round(s(151, 255) / max(1, tot), 5), share_256_650=round(s(256, 650) / max(1, tot), 5))
    log(run, row)
    return row, hist, mot

def run_one(r):
    """a dropped connection gives a short sample: retry the whole run up to 3 times"""
    for attempt in range(1, 4):
        out = run_once(r)
        if out is None or out[0]["stop"] in ("complete", "end_of_file"):
            return out
        log(f"RETRY {r['run']} attempt {attempt}: stream stopped early ({out[0]['stop']}) after {out[0]['pairs_read']:,} pairs")
    log(f"WARNING {r['run']}: kept truncated sample, {out[0]['pairs_read']:,} pairs")
    return out

# ---------- run all and write the output tables ----------
with ThreadPoolExecutor(a.jobs) as ex:
    res = [x for x in ex.map(run_one, runs) if x]
if not res:
    sys.exit("no run succeeded")
cols = list(res[0][0])
with open(os.path.join(a.out, "gse_summary.tsv"), "w") as fo:
    fo.write("\t".join(cols) + "\n" + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r, _, _ in res))
with gzip.open(os.path.join(a.out, "gse_hist.tsv.gz"), "wt") as fo:
    fo.write("run\t" + "\t".join(map(str, range(1, a.max_len + 1))) + "\n")
    fo.writelines(r["run"] + "\t" + "\t".join(map(str, h[1:])) + "\n" for r, h, _ in res)
kmers = sorted({k for _, _, m in res for k in m})
with gzip.open(os.path.join(a.out, "gse_motifs.tsv.gz"), "wt") as fo:
    fo.write("run\t" + "\t".join(kmers) + "\n")
    fo.writelines(r["run"] + "\t" + "\t".join(str(m.get(k, 0)) for k in kmers) + "\n" for r, _, m in res)
log(f"done: {len(res)} of {len(runs)} runs; results in gse_summary.tsv, gse_hist.tsv.gz, gse_motifs.tsv.gz")
