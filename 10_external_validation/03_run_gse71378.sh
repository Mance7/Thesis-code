#!/usr/bin/env bash
# 03_run_gse71378.sh
#
# Runs 02_extract_gse71378_sizes_motifs.py on all 60 GSE71378 runs (first
# 3,000,000 read pairs each, streamed from ENA, BWA-MEM to GRCh38), three runs
# at a time at the lowest CPU priority, so the job stays light on a shared server.
# Memory: about 6 GB per job (one bwa index each), so about 18 GB in total.
#
# Input : runs_gse71378.tsv (next to this script), GRCh38 FASTA with bwa index
# Output: OUT/gse_summary.tsv, gse_hist.tsv.gz, gse_motifs.tsv.gz; log in OUT/gse.log
#
# Usage:
#   bash 03_run_gse71378.sh --ref REF -o OUT [-t THREADS] [--jobs JOBS]
#
# Requires: conda env "thesis" (python 3.12, bwa 0.7.19, samtools, curl)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REF=""
OUT=""
THREADS=16
JOBS=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$REF" ] && [ -n "$OUT" ] || { echo "usage: bash $0 --ref REF -o OUT [-t THREADS] [--jobs JOBS]" >&2; exit 1; }
say() { echo "$(date '+%F %T') $*"; }

# check tools and inputs before starting
for x in bwa samtools curl python; do
  command -v "$x" >/dev/null || { say "STOP: '$x' not found. Run: conda activate thesis"; exit 1; }
done
[ -s "$REF.bwt" ] || { say "STOP: no bwa index next to the reference ($REF.bwt)"; exit 1; }
[ -s "$HERE/runs_gse71378.tsv" ] || { say "STOP: runs_gse71378.tsv missing in $HERE"; exit 1; }

mkdir -p "$OUT"
say "GSE71378: 60 runs, $THREADS threads, $JOBS jobs"
nice -n 19 python "$HERE/02_extract_gse71378_sizes_motifs.py" --runs "$HERE/runs_gse71378.tsv" \
    --ref "$REF" --out "$OUT" -t "$THREADS" --jobs "$JOBS" > "$OUT/gse.log" 2>&1
say "GSE71378 finished (exit $?), see $OUT/gse.log"
