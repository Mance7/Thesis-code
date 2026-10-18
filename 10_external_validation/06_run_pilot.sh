#!/usr/bin/env bash
# 06_run_pilot.sh
#
# Runs 05_extract_pilot_bisulfite_sizes.py on the three deep CRA001537 pilot
# samples (CRR053950 HCC, CRR053948 hepatitis, CRR053947 healthy; runs_pilot.tsv),
# first 5,000,000 read pairs each, at the lowest CPU priority. The first run also
# builds the converted chr1-8 reference and its bwa index in OUT/ref_bs (about 1 h,
# about 12 GB memory, needs about 25 GB free disk).
#
# Input : pilot FASTQ folder, GRCh38 FASTA with .fai, runs_pilot.tsv (next to this script)
# Output: OUT/ev_pilot_summary.tsv, ev_pilot_hist.tsv.gz, ev_pilot_groups.tsv; log OUT/ev_pilot.log
#
# Usage:
#   bash 06_run_pilot.sh --fastq-dir DIR --ref REF -o OUT [-t THREADS]
#
# Requires: conda env "thesis" (python 3.12, bwa 0.7.19, samtools 1.23.1)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FQ=""
REF=""
OUT=""   # best on a fast local disk: it also holds the converted reference and its index
THREADS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fastq-dir) FQ="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$FQ" ] && [ -n "$REF" ] && [ -n "$OUT" ] || { echo "usage: bash $0 --fastq-dir DIR --ref REF -o OUT [-t THREADS]" >&2; exit 1; }
say() { echo "$(date '+%F %T') $*"; }

# check tools, inputs and free disk space before starting
for x in bwa samtools python; do
  command -v "$x" >/dev/null || { say "STOP: '$x' not found. Run: conda activate thesis"; exit 1; }
done
[ -d "$FQ" ] || { say "STOP: $FQ not found"; exit 1; }
mkdir -p "$OUT"
free_gb=$(df -P --block-size=1G "$OUT" | awk 'NR==2{print $4}')
[ "$free_gb" -ge 25 ] || { say "STOP: only ${free_gb} GB free in $OUT, need 25"; exit 1; }

say "starting pilot (threads $THREADS, nice 19), output $OUT"
nice -n 19 python "$HERE/05_extract_pilot_bisulfite_sizes.py" --fastq-dir "$FQ" --ref "$REF" --out "$OUT" \
    --sheet "$HERE/runs_pilot.tsv" -t "$THREADS" --jobs 3 > "$OUT/ev_pilot.log" 2>&1
rc=$?
say "pilot finished (exit $rc), see $OUT/ev_pilot.log"
exit $rc
