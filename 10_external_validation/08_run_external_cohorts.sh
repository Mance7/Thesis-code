#!/usr/bin/env bash
# 08_run_external_cohorts.sh
#
# Runs the extraction of 07_extract_external_cohorts.py for both short-read cohorts:
#   1) GRCh38 reference and bwa index (01_download_reference.sh), if not there yet
#   2) converted reference for bisulfite reads: GRCh38 chr1-8 as a C->T copy (f_chrN)
#      and a G->A copy (r_chrN), bwa index built once (a few hours, in the background)
#   3) GSE71378: the 15 liver cancer, healthy and non-cancer disease runs, first
#      3,000,000 read pairs each from ENA, aligned to GRCh38
#   4) CRA001537 low-pass cohort (Zhang et al. 2020): all 54 runs, first 4,000,000
#      read pairs each from GSA, aligned to the converted reference
# Every step is restartable: rerun the script after a crash or dropped connection and
# it continues where it stopped (finished steps leave a marker in OUT/markers, and
# the extraction keeps per-sample checkpoints).
#
# Input : runs_gse71378.tsv, runs_zhang.tsv (next to this script), promoter BED
# Output: OUT/ref, OUT/out_gse, OUT/out_zhang (ev_summary.tsv, ev_hist.tsv.gz,
#         ev_motifs.tsv.gz, ev_promoters.tsv.gz, wig/); logs in OUT/logs
#
# Usage:
#   bash 08_run_external_cohorts.sh -o OUT [--promoters BED] [-t THREADS] [--gsa-base URL]
#
# Requires: conda env "thesis" (python 3.12, bwa 0.7.19, samtools 1.23.1, curl, wget)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
B=""
PROM=""
T=8
GSA="ftp://download.big.ac.cn/gsa/CRA001537"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) B="$2"; shift 2 ;;
    --promoters) PROM="$2"; shift 2 ;;
    -t|--threads) T="$2"; shift 2 ;;
    --gsa-base) GSA="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$B" ] || { echo "usage: bash $0 -o OUT [--promoters BED] [-t THREADS] [--gsa-base URL]" >&2; exit 1; }

R="$B/ref"; L="$B/logs"; M="$B/markers"; mkdir -p "$L" "$M"
N=GCA_000001405.15_GRCh38_no_alt_analysis_set
REF="$R/$N.fna"; BSREF="$R/ref_bs/ref_bs.fa"
PROM="${PROM:-$R/promoters_gene_level.bed}"
say() { echo "$(date '+%F %T') $*"; }

# only one copy of this script at a time
exec 9>"$B/.run_external.lock"; flock -n 9 || { say "STOP: already running"; exit 1; }
for x in bwa samtools curl wget python; do command -v $x >/dev/null || { say "STOP: $x missing"; exit 1; }; done
[ -s "$PROM" ] || { say "STOP: promoter BED not found: $PROM"; exit 1; }

# 1) reference download (resumable) and unpacking
if [ ! -f "$M/ref.done" ]; then
  bash "$HERE/01_download_reference.sh" -o "$R" >> "$L/dl_ref.log" 2>&1 || { say "STOP: reference download failed, see $L/dl_ref.log"; exit 1; }
  touch "$M/ref.done"; say "reference ready"
fi

# 2) converted reference for bisulfite reads (chr1-8, C->T and G->A copies), built in the background
build_bs() {
  [ -f "$M/bsref.done" ] && return 0
  mkdir -p "$R/ref_bs"
  if [ ! -s "$BSREF" ]; then
    say "writing converted chr1-8 reference"
    samtools faidx "$REF" chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 > "$R/ref_bs/top.fa" || return 1
    { sed -e '/^>/s/^>/>f_/' -e '/^>/!y/Cc/Tt/' "$R/ref_bs/top.fa"; sed -e '/^>/s/^>/>r_/' -e '/^>/!y/Gg/Aa/' "$R/ref_bs/top.fa"; } > "$BSREF.tmp" \
      && mv "$BSREF.tmp" "$BSREF" && rm -f "$R/ref_bs/top.fa" || return 1
  fi
  say "bwa index of converted reference (one-off, a few hours)"
  bwa index "$BSREF" > "$L/bwa_index_bs.log" 2>&1 && touch "$M/bsref.done" && say "converted index ready"
}
build_bs > "$L/build_bs.log" 2>&1 &
BSPID=$!

# 3) GSE71378: 15 runs from ENA
if [ ! -f "$M/gse.done" ]; then
  say "GSE71378 extraction"
  nice -n 10 python "$HERE/07_extract_external_cohorts.py" --mode wgs --runs "$HERE/runs_gse71378.tsv" \
     --only-groups liver_cancer,healthy,noncancer_disease --ref "$REF" --promoters "$PROM" \
     --out "$B/out_gse" -t "$T" --jobs 1 >> "$L/ev_gse.log" 2>&1
  grep -q "NOT COMPLETE" <(tail -3 "$L/ev_gse.log") || { [ -s "$B/out_gse/ev_summary.tsv" ] && touch "$M/gse.done"; }
  say "GSE71378 step finished (done marker: $([ -f "$M/gse.done" ] && echo yes || echo NO, rerun later))"
fi

# 4) CRA001537 low-pass cohort: all 54 runs from GSA (needs the converted index)
say "waiting for the converted index"; wait $BSPID
[ -f "$M/bsref.done" ] || { say "STOP: converted index failed, see $L/build_bs.log and $L/bwa_index_bs.log"; exit 1; }
if [ ! -f "$M/zhang.done" ]; then
  say "CRA001537 extraction"
  nice -n 10 python "$HERE/07_extract_external_cohorts.py" --mode bs --gsa-base "$GSA" --sheet "$HERE/runs_zhang.tsv" \
     --ref "$BSREF" --promoters "$PROM" --out "$B/out_zhang" --pairs 4000000 -t "$T" --jobs 1 >> "$L/ev_zhang.log" 2>&1
  grep -q "NOT COMPLETE" <(tail -3 "$L/ev_zhang.log") || { [ -s "$B/out_zhang/ev_summary.tsv" ] && touch "$M/zhang.done"; }
  say "CRA001537 step finished (done marker: $([ -f "$M/zhang.done" ] && echo yes || echo NO, rerun later))"
fi
[ -f "$M/gse.done" ] && [ -f "$M/zhang.done" ] && say "ALL DONE" || say "some steps incomplete: rerun this script"
