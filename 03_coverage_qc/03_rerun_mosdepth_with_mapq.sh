#!/usr/bin/env bash
# 03_rerun_mosdepth_with_mapq.sh
#
# Re-runs the mosdepth coverage QC of 01_mosdepth_global.py and
# 02_mosdepth_annotations.py with a mapping-quality cutoff (--mapq 20), so that
# reads that may be multi-mapped (low MAPQ) do not count towards the coverage.
# The original runs used no MAPQ cutoff. Same BAMs, same window and annotation
# BEDs, same --fast-mode and --thresholds; only --mapq is added.
#
# Input : RESULTS/{sample}/Alignment/*.bam (+ .bai)
#         10 kb / 100 kb window BEDs and the cleaned annotation BEDs
# Output: OUT/{sample}/alignment_qc_mapq20/{sample}[.{tag}].*  (tag = 10kb,
#         100kb or an annotation name; no tag = whole-genome summary)
#
# Usage:
#   bash 03_rerun_mosdepth_with_mapq.sh -r RESULTS -o OUT --resources-dir RESOURCES [-q MAPQ] [-t THREADS]
#
# Requires: mosdepth 0.3.14

set -euo pipefail

RESULTS=""
OUT=""
RESOURCES=""
MAPQ=20
THREADS=$(( $(nproc) * 9 / 10 )); [[ $THREADS -lt 1 ]] && THREADS=1
THRESHOLDS="1,2,3,4,5,6,7,8,9,10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--results-dir) RESULTS="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    --resources-dir) RESOURCES="$2"; shift 2 ;;
    -q|--mapq) MAPQ="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$RESULTS" && -n "$OUT" && -n "$RESOURCES" ]] || { echo "usage: $0 -r RESULTS -o OUT --resources-dir RESOURCES [-q MAPQ] [-t THREADS]" >&2; exit 1; }

CLEAN="$RESOURCES/cleaned_no_chrM"
SUBDIR="alignment_qc_mapq${MAPQ}"

# tag -> BED file (windows + the same nine annotation BEDs as 02_mosdepth_annotations.py)
declare -A BEDS=(
  [10kb]="$RESOURCES/Genome_10kb_windows.bed"
  [100kb]="$RESOURCES/Genome_100kb_windows.bed"
  [promoters]="$CLEAN/promoters/promoters_gene_level.bed"
  [gene_bodies]="$CLEAN/gene_bodies/gene_bodies_gene_level.bed"
  [tss_gene]="$CLEAN/tss/tss_gene_level.bed"
  [tss_transcripts]="$CLEAN/tss/tss_transcripts.bed"
  [tss_dedup]="$CLEAN/tss/tss_deduplicated.bed"
  [tss_collapsed]="$CLEAN/tss/tss_collapsed.bed"
  [exons_all]="$CLEAN/exons/exons_all.bed"
  [exons_merged]="$CLEAN/exons/exons_gene_merged.bed"
  [cpg_islands]="$CLEAN/cpg_islands/cpg_islands.bed"
)
TAGS="10kb 100kb promoters gene_bodies tss_gene tss_transcripts tss_dedup tss_collapsed exons_all exons_merged cpg_islands"

# check all BEDs before starting
for tag in $TAGS; do
  [[ -s "${BEDS[$tag]}" ]] || { echo "ERROR: missing BED for $tag: ${BEDS[$tag]}" >&2; exit 1; }
done

shopt -s nullglob
BAMS=("$RESULTS"/*/Alignment/*.bam)
[[ ${#BAMS[@]} -gt 0 ]] || { echo "ERROR: no BAM files in $RESULTS/*/Alignment/" >&2; exit 1; }
echo "Found ${#BAMS[@]} BAM files; mosdepth --mapq $MAPQ with $THREADS threads"

for bam in "${BAMS[@]}"; do
  sample=$(basename "$(dirname "$(dirname "$bam")")")
  [[ -s "$bam.bai" ]] || { echo "ERROR: missing index $bam.bai" >&2; exit 1; }
  qc="$OUT/$sample/$SUBDIR"
  mkdir -p "$qc"
  echo "== $sample"

  # whole-genome / per-chromosome summary
  mosdepth -t "$THREADS" --fast-mode --mapq "$MAPQ" \
    "$qc/$sample" "$bam" > "$qc/$sample.mosdepth.log" 2>&1

  # windows and annotation regions
  for tag in $TAGS; do
    echo "   $tag"
    mosdepth -t "$THREADS" --fast-mode --mapq "$MAPQ" \
      --by "${BEDS[$tag]}" --thresholds "$THRESHOLDS" \
      "$qc/$sample.$tag" "$bam" > "$qc/$sample.$tag.mosdepth.log" 2>&1
  done
done

echo "Done. Outputs in $OUT/*/$SUBDIR/"
