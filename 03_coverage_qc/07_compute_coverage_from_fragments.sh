#!/usr/bin/env bash
# 07_compute_coverage_from_fragments.sh
#
# Computes coverage from the Stage 1 fragments (MAPQ >= 20) with
# `bedtools coverage -mean`, against the same 10 kb / 100 kb window BEDs and the
# same annotation BEDs used in the mosdepth runs (01/02). This repeats the
# mosdepth coverage QC on exactly the fragment set used downstream.
#
# The window/annotation BEDs are first sorted into the chromosome order of the
# genome file (copies in OUT/sorted_beds/, the originals are not touched), so
# that the fast -sorted algorithm can be used.
#
# Input : FRAG_BED_DIR/{sample}.fragments.sorted.bed  (06_fragments_csv_to_bed.py)
# Output: OUT/{sample}/{sample}.{tag}.coverage_mean.bed
#         (chrom, start, end, mean depth per interval; tag = 10kb, 100kb or annotation)
#
# Usage:
#   bash 07_compute_coverage_from_fragments.sh -i FRAG_BED_DIR -o OUT -g GENOME --resources-dir RESOURCES [-t THREADS]
#
# Requires: bedtools 2.31.1

set -euo pipefail

RESOURCES=""
GENOME=""
IN=""
OUT=""
THREADS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--in) IN="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -g|--genome) GENOME="$2"; shift 2 ;;
    --resources-dir) RESOURCES="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$IN" && -n "$OUT" && -n "$GENOME" && -n "$RESOURCES" ]] || { echo "usage: $0 -i FRAG_BED_DIR -o OUT -g GENOME --resources-dir RESOURCES [-t THREADS]" >&2; exit 1; }

CLEAN="$RESOURCES/cleaned_no_chrM"
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

[[ -s "$GENOME" ]] || { echo "ERROR: genome file not found: $GENOME" >&2; exit 1; }
shopt -s nullglob
FRAGS=("$IN"/*.fragments.sorted.bed)
[[ ${#FRAGS[@]} -gt 0 ]] || { echo "ERROR: no *.fragments.sorted.bed in $IN" >&2; exit 1; }

# 1. sorted 3-column copies of the window/annotation BEDs (genome-file order)
SORTED="$OUT/sorted_beds"
mkdir -p "$SORTED"
for tag in $TAGS; do
  bed="${BEDS[$tag]}"
  [[ -s "$bed" ]] || { echo "ERROR: missing BED for $tag: $bed" >&2; exit 1; }
  if [[ ! -s "$SORTED/$tag.bed" ]]; then
    cut -f1-3 "$bed" | bedtools sort -g "$GENOME" -i - > "$SORTED/$tag.bed"
  fi
done

# 2. mean depth per interval for one sample and all BEDs
coverage_one() {
  local frag="$1"
  local sample
  sample=$(basename "$frag" .fragments.sorted.bed)
  mkdir -p "$OUT/$sample"
  for tag in $TAGS; do
    bedtools coverage -mean -sorted -g "$GENOME" \
      -a "$SORTED/$tag.bed" -b "$frag" > "$OUT/$sample/$sample.$tag.coverage_mean.bed"
  done
  echo "  $sample done"
}

echo "Computing coverage for ${#FRAGS[@]} samples ($THREADS in parallel)"
fail=0
for frag in "${FRAGS[@]}"; do
  while (( $(jobs -rp | wc -l) >= THREADS )); do wait -n || fail=1; done
  coverage_one "$frag" &
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || fail=1; done
[[ $fail -eq 0 ]] || { echo "ERROR: at least one sample failed" >&2; exit 1; }
echo "Done. Outputs in $OUT/{sample}/"
