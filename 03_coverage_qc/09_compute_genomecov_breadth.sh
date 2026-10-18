#!/usr/bin/env bash
# 09_compute_genomecov_breadth.sh
#
# True per-base breadth of coverage from the Stage 1 fragments (MAPQ >= 20).
# Window mean depth over-estimates breadth (a window with one deep spike has a
# high mean but most of its bases are uncovered), so we use
# `bedtools genomecov -bga`, which reports every base of the genome, including
# zero-depth stretches, as runs of equal depth.
#
# From the bedGraph we compute per sample:
#   genome_bp      total length of chr1-22, chrX, chrY
#   covered_bp     bases with depth >= MIN_DEPTH (default 1)
#   breadth_pct    100 * covered_bp / genome_bp
#   mean_depth     sum(depth * run length) / genome_bp
#
# Input : FRAG_BED_DIR/{sample}.fragments.sorted.bed  (06_fragments_csv_to_bed.py)
# Output: OUT/genomecov_breadth.tsv (one row per sample)
#         OUT/bedgraph/{sample}.bga.bedgraph.gz with --keep-bedgraph
#
# Usage:
#   bash 09_compute_genomecov_breadth.sh -i FRAG_BED_DIR -o OUT -g GENOME [-d MIN_DEPTH] [-t THREADS] [--keep-bedgraph]
#
# Requires: bedtools 2.31.1

set -euo pipefail

GENOME=""
IN=""
OUT=""
MIN_DEPTH=1
THREADS=4
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--in) IN="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -g|--genome) GENOME="$2"; shift 2 ;;
    -d|--min-depth) MIN_DEPTH="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    --keep-bedgraph) KEEP=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$IN" && -n "$OUT" && -n "$GENOME" ]] || { echo "usage: $0 -i FRAG_BED_DIR -o OUT -g GENOME [-d MIN_DEPTH] [-t THREADS] [--keep-bedgraph]" >&2; exit 1; }

[[ -s "$GENOME" ]] || { echo "ERROR: genome file not found: $GENOME" >&2; exit 1; }
shopt -s nullglob
FRAGS=("$IN"/*.fragments.sorted.bed)
[[ ${#FRAGS[@]} -gt 0 ]] || { echo "ERROR: no *.fragments.sorted.bed in $IN" >&2; exit 1; }
mkdir -p "$OUT"
[[ $KEEP -eq 1 ]] && mkdir -p "$OUT/bedgraph"

# genome file restricted to the canonical chromosomes (the only ones in Stage 1)
CANON="$OUT/genome.canonical.chrom.sizes"
awk 'BEGIN{OFS="\t"} $1 ~ /^chr([0-9]+|X|Y)$/ {print $1,$2}' "$GENOME" > "$CANON"

breadth_one() {
  local frag="$1"
  local sample
  sample=$(basename "$frag" .fragments.sorted.bed)
  # -bga: bedGraph including zero-coverage runs, so the runs tile the whole genome
  bedtools genomecov -bga -i "$frag" -g "$CANON" \
    | { if [[ $KEEP -eq 1 ]]; then tee >(gzip -c > "$OUT/bedgraph/$sample.bga.bedgraph.gz"); else cat; fi; } \
    | awk -v s="$sample" -v d="$MIN_DEPTH" 'BEGIN{OFS="\t"}
        { len = $3 - $2; g += len; b += $4 * len; if ($4 >= d) c += len }
        END { printf "%s\t%d\t%d\t%.4f\t%.6f\n", s, g, c, (g > 0) ? 100 * c / g : 0, (g > 0) ? b / g : 0 }' \
    > "$OUT/.breadth.$sample.tsv"
  echo "  $sample done"
}

echo "genomecov -bga for ${#FRAGS[@]} samples ($THREADS in parallel), breadth at >= ${MIN_DEPTH}x"
fail=0
for frag in "${FRAGS[@]}"; do
  while (( $(jobs -rp | wc -l) >= THREADS )); do wait -n || fail=1; done
  breadth_one "$frag" &
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || fail=1; done
[[ $fail -eq 0 ]] || { echo "ERROR: at least one sample failed" >&2; exit 1; }

TABLE="$OUT/genomecov_breadth.tsv"
printf "sample\tgenome_bp\tcovered_bp\tbreadth_pct\tmean_depth\n" > "$TABLE"
for frag in "${FRAGS[@]}"; do
  s=$(basename "$frag" .fragments.sorted.bed)
  cat "$OUT/.breadth.$s.tsv" >> "$TABLE"
  rm -f "$OUT/.breadth.$s.tsv"
done
echo "Done. Breadth table: $TABLE"
column -t "$TABLE"
