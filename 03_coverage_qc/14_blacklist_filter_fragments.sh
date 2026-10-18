#!/usr/bin/env bash
# 14_blacklist_filter_fragments.sh
#
# Removes fragments in problem regions of GRCh38 from the Stage 1 fragment
# tables, keeping every original column, so all downstream scripts run unchanged
# on the filtered files.
#
# Exclusion set (merged into one BED):
#   1. ENCODE blacklist v2, hg38 (Amemiya et al. 2019, Sci Rep 9:9354)
#   2. UCSC hg38 gap track (centromeres, telomeres, heterochromatin, short arms, contig gaps)
#   3. UCSC hg38 centromere model track
#   4. RepeatMasker satellite repeats (centromeric/pericentromeric alpha, beta,
#      HSAT arrays): the main source of residual pericentromeric spikes
#      (on by default; --no-satellite leaves them out)
#   5. (optional, --segdup) UCSC segmental duplications with >= 99% identity
#
# Overlap rule (--mode):
#   midpoint (default): drop a fragment if its midpoint lies in an excluded region.
#            This is length-neutral. With "any overlap", a 10 kb fragment is ~50x more
#            likely to touch an excluded interval than a 200 bp one, so the filter
#            itself would deplete long fragments, which is exactly the population
#            this thesis studies.
#   any:     drop on any 1 bp overlap (the usual short-read convention). Run once
#            as a sensitivity check; both are reported.
#
# Usage:
#   bash 14_blacklist_filter_fragments.sh -o OUTDIR [--mode midpoint|any] [--no-satellite] [--segdup] [--ref-dir DIR] [-t|--threads N] FILE [FILE ...]
#   e.g. bash 14_blacklist_filter_fragments.sh -o filtered_mid/ stage1/*.eligible_fragments.csv.gz
#        bash 14_blacklist_filter_fragments.sh -o filtered_any/ --mode any stage1/*.eligible_fragments.csv.gz
#   (a separate OUTDIR per mode; filtered files keep their original names)
#
# -t/--threads N: filter N files at the same time (default 4). Each file uses about
#   3 processes (decompress, bedtools, compress); memory per file is small.
#
# --ref-dir DIR: take the reference files from DIR (same file names as the URLs
#   below: hg38-blacklist.v2.bed.gz, gap.txt.gz, centromeres.txt.gz, rmsk.txt.gz,
#   genomicSuperDups.txt.gz) instead of downloading them, for a server without
#   access to GitHub or UCSC.
#
# Input: comma- or tab-separated (optionally gzipped) table with a header that
# contains chrom, start, end columns (names can be changed with the CHROM_COL,
# START_COL, END_COL environment variables). Coordinates are 0-based half-open
# (BED / pysam convention, as written by Stage 1); COORD_1BASED=1 handles
# 1-based closed tables.
#
# Output per input: OUTDIR/<same name> (filtered, same format) and a summary
# table OUTDIR/blacklist_filter_summary.tsv (file, mode, n_in, n_removed, pct_removed).
#
# Requires: bedtools >= 2.27 (2.31.1 used), curl or wget, gzip, awk.

set -euo pipefail

CHROM_COL="${CHROM_COL:-chrom}"
START_COL="${START_COL:-start}"
END_COL="${END_COL:-end}"
COORD_1BASED="${COORD_1BASED:-0}"
OUTDIR=""
SEGDUP=0
SATELLITE=1
MODE=midpoint
REFDIR=""
THREADS=4
FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTDIR="$2"; shift 2 ;;
    --segdup) SEGDUP=1; shift ;;
    --satellite) SATELLITE=1; shift ;;
    --no-satellite) SATELLITE=0; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --ref-dir) REFDIR="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,49p' "$0"; exit 0 ;;
    *) FILES+=("$1"); shift ;;
  esac
done
[[ -n "$OUTDIR" && ${#FILES[@]} -gt 0 ]] || { echo "usage: $0 -o OUTDIR [--mode midpoint|any] [--no-satellite] [--segdup] FILE..." >&2; exit 1; }
mkdir -p "$OUTDIR/ref"
for f in "${FILES[@]}"; do  # never overwrite an input file
  if [[ "$(cd "$(dirname "$f")" && pwd -P)" == "$(cd "$OUTDIR" && pwd -P)" ]]; then
    echo "ERROR: -o $OUTDIR is the folder of the input $f; choose a different output folder." >&2; exit 1
  fi
done
REF="$OUTDIR/ref"

fetch() {  # url dest
  [[ -s "$2" ]] && return 0
  if [[ -n "$REFDIR" && -s "$REFDIR/$(basename "$2")" ]]; then cp "$REFDIR/$(basename "$2")" "$2"; return 0; fi
  local ok=0
  if command -v curl >/dev/null; then curl -fsSL -o "$2" "$1" && ok=1; else wget -q -O "$2" "$1" && ok=1; fi
  if [[ $ok -eq 0 ]]; then
    rm -f "$2"
    echo "ERROR: could not download $1" >&2
    echo "       Download it on another machine and pass its folder with --ref-dir DIR" >&2
    echo "       (the file must be named $(basename "$2"))." >&2
    exit 1
  fi
}

echo "[1/3] building exclusion BED"
fetch https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/hg38-blacklist.v2.bed.gz "$REF/hg38-blacklist.v2.bed.gz"
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz "$REF/gap.txt.gz"
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/centromeres.txt.gz "$REF/centromeres.txt.gz"
{
  zcat "$REF/hg38-blacklist.v2.bed.gz" | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,"encode_blacklist"}'
  zcat "$REF/gap.txt.gz"          | awk 'BEGIN{OFS="\t"}{print $2,$3,$4,"gap_"$8}'
  zcat "$REF/centromeres.txt.gz"  | awk 'BEGIN{OFS="\t"}{print $2,$3,$4,"centromere"}'
  if [[ $SATELLITE -eq 1 ]]; then
    fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz "$REF/rmsk.txt.gz"
    zcat "$REF/rmsk.txt.gz" | awk 'BEGIN{OFS="\t"} $12=="Satellite" {print $6,$7,$8,"satellite"}'
  fi
  if [[ $SEGDUP -eq 1 ]]; then
    fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/genomicSuperDups.txt.gz "$REF/genomicSuperDups.txt.gz"
    zcat "$REF/genomicSuperDups.txt.gz" | awk 'BEGIN{OFS="\t"} $27>=0.99 {print $2,$3,$4,"segdup99"}'
  fi
} | sort -k1,1 -k2,2n > "$REF/exclude.unmerged.bed"
bedtools merge -i "$REF/exclude.unmerged.bed" -c 4 -o distinct > "$REF/exclude.bed"
echo "    $(wc -l < "$REF/exclude.bed") merged intervals, $(awk '{s+=$3-$2}END{printf "%.1f Mb", s/1e6}' "$REF/exclude.bed") excluded"
# per-source breakdown, useful for the methods section
cut -f4 "$REF/exclude.unmerged.bed" | sed 's/^gap_.*/gap/' | sort | uniq -c > "$REF/exclude_sources.txt"

echo "[2/3] filtering fragments"
SUMMARY="$OUTDIR/blacklist_filter_summary.tsv"
printf "file\tmode\tn_in\tn_removed\tpct_removed\n" > "$SUMMARY"

filter_one() {
  local f="$1"
  base=$(basename "$f")
  out="$OUTDIR/$base"
  cat_cmd=cat; [[ "$f" == *.gz ]] && cat_cmd=zcat
  header=$($cat_cmd "$f" | head -n1 || true)
  if [[ "$header" == *$'\t'* ]]; then SEP=$'\t'; else SEP=','; fi
  idx() { echo "$header" | awk -v FS="$SEP" -v n="$1" '{for(i=1;i<=NF;i++) if($i==n){print i; exit}}'; }
  ci=$(idx "$CHROM_COL"); si=$(idx "$START_COL"); ei=$(idx "$END_COL")
  [[ -n "$ci" && -n "$si" && -n "$ei" ]] || { echo "  $base: missing $CHROM_COL/$START_COL/$END_COL in header" >&2; return 1; }
  first_chrom=$($cat_cmd "$f" | sed -n 2p | awk -v FS="$SEP" -v c="$ci" '{print $c}' || true)
  [[ "$first_chrom" == chr* ]] || echo "  WARNING $base: chromosome '$first_chrom' has no 'chr' prefix; the UCSC/ENCODE BEDs use 'chr1' naming, so nothing would be removed" >&2

  tmp=$(mktemp)
  # prefix each row with a BED3 key, intersect, strip the key
  $cat_cmd "$f" | tail -n +2 \
    | awk -v FS="$SEP" -v OFS="\t" -v c="$ci" -v s="$si" -v e="$ei" -v one="$COORD_1BASED" \
        -v mode="$MODE" '{st=$s; if(one==1) st=st-1; en=$e;
           if(mode=="midpoint"){m=int((st+en)/2); st=m; en=m+1}
           print $c, st, en, $0}' \
    | bedtools intersect -v -a - -b "$REF/exclude.bed" \
    | cut -f4- > "$tmp"
  n_in=$($cat_cmd "$f" | tail -n +2 | wc -l)
  n_out=$(wc -l < "$tmp")
  { echo "$header"; cat "$tmp"; } | { [[ "$f" == *.gz ]] && $GZ || cat; } > "$out"
  rm -f "$tmp"
  awk -v f="$base" -v md="$MODE" -v a="$n_in" -v b="$n_out" 'BEGIN{printf "%s\t%s\t%d\t%d\t%.4f\n", f, md, a, a-b, (a>0)?100*(a-b)/a:0}' > "$OUTDIR/.summary.$base.tsv"
  echo "  $base: kept $n_out / $n_in"
}

GZ="gzip -c"; command -v pigz >/dev/null && GZ="pigz -c -p 2"
echo "    $THREADS file(s) in parallel"
fail=0
for f in "${FILES[@]}"; do
  while (( $(jobs -rp | wc -l) >= THREADS )); do wait -n || fail=1; done
  filter_one "$f" &
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || fail=1; done
wait || fail=1
for f in "${FILES[@]}"; do b=$(basename "$f"); [[ -s "$OUTDIR/.summary.$b.tsv" ]] && cat "$OUTDIR/.summary.$b.tsv" >> "$SUMMARY"; rm -f "$OUTDIR/.summary.$b.tsv"; done
[[ $fail -eq 0 ]] || { echo "ERROR: at least one file failed; see messages above" >&2; exit 1; }

echo "[3/3] done. Summary: $SUMMARY"
