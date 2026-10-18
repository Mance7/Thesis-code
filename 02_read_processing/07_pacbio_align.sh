#!/usr/bin/env bash
# PacBio CCS cfDNA reads -> aligned, de-duplicated BAMs ready for Stage 1.
#
# Follows the PacBio processing of Yu et al. 2023 (Clin Chem 69:168, Supplemental Methods):
#   "Only CCS reads with at least three full-pass subreads were aligned to the human reference
#    genome (hg19) with BWA. All duplicated reads and reads with alignment quality score below
#    30 were removed."
# Steps:
#   1. pass filter     np >= 3 on the provided *.ccs.bam (CCS reads are already built and the
#                      hairpin adapters already removed, so there is no adapter trimming for PacBio)
#   2. align           bwa mem 0.7.19 (default) to the same GRCh38 reference as the ONT data
#                      (instead of hg19); np/rq tags are carried into the BAM (samtools fastq -T + bwa mem -C)
#   3. duplicates      samtools fixmate -m | sort | markdup -r  (duplicates removed, as in the paper)
#   4. MAPQ >= 30      applied in Stage 1 (06_stage1_read_filtering.py --min-mapq 30); the BAM keeps all reads
# --aligner pbmm2 (PacBio's minimap2 wrapper) is available as an alternative aligner.
#
# Usage:
#   bash 07_pacbio_align.sh -r GRCh38.fa -o OUTDIR [-p MIN_PASSES] [--aligner bwa|pbmm2] \
#        [-j SAMPLES_IN_PARALLEL] [-t THREADS_PER_SAMPLE] FILE.ccs.bam [FILE.ccs.bam ...]
# Output per sample: OUTDIR/<sample>/<sample>.<aligner>.dedup.sorted.bam (+ .bai), .flagstat.txt,
#   .passfilter.txt, .markdup.txt, .done; plus OUTDIR/pacbio_passfilter_summary.tsv.
#   Re-running skips finished samples. Cores used = -j x -t.
# Requires: samtools >= 1.12, bwa (or pbmm2). The bwa index (REF.bwt etc.) is built once if missing.
set -euo pipefail

REF=""; OUTDIR=""; MIN_PASSES=3; ALIGNER=bwa; JOBS=2; THREADS=8; FILES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r) REF="$2"; shift 2 ;;
    -o) OUTDIR="$2"; shift 2 ;;
    -p) MIN_PASSES="$2"; shift 2 ;;
    --aligner) ALIGNER="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 1 ;;
    *) FILES+=("$1"); shift ;;
  esac
done
[[ -n "$REF" && -n "$OUTDIR" && ${#FILES[@]} -gt 0 ]] || { echo "usage: $0 -r REF.fa -o OUTDIR [options] FILE.ccs.bam ..." >&2; exit 1; }
[[ "$ALIGNER" == bwa || "$ALIGNER" == pbmm2 ]] || { echo "--aligner must be bwa or pbmm2" >&2; exit 1; }
command -v samtools >/dev/null || { echo "samtools not found" >&2; exit 1; }
command -v "$ALIGNER" >/dev/null || { echo "$ALIGNER not found (conda install -c bioconda $ALIGNER)" >&2; exit 1; }
mkdir -p "$OUTDIR"

if [[ "$ALIGNER" == bwa ]]; then
  [[ -s "$REF.bwt" ]] || { echo "[index] bwa index $REF (one-off, about an hour for hg38)"; bwa index "$REF"; }
  IDX="$REF"
else
  IDX="$OUTDIR/$(basename -- "${REF%.*}").ccs.mmi"
  [[ -s "$IDX" ]] || { echo "[index] building $IDX"; pbmm2 index --preset CCS "$REF" "$IDX"; }
fi

align_one() {
  local f="$1"
  local s; s=$(basename -- "$f"); s="${s%%.*}"
  local d="$OUTDIR/$s"; mkdir -p "$d"
  local out="$d/$s.$ALIGNER.dedup.sorted.bam"
  [[ -e "$d/$s.done" ]] && { echo "[$s] done already, skipping"; return 0; }
  if ! samtools view "$f" 2>/dev/null | head -n 1000 | grep -q $'\tnp:i:'; then
    echo "[$s] ERROR: no np tag in the first reads; is this a CCS (HiFi) BAM? Stopping this sample." >&2
    return 1
  fi
  samtools view -@ 2 -b -e "[np]>=$MIN_PASSES" -o "$d/$s.pass.bam" "$f"
  local n_in n_pass
  n_in=$(samtools view -c -@ 2 "$f"); n_pass=$(samtools view -c -@ 2 "$d/$s.pass.bam")
  printf "sample\tn_ccs\tn_pass\tmin_passes\n%s\t%s\t%s\t%s\n" "$s" "$n_in" "$n_pass" "$MIN_PASSES" > "$d/$s.passfilter.txt"

  if [[ "$ALIGNER" == bwa ]]; then
    samtools fastq -@ 2 -T np,rq "$d/$s.pass.bam" 2>/dev/null \
      | bwa mem -C -t "$THREADS" -R "@RG\tID:$s\tSM:$s\tPL:PACBIO" "$IDX" - 2> "$d/$s.bwa.log" \
      | samtools fixmate -m -@ 2 - - \
      | samtools sort -@ "$THREADS" -T "$d/$s.tmp" -o "$d/$s.sorted.bam" -
  else
    pbmm2 align "$IDX" "$d/$s.pass.bam" "$d/$s.unsorted.bam" --preset CCS -j "$THREADS" --sample "$s" > "$d/$s.pbmm2.log" 2>&1
    samtools sort -n -@ "$THREADS" "$d/$s.unsorted.bam" | samtools fixmate -m - - \
      | samtools sort -@ "$THREADS" -T "$d/$s.tmp" -o "$d/$s.sorted.bam" -
    rm -f "$d/$s.unsorted.bam"
  fi
  samtools markdup -r -s -@ "$THREADS" "$d/$s.sorted.bam" "$out" 2> "$d/$s.markdup.txt"
  samtools index -@ "$THREADS" "$out"
  samtools flagstat -@ "$THREADS" "$out" > "$d/$s.flagstat.txt"
  rm -f "$d/$s.pass.bam" "$d/$s.sorted.bam"
  touch "$d/$s.done"
  echo "[$s] CCS $n_in, >=$MIN_PASSES passes $n_pass -> $out"
}
export -f align_one
export OUTDIR IDX ALIGNER MIN_PASSES THREADS

printf '%s\n' "${FILES[@]}" | xargs -P "$JOBS" -I{} bash -c 'align_one "$@"' _ {}

SUM="$OUTDIR/pacbio_passfilter_summary.tsv"
printf "sample\tn_ccs\tn_pass\tmin_passes\n" > "$SUM"
for d in "$OUTDIR"/*/; do s=$(basename -- "$d"); [[ -s "$d/$s.passfilter.txt" ]] && tail -n +2 "$d/$s.passfilter.txt" >> "$SUM"; done
echo "Summary: $SUM. Next: python 06_stage1_read_filtering.py --bam-root \"$OUTDIR\" --bam-glob '*.$ALIGNER.dedup.sorted.bam' --min-mapq 30"
