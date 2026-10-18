#!/usr/bin/env bash
# Quick check of whether ONT reads still carry their adapters.
#
# 1. Leading soft-clip lengths of the first N alignments of a BAM, in bins
#    (0, 1-9, 10-19, ... 80+). Long leading clips on most reads point to
#    adapter sequence left on the reads.
# 2. Number of the first N reads of the FASTQ that contain the ONT
#    ligation-kit 5' adapter (AATGTACTTCGTTCAGTTACGTATTGCT).
#
# Usage: bash 01_check_adapter_trimmed.sh SAMPLE.sorted.bam SAMPLE.fq.gz [N_READS]
# Output: printed to the screen.
set -euo pipefail

BAM="${1:?usage: $0 SAMPLE.sorted.bam SAMPLE.fq.gz [N_READS]}"
FASTQ="${2:?usage: $0 SAMPLE.sorted.bam SAMPLE.fq.gz [N_READS]}"
N_READS="${3:-20000}"
ADAPTER_5P="AATGTACTTCGTTCAGTTACGTATTGCT"

echo "Leading soft-clip length (first $N_READS alignments of $BAM):"
# head closes the pipe early on purpose; do not treat that as an error
set +o pipefail
samtools view "$BAM" | head -n "$N_READS" | awk '
{
  cigar = $6
  # leading soft clip = number before an S at the start of the CIGAR
  if (match(cigar, /^[0-9]+S/)) {
    print substr(cigar, RSTART, RLENGTH-1)
  } else {
    print 0
  }
}' | awk '
{
  if ($1 == 0) bin["0"]++
  else if ($1 < 10) bin["1-9"]++
  else if ($1 < 20) bin["10-19"]++
  else if ($1 < 30) bin["20-29"]++
  else if ($1 < 40) bin["30-39"]++
  else if ($1 < 50) bin["40-49"]++
  else if ($1 < 60) bin["50-59"]++
  else if ($1 < 80) bin["60-79"]++
  else bin["80+"]++
}
END { for (b in bin) print b, bin[b] }' | sort -t- -k1,1n

echo
echo "Reads containing the 5' adapter (first $N_READS reads of $FASTQ):"
zcat "$FASTQ" | awk 'NR%4==2' | head -n "$N_READS" | \
  grep -c "$ADAPTER_5P" || true
