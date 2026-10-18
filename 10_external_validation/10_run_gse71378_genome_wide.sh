#!/usr/bin/env bash
# 10_run_gse71378_genome_wide.sh
#
# Genome-wide redo of the GSE71378 extraction for the 15 liver cancer, healthy and
# non-cancer disease runs. The ENA FASTQ files of GSE71378 are coordinate-sorted (the
# head of each file is chr1 0-7 Mb), so reads are pulled from NCBI SRA as evenly
# spaced blocks instead: GBLOCKS blocks x (GPAIRS / GBLOCKS) read pairs per run,
# aligned to GRCh38 with 09_extract_external_cohorts_genome_wide.py.
# Gives genome-wide R_sr, size histograms and 5' end motifs (too sparse for
# promoters or ichorCNA). Restartable: finished runs are kept in OUT/out_gse_v3/samples.
#
# Input : runs_gse71378.tsv (next to this script), GRCh38 FASTA with bwa index, promoter BED
# Output: OUT/out_gse_v3/ev_summary.tsv, ev_hist.tsv.gz, ev_motifs.tsv.gz; log OUT/logs/ev_gse_v3.log
#
# Usage:
#   bash 10_run_gse71378_genome_wide.sh -o OUT [--ref REF] [--promoters BED] [-t THREADS]
#                                       [--blocks GBLOCKS] [--pairs GPAIRS]
#
# Requires: conda env "thesis" (python 3.12, bwa 0.7.19) and sra-tools (fastq-dump) on PATH

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
B=""
N="GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
REF=""
PROM=""
T=8
GBLOCKS=1000
GPAIRS=300000

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) B="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --promoters) PROM="$2"; shift 2 ;;
    -t|--threads) T="$2"; shift 2 ;;
    --blocks) GBLOCKS="$2"; shift 2 ;;
    --pairs) GPAIRS="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$B" ] || { echo "usage: bash $0 -o OUT [--ref REF] [--promoters BED] [-t THREADS] [--blocks GBLOCKS] [--pairs GPAIRS]" >&2; exit 1; }
REF="${REF:-$B/ref/$N}"
PROM="${PROM:-$B/ref/promoters_gene_level.bed}"
say() { echo "$(date '+%F %T') $*"; }

# only one copy at a time; check tools before starting
mkdir -p "$B/logs"
exec 9>"$B/.gse_genome_wide.lock"; flock -n 9 || { say "STOP: already running"; exit 1; }
for x in bwa curl python fastq-dump; do command -v $x >/dev/null || { say "STOP: $x missing"; exit 1; }; done
[ -s "$REF.bwt" ] || { say "STOP: no bwa index at $REF.bwt"; exit 1; }
[ -s "$PROM" ] || { say "STOP: promoter BED not found: $PROM"; exit 1; }

say "GSE71378 genome-wide ($GBLOCKS blocks, $GPAIRS pairs per run)"
python "$HERE/09_extract_external_cohorts_genome_wide.py" --mode wgs --runs "$HERE/runs_gse71378.tsv" \
  --only-groups liver_cancer,healthy,noncancer_disease --sra-blocks "$GBLOCKS" --sra-par 8 \
  --ref "$REF" --promoters "$PROM" --out "$B/out_gse_v3" --pairs "$GPAIRS" -t "$T" --prefetch 1 \
  >> "$B/logs/ev_gse_v3.log" 2>&1
rc=$?
say "GSE71378 genome-wide finished (exit $rc): $(ls "$B"/out_gse_v3/samples/*.npz 2>/dev/null | wc -l)/15 runs; results in $B/out_gse_v3"
exit $rc
