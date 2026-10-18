#!/usr/bin/env bash
# 01_download_reference.sh
#
# Downloads the GRCh38 no-alt analysis set (GCA_000001405.15) from NCBI together
# with its .fai and the ready-made BWA index, and unpacks them. This is the
# reference the external short-read cohorts are aligned to with BWA-MEM
# (Methods 2.13). Downloads are resumable: the script retries until every file
# is complete, so it can simply be run again after a dropped connection.
#
# Output: OUT/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna (+ .fai, BWA index
#         .amb .ann .bwt .pac .sa)
#
# Usage:
#   bash 01_download_reference.sh -o OUT
#
# Requires: wget, gzip, tar

set -uo pipefail

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ -n "$OUT" ] || { echo "usage: bash $0 -o OUT" >&2; exit 1; }

B="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
N="GCA_000001405.15_GRCh38_no_alt_analysis_set"
REF="$OUT/$N.fna"
say() { echo "$(date '+%F %T') $*"; }

mkdir -p "$OUT"; cd "$OUT" || exit 1

# 1) resumable download, retry with growing pauses until each file is complete
for f in "$N.fna.fai" "$N.fna.gz" "$N.fna.bwa_index.tar.gz"; do
  n=0
  until wget -c -q --show-progress --progress=dot:giga --tries=0 --retry-connrefused --waitretry=30 \
        --read-timeout=60 --timeout=60 "$B/$f"; do
    n=$((n+1)); w=$(( n<10 ? 30*n : 300 ))
    say "retry $n for $f in ${w}s"; sleep $w
  done
  say "done $f"
done

# 2) unpack the genome and the BWA index next to it
[ -s "$REF" ] || { say "unpacking genome"; gzip -dc "$N.fna.gz" > "$REF.tmp" && mv "$REF.tmp" "$REF"; }
[ -s "$REF.bwt" ] || { say "unpacking bwa index"; tar -xzf "$N.fna.bwa_index.tar.gz" -C "$OUT"; }
ls "$REF".{amb,ann,bwt,pac,sa} >/dev/null || { say "STOP: bwa index files not found after unpacking"; exit 1; }
say "reference ready: $REF"
