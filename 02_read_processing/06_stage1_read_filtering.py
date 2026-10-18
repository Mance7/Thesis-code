#!/usr/bin/env python3
"""
Stage 1 (fragment calling): eligibility filtering of the aligned reads and
concatemer recovery, writing one fragment record per eligible read.

A primary alignment is kept only if it is mapped, on a canonical chromosome
(chr1-22, X, Y), of mapping quality >= --min-mapq and without hard clips.

  - direct fragment: no soft clip at either end (zero-clip rule, as in
    Yu et al. 2023). Fragment = reference span of the alignment. Direct
    reads that also carry a supplementary alignment are kept and flagged
    (has_incidental_sa).
  - concatemer recovery: a read that fails the clip rule but has exactly one
    supplementary alignment (SA tag) with MAPQ >= --sa-min-mapq, where both
    the primary and the supplementary segment are unclipped at their outer
    end, is split into two fragments, each with its own reference span.
    A concatemer is two molecules ligated during library preparation; its
    internal junction carries no adapter, so adapter trimming cannot detect
    it, and the zero-clip rule alone would discard both molecules.
  - putative structural variants: pairs whose two segments map to the same
    chromosome within 10 kb are not split, because they more likely reflect
    a real intramolecular rearrangement than two ligated molecules. They are
    written to a separate table and not analysed further.
  - everything else is excluded.

Settings used in the thesis:
  ONT     --min-mapq 20 (default)
  PacBio  --min-mapq 30 --bam-root <pacbio_align.sh output> --bam-glob '*.bwa.dedup.sorted.bam'
  Sensitivity run without the zero-clip rule: --no-zero-clip, which replaces it
  by a length-scaled tolerance (2% of the aligned span, at least 10 bp and at
  most 150 bp of soft clip at each end).

Input:  BAM_ROOT/<sample>/<sample>.cutadapt_trimmed.sorted.bam (from 03_align_trimmed.py)
Output (in --out):
  <sample>.eligible_fragments.csv.gz         sample, chrom, start, end, length, strand,
                                             mapq, source (direct / recovered_concatemer_*),
                                             has_incidental_sa
  <sample>.putative_intramolecular_sv.csv.gz same-chromosome pairs within 10 kb
  <sample>.done                              completion marker (finished samples are skipped on re-run)
  stage1_summary.csv                         per-sample read counts at each filter
"""

import argparse
import csv
import gzip
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

# ---------------------------------------------------------------------------
# Settings (defaults; paths and thresholds are set from the command line in main())
# ---------------------------------------------------------------------------

BAM_ROOT = None          # --bam-root
BAM_GLOB_PATTERN = "*.cutadapt_trimmed.sorted.bam"
OUT_DIR = None           # --out, or BAM_ROOT/stage1_filtering_results_<mode>

MIN_MAPQ = 20

# --- Clip tolerance ---
# "absolute" = zero-clip rule of Yu et al. 2023 (MAX_SOFT_CLIP = 0)
# "relative" = length-scaled tolerance, used only for the sensitivity run
CLIP_TOLERANCE_MODE = "absolute"
MAX_SOFT_CLIP = 0                  # bp, used only if CLIP_TOLERANCE_MODE == "absolute"
RELATIVE_CLIP_FRACTION = 0.02      # used only if CLIP_TOLERANCE_MODE == "relative"
MIN_ABS_TOLERANCE = 10             # bp floor, "relative" mode only
MAX_ABS_TOLERANCE = 150            # bp ceiling, "relative" mode only

# --- Concatemer recovery ---
ENABLE_CONCATEMER_RECOVERY = True
SA_GOOD_MAPQ = 20                  # required MAPQ on the SA entry to trust a split

# --- Same-chromosome-close exclusion (only relevant if recovery is enabled) ---
ENABLE_SV_EXCLUSION = True
SAME_CHROM_EXCLUDE_BP = 10_000     # pairs closer than this on one chromosome are putative SVs

# Canonical chromosomes only: excludes chrM (no chromatin/nucleosome
# structure, very different copy number) and alt/unplaced/decoy contigs.
CANONICAL_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

N_WORKERS = max(1, int(os.cpu_count() * 0.5))

CIGAR_STR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
REF_CONSUMING_OPS = set("MDN=X")

# ---------------------------------------------------------------------------


def allowed_clip(length):
    if CLIP_TOLERANCE_MODE == "absolute":
        return MAX_SOFT_CLIP
    return min(MAX_ABS_TOLERANCE, max(MIN_ABS_TOLERANCE, RELATIVE_CLIP_FRACTION * length))


def get_samples():
    samples = []
    for p in BAM_ROOT.iterdir():
        if not p.is_dir() or p == OUT_DIR:
            continue
        if list(p.glob(BAM_GLOB_PATTERN)):
            samples.append(p.name)
    return sorted(samples)


def bam_path(sample):
    matches = list((BAM_ROOT / sample).glob(BAM_GLOB_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No BAM matching {BAM_GLOB_PATTERN} found for {sample} in {BAM_ROOT / sample}")
    return matches[0]


def clip_lengths(cigartuples):
    """From pysam cigartuples (int op codes: 4=S, 5=H)."""
    if not cigartuples:
        return 0, 0, False
    has_hard_clip = any(op == 5 for op, _ in cigartuples)
    lead = cigartuples[0][1] if cigartuples[0][0] == 4 else 0
    trail = cigartuples[-1][1] if cigartuples[-1][0] == 4 else 0
    return lead, trail, has_hard_clip


def parse_sa_tag(sa_string):
    """SA:Z field: rname,pos,strand,CIGAR,mapQ,NM; (one or more, semicolon-sep)."""
    entries = [e for e in sa_string.strip(";").split(";") if e]
    parsed = []
    for e in entries:
        fields = e.split(",")
        if len(fields) < 6:
            continue
        rname, pos, strand, cigar_str, mapq, nm = fields[:6]
        try:
            parsed.append({
                "rname": rname,
                "pos": int(pos),
                "strand": strand,
                "cigar_str": cigar_str,
                "mapq": int(mapq),
            })
        except ValueError:
            continue
    return parsed


def cigar_str_clip_and_span(cigar_str):
    """From a CIGAR string (as in the SA tag), return (lead_clip, trail_clip, ref_span)."""
    ops = CIGAR_STR_RE.findall(cigar_str)
    if not ops:
        return 0, 0, 0
    ops = [(int(length), op) for length, op in ops]

    lead = ops[0][0] if ops[0][1] in ("S", "H") else 0
    trail = ops[-1][0] if ops[-1][1] in ("S", "H") else 0
    ref_span = sum(length for length, op in ops if op in REF_CONSUMING_OPS)

    return lead, trail, ref_span


def try_recover_concatemer(read, sample):
    """
    Attempt to split a failing read into two real fragments via its SA tag.
    Returns (recovered_fragment_rows, putative_sv_rows) -- normally one of
    these is empty. recovered_fragment_rows has 0 or 2 entries;
    putative_sv_rows has 0 or 1.
    """
    if not read.has_tag("SA"):
        return [], []

    sa_entries = parse_sa_tag(read.get_tag("SA"))
    if len(sa_entries) != 1:
        return [], []  # only trust clean, unambiguous single-junction cases

    sa = sa_entries[0]
    if sa["mapq"] < SA_GOOD_MAPQ:
        return [], []

    # --- primary segment ---
    primary_lead, primary_trail, _ = clip_lengths(read.cigartuples)
    primary_ref_span = read.reference_end - read.reference_start
    primary_true_end_clip = min(primary_lead, primary_trail)
    if primary_true_end_clip > allowed_clip(primary_ref_span):
        return [], []

    # --- SA (supplementary) segment ---
    sa_lead, sa_trail, sa_ref_span = cigar_str_clip_and_span(sa["cigar_str"])
    if sa_ref_span == 0:
        return [], []
    sa_true_end_clip = min(sa_lead, sa_trail)
    if sa_true_end_clip > allowed_clip(sa_ref_span):
        return [], []

    sa_start = sa["pos"] - 1  # SA tag position is 1-based
    sa_end = sa_start + sa_ref_span

    # Same-chromosome-close pairs: more likely a real intramolecular
    # structural event than ligation of two independent molecules (pooled
    # primary-SA distance analysis over all 14 samples). Not split into two
    # fragments, but kept in a separate table rather than discarded, since
    # they may be real structural variants in a genomically unstable cancer.
    if ENABLE_SV_EXCLUSION and read.reference_name == sa["rname"]:
        distance = min(
            abs(read.reference_start - sa_end),
            abs(sa_start - read.reference_end),
        )
        if distance <= SAME_CHROM_EXCLUDE_BP:
            return [], [{
                "sample": sample,
                "read_name": read.query_name,
                "primary_chrom": read.reference_name,
                "primary_start": read.reference_start,
                "primary_end": read.reference_end,
                "sa_chrom": sa["rname"],
                "sa_start": sa_start,
                "sa_end": sa_end,
                "distance_bp": distance,
                "sa_mapq": sa["mapq"],
            }]

    return [
        {
            "sample": sample,
            "chrom": read.reference_name,
            "start": read.reference_start,
            "end": read.reference_end,
            "length": primary_ref_span,
            "strand": "-" if read.is_reverse else "+",
            "mapq": read.mapping_quality,
            "source": "recovered_concatemer_primary",
            "has_incidental_sa": False,
        },
        {
            "sample": sample,
            "chrom": sa["rname"],
            "start": sa_start,
            "end": sa_end,
            "length": sa_ref_span,
            "strand": sa["strand"],
            "mapq": sa["mapq"],
            "source": "recovered_concatemer_supplementary",
            "has_incidental_sa": False,
        },
    ], []


FRAGMENT_FIELDNAMES = ["sample", "chrom", "start", "end", "length", "strand",
                       "mapq", "source", "has_incidental_sa"]
SV_FIELDNAMES = ["sample", "read_name", "primary_chrom", "primary_start", "primary_end",
                  "sa_chrom", "sa_start", "sa_end", "distance_bp", "sa_mapq"]


def process_sample(sample):
    bam = pysam.AlignmentFile(str(bam_path(sample)), "rb")

    frag_path = OUT_DIR / f"{sample}.eligible_fragments.csv.gz"
    sv_path = OUT_DIR / f"{sample}.putative_intramolecular_sv.csv.gz"

    n_total = 0
    n_non_canonical_chrom = 0
    n_secondary_or_supp = 0
    n_low_mapq = 0
    n_hard_clip = 0
    n_direct_eligible = 0
    n_direct_with_incidental_sa = 0
    n_recovered_reads = 0
    n_recovered_fragments = 0
    n_putative_sv = 0
    n_excluded = 0

    # Stream both outputs to disk as rows are produced, so the full
    # per-sample fragment set (tens of millions of rows) is never held in memory.
    with gzip.open(frag_path, "wt", newline="") as frag_f, \
         gzip.open(sv_path, "wt", newline="") as sv_f:

        frag_writer = csv.DictWriter(frag_f, fieldnames=FRAGMENT_FIELDNAMES)
        frag_writer.writeheader()
        sv_writer = csv.DictWriter(sv_f, fieldnames=SV_FIELDNAMES)
        sv_writer.writeheader()

        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            n_total += 1

            if read.reference_name not in CANONICAL_CHROMS:
                n_non_canonical_chrom += 1
                continue

            if read.is_secondary or read.is_supplementary:
                n_secondary_or_supp += 1
                continue

            if read.mapping_quality < MIN_MAPQ:
                n_low_mapq += 1
                continue

            lead_clip, trail_clip, has_hard = clip_lengths(read.cigartuples)

            if has_hard:
                n_hard_clip += 1
                continue

            ref_span = read.reference_end - read.reference_start
            tolerance = allowed_clip(ref_span)

            if lead_clip <= tolerance and trail_clip <= tolerance:
                n_direct_eligible += 1
                has_incidental_sa = read.has_tag("SA")
                if has_incidental_sa:
                    n_direct_with_incidental_sa += 1
                frag_writer.writerow({
                    "sample": sample,
                    "chrom": read.reference_name,
                    "start": read.reference_start,
                    "end": read.reference_end,
                    "length": ref_span,
                    "strand": "-" if read.is_reverse else "+",
                    "mapq": read.mapping_quality,
                    "source": "direct",
                    "has_incidental_sa": has_incidental_sa,
                })
                continue

            # failed direct eligibility: attempt concatemer recovery, if enabled
            if ENABLE_CONCATEMER_RECOVERY:
                recovered, putative_sv = try_recover_concatemer(read, sample)
            else:
                recovered, putative_sv = [], []

            if recovered:
                n_recovered_reads += 1
                n_recovered_fragments += len(recovered)
                for row in recovered:
                    frag_writer.writerow(row)
            elif putative_sv:
                n_putative_sv += len(putative_sv)
                for row in putative_sv:
                    sv_writer.writerow(row)
            else:
                n_excluded += 1

    bam.close()

    n_eligible_total = n_direct_eligible + n_recovered_fragments

    summary = {
        "sample": sample,
        "n_total_mapped": n_total,
        "n_non_canonical_chrom": n_non_canonical_chrom,
        "n_secondary_or_supplementary": n_secondary_or_supp,
        "n_low_mapq": n_low_mapq,
        "n_hard_clip": n_hard_clip,
        "n_direct_eligible": n_direct_eligible,
        "n_direct_with_incidental_sa": n_direct_with_incidental_sa,
        "n_recovered_reads": n_recovered_reads,
        "n_recovered_fragments": n_recovered_fragments,
        "n_putative_intramolecular_sv": n_putative_sv,
        "n_excluded": n_excluded,
        "n_eligible_total": n_eligible_total,
        "pct_eligible_total": round(100 * n_eligible_total / n_total, 3) if n_total else np.nan,
        "pct_gain_from_recovery": round(
            100 * n_recovered_fragments / n_direct_eligible, 2
        ) if n_direct_eligible else np.nan,
    }

    summary["fragment_path"] = str(frag_path)
    summary["sv_path"] = str(sv_path)

    # Marker written only after both files are fully closed, so a resumed
    # run can confirm completion without re-reading the large data files.
    (OUT_DIR / f"{sample}.done").touch()

    return summary


def is_valid_output(sample):
    """
    True only if the completion marker exists. The marker is written by
    process_sample() after both output files are closed, so a worker killed
    mid-write (e.g. out of memory) never produces one.
    """
    return (OUT_DIR / f"{sample}.done").exists()


def count_eligible_fragments(sample):
    """Fast line count via zcat (C-level) rather than Python gzip iteration."""
    frag_path = OUT_DIR / f"{sample}.eligible_fragments.csv.gz"
    result = subprocess.run(f"zcat '{frag_path}' | wc -l", shell=True,
                             capture_output=True, text=True)
    try:
        return int(result.stdout.strip()) - 1  # minus header
    except ValueError:
        return None


def main():
    global BAM_ROOT, BAM_GLOB_PATTERN, OUT_DIR, MIN_MAPQ, SA_GOOD_MAPQ
    global CLIP_TOLERANCE_MODE, N_WORKERS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bam-root", required=True,
                        help="folder with one sub-folder per sample holding the aligned, sorted BAM")
    parser.add_argument("--bam-glob", default=BAM_GLOB_PATTERN,
                        help="file pattern of the BAM inside each sample folder")
    parser.add_argument("-o", "--out", default=None,
                        help="output folder for fragment tables and summary "
                             "(default: BAM_ROOT/stage1_filtering_results_absolute, "
                             "or _relative with --no-zero-clip)")
    parser.add_argument("--min-mapq", type=int, default=MIN_MAPQ,
                        help="minimum mapping quality of the primary alignment (20 for ONT, 30 for PacBio)")
    parser.add_argument("--sa-min-mapq", type=int, default=SA_GOOD_MAPQ,
                        help="minimum mapping quality of the supplementary alignment used for concatemer recovery")
    parser.add_argument("--no-zero-clip", action="store_true",
                        help="sensitivity run: replace the zero-clip rule by the length-scaled clip tolerance")
    parser.add_argument("-t", "--threads", type=int, default=N_WORKERS,
                        help="number of samples processed in parallel")
    args = parser.parse_args()

    BAM_ROOT = Path(args.bam_root)
    BAM_GLOB_PATTERN = args.bam_glob
    MIN_MAPQ = args.min_mapq
    SA_GOOD_MAPQ = args.sa_min_mapq
    CLIP_TOLERANCE_MODE = "relative" if args.no_zero_clip else "absolute"
    if args.out:
        OUT_DIR = Path(args.out)
    else:
        OUT_DIR = BAM_ROOT / f"stage1_filtering_results_{CLIP_TOLERANCE_MODE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    N_WORKERS = max(1, args.threads)

    samples = get_samples()
    print("=" * 70)
    print("STAGE 1 -- ACTIVE CONFIGURATION")
    print("=" * 70)
    print(f"BAM source: {BAM_ROOT} ({BAM_GLOB_PATTERN})")
    print(f"MIN_MAPQ: {MIN_MAPQ}")
    print(f"CLIP_TOLERANCE_MODE: {CLIP_TOLERANCE_MODE}"
          + (f" (MAX_SOFT_CLIP={MAX_SOFT_CLIP})" if CLIP_TOLERANCE_MODE == "absolute"
             else f" (fraction={RELATIVE_CLIP_FRACTION}, floor={MIN_ABS_TOLERANCE}, ceiling={MAX_ABS_TOLERANCE})"))
    print(f"ENABLE_CONCATEMER_RECOVERY: {ENABLE_CONCATEMER_RECOVERY}"
          + (f" (SA_GOOD_MAPQ={SA_GOOD_MAPQ})" if ENABLE_CONCATEMER_RECOVERY else ""))
    print(f"ENABLE_SV_EXCLUSION: {ENABLE_SV_EXCLUSION}"
          + (f" (SAME_CHROM_EXCLUDE_BP={SAME_CHROM_EXCLUDE_BP})"
             if ENABLE_SV_EXCLUSION and ENABLE_CONCATEMER_RECOVERY else ""))
    print("=" * 70 + "\n")

    already_done = [s for s in samples if is_valid_output(s)]
    to_process = [s for s in samples if s not in already_done]

    summaries = []

    if already_done:
        print(f"Resuming: {len(already_done)} sample(s) already have valid output, skipping:")
        for s in already_done:
            n_eligible = count_eligible_fragments(s)
            print(f"  {s}: {n_eligible} eligible fragments already on disk (kept as-is)")
            summaries.append({
                "sample": s, "n_eligible_total": n_eligible,
                "resumed_from_previous_run": True,
            })
        print()

    print(f"Found {len(to_process)} sample(s) to process now, "
          f"with {N_WORKERS} parallel workers.\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_sample, s): s for s in to_process}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                summary = future.result()
            except FileNotFoundError as e:
                print(f"  [!] {sample}: {e}")
                continue
            summary["resumed_from_previous_run"] = False
            summaries.append(summary)
            print(f"  {sample}: eligible {summary['n_eligible_total']} "
                  f"(direct {summary['n_direct_eligible']} + "
                  f"recovered {summary['n_recovered_fragments']}), "
                  f"{summary['n_putative_intramolecular_sv']} flagged as putative SV, "
                  f"{summary['n_non_canonical_chrom']} non-canonical-chrom reads dropped "
                  f"= {summary['pct_eligible_total']}% of mapped reads eligible")

    summary_df = pd.DataFrame(summaries).sort_values("sample")
    summary_path = OUT_DIR / "stage1_summary.csv"
    summary_df.drop(columns=["fragment_path", "sv_path"], errors="ignore").to_csv(summary_path, index=False)

    print(f"\nWrote per-sample summary: {summary_path}")
    print(f"Wrote per-sample eligible fragment tables and putative-SV tables to: {OUT_DIR}")
    print("\nSummary:")
    cols = [c for c in ["sample", "n_direct_eligible", "n_recovered_fragments",
                         "n_putative_intramolecular_sv", "n_eligible_total",
                         "pct_gain_from_recovery", "resumed_from_previous_run"]
            if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
