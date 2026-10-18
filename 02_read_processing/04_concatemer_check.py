#!/usr/bin/env python3
"""
Concatemer check: for reads that FAIL the current soft-clip eligibility
criterion, do their clipped-off portions actually align cleanly somewhere
else in the genome (SA tag with a good MAPQ), or do they look like junk?

Biology: during ONT library prep, two separate DNA fragments can get
end-to-end ligated by accident before sequencing (a "chimeric" or
"concatenated" read). If the basecaller's mid-read adapter-splitting step
didn't catch this, one physical sequencing read actually contains two real,
independent cfDNA fragments stuck together. When aligned, minimap2 correctly
maps the first fragment as the "primary" alignment and reports the second
fragment as a separate "supplementary" (SA) alignment elsewhere in the
genome, since they don't belong at the same genomic position.

This matters because it changes what a "failing" read actually is:
  - SA tag present, good MAPQ, exactly one extra segment: strong evidence of
    a genuine two-fragment concatemer. Potentially RECOVERABLE -- the primary
    and supplementary alignments could be split back into two separate,
    legitimate fragments instead of being thrown away entirely.
  - SA tag present but multiple segments, or low MAPQ: more likely a
    repetitive/low-complexity region (e.g. telomeric repeats) causing
    genuine multi-mapping ambiguity, not a clean two-fragment ligation.
    Not a simple recovery case.
  - No SA tag at all: the clipped-off sequence didn't align anywhere
    confidently -- likely a real quality/artifact issue (basecalling noise,
    adapter remnant, degraded end), not a hidden second fragment. Correctly
    excluded, nothing to recover.

This check was run on the first (pre-trimming) alignments, where Stage 1
used a length-scaled relative clip tolerance; the same tolerance defines
"failing" here, so it checks exactly the population that rule discarded.
The results motivated the concatemer-recovery step of Stage 1
(06_stage1_read_filtering.py).

Input:  RESULTS/<sample>/Alignment/*.sorted.bam
Output: OUT/concatemer_check_summary.csv (per sample: failing reads and the
        share with no SA tag, one good SA, one low-MAPQ SA, or several SA)
"""

import argparse

from pathlib import Path
import pysam
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# paths are set from the command line in main()
RESULTS = None
OUT_DIR = None

MIN_MAPQ = 20
CLIP_TOLERANCE_MODE = "relative"
RELATIVE_CLIP_FRACTION = 0.02
MIN_ABS_TOLERANCE = 10
MAX_ABS_TOLERANCE = 150
MAX_SOFT_CLIP = 0  # only used if CLIP_TOLERANCE_MODE == "absolute"

SA_GOOD_MAPQ = 20  # MAPQ threshold for a "confident" supplementary alignment

N_WORKERS = 4

# ---------------------------------------------------------------------------


def allowed_clip(read_length):
    if CLIP_TOLERANCE_MODE == "absolute":
        return MAX_SOFT_CLIP
    return min(MAX_ABS_TOLERANCE, max(MIN_ABS_TOLERANCE, RELATIVE_CLIP_FRACTION * read_length))


def get_samples():
    return sorted(
        p.name for p in RESULTS.iterdir()
        if p.is_dir() and (p / "Alignment").exists()
    )


def bam_path(sample):
    matches = list((RESULTS / sample / "Alignment").glob("*.sorted.bam"))
    if not matches:
        raise FileNotFoundError(f"No sorted BAM found for {sample}")
    return matches[0]


def clip_lengths(cigartuples):
    if not cigartuples:
        return 0, 0, False
    has_hard_clip = any(op == 5 for op, _ in cigartuples)
    lead = cigartuples[0][1] if cigartuples[0][0] == 4 else 0
    trail = cigartuples[-1][1] if cigartuples[-1][0] == 4 else 0
    return lead, trail, has_hard_clip


def parse_sa_tag(sa_string):
    """SA:Z field is semicolon-separated entries: rname,pos,strand,CIGAR,mapQ,NM;"""
    entries = [e for e in sa_string.strip(";").split(";") if e]
    mapqs = []
    for e in entries:
        fields = e.split(",")
        if len(fields) >= 5:
            try:
                mapqs.append(int(fields[4]))
            except ValueError:
                pass
    return len(entries), mapqs


def process_sample(sample):
    bam = pysam.AlignmentFile(str(bam_path(sample)), "rb")

    n_failing = 0
    n_no_sa = 0
    n_single_sa_good = 0
    n_single_sa_low = 0
    n_multi_sa = 0

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < MIN_MAPQ:
            continue

        lead, trail, has_hard = clip_lengths(read.cigartuples)
        if has_hard:
            continue

        tolerance = allowed_clip(read.query_length or (lead + trail))
        if lead <= tolerance and trail <= tolerance:
            continue  # this read passes eligibility, not part of this check

        n_failing += 1

        if not read.has_tag("SA"):
            n_no_sa += 1
            continue

        n_entries, mapqs = parse_sa_tag(read.get_tag("SA"))

        if n_entries == 0:
            n_no_sa += 1
        elif n_entries == 1:
            if mapqs and mapqs[0] >= SA_GOOD_MAPQ:
                n_single_sa_good += 1
            else:
                n_single_sa_low += 1
        else:
            n_multi_sa += 1

    bam.close()

    total = n_failing if n_failing else 1
    return {
        "sample": sample,
        "n_failing": n_failing,
        "n_no_sa": n_no_sa,
        "pct_no_sa": round(100 * n_no_sa / total, 2),
        "n_single_sa_good_mapq": n_single_sa_good,
        "pct_single_sa_good_mapq": round(100 * n_single_sa_good / total, 2),
        "n_single_sa_low_mapq": n_single_sa_low,
        "pct_single_sa_low_mapq": round(100 * n_single_sa_low / total, 2),
        "n_multi_sa": n_multi_sa,
        "pct_multi_sa": round(100 * n_multi_sa / total, 2),
    }


def main():
    global RESULTS, OUT_DIR, N_WORKERS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True,
                        help="folder with one sub-folder per sample holding Alignment/*.sorted.bam")
    parser.add_argument("-o", "--out",
                        required=True,
                        help="output folder for the summary table")
    parser.add_argument("-t", "--threads", type=int, default=N_WORKERS,
                        help="number of samples processed in parallel")
    args = parser.parse_args()

    RESULTS = Path(args.results)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    N_WORKERS = max(1, args.threads)

    samples = get_samples()
    print(f"Running concatemer check on {len(samples)} sample(s) with {N_WORKERS} workers.\n")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
            except FileNotFoundError as e:
                print(f"  [!] {sample}: {e}")
                continue
            results.append(result)
            print(f"  {sample}: {result['n_failing']} failing reads -- "
                  f"{result['pct_single_sa_good_mapq']}% look like clean concatemers "
                  f"(single good SA), {result['pct_no_sa']}% have no SA at all")

    df = pd.DataFrame(results).sort_values("sample")
    out_path = OUT_DIR / "concatemer_check_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
