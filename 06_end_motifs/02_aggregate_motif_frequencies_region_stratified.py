#!/usr/bin/env python3
"""
Count the 256 5' end motifs per sample, size bin and region.

For every sample, motif counts are made for each size bin x region cell
(genome_wide plus is_promoter, is_gene_body, is_tss, is_exon, is_cpg_island),
the same bin x region structure as the fragment-length entropy, so the two
feature types can be compared at the same resolution (Methods 2.7).

Input:  {sample}.fragments_with_motifs.csv.gz  (01_extract_end_motifs.py; columns
        sample, chrom, start, end, length, strand, mapq, source,
        has_incidental_sa, size_bin, is_promoter, is_gene_body, is_tss, is_exon,
        is_cpg_island, motif_5p, motif_3p, motif_valid). The 5' end motif
        (motif_5p, strand-aware as in Yu et al. 2023) is counted; rows with
        motif_valid = False are dropped before counting.
Output: <out-dir>/motif_frequencies_region_stratified_long.csv with columns
        sample, group_type ("background" | "size_bin"), bin, region, motif,
        count, n_fragments, frequency_pct
        <out-dir>/per_sample/<sample>.motif_freq_region.csv (+ .done markers)

Design notes
------------
- "background" is computed once per sample, genome-wide, pooling all
  fragments regardless of size bin or region. It is the denominator of the
  observed/expected (O/E) ratio of the original motif screen
  (03_motif_stats_region_stratified.py).
- Only integer counts (Counter dicts) are kept per sample, never the full
  fragment records, so memory stays small for large fragment tables.
- Per-sample outputs + .done marker files make the run resumable.
- --source all|direct|both: "direct" counts only fragments whose `source` is
  'direct' (not recovered from concatemers). "both" reads each file once and
  writes <out-dir>/all/ and <out-dir>/direct/, each with its own per_sample/
  folder and long table (the thesis runs used --source both).
- --both-ends (off by default, not used in the thesis): count BOTH ends of
  every fragment. The far end's own 5' motif is revcomp(motif_3p) (motif_3p =
  the last 4 reference bases of the read, in read orientation). Each fragment
  then contributes two motifs, so `count`/`n_fragments` are counts of fragment
  ends (2 per fragment).

  python 02_aggregate_motif_frequencies_region_stratified.py \\
      --tagged-dir fragments \\
      --out-dir motif_region --source both -t 14
"""

import os
import sys
import glob
import argparse
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("motif_region_agg")

LENGTH_COL = "length"
SIZE_BIN_COL = "size_bin"
MOTIF_COL_DEFAULT = "motif_5p"  # 5' end-motif; override with --motif-col if needed
MOTIF_VALID_COL = "motif_valid"  # boolean; invalid (e.g. N-containing) motifs are excluded
REGION_FLAGS = ["is_promoter", "is_gene_body", "is_tss", "is_exon", "is_cpg_island"]
CHUNKSIZE = 500_000

N_WORKERS_DEFAULT = 32
PER_SAMPLE_DIR_NAME = "per_sample"


SOURCE_COL = "source"


def as_bool(s):
    """True/False read from csv may arrive as bool, 'True'/'False' strings or 1/0."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def new_acc():
    return dict(bg_counts=Counter(), bg_total=0, cell_counts={}, cell_totals=Counter(), n_invalid=0)


_RC = str.maketrans("ACGTN", "TGCAN")


def revcomp(m):
    return m.translate(_RC)[::-1] if isinstance(m, str) else m


def both_ends(chunk, motif_col):
    """Stack each fragment twice: once with its 5' motif, once with revcomp(motif_3p) (far end)."""
    far = chunk.copy()
    far[motif_col] = chunk["motif_3p"].map(revcomp)
    return pd.concat([chunk, far], ignore_index=True)


def add_chunk(acc, chunk, motif_col):
    """Add one chunk of fragments to the background and per-cell motif counters."""
    if MOTIF_VALID_COL in chunk.columns:
        before = len(chunk)
        chunk = chunk[as_bool(chunk[MOTIF_VALID_COL])]
        acc["n_invalid"] += before - len(chunk)
    if chunk.empty:
        return
    # background: every (valid-motif) fragment, regardless of bin/region
    acc["bg_counts"].update(chunk[motif_col].value_counts().to_dict())
    acc["bg_total"] += len(chunk)
    for size_bin, sub in chunk.groupby(SIZE_BIN_COL):
        key_genome = (size_bin, "genome_wide")
        acc["cell_counts"].setdefault(key_genome, Counter()).update(sub[motif_col].value_counts().to_dict())
        acc["cell_totals"][key_genome] += len(sub)
        for flag in REGION_FLAGS:
            if flag in sub.columns:
                flagged = sub.loc[as_bool(sub[flag])]
                if len(flagged):
                    key = (size_bin, flag)
                    acc["cell_counts"].setdefault(key, Counter()).update(flagged[motif_col].value_counts().to_dict())
                    acc["cell_totals"][key] += len(flagged)


def process_sample(args):
    """Count one sample's motifs (all and/or direct fragments) and write its per-sample table."""
    csv_path, motif_col, out_dirs, two_ends = args   # out_dirs: {"all"|"direct": per_sample_dir}
    sample = os.path.basename(csv_path).split(".")[0]
    to_run = {}
    for mode, d in out_dirs.items():
        if os.path.exists(os.path.join(d, f"{sample}.done")):
            log.info(f"[{sample}] {mode} already done, skipping")
        else:
            to_run[mode] = d
    outs = {mode: os.path.join(d, f"{sample}.motif_freq_region.csv") for mode, d in out_dirs.items()}
    if not to_run:
        return outs

    accs = {mode: new_acc() for mode in to_run}
    usecols_check = {SIZE_BIN_COL, motif_col, MOTIF_VALID_COL, SOURCE_COL} | set(REGION_FLAGS)
    if two_ends:
        usecols_check.add("motif_3p")
    reader = pd.read_csv(csv_path, usecols=lambda c: c in usecols_check, chunksize=CHUNKSIZE, low_memory=False)

    for chunk in reader:
        if motif_col not in chunk.columns:
            log.error(f"[{sample}] motif column '{motif_col}' not found; "
                      f"available columns: {list(chunk.columns)}")
            sys.exit(1)
        if two_ends:
            if "motif_3p" not in chunk.columns:
                log.error(f"[{sample}] --both-ends needs a 'motif_3p' column")
                sys.exit(1)
            chunk = both_ends(chunk, motif_col)
        if "all" in accs:
            add_chunk(accs["all"], chunk, motif_col)
        if "direct" in accs:
            if SOURCE_COL not in chunk.columns:
                log.error(f"[{sample}] --source direct needs a '{SOURCE_COL}' column")
                sys.exit(1)
            add_chunk(accs["direct"], chunk[chunk[SOURCE_COL].astype(str).str.strip() == "direct"], motif_col)

    for mode, acc in accs.items():
        write_sample(sample, acc, outs[mode], os.path.join(to_run[mode], f"{sample}.done"), mode, motif_col)
    return outs


def write_sample(sample, acc, out_path, done_marker, mode, motif_col):
    """Write the background and bin x region motif rows of one sample, then the .done marker."""
    bg_counts, bg_total = acc["bg_counts"], acc["bg_total"]
    cell_counts, cell_totals = acc["cell_counts"], acc["cell_totals"]
    n_invalid_dropped = acc["n_invalid"]
    rows = []
    # background rows
    for motif, count in bg_counts.items():
        rows.append({
            "sample": sample, "group_type": "background", "bin": "background",
            "region": "genome_wide", "motif": motif,
            "count": count, "n_fragments": bg_total,
            "frequency_pct": 100.0 * count / bg_total if bg_total else 0.0,
        })

    # size_bin x region rows
    for (size_bin, region), counter in cell_counts.items():
        total = cell_totals[(size_bin, region)]
        for motif, count in counter.items():
            rows.append({
                "sample": sample, "group_type": "size_bin", "bin": size_bin,
                "region": region, "motif": motif,
                "count": count, "n_fragments": total,
                "frequency_pct": 100.0 * count / total if total else 0.0,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    with open(done_marker, "w") as f:
        f.write("done\n")

    log.info(f"[{sample}] {mode}: dropped {n_invalid_dropped} fragments with invalid motif "
             f"({motif_col}); wrote {len(out_df)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate end-motif frequencies stratified by size_bin x region."
    )
    parser.add_argument("--tagged-dir", required=True,
                         help="Directory containing {sample}.fragments_with_motifs.csv.gz")
    parser.add_argument("-o", "--out-dir", required=True, help="output folder")
    parser.add_argument("--motif-col", default=MOTIF_COL_DEFAULT,
                         help="Name of the 5' end-motif column in the input files")
    parser.add_argument("-t", "--threads", "--workers", dest="workers", type=int, default=N_WORKERS_DEFAULT,
                        help="samples processed in parallel")
    parser.add_argument("--source", choices=["all", "direct", "both"], default="all",
                        help="all fragments, direct (non-recovered) only, or both in one pass "
                             "(writes <out-dir>/all/ and <out-dir>/direct/)")
    parser.add_argument("--pattern", default="*.fragments_with_motifs.csv.gz",
                        help="file glob inside --tagged-dir")
    parser.add_argument("--both-ends", action="store_true",
                        help="also count revcomp(motif_3p), the far end of each fragment (counts become ends)")
    args = parser.parse_args()

    modes = ["all", "direct"] if args.source == "both" else [args.source]
    mode_dirs = {m: (os.path.join(args.out_dir, m) if args.source == "both" else args.out_dir) for m in modes}
    out_dirs = {}
    for m, d in mode_dirs.items():
        out_dirs[m] = os.path.join(d, PER_SAMPLE_DIR_NAME)
        os.makedirs(out_dirs[m], exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.tagged_dir, args.pattern)))
    if not csv_paths:
        log.error(f"No input files found under {args.tagged_dir}")
        sys.exit(1)

    log.info(f"Found {len(csv_paths)} sample files. Running with {args.workers} workers.")

    tasks = [(p, args.motif_col, out_dirs, args.both_ends) for p in csv_paths]
    out_paths = {m: [] for m in modes}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_sample, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                for m, p in fut.result().items():
                    out_paths[m].append(p)
            except Exception as e:
                log.error(f"FAILED on {src}: {e}")
                raise

    for m in modes:
        log.info(f"Aggregating per-sample outputs ({m})...")
        long_df = pd.concat([pd.read_csv(p) for p in sorted(out_paths[m])], ignore_index=True)
        long_out = os.path.join(mode_dirs[m], "motif_frequencies_region_stratified_long.csv")
        long_df.to_csv(long_out, index=False)
        log.info(f"Wrote master long-format table: {long_out}  ({len(long_df)} rows)")


if __name__ == "__main__":
    main()