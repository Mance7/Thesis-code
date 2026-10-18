#!/usr/bin/env python3
"""
Validate the annotation BED files against the real chromosome lengths,
drop mitochondrial intervals and clip intervals to chromosome ends.

Mitochondrial genes are excluded rather than clipped: mtDNA has no
chromatin/nucleosome structure and no canonical promoter architecture,
so promoter, TSS and gene-body annotations for MT genes are not
meaningful for fragment and coverage analysis, and the very high copy
number of mtDNA would distort depth summaries.

For every BED file in ANNOTATIONS:
  - drops all chrM / MT / M rows (reported separately)
  - clips any remaining interval whose start < 0 or end > chromosome length
  - drops intervals that become zero/negative length after clipping
  - drops intervals on chromosomes not present in the chrom sizes file
    (unless --keep-unknown-chroms is given)
  - writes a cleaned copy (chrom, start, end) to --out, keeping the
    sub-folder layout, and leaves the original untouched
  - prints a summary of everything it changed or removed

Input:  BED resources from 01_making_bed_resources.py / 03 / 04 and
        GRCh38.chrom.sizes from 02_build_reference_resources.py
Output: cleaned BED files (no chrM, in bounds) in --out
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and options
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--chrom-sizes",
                    required=True,
                    help="chromosome sizes file, two columns: chrom<TAB>length")
parser.add_argument("--resources", required=True,
                    help="folder with the (sorted) annotation BED files")
parser.add_argument("-o", "--out", required=True,
                    help="output folder for the cleaned BED files")
parser.add_argument("--keep-unknown-chroms", action="store_true",
                    help="keep rows on chromosomes missing from the chrom sizes file (unchecked)")
args = parser.parse_args()

CHROM_SIZES_FILE = Path(args.chrom_sizes)

RESOURCES = Path(args.resources)

ANNOTATIONS = {
    "promoters": RESOURCES / "promoters" / "promoters_gene_level.bed",
    "gene_bodies": RESOURCES / "gene_bodies" / "gene_bodies_gene_level.bed",
    "tss_gene": RESOURCES / "tss" / "tss_gene_level.bed",
    "tss_transcripts": RESOURCES / "tss" / "tss_transcripts.bed",
    "tss_dedup": RESOURCES / "tss" / "tss_deduplicated.bed",
    "tss_collapsed": RESOURCES / "tss" / "tss_collapsed.bed",
    "exons_all": RESOURCES / "exons" / "exons_all.bed",
    "exons_merged": RESOURCES / "exons" / "exons_gene_merged.bed",
    "cpg_islands": RESOURCES / "cpg_islands" / "cpg_islands.bed",
}

# Cleaned copies are written here; originals are kept.
OUTPUT_DIR = Path(args.out)

# Any chrom name spelling treated as mitochondrial and dropped.
MITO_CHROM_NAMES = {"chrM", "chrMT", "MT", "M"}

# If True, rows whose chrom is not in the chrom sizes file are dropped entirely.
# If False, those rows are passed through unchanged (unchecked).
DROP_UNKNOWN_CHROMS = not args.keep_unknown_chroms

# ---------------------------------------------------------------------------


def load_chrom_sizes(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "length"], comment="#")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    bad = df[df["length"].isna()]
    if len(bad):
        raise ValueError(
            f"Non-numeric length value(s) in {path}:\n{bad}\n"
            "Check for a header row or stray text in this file."
        )
    df["length"] = df["length"].astype(int)
    return dict(zip(df["chrom"], df["length"]))


# Feature-type values that mark a file as GTF/GFF-style rather than plain BED.
GTF_FEATURE_TYPES = {
    "gene", "transcript", "exon", "CDS", "UTR", "five_prime_UTR",
    "three_prime_UTR", "start_codon", "stop_codon", "Selenocysteine",
}


def detect_format(df):
    """
    Returns "bed" if this looks like a standard BED file
    (col 1 / col 2 are already numeric start/end, 0-based).

    Returns "gtf" if this looks like GTF/GFF (>=9 cols, col index 2
    holds a recognizable feature type) -> start/end live in cols 3/4
    (0-indexed), 1-based, need -1 on start to become BED-style 0-based.
    """
    if df.shape[1] >= 9:
        col2_values = set(df[2].astype(str).unique())
        if col2_values & GTF_FEATURE_TYPES:
            return "gtf"
    return "bed"


def clean_bed(bed_path, chrom_sizes, out_path):
    raw = pd.read_csv(bed_path, sep="\t", header=None, comment="#")

    fmt = detect_format(raw)

    if fmt == "gtf":
        print(f"  [i] detected GTF/GFF-style file (col2 feature types found) "
              f"-> using cols 4/5 as 1-based start/end, converting to 0-based")
        df = pd.DataFrame({
            0: raw[0],                                      # chrom
            1: pd.to_numeric(raw[3], errors="coerce") - 1,   # start, GTF 1-based -> BED 0-based
            2: pd.to_numeric(raw[4], errors="coerce"),       # end (already correct as BED end)
        })
    else:
        df = pd.DataFrame({
            0: raw[0],
            1: pd.to_numeric(raw[1], errors="coerce"),
            2: pd.to_numeric(raw[2], errors="coerce"),
        })

    bad_rows = df[df[1].isna() | df[2].isna()]
    if len(bad_rows):
        print(f"  [!] {len(bad_rows)} row(s) with non-numeric start/end dropped "
              f"(likely a header row or malformed line):")
        print(bad_rows.head(10).to_string())
        if len(bad_rows) > 10:
            print(f"      ... and {len(bad_rows) - 10} more")
        df = df.dropna(subset=[1, 2]).copy()

    df[1] = df[1].astype(int)
    df[2] = df[2].astype(int)

    n_total = len(df)
    report = {
        "file": bed_path.name,
        "total_rows": n_total,
        "mito_dropped": 0,
        "unknown_chrom_dropped": 0,
        "negative_start_clipped": 0,
        "end_exceeds_length_clipped": 0,
        "zero_or_negative_after_clip_dropped": 0,
        "rows_written": 0,
    }

    # drop mitochondrial rows entirely
    mito_mask = df[0].isin(MITO_CHROM_NAMES)
    n_mito = int(mito_mask.sum())
    if n_mito:
        print(f"  [i] dropping {n_mito} mitochondrial row(s) "
              f"({sorted(df.loc[mito_mask, 0].unique())})")
        report["mito_dropped"] = n_mito
        df = df.loc[~mito_mask].copy()

    known_mask = df[0].isin(chrom_sizes.keys())
    n_unknown = (~known_mask).sum()

    if n_unknown:
        offenders = sorted(df.loc[~known_mask, 0].unique())
        print(f"  [!] {n_unknown} row(s) on chrom(s) not in chrom sizes file: {offenders}")
        if DROP_UNKNOWN_CHROMS:
            df = df.loc[known_mask].copy()
            report["unknown_chrom_dropped"] = int(n_unknown)
        else:
            print("      (kept, unchecked, --keep-unknown-chroms)")

    lengths = df[0].map(chrom_sizes)

    # clip negative starts
    neg_start_mask = df[1] < 0
    report["negative_start_clipped"] = int(neg_start_mask.sum())
    df.loc[neg_start_mask, 1] = 0

    # clip ends exceeding chrom length (only where we know the length)
    known_here = df[0].isin(chrom_sizes.keys())
    exceeds_mask = known_here & (df[2] > lengths)
    if exceeds_mask.any():
        exceed_rows = df.loc[exceeds_mask]
        for chrom in sorted(exceed_rows[0].unique()):
            sub = exceed_rows[exceed_rows[0] == chrom]
            print(
                f"  [!] {len(sub)} row(s) on {chrom} exceed length "
                f"{chrom_sizes[chrom]} (max end found: {sub[2].max()}) -> clipped"
            )
        report["end_exceeds_length_clipped"] = int(exceeds_mask.sum())
    df.loc[exceeds_mask, 2] = lengths[exceeds_mask]

    # drop anything that's now zero/negative length
    bad_len_mask = df[2] <= df[1]
    report["zero_or_negative_after_clip_dropped"] = int(bad_len_mask.sum())
    if bad_len_mask.any():
        print(f"  [!] {bad_len_mask.sum()} row(s) dropped (zero/negative length after clipping)")
    df = df.loc[~bad_len_mask].copy()

    report["rows_written"] = len(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", header=False, index=False)

    return report


def main():
    chrom_sizes = load_chrom_sizes(CHROM_SIZES_FILE)
    print(f"Loaded {len(chrom_sizes)} chromosome sizes from {CHROM_SIZES_FILE}\n")

    reports = []

    for annotation_name, bed_path in ANNOTATIONS.items():
        print("=" * 70)
        print(f"{annotation_name}: {bed_path}")
        print("=" * 70)

        if not bed_path.exists():
            raise FileNotFoundError(bed_path)

        out_path = OUTPUT_DIR / bed_path.relative_to(RESOURCES)
        report = clean_bed(bed_path, chrom_sizes, out_path)
        reports.append(report)

        print(f"  -> wrote cleaned bed (no chrM, in-bounds): {out_path}\n")

    # summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(reports)
    print(summary_df.to_string(index=False))

    total_issues = summary_df[
        [
            "mito_dropped",
            "unknown_chrom_dropped",
            "negative_start_clipped",
            "end_exceeds_length_clipped",
            "zero_or_negative_after_clip_dropped",
        ]
    ].sum().sum()

    if total_issues == 0:
        print("\nAll BED files were already clean (no chrM, all within chromosome bounds).")
    else:
        print(f"\n{int(total_issues)} total row-level issue(s) found and handled.")
        print(f"Cleaned BED files are in: {OUTPUT_DIR}")
        print("Originals were left untouched.")


if __name__ == "__main__":
    main()
