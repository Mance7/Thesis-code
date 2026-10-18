#!/usr/bin/env python3
"""
End-motif extraction: the 4-nucleotide 5' end motif of every fragment.

For each region-tagged, size-binned fragment, the first 4 reference bases at
its 5' end are read from the GRCh38 FASTA (Jiang et al. 2020, Jin et al.
2021, Guo et al. 2022). Only this 5' motif (column motif_5p) is used in the
thesis analyses (Methods 2.7). The 4 bases at the 3' end (motif_3p) are
written to the same table but were not analysed.

The 5' end depends on strand: for a "-" strand fragment the biological 5'
end is the genomic END coordinate, reverse-complemented (the read is the
reverse complement of the + strand reference there):
  + strand: 5' motif = reference[start : start+4]
  - strand: 5' motif = revcomp(reference[end-4 : end])
The 3' end is the mirror case:
  + strand: 3' motif = reference[end-4 : end]
  - strand: 3' motif = revcomp(reference[start : start+4])

Fragments shorter than 4 bp, or whose motif contains an ambiguous base (N),
are flagged motif_valid = False (kept in the table, so the number excluded
can be checked) and are left out of all downstream motif counting.

Input : <tagged-dir>/<sample>.fragments_region_tagged.csv.gz (04_size_bins_regions/02_tag_fragments_with_regions.py)
        reference FASTA indexed with `samtools faidx` (the .fai file is needed for random access)
Output: <out-dir>/<sample>.fragments_with_motifs.csv.gz (adds motif_5p, motif_3p, motif_valid)
        <out-dir>/motif_extraction_summary.csv
Samples run in parallel (--threads). For PacBio, point --tagged-dir and --out-dir
to the PacBio folders; the reference is the same.

  python 01_extract_end_motifs.py --tagged-dir size_bins/region_tagged \\
      --reference reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna -t 14
"""

from pathlib import Path
import argparse
import pysam
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
PATTERN = "*.fragments_region_tagged.csv.gz"

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

# ---------------------------------------------------------------------------


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def get_tagged_files(tagged_dir, pattern):
    return sorted(Path(tagged_dir).glob(pattern))


def extract_motifs(df, fasta):
    """Add motif_5p, motif_3p and motif_valid columns, one fragment at a time."""
    motifs_5p = []
    motifs_3p = []
    valid = []

    for chrom, start, end, strand in zip(df["chrom"], df["start"], df["end"], df["strand"]):
        length = end - start
        if length < 4:
            motifs_5p.append(None)
            motifs_3p.append(None)
            valid.append(False)
            continue

        try:
            if strand == "+":
                m5 = fasta.fetch(chrom, start, start + 4).upper()
                m3 = fasta.fetch(chrom, end - 4, end).upper()
            else:
                m5 = revcomp(fasta.fetch(chrom, end - 4, end).upper())
                m3 = revcomp(fasta.fetch(chrom, start, start + 4).upper())
        except (KeyError, ValueError):
            # chrom not in fasta, or coordinates out of range
            motifs_5p.append(None)
            motifs_3p.append(None)
            valid.append(False)
            continue

        is_valid = (
            m5 is not None and m3 is not None
            and len(m5) == 4 and len(m3) == 4
            and "N" not in m5 and "N" not in m3
        )
        motifs_5p.append(m5)
        motifs_3p.append(m3)
        valid.append(is_valid)

    df["motif_5p"] = motifs_5p
    df["motif_3p"] = motifs_3p
    df["motif_valid"] = valid
    return df


def process_sample(f, reference_fasta, out_dir):
    """Process a single sample file: load, extract motifs, write output, return summary."""
    sample = f.name.split(".")[0]

    # Each worker opens its own FASTA handle
    fasta = pysam.FastaFile(str(reference_fasta))

    try:
        df = pd.read_csv(f, compression="gzip")
        df = extract_motifs(df, fasta)

        n_total = len(df)
        n_valid = int(df["motif_valid"].sum())

        out_path = out_dir / f"{sample}.fragments_with_motifs.csv.gz"
        df.to_csv(out_path, index=False, compression="gzip")

        return {
            "sample": sample,
            "n_fragments": n_total,
            "n_valid_motif": n_valid,
            "pct_valid_motif": round(100 * n_valid / n_total, 2) if n_total else np.nan,
            "output": str(out_path)
        }
    finally:
        fasta.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagged-dir", required=True,
                    help="folder with the region-tagged fragment tables (csv.gz)")
    ap.add_argument("--pattern", default=PATTERN, help="file name pattern inside --tagged-dir")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output folder (default: <tagged-dir>/../end_motifs)")
    ap.add_argument("--reference", required=True,
                    help="GRCh38 reference FASTA, indexed with samtools faidx")
    ap.add_argument("-t", "--threads", type=int, default=max(1, int(os.cpu_count() * 0.6)),
                    help="samples processed in parallel")
    a = ap.parse_args()
    REFERENCE = Path(a.reference)
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.tagged_dir).parent / "end_motifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    N_WORKERS = max(1, a.threads)
    files = get_tagged_files(a.tagged_dir, a.pattern)
    print(f"Found {len(files)} region-tagged fragment tables.\n")

    fai_path = Path(str(REFERENCE) + ".fai")
    if not fai_path.exists():
        print(f"[!] No .fai index found for {REFERENCE}.")
        print(f"    Index it first: samtools faidx {REFERENCE}")
        return

    print(f"Processing {len(files)} samples in parallel with {N_WORKERS} workers...\n")

    summary_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit all jobs
        futures = {
            executor.submit(process_sample, f, REFERENCE, out_dir): f
            for f in files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            f = futures[future]
            try:
                result = future.result()
                summary_rows.append(result)
                print(f"  {result['sample']}: {result['n_valid_motif']}/{result['n_fragments']} "
                      f"fragments with a valid 5'/3' motif -> {result['output']}")
            except Exception as exc:
                print(f"  {f.name} generated an exception: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "motif_extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")
    print(summary_df[["sample", "n_fragments", "n_valid_motif", "pct_valid_motif"]].to_string(index=False))


if __name__ == "__main__":
    main()
