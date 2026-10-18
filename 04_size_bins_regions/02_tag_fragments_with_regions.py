#!/usr/bin/env python3
"""
Tag every size-binned fragment with the genomic regions it overlaps.

Each sample's binned fragments are intersected with the cleaned annotation
BED files (chrM removed, clipped to chromosome ends; Methods 2.5) and one
boolean column is added per region: is_promoter, is_gene_body, is_tss,
is_exon and is_cpg_island. A fragment gets a flag if it overlaps the region
by at least 1 bp (bedtools intersect -u).

The flags are non-exclusive, as in Esfahani et al. 2022 (EPIC-seq), which
computed fragmentation entropy separately per region type: a fragment that
overlaps both a promoter and a CpG island is True for both. "Genome-wide"
is simply the full fragment set. ONT and PacBio use the same BED files.

Input : <binned-dir>/<sample>.fragments_binned.csv.gz (01_assign_fragments_to_bins.py)
        region BED files under --resources-dir (see REGIONS below)
Output: <out-dir>/<sample>.fragments_region_tagged.csv.gz
        <out-dir>/region_overlap_summary.csv (percent of fragments per region)
Temporary BED files are written to <out-dir>/_tmp_beds and removed afterwards.

Requires bedtools (2.31.1) on PATH.
  python 02_tag_fragments_with_regions.py --binned-dir STAGE1/size_bins \\
      --resources-dir resources/cleaned_no_chrM -t 14
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
# One canonical BED per region category (the redundant TSS variants
# tss_gene / tss_transcripts / tss_dedup and the un-merged exons_all are not
# used). Paths are relative to --resources-dir.
REGIONS = {
    "promoter": Path("promoters") / "promoters_gene_level.bed",
    "gene_body": Path("gene_bodies") / "gene_bodies_gene_level.bed",
    "tss": Path("tss") / "tss_collapsed.bed",
    "exon": Path("exons") / "exons_gene_merged.bed",
    "cpg_island": Path("cpg_islands") / "cpg_islands.bed",
}

# ---------------------------------------------------------------------------


def tag_sample(args):
    """Add the is_<region> columns to one sample's fragment table."""
    f, out_dir, tmp_dir, regions = args
    sample = f.name.split(".")[0]
    df = pd.read_csv(f, compression="gzip")
    df["fragment_id"] = df.index.astype(str)

    # fragments as a BED file, with the row number as the name column
    frag_bed_path = tmp_dir / f"{sample}.fragments.bed"
    df[["chrom", "start", "end", "fragment_id"]].to_csv(
        frag_bed_path, sep="\t", header=False, index=False
    )

    msgs = []
    for region_name, region_bed in regions.items():
        if not region_bed.exists():
            msgs.append(f"  [!] missing region bed: {region_bed} -- skipping {region_name} "
                        f"(check --resources-dir)")
            df[f"is_{region_name}"] = False
            continue

        # -u: report each fragment once if it overlaps the region by >= 1 bp
        result = subprocess.run(
            ["bedtools", "intersect", "-a", str(frag_bed_path), "-b", str(region_bed), "-u"],
            capture_output=True, text=True, check=True,
        )
        overlapping_ids = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            overlapping_ids.add(line.split("\t")[3])  # fragment_id column

        df[f"is_{region_name}"] = df["fragment_id"].isin(overlapping_ids)

    df = df.drop(columns=["fragment_id"])

    out_path = out_dir / f"{sample}.fragments_region_tagged.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")

    frag_bed_path.unlink()

    n_total = len(df)
    row = {"sample": sample, "n_fragments": n_total}
    for region_name in regions:
        n_overlap = int(df[f"is_{region_name}"].sum())
        row[f"pct_{region_name}"] = round(100 * n_overlap / n_total, 2) if n_total else 0
    msgs.append(f"  {sample}: tagged {n_total} fragments -> {out_path}")
    return row, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binned-dir", required=True,
                    help="folder with the size-binned fragment tables (csv.gz)")
    ap.add_argument("--pattern", default="*.fragments_binned.csv.gz",
                    help="file name pattern inside --binned-dir")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output folder (default: <binned-dir>/region_tagged)")
    ap.add_argument("--resources-dir", required=True,
                    help="folder with the cleaned region BED files (chrM removed)")
    ap.add_argument("-t", "--threads", type=int, default=max(1, int(os.cpu_count() * 0.5)),
                    help="samples processed in parallel")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else Path(a.binned_dir) / "region_tagged"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_beds"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    regions = {k: Path(a.resources_dir) / v for k, v in REGIONS.items()}
    if shutil.which("bedtools") is None:
        raise SystemExit("bedtools not on PATH (activate the conda env that has it)")

    files = sorted(Path(a.binned_dir).glob(a.pattern))
    print(f"Found {len(files)} binned fragment tables.\n")

    summary_rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(a.threads, len(files) or 1))) as ex:
        for row, msgs in ex.map(tag_sample, [(f, out_dir, tmp_dir, regions) for f in files]):
            print("\n".join(msgs))
            summary_rows.append(row)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "region_overlap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nWrote summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
