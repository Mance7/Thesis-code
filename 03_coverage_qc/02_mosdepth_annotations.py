#!/usr/bin/env python3
"""
Runs mosdepth on every aligned BAM once per annotation BED (promoters, gene
bodies, the TSS variants, exons, CpG islands), so the mean depth inside each
region type can be compared with the genome-wide background
(05_assess_wgs_vs_targeted_mosdepth.py).

mosdepth 0.3.14, --fast-mode, no MAPQ cutoff (same settings as
01_mosdepth_global.py).

Input : RESULTS/{sample}/Alignment/*.bam (+ .bam.bai)
        cleaned annotation BEDs (chrM removed) in RESOURCES/cleaned_no_chrM/
Output: OUT/{sample}/alignment_qc/{sample}.{annotation}.* and a .log per call
"""

import argparse
import os
import subprocess
from pathlib import Path

# annotation name -> BED path relative to --resources-dir
ANNOTATIONS = {
    "promoters": Path("promoters") / "promoters_gene_level.bed",
    "gene_bodies": Path("gene_bodies") / "gene_bodies_gene_level.bed",
    "tss_gene": Path("tss") / "tss_gene_level.bed",
    "tss_transcripts": Path("tss") / "tss_transcripts.bed",
    "tss_dedup": Path("tss") / "tss_deduplicated.bed",
    "tss_collapsed": Path("tss") / "tss_collapsed.bed",
    "exons_all": Path("exons") / "exons_all.bed",
    "exons_merged": Path("exons") / "exons_gene_merged.bed",
    "cpg_islands": Path("cpg_islands") / "cpg_islands.bed",
}

THRESHOLDS = "1,2,3,4,5,6,7,8,9,10"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="folder with one sub-folder per sample holding Alignment/*.bam")
    p.add_argument("--bam-glob", default="*/Alignment/*.bam",
                   help="glob (relative to --results-dir) that finds the BAM files")
    p.add_argument("--resources-dir", type=Path, required=True,
                   help="folder with the cleaned (no chrM) annotation BED files")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output root; results go to OUT/{sample}/alignment_qc/")
    p.add_argument("-t", "--threads", type=int,
                   default=max(1, int((os.cpu_count() or 1) * 0.90)),
                   help="threads per mosdepth call (default: 90%% of the cores)")
    return p.parse_args()


def run_command(cmd, log_file):
    """Run one command, writing its stdout/stderr to a log file."""
    print("Running:")
    print(" ".join(map(str, cmd)))
    with open(log_file, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True)


def find_samples(results_dir, bam_glob, out_root):
    bam_files = sorted(results_dir.glob(bam_glob))
    if len(bam_files) == 0:
        raise RuntimeError(f"No BAM files found with {results_dir}/{bam_glob}")
    print(f"\nFound {len(bam_files)} BAM files.\n")

    samples = []
    for bam in bam_files:
        sample = bam.parent.parent.name
        bai = bam.with_suffix(".bam.bai")
        if not bai.exists():
            raise FileNotFoundError(f"Missing BAM index:\n{bai}")
        qc_dir = out_root / sample / "alignment_qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        samples.append({"sample": sample, "bam": bam, "qc_dir": qc_dir})
    return samples


def run_annotation_mosdepth(sample, annotation_name, bed, threads):
    name = sample["sample"]
    qc_dir = sample["qc_dir"]
    prefix = qc_dir / f"{name}.{annotation_name}"
    run_command(
        ["mosdepth", "-t", str(threads), "--fast-mode",
         "--by", str(bed), "--thresholds", THRESHOLDS,
         str(prefix), str(sample["bam"])],
        qc_dir / f"{name}.{annotation_name}.mosdepth.log")


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Available CPU cores : {os.cpu_count()}")
    print(f"Using {args.threads} threads")
    print("=" * 60)

    samples = find_samples(args.results_dir, args.bam_glob, args.out)

    for sample in samples:
        print("\n" + "=" * 70)
        print(sample["sample"])
        print("=" * 70)

        for annotation, rel_bed in ANNOTATIONS.items():
            print(f"  {annotation}")
            bed = args.resources_dir / rel_bed
            if not bed.exists():
                raise FileNotFoundError(bed)
            run_annotation_mosdepth(sample, annotation, bed, args.threads)

    print("\nFinished annotation mosdepth.")


if __name__ == "__main__":
    main()
