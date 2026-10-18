#!/usr/bin/env python3
"""
Runs mosdepth on every aligned BAM to get the genome-wide coverage QC:
  - mean depth per 10 kb and per 100 kb window (+ % of each window covered
    at 1x..10x)
  - whole-genome / per-chromosome summary (no windows)

mosdepth 0.3.14 is run in --fast-mode with no MAPQ cutoff, i.e. on all
primary alignments (this is the "original" coverage output; the MAPQ >= 20
re-run is 03_rerun_mosdepth_with_mapq.sh).

Input : RESULTS/{sample}/Alignment/*.bam (+ .bam.bai)
        10 kb and 100 kb genome window BED files
Output: OUT/{sample}/alignment_qc/{sample}.10kb.*, {sample}.100kb.*,
        {sample}.mosdepth.* and one .log file per mosdepth call
"""

import argparse
import os
import subprocess
from pathlib import Path

# % of each window covered at >= 1x, 2x, ... 10x
THRESHOLDS = "1,2,3,4,5,6,7,8,9,10"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="folder with one sub-folder per sample holding Alignment/*.bam")
    p.add_argument("--bam-glob", default="*/Alignment/*.bam",
                   help="glob (relative to --results-dir) that finds the BAM files")
    p.add_argument("--window10", type=Path, required=True,
                   help="BED of 10 kb genome windows")
    p.add_argument("--window100", type=Path, required=True,
                   help="BED of 100 kb genome windows")
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
    """One entry per BAM: sample name = name of the folder above Alignment/."""
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


def run_mosdepth(sample, threads, window10, window100):
    name = sample["sample"]
    bam = sample["bam"]
    qc_dir = sample["qc_dir"]

    print("\n" + "=" * 60)
    print(f"Mosdepth: {name}")
    print("=" * 60)

    out_prefix = qc_dir / name

    # 10 kb windows
    run_command(
        ["mosdepth", "-t", str(threads), "--fast-mode",
         "--by", str(window10), "--thresholds", THRESHOLDS,
         str(out_prefix) + ".10kb", str(bam)],
        qc_dir / f"{name}.10kb.mosdepth.log")

    # 100 kb windows
    run_command(
        ["mosdepth", "-t", str(threads), "--fast-mode",
         "--by", str(window100), "--thresholds", THRESHOLDS,
         str(out_prefix) + ".100kb", str(bam)],
        qc_dir / f"{name}.100kb.mosdepth.log")

    # chromosome / whole-genome summary
    run_command(
        ["mosdepth", "-t", str(threads), "--fast-mode",
         str(out_prefix), str(bam)],
        qc_dir / f"{name}.mosdepth.log")

    print(f"Finished mosdepth for {name}")


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Available CPU cores : {os.cpu_count()}")
    print(f"Using {args.threads} threads")
    print("=" * 60)

    # window BEDs must exist before we start
    for f in [args.window10, args.window100]:
        if not f.exists():
            raise FileNotFoundError(f)

    samples = find_samples(args.results_dir, args.bam_glob, args.out)
    for i, sample in enumerate(samples, start=1):
        print(f"{i}/{len(samples)} : {sample['sample']}")
        run_mosdepth(sample, args.threads, args.window10, args.window100)

    print("\nAll mosdepth analyses completed.")


if __name__ == "__main__":
    main()
