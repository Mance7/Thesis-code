#!/usr/bin/env python3
"""
Align the Cutadapt-trimmed ONT reads (from 02_cutadapt_trim.py) to GRCh38
(GCA_000001405.15 no-alt analysis set) with minimap2 2.31, then sort and
index with samtools 1.23.1.

Command run per sample:
  minimap2 -ax map-ont -t T REF.mmi SAMPLE.trimmed.fastq.gz | samtools sort -@ T -o SAMPLE.cutadapt_trimmed.sorted.bam
  samtools index -@ T SAMPLE.cutadapt_trimmed.sorted.bam

Input:  TRIM_DIR/<sample>/<sample>.trimmed.fastq.gz and a minimap2 index of GRCh38
Output: TRIM_DIR/<sample>/<sample>.cutadapt_trimmed.sorted.bam (+ .bai), next to
        the trimmed reads. Samples whose BAM already exists are skipped.
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# set from the command line in main()
CUTADAPT_DIR = None
REFERENCE = None
THREADS_PER_JOB = 8   # threads per minimap2 / samtools call
N_WORKERS = 4         # samples aligned in parallel (total cores = THREADS_PER_JOB x N_WORKERS)

# ---------------------------------------------------------------------------


def get_trimmed_fastqs():
    return sorted(CUTADAPT_DIR.glob("*/*.trimmed.fastq.gz"))


def align_sample(fastq_path):
    sample = fastq_path.name.split(".")[0]
    sample_dir = fastq_path.parent
    bam_path = sample_dir / f"{sample}.cutadapt_trimmed.sorted.bam"

    if bam_path.exists():
        return {"sample": sample, "status": "SKIPPED (already exists)", "bam_path": str(bam_path)}

    command = (
        f"minimap2 -ax map-ont -t {THREADS_PER_JOB} "
        f"'{REFERENCE}' '{fastq_path}' "
        f"| samtools sort -@ {THREADS_PER_JOB} -o '{bam_path}'"
    )

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        return {"sample": sample, "status": "FAILED", "bam_path": None}

    subprocess.run(f"samtools index -@ {THREADS_PER_JOB} '{bam_path}'", shell=True)

    return {"sample": sample, "status": "OK", "bam_path": str(bam_path)}


def main():
    global CUTADAPT_DIR, REFERENCE, THREADS_PER_JOB, N_WORKERS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trimmed-dir", required=True,
                        help="folder with one sub-folder per sample holding the trimmed reads (trimmed.fastq.gz); "
                             "BAMs are written next to them")
    parser.add_argument("--reference", required=True,
                        help="minimap2 index (mmi) of the GRCh38 no-alt analysis set")
    parser.add_argument("-t", "--threads", type=int, default=THREADS_PER_JOB,
                        help="threads per minimap2 / samtools call")
    parser.add_argument("-j", "--jobs", type=int, default=N_WORKERS,
                        help="number of samples aligned in parallel")
    args = parser.parse_args()

    CUTADAPT_DIR = Path(args.trimmed_dir)
    REFERENCE = Path(args.reference)
    THREADS_PER_JOB = args.threads
    N_WORKERS = max(1, args.jobs)

    fastqs = get_trimmed_fastqs()
    print(f"Found {len(fastqs)} Cutadapt-trimmed fastq files. "
          f"Aligning with {N_WORKERS} parallel jobs x {THREADS_PER_JOB} threads each.\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(align_sample, f): f for f in fastqs}
        for future in as_completed(futures):
            result = future.result()
            print(f"  {result['sample']}: {result['status']} -> {result['bam_path']}")

    print("\nDone. Next step: 06_stage1_read_filtering.py on these BAMs.")


if __name__ == "__main__":
    main()
