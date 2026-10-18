#!/usr/bin/env python3
"""
Trim the ONT ligation-kit adapters from the raw ONT FASTQ files with Cutadapt 5.2,
following the adapter step of Yu et al. (2023).

Command run per sample:
  cutadapt -g AATGTACTTCGTTCAGTTACGTATTGCT...AGCAATACGTAACTGAACGAAGT \\
           -e 0.3 --discard-untrimmed -o {out}.trimmed.fastq.gz {in}.fq.gz

-g "adapter1...adapter2" is Cutadapt's linked-adapter syntax (ONT LSK109
adapters at the 5' and 3' end of the read). -e 0.3 allows a 30% error rate
in the adapter match, to tolerate ONT sequencing errors. --discard-untrimmed
keeps only reads in which both linked adapters were found and removed.

Input:  raw ONT FASTQ files, one folder per sample: RAW_DIR/<sample>/<sample>.fq.gz
Output: OUT/<sample>/<sample>.trimmed.fastq.gz, OUT/<sample>/<sample>.cutadapt.log
        and OUT/cutadapt_summary.csv (reads in, reads kept, % kept per sample)
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
ADAPTER_5P = "AATGTACTTCGTTCAGTTACGTATTGCT"
ADAPTER_3P = "AGCAATACGTAACTGAACGAAGT"
LINKED_ADAPTER = f"{ADAPTER_5P}...{ADAPTER_3P}"

# set from the command line in main()
RAW_DIR = None
OUT_DIR = None
ERROR_RATE = 0.3
N_WORKERS = max(1, int(os.cpu_count() * 0.5))

# ---------------------------------------------------------------------------


def get_raw_fastqs():
    return sorted(RAW_DIR.glob("*/*.fq.gz")) or sorted(RAW_DIR.glob("*/*.fastq.gz"))


def run_cutadapt(fastq_path):
    sample = fastq_path.name.split(".")[0]
    sample_out_dir = OUT_DIR / sample
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    trimmed_path = sample_out_dir / f"{sample}.trimmed.fastq.gz"
    log_path = sample_out_dir / f"{sample}.cutadapt.log"

    cmd = [
        "cutadapt",
        "-g", LINKED_ADAPTER,
        "-e", str(ERROR_RATE),
        "--discard-untrimmed",
        "-o", str(trimmed_path),
        str(fastq_path),
    ]

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        return {"sample": sample, "status": "FAILED", "trimmed_path": None,
                "n_input": None, "n_output": None, "pct_retained": None}

    # read Cutadapt's own summary from the log
    n_input, n_output = None, None
    with open(log_path) as f:
        for line in f:
            if "Total reads processed:" in line:
                n_input = int(line.split(":")[1].strip().replace(",", ""))
            if "Reads written (passing filters):" in line:
                # format: "Reads written (passing filters):   1,234,567 (45.6%)"
                n_output = int(line.split(":")[1].strip().split()[0].replace(",", ""))

    pct_retained = round(100 * n_output / n_input, 2) if (n_input and n_output) else None

    return {
        "sample": sample,
        "status": "OK",
        "trimmed_path": str(trimmed_path),
        "n_input": n_input,
        "n_output": n_output,
        "pct_retained": pct_retained,
    }


def main():
    global RAW_DIR, OUT_DIR, ERROR_RATE, N_WORKERS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True,
                        help="folder with one sub-folder per sample holding the raw ONT reads (fq.gz)")
    parser.add_argument("-o", "--out", required=True,
                        help="output folder for the trimmed reads, logs and summary")
    parser.add_argument("-e", "--error-rate", type=float, default=0.3,
                        help="maximum error rate in the adapter match (cutadapt -e)")
    parser.add_argument("-t", "--threads", type=int, default=N_WORKERS,
                        help="number of samples trimmed in parallel")
    args = parser.parse_args()

    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_RATE = args.error_rate
    N_WORKERS = max(1, args.threads)

    fastqs = get_raw_fastqs()
    print(f"Found {len(fastqs)} raw fastq files. Running Cutadapt with {N_WORKERS} workers.\n")
    print(f"Linked adapter: {LINKED_ADAPTER}")
    print(f"Error rate: {ERROR_RATE}, discard-untrimmed: True\n")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_cutadapt, f): f for f in fastqs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  {result['sample']}: {result['status']} -- "
                  f"{result['n_output']}/{result['n_input']} reads retained "
                  f"({result['pct_retained']}%)")

    summary_df = pd.DataFrame(results)
    summary_path = OUT_DIR / "cutadapt_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")
    print(summary_df.to_string(index=False))
    print("\npct_retained is the share of reads in which both adapters were found. "
          "Next step: 03_align_trimmed.py.")


if __name__ == "__main__":
    main()
