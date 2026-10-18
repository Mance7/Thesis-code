#!/usr/bin/env python3
"""
Assign every eligible fragment to one of the eleven size bins.

Each fragment from the Stage 1 eligible-fragment tables is put into exactly
one primary size bin (Methods 2.4):
  <90, 90-125, 126-135, 136-150, 151-255, 256-420, 421-650, 651-1000,
  1001-2000, 2001-5000 and >5000 bp.
The bins are contiguous and do not overlap. The 151-255 / 256-420 edge uses
this cohort's own valley in the length distribution (mean 254 bp across the
14 ONT samples) rather than a value taken from the literature. ONT and
PacBio use the same bin edges.

The two Le et al. 2022 windows (126-135 and 240-324 bp) are reported on the
screen as a separate, overlapping comparison; they are not bins.

Input : <sample>.eligible_fragments.csv.gz (Stage 1 output; comma or tab
        separated, delimiter detected automatically)
Output: <out-dir>/<sample>.<out-suffix>.csv.gz  (same table plus a size_bin column)
        <out-dir>/all_samples_bin_counts.csv    (fragments per sample and bin,
                                                 split into direct / recovered)

Genome-wide run (the input of 02_tag_fragments_with_regions.py):
  python 01_assign_fragments_to_bins.py --in-dir STAGE1 --out-dir STAGE1/size_bins -t 14
The same script also bins the promoter-tagged tables used for the
individual-promoter tests (--pattern "*.promoter_tagged_fragments_with_tss.csv.gz"
--out-suffix "fragments(promoters)_binned").
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
PATTERN = "*.eligible_fragments.csv.gz"
OUT_SUFFIX = "fragments_binned"

# Primary structural bins: (lower_inclusive, upper_inclusive, label, citation)
BIN_DEFINITIONS = [
    (0,      89,   "<90bp",          "exploratory"),
    (90,     125,  "90-125bp",       "Mouliere 2019; Underhill 2018"),
    (126,    135,  "126-135bp",      "Le et al. 2022 window 1"),
    (136,    150,  "136-150bp",      "Mouliere 2019; Underhill 2018"),
    (151,    255,  "151-255bp",      "mono-nucleosome (empirical valley, this cohort)"),
    (256,    420,  "256-420bp",      "di-nucleosome (empirical lower edge; Snyder 2016 upper)"),
    (421,    650,  "421-650bp",      "tri-nucleosome (Snyder 2016, lower-confidence)"),
    (651,    1000, "651-1000bp",     "oligo-nucleosome (Snyder 2016)"),
    (1001,   2000, "1001-2000bp",    "Yu et al. 2023"),
    (2001,   5000, "2001-5000bp",    "Berman 2024; Chan/Lo 2015 PNAS"),
    (5001,   10**9,">5000bp",        "exploratory"),
]

# Separate, OVERLAPPING window(s) reported alongside (not instead of) the
# primary bins, for direct comparison to specific published findings.
LE_ET_AL_WINDOWS = [
    (126, 135, "Le_2022_window1"),
    (240, 324, "Le_2022_window2"),
]

# ---------------------------------------------------------------------------


def detect_sep(path):
    """Comma or tab, whichever appears more often in the header line."""
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        first = fh.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def assign_bins(length):
    """Vectorised bin assignment: one label per fragment length."""
    length = np.asarray(length)
    out = np.full(length.shape, "unbinned", dtype=object)
    for lo, hi, label, _ in BIN_DEFINITIONS:
        out[(length >= lo) & (length <= hi)] = label
    return out


def bin_sample(args):
    """Bin one sample's fragments, write the binned table and return its counts."""
    f, out_dir, out_suffix = args
    sample = f.name.split(".")[0]
    df = pd.read_csv(f, sep=detect_sep(f), compression="gzip")
    if len(df) == 0:
        return sample, None, [f"  [!] {sample}: no eligible fragments, skipping"]
    if "length" not in df.columns:
        df["length"] = df["end"] - df["start"]
    df["size_bin"] = assign_bins(df["length"].to_numpy())

    # save the binned fragment table (adds size_bin column)
    binned_path = out_dir / f"{sample}.{out_suffix}.csv.gz"
    df.to_csv(binned_path, index=False, compression="gzip")

    # fragments per bin, split by origin (direct read or recovered from a concatemer)
    rows = []
    bin_counts = df["size_bin"].value_counts()
    has_source = "source" in df.columns
    for label in [b[2] for b in BIN_DEFINITIONS]:
        in_bin = df["size_bin"] == label
        rows.append({
            "sample": sample,
            "size_bin": label,
            "n_fragments": int(bin_counts.get(label, 0)),
            "n_direct": int((in_bin & (df["source"] == "direct")).sum()) if has_source else np.nan,
            "n_recovered": int((in_bin & df["source"].astype(str).str.startswith("recovered")).sum()) if has_source else np.nan,
        })

    # Le et al. 2022 windows, reported on screen only
    msgs = []
    for lo, hi, name in LE_ET_AL_WINDOWS:
        n_in_window = int(((df["length"] >= lo) & (df["length"] <= hi)).sum())
        pct = round(100 * n_in_window / len(df), 3)
        msgs.append(f"  {sample}: {name} ({lo}-{hi}bp) -- {n_in_window} fragments ({pct}%)")
    msgs.append(f"  {sample}: total {len(df)} fragments binned -> {binned_path}\n")
    return sample, rows, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="folder with the Stage 1 fragment tables")
    ap.add_argument("--pattern", default=PATTERN,
                    help="file name pattern of the fragment tables inside --in-dir")
    ap.add_argument("-o", "--out-dir", required=True,
                    help="output folder for the binned tables and the bin-count summary")
    ap.add_argument("--out-suffix", default=OUT_SUFFIX,
                    help="output = <sample>.<suffix>.csv.gz; 'fragments_binned' is the name "
                         "02_tag_fragments_with_regions.py reads")
    ap.add_argument("-t", "--threads", type=int, default=max(1, int(os.cpu_count() * 0.5)),
                    help="samples processed in parallel")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.in_dir).glob(a.pattern))
    print(f"Found {len(files)} eligible-fragment tables.\n")

    all_summary_rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(a.threads, len(files) or 1))) as ex:
        for sample, rows, msgs in ex.map(bin_sample, [(f, out_dir, a.out_suffix) for f in files]):
            print("\n".join(msgs))
            if rows:
                all_summary_rows.extend(rows)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = out_dir / "all_samples_bin_counts.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote per-sample bin counts: {summary_path}\n")

    pivot = summary_df.pivot(index="sample", columns="size_bin", values="n_fragments")
    pivot = pivot.reindex(columns=[b[2] for b in BIN_DEFINITIONS])
    print("--- Fragment counts per bin (all samples) ---")
    print(pivot.to_string())

    print("\nCells below ~500-1000 fragments may be too sparse for a stable "
          "entropy/motif estimate in that bin for that sample.")


if __name__ == "__main__":
    main()
