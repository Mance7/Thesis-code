#!/usr/bin/env python3
"""
Merge the 11-bin motif counts into five coarser size bins.

Re-bins the 11-bin motif long table (02_aggregate_motif_frequencies_region_stratified.py
output) by summing raw counts, and writes a new table (the input is never changed).
Scheme 5bin (exact edges, because every edge used is an edge of the 11 bins):
    <=255bp      = <90, 90-125, 126-135, 136-150, 151-255     (mono-nucleosome and shorter)
    256-650bp    = 256-420, 421-650                            (di- and tri-nucleosome, = numerator of R)
    651-2000bp   = 651-1000, 1001-2000
    2001-5000bp, >5000bp unchanged
The genome-wide background rows are dropped. For comparison, the 4-bin merge of
05_motif_reduced_family_stats.py (--merge-4bin) puts 421-650 into "601-2000bp",
so its "<=600bp" bin really ends at 420 bp.

Input : motif_frequencies_region_stratified_long.csv
Output: <out>/motif_frequencies_5bin_long.csv (sample, bin, region, motif, count,
        n_fragments, frequency_pct), the input of 05_motif_reduced_family_stats.py

  python 04_remap_motif_bins.py --table "motif_region/all/motif_frequencies_region_stratified_long.csv" \\
      -o "motif5/ONT_all/table" --scheme 5bin
"""
import argparse, os
import pandas as pd

SCHEMES = {"5bin": {"<90bp": "<=255bp", "90-125bp": "<=255bp", "126-135bp": "<=255bp", "136-150bp": "<=255bp",
                    "151-255bp": "<=255bp", "256-420bp": "256-650bp", "421-650bp": "256-650bp",
                    "651-1000bp": "651-2000bp", "1001-2000bp": "651-2000bp",
                    "2001-5000bp": "2001-5000bp", ">5000bp": ">5000bp"}}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--table", required=True, help="11-bin motif long table (csv)")
ap.add_argument("-o", "--out", required=True, help="output folder")
ap.add_argument("--scheme", default="5bin", choices=list(SCHEMES), help="bin merge scheme")
ap.add_argument("--bin-col", default="bin", help="size bin column of --table")
ap.add_argument("--count-col", default="count", help="raw motif count column of --table")
ap.add_argument("-t", "--threads", type=int, default=1, help="not used; the job is a single groupby")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.table)
df = df[df[a.bin_col].astype(str) != "background"]
unknown = set(df[a.bin_col]) - set(SCHEMES[a.scheme])
if unknown:
    raise SystemExit(f"bins not in scheme {a.scheme}: {sorted(unknown)}")
df[a.bin_col] = df[a.bin_col].map(SCHEMES[a.scheme])
# sum raw counts into the new bins, then recompute cell sizes and percentages
out = df.groupby(["sample", a.bin_col, "region", "motif"], as_index=False)[a.count_col].sum()
out["n_fragments"] = out.groupby(["sample", a.bin_col, "region"])[a.count_col].transform("sum")
out["frequency_pct"] = 100 * out[a.count_col] / out["n_fragments"]
fn = os.path.join(a.out, f"motif_frequencies_{a.scheme}_long.csv")
out.to_csv(fn, index=False)
print(f"wrote {fn}: {len(out)} rows; bins: {sorted(out[a.bin_col].unique())}")
