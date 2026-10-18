#!/usr/bin/env python3
"""
Checks which recurrent coverage spikes fall in known problem regions of
GRCh38, with `bedtools intersect -wao`.

The first check used the ENCODE blacklist v2 for GRCh38 (Amemiya et al.
2019) alone. Spikes it did not explain sat mostly at centromeres,
pericentromeric blocks and acrocentric short arms, which the blacklist does
not fully cover, so the UCSC gap track, the UCSC centromere model and
(optionally) RepeatMasker satellite repeats can be added as further tracks.
Each spike is reported with the tracks it overlaps and the overlap length.

Input : recurrent_spikes.bed from 11_extract_spike_coordinates.py
        --blacklist  hg38-blacklist.v2.bed(.gz)        (ENCODE, BED)
        --gap        gap.txt.gz                         (UCSC table, optional)
        --centromeres centromeres.txt.gz                (UCSC table, optional)
        --rmsk       rmsk.txt.gz, satellites only       (UCSC table, optional)
Output: OUT/problem_tracks.bed                  all tracks as one sorted BED4
        OUT/recurrent_spikes_vs_tracks.wao.tsv  raw bedtools intersect -wao output
        OUT/recurrent_spikes_annotated.csv      one row per spike + overlap per track
        OUT/recurrent_spikes_match_summary.csv  spikes matched per track
"""

import argparse
import gzip
import subprocess
from pathlib import Path

import pandas as pd



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spikes", type=Path, required=True,
                   help="recurrent spike bins (BED4: chrom, start, end, n_samples)")
    p.add_argument("--blacklist", type=Path, required=True,
                   help="ENCODE blacklist v2 for GRCh38 (hg38-blacklist.v2.bed.gz)")
    p.add_argument("--gap", type=Path, default=None,
                   help="UCSC hg38 gap table (gap.txt.gz)")
    p.add_argument("--centromeres", type=Path, default=None,
                   help="UCSC hg38 centromere model table (centromeres.txt.gz)")
    p.add_argument("--rmsk", type=Path, default=None,
                   help="UCSC hg38 RepeatMasker table (rmsk.txt.gz); only class 'Satellite' is used")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output folder")
    return p.parse_args()


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_track(path, chrom_col, start_col, end_col, label, keep=None):
    """Read chrom/start/end columns (0-based column numbers) of a BED or UCSC table."""
    rows = []
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if keep is not None and not keep(f):
                continue
            rows.append((f[chrom_col], int(f[start_col]), int(f[end_col]), label))
    return rows


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. all problem-region tracks in one BED4 (4th column = track name)
    rows = read_track(args.blacklist, 0, 1, 2, "encode_blacklist")
    tracks = ["encode_blacklist"]
    if args.gap:
        rows += read_track(args.gap, 1, 2, 3, "ucsc_gap")
        tracks.append("ucsc_gap")
    if args.centromeres:
        rows += read_track(args.centromeres, 1, 2, 3, "ucsc_centromere")
        tracks.append("ucsc_centromere")
    if args.rmsk:
        rows += read_track(args.rmsk, 5, 6, 7, "satellite", keep=lambda f: f[11] == "Satellite")
        tracks.append("satellite")
    track_bed = args.out / "problem_tracks.bed"
    (pd.DataFrame(rows).sort_values([0, 1, 2])
       .to_csv(track_bed, sep="\t", header=False, index=False))
    print(f"Problem tracks: {', '.join(tracks)} ({len(rows):,} intervals)")

    # 2. every spike with every overlapping track interval (-wao keeps spikes with no overlap)
    wao = args.out / "recurrent_spikes_vs_tracks.wao.tsv"
    with open(wao, "w") as out:
        subprocess.run(["bedtools", "intersect", "-wao", "-a", str(args.spikes),
                        "-b", str(track_bed)], stdout=out, check=True)

    cols = ["chrom", "start", "end", "n_samples", "t_chrom", "t_start", "t_end", "track", "overlap_bp"]
    hits = pd.read_csv(wao, sep="\t", header=None, names=cols, dtype={"chrom": str, "t_chrom": str})
    spikes = hits[["chrom", "start", "end", "n_samples"]].drop_duplicates()

    # 3. overlap bp per spike and track (bedtools writes '.' / 0 when nothing overlaps)
    matched = hits[hits["overlap_bp"] > 0]
    per_track = (matched.groupby(["chrom", "start", "end", "track"])["overlap_bp"].sum()
                 .unstack(fill_value=0).reindex(columns=tracks, fill_value=0))
    ann = spikes.merge(per_track.reset_index(), on=["chrom", "start", "end"], how="left")
    ann[tracks] = ann[tracks].fillna(0).astype(int)
    ann["matched_tracks"] = ann[tracks].apply(
        lambda r: ",".join(t for t in tracks if r[t] > 0), axis=1)
    ann["any_match"] = ann["matched_tracks"] != ""
    ann.to_csv(args.out / "recurrent_spikes_annotated.csv", index=False)

    # 4. summary: how many spikes each track explains
    n = len(ann)
    n_all = ann["n_samples"].max()
    universal = ann[ann["n_samples"] == n_all]
    summary = [{"track": t, "n_spikes_matched": int((ann[t] > 0).sum()),
                "pct_spikes_matched": 100 * (ann[t] > 0).mean(),
                "n_universal_matched": int((universal[t] > 0).sum())} for t in tracks]
    summary.append({"track": "any", "n_spikes_matched": int(ann["any_match"].sum()),
                    "pct_spikes_matched": 100 * ann["any_match"].mean(),
                    "n_universal_matched": int(universal["any_match"].sum())})
    summary = pd.DataFrame(summary)
    summary["n_spikes"] = n
    summary["n_universal_spikes"] = len(universal)
    summary.to_csv(args.out / "recurrent_spikes_match_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"(universal = spike in the largest number of samples seen, {n_all})")
    print(f"Wrote {args.out}/recurrent_spikes_annotated.csv")


if __name__ == "__main__":
    main()
