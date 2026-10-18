#!/usr/bin/env python3
"""Put the ichorCNA genome-wide copy-number plots of three samples on both platforms into one figure.

Figure:
  Figure 3.22  fig4_cna_cross_platform.png   log2 copy ratio in 1 Mb windows from ichorCNA (red gain, green loss, blue neutral)
    A, B  TBR3976 (HCC), ONT and PacBio
    C, D  TBR2967 (HCC), ONT and PacBio
    E, F  TBR4483 (HBV carrier), ONT and PacBio

The panels are ichorCNA's own <sample>_genomeWide.png plots, unchanged, laid out in a 3 x 2 grid with a panel
label above each. ONT uses the primary lowTF setting and PacBio the std setting (the primary PacBio setting
failed to fit most samples; Section 3.8).

Input: the tumour-fraction output folder of the ichorCNA runs, with one folder per platform and setting
(ichor_<ont|pacbio>_<lowTF|std>/<sample>/...); the plot is searched for anywhere below the sample folder.
Output: fig4_cna_cross_platform.png (300 dpi) in the --out folder.

Usage: python 12_plot_cna_cross_platform.py --tf-dir "results/tumour_fraction" -o "results/figures"
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# (sample, group label) in row order
SAMPLES = [("TBR3976", "HCC"), ("TBR2967", "HCC"), ("TBR4483", "HBV")]


def find_plot(tf_dir, platform, setting, sample):
    """path of <sample>_genomeWide.png in the ichorCNA output of one platform and setting"""
    hits = sorted(glob.glob(os.path.join(tf_dir, f"ichor_{platform}_{setting}", sample, "**", f"{sample}_genomeWide.png"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"no {sample}_genomeWide.png under {tf_dir}/ichor_{platform}_{setting}/{sample}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description="Cross-platform copy-number figure (Figure 3.22).")
    ap.add_argument("--tf-dir", required=True,
                    help="ichorCNA output folder with ichor_<ont|pacbio>_<lowTF|std>/<sample>/ subfolders")
    ap.add_argument("--ont-setting", default="lowTF", help="ichorCNA setting shown for ONT (default lowTF, the primary setting)")
    ap.add_argument("--pacbio-setting", default="std", help="ichorCNA setting shown for PacBio (default std)")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if missing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(20, 9.5))
    letters = iter("ABCDEF")
    for row, (sample, group) in enumerate(SAMPLES):
        for col, (platform, name, setting) in enumerate([("ont", "ONT", a.ont_setting), ("pacbio", "PacBio", a.pacbio_setting)]):
            ax = axes[row, col]
            ax.imshow(mpimg.imread(find_plot(a.tf_dir, platform, setting, sample)))
            ax.axis("off")
            ax.set_title(f"{next(letters)}  {sample} ({group}), {name}", loc="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.12)
    png = os.path.join(a.out, "fig4_cna_cross_platform.png")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
