# Thesis-code

Analysis code for my MSc thesis: long cell-free DNA fragmentomics in hepatocellular carcinoma (HCC) versus HBV carriers, using the ONT and PacBio data of Yu et al. (2023, *Clin Chem* 69:168).

The folders are numbered in the order the steps were run on the analysis server, and the scripts inside each folder are numbered in the same way. The thesis Methods (Chapter 2) describe each step, Section 2.14 gives the software versions and Appendix A lists the scripts.

## Folders

| Folder | What it does | Thesis section |
|---|---|---|
| `00_environment` | Conda environment of the analysis server and the offline install of ichorCNA | 2.14 |
| `01_annotation` | GENCODE v49 promoter, TSS, gene body, exon and CpG island BED files; chromosome sizes; promoter CpG classes | 2.5, 2.8 |
| `02_read_processing` | Cutadapt adapter trimming, minimap2 alignment, Stage 1 fragment calling with concatemer recovery, PacBio CCS alignment | 2.2, 2.11 |
| `03_coverage_qc` | Whole-genome versus targeted checks, coverage spikes and problem-region filtering | 2.3 |
| `04_size_bins_regions` | Size-bin assignment, region flags and the size-bin merge | 2.4, 2.5 |
| `05_entropy_size` | Fragment-length entropy, rarefaction, length histograms and size-ratio analyses | 2.6, 4.3 |
| `06_end_motifs` | 5′ end-motif extraction, motif tables and motif tests | 2.7, 2.9 |
| `07_promoters_genesets` | Individual-promoter metrics, gene-set pooling, random-set nulls, Hallmark map and TSS profiles | 2.8 |
| `08_classifiers` | Sample covariates, feature tables, nested cross-validation and the locked ONT model applied to PacBio | 2.3.3, 2.10 |
| `09_tumour_fraction` | Read-count WIG files and ichorCNA | 2.12 |
| `10_external_validation` | GSE71378 and CRA001537: download, alignment, fragment sizes, end motifs and pre-registered tests | 2.13 |
| `11_figures` | Thesis figures, in figure-number order | all chapters |

Figures 2.5, 4.3 and 4.4 are drawn by the analysis scripts themselves (`08_classifiers/05_explain_depth_adjustment.py` and `05_entropy_size/06_size_bin_fold_and_window.py`).

## Running

Every script has `--help`. Input files and output folders are given on the command line; no paths are built in. Scripts that run in parallel take `--threads`, and every script writes to its own output folder.

Environments: `thesis` (Python 3.12, pysam, pandas, SciPy, scikit-learn, Cutadapt 5.2, minimap2 2.31, BWA 0.7.19, samtools 1.23.1, bedtools 2.31.1, mosdepth 0.3.14) and `ichor` (R 4.3.3, ichorCNA 0.5.1).

## Data

This repository holds code only. The discovery data are controlled access at the European Genome-phenome Archive (ONT: EGAS00001006328; PacBio: EGAS00001006609). The external validation data are open (GEO GSE71378, SRA PRJNA291063; Genome Sequence Archive CRA001537).
