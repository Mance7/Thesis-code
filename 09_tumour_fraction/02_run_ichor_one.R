#!/usr/bin/env Rscript
# 02_run_ichor_one.R
# ------------------
# One ichorCNA 0.5.1 run on one sample (Methods 2.12), through the package's own
# run_ichorCNA() function. (The runIchorCNA.R script on GitHub belongs to the older
# 0.3.2 package and fails in runEM with 0.5.1.)
#
# Input:  the 1 Mb WIG of one sample from 01_fragments_to_wig.py and the hg38 reference
#         files that come with ichorCNA (GC, mappability, replication timing, the panel of
#         normal plasma HD_ULP_PoN_hg38_1Mb, centromeres).
# Output: the usual ichorCNA files in OUTDIR (<ID>.params.txt holds tumour fraction and ploidy).
# Two settings (CONFIG):
#   lowTF  primary, ichorCNA's setting for tumour fractions below about 5%: normal fraction
#          0.95-0.999, ploidy 2, copy number up to 3, no subclones
#   std    sensitivity: normal fraction 0.5-0.9, ploidy 2 or 3, copy number up to 5, subclones
# Autosomes and chrX are analysed; training and normalisation use the autosomes.
#
# Usage: Rscript 02_run_ichor_one.R ID WIG OUTDIR CONFIG(lowTF|std) GC MAP PON CEN REPTIME
#        (REPTIME = NULL to run without the replication-timing correction)
a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 9) stop("usage: ID WIG OUTDIR CONFIG GC MAP PON CEN REPTIME")
names(a)[1:9] <- c("id", "wig", "out", "cfg", "gc", "map", "pon", "cen", "rep")
suppressPackageStartupMessages(library(ichorCNA))
# Offline seqinfo: ichorCNA looks for BSgenome.Hsapiens.UCSC.<build> and otherwise fetches
# chromosome sizes online. The GRCh38 primary-assembly lengths are fixed, so they are set here.
hg38_len <- c(248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636,
              138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345,
              83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895, 57227415)
names(hg38_len) <- paste0("chr", c(1:22, "X", "Y"))
offline_seqinfo <- function(genomeBuild = "hg38", genomeStyle = "UCSC", chrs = names(hg38_len)) {
  chrs <- as.character(chrs)
  GenomeInfoDb::seqlevelsStyle(chrs) <- "UCSC"
  GenomeInfoDb::Seqinfo(seqnames = chrs, seqlengths = unname(hg38_len[chrs]),
                        isCircular = rep(FALSE, length(chrs)), genome = "hg38")
}
assignInNamespace("getSeqInfo", offline_seqinfo, ns = "ichorCNA")
dir.create(a[["out"]], recursive = TRUE, showWarnings = FALSE)
# settings shared by both configurations
common <- list(tumor_wig = a[["wig"]], gcWig = a[["gc"]], mapWig = a[["map"]],
               repTimeWig = if (a[["rep"]] %in% c("", "NULL")) NULL else a[["rep"]],
               normal_panel = a[["pon"]], centromere = a[["cen"]], id = a[["id"]], outDir = a[["out"]],
               includeHOMD = FALSE, chrs = "c(paste0('chr', 1:22), 'chrX')",
               chrTrain = "paste0('chr', 1:22)", chrNormalize = "paste0('chr', 1:22)",
               genomeStyle = "UCSC", genomeBuild = "hg38", txnE = 0.9999, txnStrength = 10000, cores = 1)
cfg <- if (a[["cfg"]] == "lowTF") {
  # ichorCNA recommendation for tumour fraction below ~5%
  list(normal = "c(0.95, 0.99, 0.995, 0.999)", ploidy = "c(2)", maxCN = 3,
       estimateScPrevalence = FALSE, scStates = "c()")
} else {
  list(normal = "c(0.5, 0.6, 0.7, 0.8, 0.9)", ploidy = "c(2, 3)", maxCN = 5,
       estimateScPrevalence = TRUE, scStates = "c(1, 3)")
}
do.call(run_ichorCNA, c(common, cfg))
message("DONE ", a[["id"]], " ", a[["cfg"]])
