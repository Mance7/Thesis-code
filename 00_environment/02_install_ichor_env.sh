#!/usr/bin/env bash
# 02_install_ichor_env.sh
# ----------------------
# Environment setup only: installs ichorCNA 0.5.1 (R 4.3.3, bioconda r-ichorcna) into its own
# conda environment "ichor" on a server without internet access (Methods 2.12, 2.14).
#
# Needs:
#   PK  a folder with all conda packages of the environment (.conda / .tar.bz2), downloaded
#       on a machine with internet, plus bioc_data_pkgs.txt: the list of Bioconductor data
#       packages in that folder that would download their data at install time (made with
#       01_list_bioc_data_urls.sh, which also writes their download URLs)
#   D   a folder with the source tarballs of those data packages (downloaded from those URLs)
# The data packages are skipped in the conda install and installed from their tarballs with
# R CMD INSTALL instead. 09_tumour_fraction/02_run_ichor_one.R then runs inside this environment.
#
# Usage: bash 02_install_ichor_env.sh PK D
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: bash 02_install_ichor_env.sh PK D   (PK: conda package folder, D: data package tarballs)"; exit 1; }
PK="$1"
D="$2"
source "$(conda info --base)/etc/profile.d/conda.sh"

# 0) remove an earlier single-package ichorCNA install from the analysis env "thesis" (unused there)
conda remove -n thesis --offline --force -y r-ichorcna 2>/dev/null || true

# 1) all packages except the data packages that download at install time
cd "$PK"
[ -s bioc_data_pkgs.txt ] || { echo "bioc_data_pkgs.txt missing in $PK (run 01_list_bioc_data_urls.sh first)"; exit 1; }
ls "$D"/*.tar.gz >/dev/null 2>&1 || { echo "MISSING: data package tarballs in $D"; exit 1; }
echo "@EXPLICIT" > ichor_explicit.txt
ls "$PWD"/*.conda "$PWD"/*.tar.bz2 2>/dev/null | grep -v -F -f bioc_data_pkgs.txt | sed 's|^|file://|' >> ichor_explicit.txt
echo "$(($(wc -l < ichor_explicit.txt)-1)) packages"
conda env remove -n ichor -y 2>/dev/null || true
conda create -n ichor --offline --file ichor_explicit.txt -y

# 2) the data packages from their tarballs (GenomeInfoDbData first: the others load it)
conda activate ichor
for f in $(ls "$D"/GenomeInfoDbData_*.tar.gz 2>/dev/null) $(ls "$D"/*.tar.gz | grep -v GenomeInfoDbData_); do
  echo "== R CMD INSTALL $(basename "$f")"; R CMD INSTALL --library="$CONDA_PREFIX/lib/R/library" "$f"
done

# 3) check that ichorCNA loads
Rscript -e 'library(ichorCNA); cat("ichorCNA OK\n")'
find "$CONDA_PREFIX" -name runIchorCNA.R || true
