#!/usr/bin/env bash
# 01_list_bioc_data_urls.sh
# -------------------------
# The analysis server has no direct access to bioconductor.org, so the conda packages for
# the ichorCNA environment are downloaded on another machine and copied over (folder PK).
# Some Bioconductor data packages download their data at install time; this script lists
# them (bioc_data_pkgs.txt) and writes their download URLs (bioc_data_urls.txt) so the
# tarballs can be fetched separately. 02_install_ichor_env.sh uses both.
#
# Usage: bash 01_list_bioc_data_urls.sh PK
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash 01_list_bioc_data_urls.sh PK   (PK: folder with the downloaded conda packages)"; exit 1; }
PK="$1"
cd "$PK"
JSON=$(find . -path "*bioconductor-data-packages*" -name dataURLs.json | head -1)
[ -n "$JSON" ] || { echo "dataURLs.json not found under $PK"; exit 1; }
grep -l installBiocDataPackage */bin/.*post-link.sh 2>/dev/null | cut -d/ -f1 | sort -u > bioc_data_pkgs.txt
echo "data packages that download at install:"; cat bioc_data_pkgs.txt
python3 - "$JSON" bioc_data_pkgs.txt > bioc_data_urls.txt <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
for line in open(sys.argv[2]):
    m = re.match(r"bioconductor-(.+)-(\d[\w.]*)-r\d", line.strip())
    if not m: continue
    key = f"{m.group(1)}-{m.group(2)}"
    e = d.get(key)
    if e is None: print(f"# NOT IN JSON: {key}"); continue
    urls = e["urls"]; u = next((x for x in urls if "bioconductor.org" in x), urls[0])
    print(u)
    print(f"# md5 {e['md5']}  alt: " + " ".join(x for x in urls if x != u))
PY
# ichorCNA with --genomeBuild hg38 looks for BSgenome.Hsapiens.UCSC.hg38 and otherwise fetches chromosome info online
grep -q -i "ucsc.hg38" bioc_data_urls.txt || echo "https://bioconductor.org/packages/3.22/data/annotation/src/contrib/BSgenome.Hsapiens.UCSC.hg38_1.4.5.tar.gz" >> bioc_data_urls.txt
echo; echo "== $PK/bioc_data_urls.txt"; cat bioc_data_urls.txt
