#!/bin/bash
# Download every public dataset and model file the project needs into the data directory
# (DATA_DIR, default ../data next to the repo; same as AT_DATA in src/paths.py). Already complete files are skipped.
#
#   bash scripts/download_data.sh            ICBHI, KAUH, AST weights, reference repositories
#   bash scripts/download_data.sh --hf-lung  also HF_Lung_V1 (large; only needed for the archived early experiments)
#
# Sources
#   ICBHI 2017 respiratory sound database (1.98 GB zip, md5-checked):
#       official site https://bhichallenge.med.auth.gr (its certificate has expired, so the zip is verified by md5),
#       mirror https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HT6PKI (also on Kaggle)
#   KAUH lung sounds (Fraiwan et al.), Mendeley Data jwyy9np4gv version 3: Audio Files.zip, Data annotation.xlsx
#   AST AudioSet weights (mAP 0.4593) from the AST authors
#   HF_Lung_V1: https://gitlab.com/techsupportHF/HF_Lung_V1 (git clone, then unzip train and test)
# Then run: python scripts/prepare_data.py
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR=${DATA_DIR:-${AT_DATA:-../data}}
REPOS_DIR=${REPOS_DIR:-${AT_REPOS:-../repos}}
ICBHI_URL=https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
ICBHI_MIRROR=https://dataverse.harvard.edu/api/access/datafile/7127117
ICBHI_MD5=f6a5d0ea40bbcd25e17754f9d5ddae48
MENDELEY=https://data.mendeley.com/public-files/datasets/jwyy9np4gv/files
AST_URL='https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'

fetch() {  # fetch <url> <file> [extra curl args]; resumes partial downloads
  local url=$1 file=$2; shift 2
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 3 -C - -o "$file" "$@" "$url"
}

# ICBHI
zip=$DATA_DIR/raw/icbhi/ICBHI_final_database.zip
if [ ! -d "$DATA_DIR/raw/icbhi/ICBHI_final_database" ]; then
  fetch "$ICBHI_URL" "$zip" -k || fetch "$ICBHI_MIRROR" "$zip"
  [ "$(md5sum "$zip" | cut -d' ' -f1)" = "$ICBHI_MD5" ] || { echo "ICBHI zip md5 mismatch: delete $zip and retry"; exit 1; }
  unzip -q -n "$zip" -d "$DATA_DIR/raw/icbhi" && rm "$zip"
fi

# KAUH
kauh=$DATA_DIR/raw/kauh
if [ ! -d "$kauh/Audio Files" ]; then
  fetch "$MENDELEY/99d7bd63-eb5d-4000-9c2a-a8dd31168cbc/file_downloaded" "$kauh/Audio Files.zip"
  unzip -q -n "$kauh/Audio Files.zip" -d "$kauh" && rm "$kauh/Audio Files.zip"
fi
[ -f "$kauh/Data annotation.xlsx" ] || fetch "$MENDELEY/bd290409-a700-4c03-ac0f-528c81d8e5c4/file_downloaded" "$kauh/Data annotation.xlsx"

# AST weights (the reference code reads ./pretrained_models/ relative to this folder)
ast=$DATA_DIR/models/ast/pretrained_models/audioset_10_10_0.4593.pth
[ -f "$ast" ] || fetch "$AST_URL" "$ast"

# Reference repositories (read-only; Patch-Mix CL provides the model, the loss and the official ICBHI split)
mkdir -p "$REPOS_DIR"
[ -d "$REPOS_DIR/patch-mix_contrastive_learning" ] || git clone https://github.com/raymin0223/patch-mix_contrastive_learning "$REPOS_DIR/patch-mix_contrastive_learning"

if [ "${1:-}" = "--hf-lung" ] && [ ! -d "$DATA_DIR/raw/hf_lung/train" ]; then
  git clone https://gitlab.com/techsupportHF/HF_Lung_V1 "$DATA_DIR/raw/hf_lung_download"
  echo "Unzip the archives in $DATA_DIR/raw/hf_lung_download so that train/ and test/ end up in $DATA_DIR/raw/hf_lung/"
fi
echo "done; next: python scripts/prepare_data.py"
