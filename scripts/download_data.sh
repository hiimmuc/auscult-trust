#!/bin/bash
# Download every public dataset and model file the project needs into the data directory
# (DATA_DIR, default ../data next to the repo; same as AT_DATA in src/paths.py). Already complete files are skipped.
#
#   bash scripts/download_data.sh            ICBHI, KAUH, encoder weights, reference repositories
#   bash scripts/download_data.sh --hf-lung  also HF_Lung_V1 (large; only needed for the archived early experiments)
#
# Sources
#   ICBHI 2017 respiratory sound database (1.98 GB zip, md5-checked):
#       official site https://bhichallenge.med.auth.gr (its certificate has expired, so the zip is verified by md5),
#       mirror https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HT6PKI (also on Kaggle)
#   KAUH lung sounds (Fraiwan et al.), Mendeley Data jwyy9np4gv version 3: Audio Files.zip, Data annotation.xlsx
#   Encoder weights: AST AudioSet (AST authors), OPERA-CT (Hugging Face evelyn0414/OPERA), CLAP (laion/clap-htsat-unfused) and
#       HeAR (google/hear-pytorch, gated: request access on its Hugging Face page and run `hf auth login` first) into
#       DATA_DIR/models/huggingface; HTS-AT AudioSet weights are a manual download (Google Drive folder of the HTS-AT authors,
#       https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6) to DATA_DIR/models/htsat/AudioSet/
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

# DASS weights (medium, AudioSet)
dass=$DATA_DIR/models/dass/pretrained_models/DASS_medium_v2.pth
[ -f "$dass" ] || fetch https://github.com/Saurabhbhati/DASS/releases/download/v0.2/DASS_medium_v2.pth "$dass"

# OPERA-CT weights
opera=$DATA_DIR/models/opera/encoder-operaCT.ckpt
[ -f "$opera" ] || fetch "https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCT.ckpt?download=true" "$opera"

# CLAP and HeAR through the Hugging Face hub (HeAR needs an authorised account; skipped with a message otherwise)
hub=$DATA_DIR/models/huggingface/hub
HF=${HF:-hf}
$HF download laion/clap-htsat-unfused --cache-dir "$hub" > /dev/null || echo "CLAP download failed"
$HF download google/hear-pytorch --cache-dir "$hub" > /dev/null || echo "HeAR download failed: request access at https://huggingface.co/google/hear-pytorch and run 'hf auth login'"
[ -f "$DATA_DIR/models/htsat/AudioSet/HTSAT_AudioSet_Saved_1.ckpt" ] || echo "HTS-AT weights missing: download HTSAT_AudioSet_Saved_1.ckpt (link in the header) to DATA_DIR/models/htsat/AudioSet/"

# Reference repositories (read-only): Patch-Mix CL (model, loss, official ICBHI split), OPERA (HTS-AT network), HeAR (preprocessing),
# Lung-SRAD (DASS), RespireNet and SG-SCL (baselines), HTS-AT
mkdir -p "$REPOS_DIR"
clone() { [ -d "$REPOS_DIR/$2" ] || git clone "$1" "$REPOS_DIR/$2"; }
clone https://github.com/raymin0223/patch-mix_contrastive_learning patch-mix_contrastive_learning
clone https://github.com/evelyn0414/OPERA OPERA
clone https://github.com/Google-Health/hear hear
clone https://github.com/RSC-Toolkit/Lung-SRAD Lung-SRAD
clone https://github.com/microsoft/RespireNet RespireNet
clone https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning SG-SCL
clone https://github.com/RetroCirce/HTS-Audio-Transformer HTS-Audio-Transformer
clone https://github.com/Saurabhbhati/DASS DASS  # selective-scan CUDA kernel, see scripts/build_dass_env.sh

if [ "${1:-}" = "--hf-lung" ] && [ ! -d "$DATA_DIR/raw/hf_lung/train" ]; then
  git clone https://gitlab.com/techsupportHF/HF_Lung_V1 "$DATA_DIR/raw/hf_lung_download"
  echo "Unzip the archives in $DATA_DIR/raw/hf_lung_download so that train/ and test/ end up in $DATA_DIR/raw/hf_lung/"
fi
echo "done; next: python scripts/prepare_data.py"
