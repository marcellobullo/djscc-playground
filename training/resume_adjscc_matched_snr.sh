#!/usr/bin/env bash
#
# Resume the ADJSCC "matched SNR" fine-tune (adjscc-cr6-awgn, encoder + decoder fed the true SNR,
# AWGN SNR ~ U[-2, 30] dB, checkpoint selected on the mean val PSNR over 0..28 dB).
# Epoch 0 was trained on a Mac (22 min/epoch on MPS); this continues to 20 epochs on a GPU.
#
# Needs, relative to the repo root (both are git-ignored):
#   ckpts/adjscc-matched-snr/last.pth   network + optimizer + epoch (resume state)
#   ckpts/adjscc-matched-snr/best/      HF folder (config, code, weights) used to build the model
#   data/DIV2K_train_HR, data/DIV2K_valid_HR
# If DIV2K is missing:
#   mkdir -p data && cd data
#   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
#   unzip -q DIV2K_train_HR.zip && unzip -q DIV2K_valid_HR.zip && cd ..
# Python deps: pip install -r requirements.txt  (torch with CUDA, kaira, compressai, transformers)
#
# Usage: training/resume_adjscc_matched_snr.sh [extra train.py args, e.g. --batch-size 8]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=ckpts/adjscc-matched-snr
for f in "$OUT/last.pth" "$OUT/best/config.json" data/DIV2K_train_HR data/DIV2K_valid_HR; do
  [[ -e "$f" ]] || { echo "[!] missing $f (see the header of this script)" >&2; exit 2; }
done
python training/train.py \
  --model "$OUT/best" --resume "$OUT/last.pth" \
  --train-data-dir data/DIV2K_train_HR --val-data-dir data/DIV2K_valid_HR \
  --channel awgn --snr-min -2 --snr-max 30 --val-snrs 0 4 8 12 16 20 24 28 \
  --epochs 20 --batch-size 4 --lr 5e-5 --device auto --out "$OUT" "$@"
