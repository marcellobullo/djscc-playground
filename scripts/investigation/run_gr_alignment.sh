#!/usr/bin/env bash
# Semantic-alignment experiment in the offline GNU Radio replica (no radios): the four runs of
# docs/alignment-ota.md (matched / aligned / misaligned / target) on the 24 Kodak images over an
# SNR sweep, deployed TX/RX chain with the 16-sample frame gap and the CFO-ramp corrector.
# Encoder and decoder CSI follow the channel SNR (as with --use-live-snr on hardware).
#   results/gr-align/<run>.json, <run>.log
# Usage: scripts/investigation/run_gr_alignment.sh [SNRs...]   (default: 5 10 15 20 25 30)
#   env: KODAK=<dir with kodim*.png>  ALIGNER=<aligner folder>  OUT=<dir>  IMAGES=24
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=~/miniconda3/envs/demo
export DYLD_FALLBACK_LIBRARY_PATH=$D/lib PYTHONPATH=$D/lib/python3.11/site-packages PATH=$D/bin:$PATH HF_HUB_OFFLINE=1
KODAK="${KODAK:-$HOME/Downloads/kodak_dataset}"
OUT="${OUT:-$ROOT/results/gr-align}"
IMAGES="${IMAGES:-24}"
SRC="$ROOT/ckpts/adjscc-matched-snr-final/best"
TGT=marcellobullo/djscc-convnext-cr6-awgn
ALIGNER="${ALIGNER:-$ROOT/ckpts/adjscc-matched-snr-final/aligner-convnext-awgn}"
SNRS=("$@"); [[ ${#SNRS[@]} -eq 0 ]] && SNRS=(5 10 15 20 25 30)
mkdir -p "$OUT"
COMMON=(--kodak "$KODAK" --images "$IMAGES" --snr "${SNRS[@]}" --gr-gap 16 --gr-cpe --csi-follow-snr --out "$OUT")
E2E="$ROOT/scripts/investigation/e2e_link.py"

python "$E2E" "${COMMON[@]}" --model "$SRC" --tag matched > "$OUT/matched.log" 2>&1 &
python "$E2E" "${COMMON[@]}" --model "$SRC" --rx-model "$TGT" --aligner "$ALIGNER" --tag aligned > "$OUT/aligned.log" 2>&1 &
python "$E2E" "${COMMON[@]}" --model "$SRC" --rx-model "$TGT" --tag misaligned > "$OUT/misaligned.log" 2>&1 &
python "$E2E" "${COMMON[@]}" --model "$TGT" --tag target > "$OUT/target.log" 2>&1 &
wait
for t in matched aligned misaligned target; do echo "== $t"; grep '^SNR' "$OUT/$t.log" || tail -3 "$OUT/$t.log"; done
