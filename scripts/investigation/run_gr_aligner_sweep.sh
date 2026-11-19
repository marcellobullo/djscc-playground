#!/usr/bin/env bash
# Score several aligners in the offline GNU Radio replica, same setup as run_gr_alignment.sh
# (24 Kodak images, frame gap + CFO-ramp corrector, CSI follows the channel SNR). Only the
# aligned run is repeated per aligner; matched / misaligned / target come from run_gr_alignment.sh.
#   results/gr-align/aligned_<name>.json
# Usage: scripts/investigation/run_gr_aligner_sweep.sh name=folder [name=folder ...]
#   env: SNRS="5 10 15 20 25 30"  KODAK=...  OUT=...  IMAGES=24  JOBS=3
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=~/miniconda3/envs/demo
export DYLD_FALLBACK_LIBRARY_PATH=$D/lib PYTHONPATH=$D/lib/python3.11/site-packages PATH=$D/bin:$PATH HF_HUB_OFFLINE=1
KODAK="${KODAK:-$HOME/Downloads/kodak_dataset}"
OUT="${OUT:-$ROOT/results/gr-align}"
IMAGES="${IMAGES:-24}"; JOBS="${JOBS:-3}"
read -r -a SNRS <<< "${SNRS:-5 10 15 20 25 30}"
SRC="$ROOT/ckpts/adjscc-matched-snr-final/best"
TGT=marcellobullo/djscc-convnext-cr6-awgn
mkdir -p "$OUT"
for spec in "$@"; do
  name="${spec%%=*}"; folder="${spec#*=}"
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done
  python "$ROOT/scripts/investigation/e2e_link.py" --kodak "$KODAK" --images "$IMAGES" --snr "${SNRS[@]}" \
    --gr-gap 16 --gr-cpe --csi-follow-snr --out "$OUT" \
    --model "$SRC" --rx-model "$TGT" --aligner "$folder" --tag "aligned_$name" > "$OUT/aligned_$name.log" 2>&1 &
done
wait
for spec in "$@"; do name="${spec%%=*}"; echo "== $name"; grep '^SNR' "$OUT/aligned_$name.log" || tail -3 "$OUT/aligned_$name.log"; done
