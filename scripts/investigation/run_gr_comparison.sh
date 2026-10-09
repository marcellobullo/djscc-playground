#!/usr/bin/env bash
# Re-run the wired-channel experiment (24 Kodak images, SNR 0..28 dB) with the AWGN channel
# inside the offline GNU Radio replica instead of the lab channel emulator.
#   results/gr-channel/sscc_asis      SSCC (JPEG+LDPC, 9 LDPC/mod combos), TX as deployed
#   results/gr-channel/sscc_gap16     SSCC, TX with the 16-sample frame gap
#   results/gr-channel/<model>_asis   DJSCC chain as deployed
#   results/gr-channel/<model>_fixed  DJSCC chain with frame gap + CFO-ramp corrector
# Usage: scripts/investigation/run_gr_comparison.sh [sscc|djscc|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=~/miniconda3/envs/demo
export DYLD_FALLBACK_LIBRARY_PATH=$D/lib PYTHONPATH=$D/lib/python3.11/site-packages PATH=$D/bin:$PATH HF_HUB_OFFLINE=1
OUT="$ROOT/results/gr-channel"; INV="$ROOT/scripts/investigation"; mkdir -p "$OUT"
WHAT="${1:-all}"
if [[ "$WHAT" == sscc || "$WHAT" == all ]]; then
  $D/bin/python $INV/e2e_sscc.py --images 24 --out "$OUT/sscc_asis"  --tag sscc_asis  > "$OUT/sscc_asis.log" 2>&1 &
  $D/bin/python $INV/e2e_sscc.py --images 24 --out "$OUT/sscc_gap16" --tag sscc_gap16 --frame-gap 16 > "$OUT/sscc_gap16.log" 2>&1 &
fi
if [[ "$WHAT" == djscc || "$WHAT" == all ]]; then
  # name | HF model | decoder CSI (as used on hardware / in the earlier replica runs)
  for m in "deployed|marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi|19" \
           "convnext-awgn|marcellobullo/djscc-convnext-cr6-awgn|19" \
           "adjscc-awgn|marcellobullo/adjscc-cr6-awgn|5"; do
    IFS='|' read -r name model csi <<<"$m"
    $D/bin/python $INV/e2e_link.py --model "$model" --csi "$csi" --images 24 --tag "${name}_asis" \
        --save-dir "$OUT/${name}_asis/images" --out "$OUT/${name}_asis" > "$OUT/${name}_asis.log" 2>&1 &
    $D/bin/python $INV/e2e_link.py --model "$model" --csi "$csi" --images 24 --tag "${name}_fixed" --gr-gap 16 --gr-cpe \
        --save-dir "$OUT/${name}_fixed/images" --out "$OUT/${name}_fixed" > "$OUT/${name}_fixed.log" 2>&1 &
  done
fi
wait
