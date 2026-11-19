#!/usr/bin/env bash
# Dry run of the over-the-air protocol through the GNU Radio replica, on the 500-image COCO subset.
# Mirrors the planned OTA sessions: per SNR ("position"), one capture per TX configuration, every
# RX-side configuration decoded offline from the same received latents, then YOLO + PSNR/SSIM.
#
#   captures:  A        ADJSCC raw                -> matched, misaligned, RX aligners
#              A_txres  ADJSCC + TX ResCNN aligner -> tx_rescnn, both_rescnn
#              A_txconv ADJSCC + TX conv aligner   -> tx_conv, both_conv
#              B        ConvNeXt raw               -> target
#
# Usage: scripts/investigation/run_coco_replica.sh capture A B      (any subset of captures)
#        scripts/investigation/run_coco_replica.sh decode
#        scripts/investigation/run_coco_replica.sh eval
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
SUB=data/coco_subset500
OUT="${OUT:-out/coco_replica}"
SNRS=(${SNRS:-5 10 15 20 25})
N=500; CHUNK=50
A=ckpts/adjscc-matched-snr-final/best
B=marcellobullo/djscc-convnext-cr6-awgn
AL=ckpts/aligners
CONV=ckpts/adjscc-matched-snr-final/aligner-convnext-awgn

capture() {  # name tx_model [tx_aligner]
  local name=$1 model=$2 txal=${3:-}
  local extra=(); [[ -n "$txal" ]] && extra=(--tx-aligner "$txal")
  for ((s = 0; s < N; s += CHUNK)); do
    $PY scripts/investigation/e2e_link.py --kodak $SUB/tx --pattern '*.png' --start $s --images $CHUNK \
      --gr-gap 16 --gr-cpe --csi-follow-snr --snr "${SNRS[@]}" --model "$model" --rx-model $B \
      ${extra[@]+"${extra[@]}"} --no-decode --dump-dir $OUT/dump/$name --out $OUT/logs --tag ${name}_$s \
      2>&1 | grep -E "^SNR|\[\*\] TX aligner" || true
  done
}

case "${1:-}" in
  capture)
    shift
    for c in "$@"; do
      case $c in
        A) capture A $A ;;
        A_txres) capture A_txres $A $AL/tx_rescnn ;;
        A_txconv) capture A_txconv $A $AL/tx_conv ;;
        B) capture B $B ;;
      esac
    done ;;
  decode)
    for snr in "${SNRS[@]}"; do
      d=snr_${snr}dB; o=$OUT/dec/$d
      $PY scripts/decode_ota_dumps.py --dump $OUT/dump/A/$d --kodak $SUB/tx --pattern '*.png' --model $B \
        --run matched=none@$A --run misaligned=none --run rx_conv=$CONV --run rx_wiener=$AL/wiener5 \
        --run rx_jwiener=$AL/jw_t1_best --run rx_rescnn=$AL/rescnn --out $o/A
      [[ -d $OUT/dump/A_txres/$d ]] && $PY scripts/decode_ota_dumps.py --dump $OUT/dump/A_txres/$d --kodak $SUB/tx \
        --pattern '*.png' --model $B --run tx_rescnn=none --run both_rescnn=$AL/both_rescnn --out $o/A_txres
      [[ -d $OUT/dump/A_txconv/$d ]] && $PY scripts/decode_ota_dumps.py --dump $OUT/dump/A_txconv/$d --kodak $SUB/tx \
        --pattern '*.png' --model $B --run tx_conv=none --run both_conv=$AL/both_conv --out $o/A_txconv
      [[ -d $OUT/dump/B/$d ]] && $PY scripts/decode_ota_dumps.py --dump $OUT/dump/B/$d --kodak $SUB/tx \
        --pattern '*.png' --model $B --run target=none --out $o/B
    done ;;
  eval)
    for snr in "${SNRS[@]}"; do
      runs=()
      for dir in $OUT/dec/snr_${snr}dB/*/*/; do
        runs+=(--run "$(basename $dir)=$dir")
      done
      $PY scripts/task/eval_yolo.py --subset $SUB "${runs[@]}" --out $OUT/yolo_snr${snr}.json
    done ;;
  *) echo "usage: $0 capture A A_txres A_txconv B | decode | eval"; exit 1 ;;
esac
