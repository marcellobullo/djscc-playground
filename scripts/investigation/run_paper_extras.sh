#!/usr/bin/env bash
# Extra replica experiments for the paper (all on the laptop; same set-up as run_gr_alignment.sh:
# 24 Kodak images, frame gap + CFO-ramp corrector, CSI follows the channel SNR).
#   qual    reconstructions saved for the image grid (10 and 25 dB)
#   sscc    JPEG + LDPC + OFDM baseline through the same replica
#   fits    closed-form aligners for the pilot-budget and fitting-SNR studies
#   budget  pilot budget: aligners fitted on N pilot images, scored at 10 and 25 dB
#   fitsnr  Wiener / J-Wiener fitted at 5 / 15 dB, scored over the full SNR sweep
#   robust  multipath, CFO, phase noise and packet loss at 15 and 25 dB
# Usage: scripts/investigation/run_paper_extras.sh [section ...]   (default: all)
#   env: JOBS=3 KODAK=... DIV2K=...
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
D=~/miniconda3/envs/demo
export DYLD_FALLBACK_LIBRARY_PATH=$D/lib PYTHONPATH=$D/lib/python3.11/site-packages PATH=$D/bin:$PATH HF_HUB_OFFLINE=1
KODAK="${KODAK:-$HOME/Downloads/kodak_dataset}"
DIV2K="${DIV2K:-$HOME/research/djscc-demo-alignment/data/DIV2K_train_HR}"
JOBS="${JOBS:-3}"
OUT=results/gr-align
A=ckpts/aligners
SRC=ckpts/adjscc-matched-snr-final/best
TGT=marcellobullo/djscc-convnext-cr6-awgn
CONV=ckpts/adjscc-matched-snr-final/aligner-convnext-awgn
SNRS_ALL=(5 10 15 20 25 30)
SECTIONS=("${@:-qual sscc fits budget fitsnr robust}")
read -r -a SECTIONS <<< "${SECTIONS[*]}"

throttle() { while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done; }
# e2e <out_dir> <tag> <tx_model> <rx_model|""> <aligner|""> <extra args...>
e2e() {
  local od=$1 tag=$2 tx=$3 rx=$4 al=$5; shift 5
  [[ -f "$od/$tag.json" ]] && { echo "skip $od/$tag"; return; }
  mkdir -p "$od"; throttle
  local args=(--kodak "$KODAK" --images 24 --gr-gap 16 --gr-cpe --csi-follow-snr --out "$od" --model "$tx" --tag "$tag")
  [[ -n "$rx" ]] && args+=(--rx-model "$rx")
  [[ -n "$al" ]] && args+=(--aligner "$al")
  python scripts/investigation/e2e_link.py "${args[@]}" "$@" > "$od/$tag.log" 2>&1 &
}
# the six runs of the four-run protocol with the three main aligners
six() {  # out_dir suffix extra...
  local od=$1 sfx=$2; shift 2
  e2e "$od" matched$sfx "$SRC" "" "" "$@"
  e2e "$od" target$sfx "$TGT" "" "" "$@"
  e2e "$od" misaligned$sfx "$SRC" "$TGT" "" "$@"
  e2e "$od" conv$sfx "$SRC" "$TGT" "$CONV" "$@"
  e2e "$od" rescnn$sfx "$SRC" "$TGT" "$A/rescnn" "$@"
  e2e "$od" jw$sfx "$SRC" "$TGT" "$A/jw_t1_best" "$@"
}
fit() {  # method N snr out extra...
  local m=$1 n=$2 s=$3 o=$4; shift 4
  [[ -f "$o/aligner.pth" ]] && { echo "skip fit $o"; return; }
  python training/fit_closed_form_aligner.py --method $m --kernel 5 --images $n --snr-db $s \
    --src-model $SRC --tgt-model $TGT --data-dir "$DIV2K" --out "$o" "$@" > "$o.log" 2>&1
}

for sec in "${SECTIONS[@]}"; do
  echo "=== $sec $(date +%H:%M:%S)"
  case $sec in
    qual)
      od=$OUT/qual
      e2e $od matched "$SRC" "" "" --snr 10 25 --save-dir $od/img/matched
      e2e $od target "$TGT" "" "" --snr 10 25 --save-dir $od/img/target
      e2e $od misaligned "$SRC" "$TGT" "" --snr 10 25 --save-dir $od/img/misaligned
      e2e $od conv "$SRC" "$TGT" "$CONV" --snr 10 25 --save-dir $od/img/conv
      e2e $od rescnn "$SRC" "$TGT" "$A/rescnn" --snr 10 25 --save-dir $od/img/rescnn
      e2e $od jw "$SRC" "$TGT" "$A/jw_t1_best" --snr 10 25 --save-dir $od/img/jw
      ;;
    sscc)
      throttle
      mkdir -p $OUT/sscc
      [[ -f $OUT/sscc/data.json ]] || python scripts/investigation/e2e_sscc.py --kodak "$KODAK" --images 24 \
        --snr "${SNRS_ALL[@]}" --frame-gap 16 --out $OUT/sscc > $OUT/sscc.log 2>&1 &
      ;;
    fits)   # sequential (one GPU); the replica sections can run alongside
      mkdir -p $A/budget $A/fitsnr
      for n in 1 2 5 10 25 50 400; do fit wiener $n 25 $A/budget/wiener_n$n; done
      for n in 1 5 10 25 50; do fit jwiener $n 25 $A/budget/jw_n$n --alpha 0.3; done
      for s in 5 15; do
        fit wiener 100 $s $A/fitsnr/wiener_s$s
        fit jwiener 100 $s $A/fitsnr/jw_s$s --alpha 0.3
      done
      ;;
    budget)
      od=$OUT/budget
      for f in $A/budget/*/; do n=$(basename "$f"); e2e $od "$n" "$SRC" "$TGT" "$f" --snr 10 25; done
      e2e $od wiener_n100 "$SRC" "$TGT" "$A/wiener5_100" --snr 10 25
      e2e $od wiener_n800 "$SRC" "$TGT" "$A/wiener5" --snr 10 25
      e2e $od jw_n100 "$SRC" "$TGT" "$A/jw_t1_best" --snr 10 25
      e2e $od otp_n1 "$SRC" "$TGT" "$A/otpilot1_p1" --snr 10 25
      e2e $od otp_n8 "$SRC" "$TGT" "$A/otpilot1_p8" --snr 10 25
      for f in $A/conv_n* $A/rescnn_n*; do [[ -d $f ]] && e2e $od "$(basename $f)" "$SRC" "$TGT" "$f" --snr 10 25; done
      ;;
    fitsnr)
      od=$OUT/fitsnr
      for f in $A/fitsnr/*/; do e2e $od "$(basename $f)" "$SRC" "$TGT" "$f" --snr "${SNRS_ALL[@]}"; done
      ;;
    robust)
      od=$OUT/robust
      six $od _mp3 --snr 15 25 --taps 3
      six $od _mp8 --snr 15 25 --taps 8
      six $od _cfo --snr 15 25 --cfo-hz 500
      six $od _pn --snr 15 25 --pn-linewidth 100
      for r in 05 10 20; do six $od _drop$r --snr 15 25 --drop-rate 0.$r; done
      ;;
  esac
done
wait
echo "=== done $(date +%H:%M:%S)"
