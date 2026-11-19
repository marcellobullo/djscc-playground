#!/usr/bin/env bash
# Second round of aligners (GPU machine, repo root):
#   GPU $GPU_A: two more source models into the same target (ConvNeXt-AWGN), same recipe as the
#               first pair: conv 5x5 (end2end), rescnn warm-started from it, closed-form Wiener 5x5
#               and Jacobian-weighted Wiener (alpha picked on DIV2K valid), 100 pilot images.
#                 b2019     Bourtsoulatze-2019 DeepJSCC (OFDM-trained)      -> ConvNeXt-AWGN
#                 spatialcsi ConvNeXt spatial-CSI (OFDM-trained, same family) -> ConvNeXt-AWGN
#   GPU $GPU_B: data budget for ADJSCC -> ConvNeXt-AWGN: conv and rescnn trained on N images
#               (subset repeated, so every run has the same number of steps).
# Outputs: ckpts/aligners/<name>/, logs ckpts/aligners/<name>.log
# Usage: nohup scripts/train_aligner_suite2.sh > suite2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"; EPOCHS="${EPOCHS:-30}"
TGT=marcellobullo/djscc-convnext-cr6-awgn
TR=data/DIV2K_train_HR; VA=data/DIV2K_valid_HR
O=ckpts/aligners; mkdir -p $O
REC=(--channel awgn --snr-db 25 --drop-min 0 --drop-max 0.1 --loss mse
     --train-data-dir $TR --val-data-dir $VA --epochs $EPOCHS)

pair() {  # name src_model
  local n=$1 src=$2
  CUDA_VISIBLE_DEVICES=$GPU_A $PY training/train_aligner.py --src-model $src --tgt-model $TGT "${REC[@]}" \
    --kind conv --out $O/${n}_conv > $O/${n}_conv.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU_A $PY training/train_aligner.py --src-model $src --tgt-model $TGT "${REC[@]}" \
    --kind rescnn --init-aligner $O/${n}_conv --out $O/${n}_rescnn > $O/${n}_rescnn.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU_A $PY training/fit_closed_form_aligner.py --method wiener --kernel 5 --images 100 \
    --src-model $src --tgt-model $TGT --data-dir $TR --snr-db 25 --out $O/${n}_wiener5 > $O/${n}_wiener5.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU_A $PY training/fit_closed_form_aligner.py --method jwiener --kernel 5 --images 100 \
    --alpha 0.3 --save-stats $O/${n}_jw_stats.pt \
    --src-model $src --tgt-model $TGT --data-dir $TR --snr-db 25 --out $O/${n}_jw_a03 > $O/${n}_jw.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU_A $PY scripts/investigation/sweep_jwiener_alpha.py $O/${n}_jw_stats.pt \
    --src-model $src --tgt-model $TGT --val-dir $VA --out $O/${n}_jw_best > $O/${n}_jw_sweep.log 2>&1
  rm -f $O/${n}_jw_stats.pt
}

(
  pair b2019 marcellobullo/djscc-bourtsoulatze-cr6-ofdm
  pair spatialcsi marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi
) &

(
  SRC=ckpts/adjscc-matched-snr-final/best
  for N in 10 50 200; do
    CUDA_VISIBLE_DEVICES=$GPU_B $PY training/train_aligner.py --src-model $SRC --tgt-model $TGT "${REC[@]}" \
      --kind conv --train-images $N --out $O/conv_n$N > $O/conv_n$N.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_B $PY training/train_aligner.py --src-model $SRC --tgt-model $TGT "${REC[@]}" \
      --kind rescnn --train-images $N --init-aligner $O/conv_n$N --out $O/rescnn_n$N > $O/rescnn_n$N.log 2>&1
  done
) &
wait
echo SUITE2_DONE
