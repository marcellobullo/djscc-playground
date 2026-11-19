#!/usr/bin/env bash
# Train / fit the aligner comparison set for ADJSCC matched-SNR -> ConvNeXt-AWGN, same recipe
# as the deployed conv aligner (docs/alignment-ota.md step 1: AWGN 25 dB, 0-10 % drops, MSE).
#   GPU $GPU_A: twoconv (end2end), then the closed-form fits (wiener 5x5, parseval 1x1 / 5x5)
#   GPU $GPU_B: rescnn (end2end, linear path warm-started from the deployed conv aligner)
# Outputs: ckpts/aligners/<name>/ (aligner.pth + aligner_config.json), logs in ckpts/aligners/*.log
# Usage (on the GPU machine, repo root): nohup scripts/train_aligner_suite.sh > suite.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"; EPOCHS="${EPOCHS:-30}"
SRC=ckpts/adjscc-matched-snr-final/best
TGT=marcellobullo/djscc-convnext-cr6-awgn
CONV=ckpts/adjscc-matched-snr-final/aligner-convnext-awgn
TR=data/DIV2K_train_HR; VA=data/DIV2K_valid_HR
O=ckpts/aligners; mkdir -p $O
E2E=(--src-model $SRC --tgt-model $TGT --channel awgn --snr-db 25 --drop-min 0 --drop-max 0.1 --loss mse
     --train-data-dir $TR --val-data-dir $VA --epochs $EPOCHS)

(
  CUDA_VISIBLE_DEVICES=$GPU_A $PY training/train_aligner.py "${E2E[@]}" --kind twoconv --out $O/twoconv > $O/twoconv.log 2>&1
  for spec in "wiener 5" "parseval 1" "parseval 5"; do
    set -- $spec
    CUDA_VISIBLE_DEVICES=$GPU_A $PY training/fit_closed_form_aligner.py --method $1 --kernel $2 \
      --src-model $SRC --tgt-model $TGT --data-dir $TR --snr-db 25 --out $O/$1$2 > $O/$1$2.log 2>&1
  done
) &
CUDA_VISIBLE_DEVICES=$GPU_B $PY training/train_aligner.py "${E2E[@]}" --kind rescnn --init-aligner $CONV --out $O/rescnn > $O/rescnn.log 2>&1 &
wait
echo SUITE_DONE
