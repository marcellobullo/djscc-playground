#!/usr/bin/env bash
# Aligner placement study (GPU machine, repo root): ADJSCC matched-SNR -> ConvNeXt-AWGN.
#   TX  : aligner on the encoder latent before the channel (power-matched), warm-started from the
#         RX aligner of the same family.
#   both: an RX aligner trained on top of the frozen TX aligner (cascade), identity-initialised,
#         so it starts exactly at the TX-only result.
# GPU $GPU_A: ResCNN family; GPU $GPU_B: linear conv family. Same recipe as the RX aligners.
# Outputs ckpts/aligners/{tx,both}_{rescnn,conv}/, logs alongside.
# Usage: nohup scripts/train_aligner_placement.sh > placement.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"; EPOCHS="${EPOCHS:-30}"
SRC=ckpts/adjscc-matched-snr-final/best
TGT=marcellobullo/djscc-convnext-cr6-awgn
TR=data/DIV2K_train_HR; VA=data/DIV2K_valid_HR
O=ckpts/aligners; mkdir -p $O
REC=(--src-model $SRC --tgt-model $TGT --channel awgn --snr-db 25 --drop-min 0 --drop-max 0.1 --loss mse
     --train-data-dir $TR --val-data-dir $VA --epochs $EPOCHS)

family() {  # gpu kind rx_init
  local g=$1 k=$2 init=$3
  CUDA_VISIBLE_DEVICES=$g $PY training/train_aligner.py "${REC[@]}" --kind $k --placement tx \
    --init-aligner $init --out $O/tx_$k > $O/tx_$k.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY training/train_aligner.py "${REC[@]}" --kind $k --tx-aligner $O/tx_$k \
    --init-identity --out $O/both_$k > $O/both_$k.log 2>&1
}

family $GPU_A rescnn $O/rescnn &
family $GPU_B conv ckpts/adjscc-matched-snr-final/aligner-convnext-awgn &
wait
echo PLACEMENT_DONE
