#!/usr/bin/env bash
# Score a second source model into ConvNeXt-AWGN in the replica (same set-up as run_gr_alignment.sh).
# Usage: scripts/investigation/run_pair.sh <name> <src_model>   (aligners in ckpts/aligners/<name>_{conv,rescnn,wiener5,jw_best})
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
D=~/miniconda3/envs/demo
export DYLD_FALLBACK_LIBRARY_PATH=$D/lib PYTHONPATH=$D/lib/python3.11/site-packages PATH=$D/bin:$PATH HF_HUB_OFFLINE=1
N=$1; SRC=$2; TGT=marcellobullo/djscc-convnext-cr6-awgn; OD=results/gr-align/pair_$N; JOBS=${JOBS:-3}
mkdir -p $OD
run() { local tag=$1; shift; [[ -f $OD/$tag.json ]] && return
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done
  python scripts/investigation/e2e_link.py --kodak ~/Downloads/kodak_dataset --images 24 --snr 5 10 15 20 25 30 \
    --gr-gap 16 --gr-cpe --csi-follow-snr --out $OD --model $SRC --tag $tag "$@" > $OD/$tag.log 2>&1 & }
run matched
run misaligned --rx-model $TGT
for a in conv rescnn wiener5 jw_best; do [[ -d ckpts/aligners/${N}_$a ]] && run $a --rx-model $TGT --aligner ckpts/aligners/${N}_$a; done
wait; echo PAIR_DONE $N
