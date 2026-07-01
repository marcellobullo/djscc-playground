#!/usr/bin/env bash
#
# run_tx_experiment_conventional.sh — transmit a dataset folder once, for a
# single (SNR, LDPC rate, modulation) point, using the conventional
# (JPEG/JPEG2000 + LDPC + OFDM) baseline.
#
# Companion to run_rx_experiment_conventional.sh. The SNR is set MANUALLY on
# the lab channel emulator; this script only labels the run and (optionally)
# launches the GNU Radio TX flowgraph. Re-run it once per (SNR, ldpc-n,
# ldpc-k, mod-order) combination, after re-setting the emulator and starting
# the matching RX.
#
# Usage:
#   scripts/run_tx_experiment_conventional.sh <SNR_dB> [options]
#
# Examples:
#   # flowgraph already running (default); rate-1/2, BPSK, JPEG
#   scripts/run_tx_experiment_conventional.sh 10 --dataset ./dataset \
#       --ldpc-n 2304 --ldpc-k 1152 --mod-order 1
#
#   # also launch the GNU Radio TX flowgraph for this run (needed whenever
#   # --mod-order changes, since modulation is fixed at flowgraph construction)
#   scripts/run_tx_experiment_conventional.sh 10 --dataset ./dataset \
#       --ldpc-n 2304 --ldpc-k 1152 --mod-order 1 --launch-fg
#
# Options:
#   --dataset DIR      folder of images to transmit          (default: ./dataset)
#   --codec NAME       jpeg | jpeg2000                        (default: jpeg)
#   --codec-quality N  codec quality/ratio knob                (default: script default, 48)
#   --comp-ratio N     DJSCC-equivalent inverse compression ratio (default: 6)
#   --ldpc-n N         LDPC codeword length n                  (default: 2304)
#   --ldpc-k N         LDPC message length k                   (default: 1152)
#   --mod-order N      bits/symbol: 1=BPSK 2=QPSK 4=16QAM       (default: 1)
#   --no-interleave    disable symbol interleaving (default: ON, must match RX)
#   --no-warmup        skip warm-up frames before the timed run
#   --interval SEC     seconds between images                  (default: 3.0)
#   --launch-fg        compile + run conventional_tx.grc in conda env `demo`
#                         for this run (mod-order baked into the flowgraph)
#   --no-wait          don't pause for the "set emulator + start RX" confirmation
#   --                 pass any remaining args straight to socket_conventional_tx.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET="$ROOT/dataset"
CODEC="jpeg"
CODEC_QUALITY=""
COMP_RATIO=6
LDPC_N=2304
LDPC_K=1152
MOD_ORDER=1
INTERLEAVE=1
WARMUP=1
INTERVAL=3.0
LAUNCH_FG=0
WAIT=1
SNR=""
EXTRA=()

# GNU Radio flowgraph (only used with --launch-fg)
GRC="$ROOT/transmitter/gnu_radio/conventional_tx.grc"
DEMO_ENV="$HOME/miniconda3/envs/demo"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)       DATASET="$2";       shift 2 ;;
    --codec)         CODEC="$2";         shift 2 ;;
    --codec-quality) CODEC_QUALITY="$2"; shift 2 ;;
    --comp-ratio)    COMP_RATIO="$2";    shift 2 ;;
    --ldpc-n)        LDPC_N="$2";        shift 2 ;;
    --ldpc-k)        LDPC_K="$2";        shift 2 ;;
    --mod-order)     MOD_ORDER="$2";     shift 2 ;;
    --no-interleave) INTERLEAVE=0;       shift ;;
    --no-warmup)     WARMUP=0;           shift ;;
    --interval)      INTERVAL="$2";      shift 2 ;;
    --launch-fg)     LAUNCH_FG=1;        shift ;;
    --no-wait)       WAIT=0;             shift ;;
    --)              shift; EXTRA+=("$@"); break ;;
    -h|--help)       sed -n '2,41p' "$0"; exit 0 ;;
    -*)              echo "[!] unknown option: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$SNR" ]]; then SNR="$1"; shift
      else echo "[!] unexpected argument: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$SNR" ]]; then
  echo "[!] missing required <SNR_dB> argument" >&2
  echo "    usage: $0 <SNR_dB> [--dataset DIR] [--ldpc-n N] [--ldpc-k N] [--mod-order N] [--launch-fg]" >&2
  exit 2
fi
if [[ ! -d "$DATASET" ]]; then
  echo "[!] dataset folder not found: $DATASET" >&2
  exit 2
fi
case "$MOD_ORDER" in
  1|2|4) ;;
  *) echo "[!] --mod-order must be 1 (BPSK), 2 (QPSK), or 4 (16QAM)" >&2; exit 2 ;;
esac
MOD_NAME="16QAM"; [[ "$MOD_ORDER" -eq 1 ]] && MOD_NAME="BPSK"; [[ "$MOD_ORDER" -eq 2 ]] && MOD_NAME="QPSK"
RATE="$(awk -v n="$LDPC_N" -v k="$LDPC_K" 'BEGIN{printf "%.3f", k/n}')"

# ── optionally launch the GNU Radio TX flowgraph ──────────────────────────────
FG_PID=""
cleanup() {
  if [[ -n "$FG_PID" ]] && kill -0 "$FG_PID" 2>/dev/null; then
    echo "[*] stopping TX flowgraph (pid $FG_PID)..."
    kill "$FG_PID" 2>/dev/null || true
    wait "$FG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$LAUNCH_FG" -eq 1 ]]; then
  if [[ ! -x "$DEMO_ENV/bin/python" ]]; then
    echo "[!] conda env not found at $DEMO_ENV (needed for --launch-fg)" >&2
    exit 2
  fi
  GEN_DIR="$(mktemp -d)"
  echo "[*] compiling $GRC with grcc (env: demo)..."
  DYLD_FALLBACK_LIBRARY_PATH="$DEMO_ENV/lib" \
  PYTHONPATH="$DEMO_ENV/lib/python3.11/site-packages" \
  PATH="$DEMO_ENV/bin:$PATH" \
    "$DEMO_ENV/bin/grcc" "$GRC" -o "$GEN_DIR"
  GEN_PY="$(ls "$GEN_DIR"/*.py | head -n1)"
  echo "[*] launching TX flowgraph: $GEN_PY (mod-order=$MOD_ORDER / $MOD_NAME)"
  DYLD_FALLBACK_LIBRARY_PATH="$DEMO_ENV/lib" \
  PYTHONPATH="$DEMO_ENV/lib/python3.11/site-packages" \
  PATH="$DEMO_ENV/bin:$PATH" \
    "$DEMO_ENV/bin/python" "$GEN_PY" --mod-order "$MOD_ORDER" &
  FG_PID=$!
  echo "[*] TX flowgraph pid $FG_PID — giving it 5s to come up..."
  sleep 5
fi

# ── confirmation gate ─────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  SNR point   : ${SNR} dB"
echo "  dataset     : $DATASET"
echo "  codec       : $CODEC"
echo "  ldpc (n,k)  : ($LDPC_N, $LDPC_K)  rate=$RATE"
echo "  mod order   : $MOD_ORDER ($MOD_NAME)"
echo "  comp ratio  : $COMP_RATIO"
echo "  interleave  : $([[ $INTERLEAVE -eq 1 ]] && echo on || echo off)"
echo "  warmup      : $([[ $WARMUP -eq 1 ]] && echo on || echo off)"
echo "  flowgraph   : $([[ $LAUNCH_FG -eq 1 ]] && echo 'launched by script' || echo 'assumed running')"
echo "════════════════════════════════════════════════════════════════"
echo "  ▸ Set the channel emulator to ${SNR} dB."
echo "  ▸ Start the RX for ${SNR} dB first, e.g.:"
echo "      scripts/run_rx_experiment_conventional.sh ${SNR} --ldpc-n $LDPC_N --ldpc-k $LDPC_K --mod-order $MOD_ORDER"
echo "════════════════════════════════════════════════════════════════"
if [[ "$WAIT" -eq 1 ]]; then
  read -r -p "Press Enter when the emulator is set and the RX is listening... "
fi

# ── transmit the dataset once ─────────────────────────────────────────────────
TX_ARGS=(
  "$ROOT/transmitter/socket_conventional_tx.py"
  --source folder
  --path "$DATASET"
  --codec "$CODEC"
  --comp-ratio "$COMP_RATIO"
  --ldpc-n "$LDPC_N"
  --ldpc-k "$LDPC_K"
  --bits-per-symbol "$MOD_ORDER"
  --interval "$INTERVAL"
)
[[ -n "$CODEC_QUALITY" ]] && TX_ARGS+=(--codec-quality "$CODEC_QUALITY")
[[ "$INTERLEAVE" -eq 1 ]] && TX_ARGS+=(--interleave)
[[ "$WARMUP" -eq 0 ]]     && TX_ARGS+=(--no-warmup)
[[ ${#EXTRA[@]} -gt 0 ]]  && TX_ARGS+=("${EXTRA[@]}")

echo "[*] transmitting dataset at ${SNR} dB ..."
python "${TX_ARGS[@]}"
echo "[*] done — SNR ${SNR} dB complete."
