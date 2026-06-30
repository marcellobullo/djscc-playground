#!/usr/bin/env bash
#
# run_tx_experiment.sh — transmit a dataset folder once, for a single SNR point.
#
# The SNR is set MANUALLY on the lab channel emulator; this script only labels
# the run and (optionally) launches the GNU Radio TX flowgraph. Re-run it once
# per SNR after re-setting the emulator and starting the matching RX.
#
# Usage:
#   scripts/run_tx_experiment.sh <SNR_dB> [options]
#
# Examples:
#   # flowgraph already running (default); transmit ./dataset at 10 dB
#   scripts/run_tx_experiment.sh 10 --dataset ./dataset
#
#   # also launch the GNU Radio TX flowgraph for this run
#   scripts/run_tx_experiment.sh 10 --dataset ./dataset --launch-fg
#
# Options:
#   --dataset DIR     folder of images to transmit            (default: ./dataset)
#   --model NAME      HF id / alias / .pth for the encoder
#                       (default: marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi)
#   --no-interleave   disable symbol interleaving (default: ON, must match RX)
#   --interval SEC    seconds between images                  (default: 3.0)
#   --launch-fg       compile + run djscc_tx.grc in conda env `demo` for this run
#   --no-wait         don't pause for the "set emulator + start RX" confirmation
#   --                pass any remaining args straight to socket_tx.py
#
set -euo pipefail

# ── resolve repo root (script lives in scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi"
DATASET="$ROOT/dataset"
INTERLEAVE=1
INTERVAL=3.0
LAUNCH_FG=0
WAIT=1
SNR=""
EXTRA=()

# GNU Radio flowgraph (only used with --launch-fg)
GRC="$ROOT/transmitter/gnu_radio/djscc_tx.grc"
DEMO_ENV="$HOME/miniconda3/envs/demo"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)       DATASET="$2"; shift 2 ;;
    --model)         MODEL="$2";   shift 2 ;;
    --interval)      INTERVAL="$2"; shift 2 ;;
    --no-interleave) INTERLEAVE=0; shift ;;
    --launch-fg)     LAUNCH_FG=1;  shift ;;
    --no-wait)       WAIT=0;       shift ;;
    --)              shift; EXTRA+=("$@"); break ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    -*)              echo "[!] unknown option: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$SNR" ]]; then SNR="$1"; shift
      else echo "[!] unexpected argument: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$SNR" ]]; then
  echo "[!] missing required <SNR_dB> argument" >&2
  echo "    usage: $0 <SNR_dB> [--dataset DIR] [--model NAME] [--launch-fg]" >&2
  exit 2
fi
if [[ ! -d "$DATASET" ]]; then
  echo "[!] dataset folder not found: $DATASET" >&2
  exit 2
fi

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
  echo "[*] launching TX flowgraph: $GEN_PY"
  DYLD_FALLBACK_LIBRARY_PATH="$DEMO_ENV/lib" \
  PYTHONPATH="$DEMO_ENV/lib/python3.11/site-packages" \
  PATH="$DEMO_ENV/bin:$PATH" \
    "$DEMO_ENV/bin/python" "$GEN_PY" &
  FG_PID=$!
  echo "[*] TX flowgraph pid $FG_PID — giving it 5s to come up..."
  sleep 5
fi

# ── confirmation gate ─────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  SNR point   : ${SNR} dB"
echo "  dataset     : $DATASET"
echo "  model       : $MODEL"
echo "  interleave  : $([[ $INTERLEAVE -eq 1 ]] && echo on || echo off)"
echo "  flowgraph   : $([[ $LAUNCH_FG -eq 1 ]] && echo 'launched by script' || echo 'assumed running')"
echo "════════════════════════════════════════════════════════════════"
echo "  ▸ Set the channel emulator to ${SNR} dB."
echo "  ▸ Start the RX for ${SNR} dB first, e.g.:"
echo "      scripts/run_rx_experiment.sh ${SNR} --count <N>"
echo "════════════════════════════════════════════════════════════════"
if [[ "$WAIT" -eq 1 ]]; then
  read -r -p "Press Enter when the emulator is set and the RX is listening... "
fi

# ── transmit the dataset once ─────────────────────────────────────────────────
TX_ARGS=(
  "$ROOT/transmitter/socket_tx.py"
  --model "$MODEL"
  --source folder
  --path "$DATASET"
  --interval "$INTERVAL"
)
[[ "$INTERLEAVE" -eq 1 ]] && TX_ARGS+=(--interleave)
[[ ${#EXTRA[@]} -gt 0 ]] && TX_ARGS+=("${EXTRA[@]}")

echo "[*] transmitting dataset at ${SNR} dB ..."
python "${TX_ARGS[@]}"
echo "[*] done — SNR ${SNR} dB complete."
