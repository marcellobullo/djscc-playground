#!/usr/bin/env bash
#
# run_rx_experiment.sh — receive + save a dataset for a single SNR point.
#
# Companion to run_tx_experiment.sh. The SNR is set MANUALLY on the lab channel
# emulator; this script only uses it to name the per-SNR output folder/files and
# (optionally) launch the GNU Radio RX flowgraph. Run it once per SNR, then run
# the TX side at the matching SNR.
#
# Output layout (default):
#   <results>/snr_<SNR>dB/<prefix>_001.png, _002.png, ...
# where <prefix> defaults to "snr<SNR>".
#
# Usage:
#   scripts/run_rx_experiment.sh <SNR_dB> [options]
#
# Examples:
#   # flowgraph already running (default); receive 50 images at 10 dB
#   scripts/run_rx_experiment.sh 10 --count 50 --results ./results
#
#   # also launch the GNU Radio RX flowgraph for this run
#   scripts/run_rx_experiment.sh 10 --count 50 --launch-fg
#
# Options:
#   --results DIR     parent results folder; output goes to DIR/snr_<SNR>dB
#                                                         (default: ./results)
#   --prefix NAME     filename prefix          (default: snr<SNR>)
#   --count N         stop after N unique images (default: 0 = run until Ctrl-C)
#   --model NAME      HF id / alias / .pth for the decoder
#                       (default: marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi)
#   --aligner NAME    use the named aligner    (default: none / disabled)
#   --no-interleave   disable de-interleaving (default: ON, must match TX)
#   --launch-fg       compile + run djscc_rx.grc in conda env `demo` for this run
#   --keep-timestamp  keep the timestamp in filenames (default: clean sequential)
#   --                pass any remaining args straight to socket_rx.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi"
RESULTS="$ROOT/results"
PREFIX=""
COUNT=0
ALIGNER=""           # empty = no aligner
INTERLEAVE=1
LAUNCH_FG=0
TIMESTAMP=0          # 0 = clean sequential names (--no-timestamp on RX)
SNR=""
EXTRA=()

GRC="$ROOT/receiver/gnu_radio/djscc_rx.grc"
DEMO_ENV="$HOME/miniconda3/envs/demo"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results)        RESULTS="$2"; shift 2 ;;
    --prefix)         PREFIX="$2";  shift 2 ;;
    --count)          COUNT="$2";   shift 2 ;;
    --model)          MODEL="$2";   shift 2 ;;
    --aligner)        ALIGNER="$2"; shift 2 ;;
    --no-interleave)  INTERLEAVE=0; shift ;;
    --launch-fg)      LAUNCH_FG=1;  shift ;;
    --keep-timestamp) TIMESTAMP=1;  shift ;;
    --)               shift; EXTRA+=("$@"); break ;;
    -h|--help)        sed -n '2,36p' "$0"; exit 0 ;;
    -*)               echo "[!] unknown option: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$SNR" ]]; then SNR="$1"; shift
      else echo "[!] unexpected argument: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$SNR" ]]; then
  echo "[!] missing required <SNR_dB> argument" >&2
  echo "    usage: $0 <SNR_dB> [--results DIR] [--count N] [--launch-fg]" >&2
  exit 2
fi
[[ -z "$PREFIX" ]] && PREFIX="snr${SNR}"
OUTDIR="$RESULTS/snr_${SNR}dB"

# ── optionally launch the GNU Radio RX flowgraph ──────────────────────────────
FG_PID=""
cleanup() {
  if [[ -n "$FG_PID" ]] && kill -0 "$FG_PID" 2>/dev/null; then
    echo "[*] stopping RX flowgraph (pid $FG_PID)..."
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
  echo "[*] launching RX flowgraph: $GEN_PY"
  DYLD_FALLBACK_LIBRARY_PATH="$DEMO_ENV/lib" \
  PYTHONPATH="$DEMO_ENV/lib/python3.11/site-packages" \
  PATH="$DEMO_ENV/bin:$PATH" \
    "$DEMO_ENV/bin/python" "$GEN_PY" &
  FG_PID=$!
  echo "[*] RX flowgraph pid $FG_PID — giving it 5s to come up..."
  sleep 5
fi

# ── banner ────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  SNR point   : ${SNR} dB"
echo "  output dir  : $OUTDIR"
echo "  name prefix : ${PREFIX}  ->  ${PREFIX}_001$([[ $TIMESTAMP -eq 1 ]] && echo '_<stamp>').png"
echo "  model       : $MODEL"
echo "  aligner     : ${ALIGNER:-none}"
echo "  interleave  : $([[ $INTERLEAVE -eq 1 ]] && echo on || echo off)"
echo "  stop after  : $([[ $COUNT -gt 0 ]] && echo "${COUNT} images" || echo 'Ctrl-C / idle timeout')"
echo "  flowgraph   : $([[ $LAUNCH_FG -eq 1 ]] && echo 'launched by script' || echo 'assumed running')"
echo "════════════════════════════════════════════════════════════════"
echo "  ▸ Make sure the channel emulator is set to ${SNR} dB, then start the TX."
echo "════════════════════════════════════════════════════════════════"

# ── receive ───────────────────────────────────────────────────────────────────
RX_ARGS=(
  "$ROOT/receiver/socket_rx.py"
  --model "$MODEL"
  --output-dir "$OUTDIR"
  --name-prefix "$PREFIX"
  --count "$COUNT"
)
[[ -n "$ALIGNER" ]]       && RX_ARGS+=(--aligner "$ALIGNER")
[[ "$INTERLEAVE" -eq 1 ]] && RX_ARGS+=(--interleave)
[[ "$TIMESTAMP" -eq 0 ]] && RX_ARGS+=(--no-timestamp)
[[ ${#EXTRA[@]} -gt 0 ]] && RX_ARGS+=("${EXTRA[@]}")

echo "[*] receiving at ${SNR} dB -> $OUTDIR ..."
python "${RX_ARGS[@]}"
echo "[*] done — SNR ${SNR} dB saved to $OUTDIR"
