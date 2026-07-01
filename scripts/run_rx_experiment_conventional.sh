#!/usr/bin/env bash
#
# run_rx_experiment_conventional.sh — receive + save a dataset for a single
# (SNR, LDPC rate, modulation) point, using the conventional (JPEG/JPEG2000 +
# LDPC + OFDM) baseline.
#
# Companion to run_tx_experiment_conventional.sh. The SNR is set MANUALLY on
# the lab channel emulator; this script uses it (together with the LDPC rate
# and modulation order) to name the per-run output folder and — when
# --launch-fg is used — to seed the soft LDPC decoder's noise-power estimate
# (npwr = 10**(-SNR/10)) inside the flowgraph. Run it once per (SNR, ldpc-n,
# ldpc-k, mod-order) combination, then run the matching TX.
#
# Note: unlike the DJSCC socket_rx.py, socket_conventional_rx.py has no
# --count flag — it saves one image per received transmission and stops on
# Ctrl-C or after --timeout seconds of inactivity (0 = never).
#
# Output layout (default):
#   <results>/snr_<SNR>dB/ldpc_n<N>_k<K>_mod<MOD>/
# (with --exp-id-mode: image_<order>.png + manifest.csv in that folder)
#
# Usage:
#   scripts/run_rx_experiment_conventional.sh <SNR_dB> [options]
#
# Examples:
#   # flowgraph already running (default); rate-1/2, BPSK, JPEG
#   scripts/run_rx_experiment_conventional.sh 10 --ldpc-n 2304 --ldpc-k 1152 \
#       --mod-order 1 --exp-id-mode --timeout 30
#
#   # also launch the GNU Radio RX flowgraph for this run (needed whenever
#   # --mod-order or SNR changes, since both are baked in at construction)
#   scripts/run_rx_experiment_conventional.sh 10 --ldpc-n 2304 --ldpc-k 1152 \
#       --mod-order 1 --launch-fg
#
# Options:
#   --results DIR       parent results folder      (default: ./results)
#   --output-dir DIR    exact output dir, overrides --results-derived path
#   --codec NAME         jpeg | jpeg2000              (default: jpeg)
#   --comp-ratio N       DJSCC-equivalent inverse compression ratio (default: 6)
#   --ldpc-n N           LDPC codeword length n       (default: 2304)
#   --ldpc-k N           LDPC message length k        (default: 1152)
#   --bp-iters N         belief-propagation iterations (default: 50)
#   --demap NAME         soft | hard                  (default: soft)
#   --mod-order N        bits/symbol: 1=BPSK 2=QPSK 4=16QAM (default: 1)
#   --device NAME        auto | cpu | mps | cuda       (default: cpu)
#   --no-interleave      disable de-interleaving (default: ON, must match TX)
#   --exp-id-mode        save exactly one PNG per transmitted image
#                          (image_<order>.png) + manifest.csv (recommended
#                          for sweeps; pair with TX's --no-warmup)
#   --timeout SEC        auto-exit after N seconds with no new image (default: 0 = never)
#   --launch-fg          compile + run conventional_rx.grc in conda env `demo`
#                          for this run, with --mod-order/--snr-db matching
#   --                    pass any remaining args straight to socket_conventional_rx.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
RESULTS="$ROOT/results"
OUTDIR_OVERRIDE=""
CODEC="jpeg"
COMP_RATIO=6
LDPC_N=2304
LDPC_K=1152
BP_ITERS=50
DEMAP="soft"
MOD_ORDER=1
DEVICE="cpu"
INTERLEAVE=1
EXP_ID_MODE=0
TIMEOUT=0
LAUNCH_FG=0
SNR=""
EXTRA=()

# GNU Radio flowgraph (only used with --launch-fg)
GRC="$ROOT/receiver/gnu_radio/conventional_rx.grc"
DEMO_ENV="$HOME/miniconda3/envs/demo"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results)        RESULTS="$2";        shift 2 ;;
    --output-dir)      OUTDIR_OVERRIDE="$2"; shift 2 ;;
    --codec)           CODEC="$2";          shift 2 ;;
    --comp-ratio)      COMP_RATIO="$2";     shift 2 ;;
    --ldpc-n)          LDPC_N="$2";         shift 2 ;;
    --ldpc-k)          LDPC_K="$2";         shift 2 ;;
    --bp-iters)        BP_ITERS="$2";       shift 2 ;;
    --demap)           DEMAP="$2";          shift 2 ;;
    --mod-order)       MOD_ORDER="$2";      shift 2 ;;
    --device)          DEVICE="$2";         shift 2 ;;
    --no-interleave)   INTERLEAVE=0;        shift ;;
    --exp-id-mode)     EXP_ID_MODE=1;       shift ;;
    --timeout)         TIMEOUT="$2";        shift 2 ;;
    --launch-fg)       LAUNCH_FG=1;         shift ;;
    --)                shift; EXTRA+=("$@"); break ;;
    -h|--help)         sed -n '2,53p' "$0"; exit 0 ;;
    -*)                echo "[!] unknown option: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$SNR" ]]; then SNR="$1"; shift
      else echo "[!] unexpected argument: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$SNR" ]]; then
  echo "[!] missing required <SNR_dB> argument" >&2
  echo "    usage: $0 <SNR_dB> [--ldpc-n N] [--ldpc-k N] [--mod-order N] [--launch-fg]" >&2
  exit 2
fi
case "$MOD_ORDER" in
  1|2|4) ;;
  *) echo "[!] --mod-order must be 1 (BPSK), 2 (QPSK), or 4 (16QAM)" >&2; exit 2 ;;
esac
MOD_NAME="16QAM"; [[ "$MOD_ORDER" -eq 1 ]] && MOD_NAME="BPSK"; [[ "$MOD_ORDER" -eq 2 ]] && MOD_NAME="QPSK"
RATE="$(awk -v n="$LDPC_N" -v k="$LDPC_K" 'BEGIN{printf "%.3f", k/n}')"

if [[ -n "$OUTDIR_OVERRIDE" ]]; then
  OUTDIR="$OUTDIR_OVERRIDE"
else
  OUTDIR="$RESULTS/snr_${SNR}dB/ldpc_n${LDPC_N}_k${LDPC_K}_mod${MOD_ORDER}"
fi

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
  echo "[*] launching RX flowgraph: $GEN_PY (mod-order=$MOD_ORDER / $MOD_NAME, snr-db=$SNR)"
  DYLD_FALLBACK_LIBRARY_PATH="$DEMO_ENV/lib" \
  PYTHONPATH="$DEMO_ENV/lib/python3.11/site-packages" \
  PATH="$DEMO_ENV/bin:$PATH" \
    "$DEMO_ENV/bin/python" "$GEN_PY" --mod-order "$MOD_ORDER" --snr-db "$SNR" &
  FG_PID=$!
  echo "[*] RX flowgraph pid $FG_PID — giving it 5s to come up..."
  sleep 5
fi

# ── banner ────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  SNR point   : ${SNR} dB"
echo "  output dir  : $OUTDIR"
echo "  codec       : $CODEC"
echo "  ldpc (n,k)  : ($LDPC_N, $LDPC_K)  rate=$RATE"
echo "  bp-iters    : $BP_ITERS"
echo "  demap       : $DEMAP"
echo "  mod order   : $MOD_ORDER ($MOD_NAME)"
echo "  comp ratio  : $COMP_RATIO"
echo "  device      : $DEVICE"
echo "  interleave  : $([[ $INTERLEAVE -eq 1 ]] && echo on || echo off)"
echo "  exp-id-mode : $([[ $EXP_ID_MODE -eq 1 ]] && echo on || echo off)"
echo "  stop after  : $([[ $TIMEOUT != 0 ]] && echo "${TIMEOUT}s idle" || echo 'Ctrl-C')"
echo "  flowgraph   : $([[ $LAUNCH_FG -eq 1 ]] && echo 'launched by script' || echo 'assumed running')"
echo "════════════════════════════════════════════════════════════════"
echo "  ▸ Make sure the channel emulator is set to ${SNR} dB, then start the TX."
echo "════════════════════════════════════════════════════════════════"

# ── receive ───────────────────────────────────────────────────────────────────
RX_ARGS=(
  "$ROOT/receiver/socket_conventional_rx.py"
  --codec "$CODEC"
  --comp-ratio "$COMP_RATIO"
  --ldpc-n "$LDPC_N"
  --ldpc-k "$LDPC_K"
  --bp-iters "$BP_ITERS"
  --bits-per-symbol "$MOD_ORDER"
  --demap "$DEMAP"
  --device "$DEVICE"
  --output-dir "$OUTDIR"
  --timeout "$TIMEOUT"
)
[[ "$INTERLEAVE" -eq 1 ]]  && RX_ARGS+=(--interleave)
[[ "$EXP_ID_MODE" -eq 1 ]] && RX_ARGS+=(--exp-id-mode)
[[ ${#EXTRA[@]} -gt 0 ]]   && RX_ARGS+=("${EXTRA[@]}")

echo "[*] receiving at ${SNR} dB -> $OUTDIR ..."
python "${RX_ARGS[@]}"
echo "[*] done — SNR ${SNR} dB saved to $OUTDIR"
