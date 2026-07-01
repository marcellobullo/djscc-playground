#!/usr/bin/env python3
"""
Conventional (JPEG/JPEG2000 + LDPC) Image Transmitter — companion to
``image_tx_encoded.py`` for fair PSNR-vs-SNR comparison against DeepJSCC.

Same capture UX (camera / file / folder, interactive or batch). Differences:

  * Source coding is classical (PIL JPEG2000 or JPEG) instead of a neural
    encoder.
  * Channel coding is LDPC via Kaira's RPTU code database.
  * Publishes **uint8 bytes** to ZMQ (default port 5557) — NOT complex
    symbols. The companion GR flowgraph (``transmitter/gnu_radio/
    tx_conventional.grc``) takes the bytes through ``repack_bits`` and
    ``chunks_to_symbols`` (modulation), then OFDM + USRP.

Split of responsibilities (matches the user's preference: maximum
flexibility, source + channel coding fully CLI-driven here, modulation
handled in the .grc):

  Python here          .grc
  ─────────────         ─────────────────────────────────────────
  capture frame  →      —
  JPEG / J2K     →      —
  LDPC encode    →      —
  byte-align     →  ZMQ source (byte) → repack_bits → chunks_to_symbols
                        → stream_to_tagged_stream → OFDM allocator → FFT
                        → cyclic prefix → multiply_const → USRP

The Python tool needs the modulation order ``M`` only to compute the
byte budget that, after LDPC and modulation, produces ``N_target``
complex symbols at the .grc output. **You are responsible for keeping
``--bits-per-symbol`` here in sync with ``repack`` and ``payload_mod``
in the .grc.** The script logs an explicit reminder on startup.

DJSCC channel-use equivalence (image 768×512×3):

  comp_ratio | N_target  | (mod, LDPC)            | JPEG byte budget
  -----------|-----------|------------------------|------------------
  1/6        | 196,608   | QPSK + (960, 640) 2/3  | 32,768
  1/6        | 196,608   | QPSK + (1920, 960) 1/2 | 24,576
  1/12       |  98,304   | QPSK + (960, 640) 2/3  | 16,384

Examples::

  # Default: QPSK + LDPC 2/3 + JPEG2000 at comp_ratio 1/6 (matches
  # AWGN_rate_16 DJSCC checkpoint's channel use).
  python image_tx_encoded_conventional.py --comp-ratio 6

  # 16QAM + LDPC 1/2 + JPEG, file source.
  python image_tx_encoded_conventional.py --comp-ratio 6 \\
      --codec jpeg --bits-per-symbol 4 --ldpc-n 1920 --ldpc-k 960 \\
      --source file --path photo.jpg
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pmt
import torch
import zmq
import scipy.sparse
from PIL import Image

import ldpc.code_util

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from utils.parity_matrix_helper import get_generator_and_info_bits


# ── UI constants ────────────────────────────────────────────────────────────
BTN_COLOR_IDLE    = (34, 139, 34)    # green
BTN_COLOR_HOVER   = (0, 200, 0)      # bright green
BTN_COLOR_ACTIVE  = (0, 80, 200)     # blue — "sending"
BTN_COLOR_DONE    = (0, 180, 180)    # cyan — brief "sent!" flash
OVERLAY_ALPHA     = 0.55
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
# Config
# =====================================================================
@dataclass
class TxConventionalConfig:
    # capture / image
    width: int
    height: int
    channel: int

    # source coding
    codec: str               # "jpeg2000" | "jpeg"
    codec_quality: int       # JPEG: quality 1..95; J2K: compression ratio (raw/target)
    target_bytes: int        # explicit budget; 0 = derive from layout
    fit_to_budget: bool      # True: rate-target the codec to ≤ budget; False: free quality

    # channel coding
    ldpc_n: int              # codeword length (RPTU database)
    ldpc_k: int              # message length

    # modulation hint (used only for byte-budget arithmetic)
    bits_per_symbol: int
    target_complex_symbols: int
    packet_len: int

    # transmission
    repeat: int
    repeat_interval: float
    warmup_frames: int
    warmup_interval: float

    # zmq
    port: str
    bind_host: str
    topic: bytes

    # debug: skip GR — PUSH PDUs straight into the RX's PULL on port 5559
    direct_zmq: bool

    # interleaver: spread LDPC codeword bits across PDU slots so a lost slot
    # erases ~n/num_slots bits per LDPC block instead of whole blocks.
    # Must match the RX --interleave flag.
    interleave: bool


# =====================================================================
# Layout math — pick LDPC block count + budgets that align with OFDM packets
# =====================================================================
@dataclass
class Layout:
    num_ldpc_blocks: int
    jpeg_budget_bytes: int       # bytes the source coder must hit (will pad up to this)
    ldpc_input_bytes: int        # = num_blocks * (k/8)
    ldpc_output_bytes: int       # = num_blocks * (n/8); this is what's published
    output_complex_symbols: int  # what the .grc will emit per image
    packet_align_bytes: int      # = packet_len * bits_per_symbol / 8


def compute_layout(cfg: TxConventionalConfig) -> Layout:
    """Pick the smallest LDPC block count whose codeword stream aligns to OFDM
    packets. Forces a clean N_complex multiple of packet_len at the radio side."""
    if cfg.ldpc_n % 8 or cfg.ldpc_k % 8:
        raise ValueError(
            f"LDPC (n,k)=({cfg.ldpc_n},{cfg.ldpc_k}) must both be multiples of 8")

    bytes_per_codeword = cfg.ldpc_n // 8
    bytes_per_message  = cfg.ldpc_k // 8
    packet_align_bytes = cfg.packet_len * cfg.bits_per_symbol // 8

    # Smallest #blocks that produces ≥ target_complex_symbols channel uses.
    target_bits_after_ldpc = cfg.target_complex_symbols * cfg.bits_per_symbol
    num_blocks_min = (target_bits_after_ldpc + cfg.ldpc_n - 1) // cfg.ldpc_n

    num_blocks = num_blocks_min
    # Round up so total LDPC byte stream is a multiple of an OFDM packet.
    while (num_blocks * bytes_per_codeword) % packet_align_bytes != 0:
        num_blocks += 1

    ldpc_input_bytes  = num_blocks * bytes_per_message
    ldpc_output_bytes = num_blocks * bytes_per_codeword
    output_complex_symbols = ldpc_output_bytes * 8 // cfg.bits_per_symbol

    jpeg_budget_bytes = (cfg.target_bytes if cfg.target_bytes > 0
                         else ldpc_input_bytes)
    if jpeg_budget_bytes > ldpc_input_bytes:
        raise ValueError(
            f"--target-bytes={jpeg_budget_bytes} exceeds the LDPC input window "
            f"of {ldpc_input_bytes} bytes for {num_blocks} blocks of "
            f"({cfg.ldpc_n},{cfg.ldpc_k}). Increase target_complex_symbols or "
            f"reduce target_bytes.")

    return Layout(
        num_ldpc_blocks=num_blocks,
        jpeg_budget_bytes=jpeg_budget_bytes,
        ldpc_input_bytes=ldpc_input_bytes,
        ldpc_output_bytes=ldpc_output_bytes,
        output_complex_symbols=output_complex_symbols,
        packet_align_bytes=packet_align_bytes,
    )


# =====================================================================
# Source codec — fit a budget
# =====================================================================
def _compress_jpeg2000(img_pil: Image.Image, raw_bytes: int,
                       budget_bytes: int) -> bytes:
    ratio = max(1.0, raw_bytes / budget_bytes)
    buf = io.BytesIO()
    img_pil.save(
        buf, "JPEG2000",
        quality_mode="rates",
        quality_layers=[ratio],
        irreversible=True,
        codeblock_size=(32, 32),
        precinct_size=[(128, 128)],
        progression="RPCL",
        num_resolutions=3,
        mct=1,
        mode="tiled",
    )
    data = buf.getvalue()
    print(f"  Compressed JPEG2000 to {len(data)} bytes with ratio {ratio:.2f} "
          f"for budget {budget_bytes} bytes.")
    return data


def _compress_jpeg(img_pil: Image.Image, budget_bytes: int,
                   default_quality: int) -> bytes:
    """Binary-search JPEG quality 1..95 to land at the largest payload ≤ budget."""
    if budget_bytes <= 0:
        return b""

    lo, hi = 1, 95
    best: Optional[bytes] = None
    for _ in range(8):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        img_pil.save(buf, "JPEG", quality=mid, optimize=True)
        data = buf.getvalue()
        if len(data) <= budget_bytes:
            best = data
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        # Even quality=1 didn't fit: take it and let LDPC pad eat the rest.
        buf = io.BytesIO()
        img_pil.save(buf, "JPEG", quality=1, optimize=True)
        best = buf.getvalue()
        if len(best) > budget_bytes:
            print(f"[!] JPEG quality=1 produces {len(best)} bytes (>{budget_bytes}); "
                  "image will not fit the channel-use budget and will be truncated.")
            best = best[:budget_bytes]
    else:
        print(f"  Found JPEG quality={mid} with {len(best)} bytes "
              f"for budget {budget_bytes} bytes.")
    return best


def compress_to_budget(img_bgr: np.ndarray, cfg: TxConventionalConfig,
                       budget_bytes: int) -> bytes:
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if img_pil.size != (cfg.width, cfg.height):
        img_pil = img_pil.resize((cfg.width, cfg.height))

    raw_bytes = len(img_pil.tobytes())
    #print(f"  Raw image size: {raw_bytes} bytes. Codec budget: {budget_bytes} bytes.")
    #raw_bytes_computed = cfg.width * cfg.height * 3
    #print(f"  Computed raw bytes: {raw_bytes_computed} bytes.")

    if not cfg.fit_to_budget:
        # Free-quality mode: encode at user quality, no rate target.
        buf = io.BytesIO()
        if cfg.codec == "jpeg2000":
            img_pil.save(buf, "JPEG2000",
                         quality_mode="rates",
                         quality_layers=[cfg.codec_quality],
                         irreversible=True, mct=1)
        else:
            img_pil.save(buf, "JPEG", quality=cfg.codec_quality, optimize=True)
        return buf.getvalue()

    if cfg.codec == "jpeg2000":
        return _compress_jpeg2000(img_pil, raw_bytes, budget_bytes)
    elif cfg.codec == "jpeg":
        return _compress_jpeg(img_pil, budget_bytes, cfg.codec_quality)
    raise ValueError(f"Unknown codec: {cfg.codec}")


# =====================================================================
# Channel codec — Kaira LDPC, batched
# =====================================================================
# Block interleaver — shared with image_rx_decoded_conventional.py
# =====================================================================
def _make_interleaver(num_bits: int, num_slots: int):
    """Return (perm_fwd, perm_inv) for a block interleaver over num_bits bits.

    The bits are laid out as a (num_slots × bits_per_slot) matrix filled
    row-by-row, then read column-by-column.  Each slot of bits_per_slot
    consecutive TX positions therefore contains ~n/num_slots bits from every
    LDPC block, so a single lost PDU slot only erases ~1/num_slots of each
    block rather than wiping whole blocks.

      perm_fwd[ldpc_pos] = tx_pos   — apply at RX: ldpc_llrs = rx_llrs[perm_fwd]
      perm_inv[tx_pos]   = ldpc_pos — apply at TX: tx_bits   = ldpc_bits[perm_inv]
    """
    bps = num_bits // num_slots  # bits per slot
    p = np.arange(num_bits, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_bits, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


# =====================================================================
class LDPCBatchEncoder:
    def __init__(self, ldpc_n: int, ldpc_k: int):
        self.n = ldpc_n
        self.k = ldpc_k
        
        # Get systematic generator matrix directly
        G, _ = get_generator_and_info_bits(ldpc_n, ldpc_k)
        self.G = G.astype(np.uint16)

    def encode_bytes(self, padded_msg: np.ndarray) -> np.ndarray:
        """``padded_msg`` length must equal num_blocks * (k/8). Returns
        num_blocks * (n/8) bytes."""
        if padded_msg.dtype != np.uint8:
            padded_msg = padded_msg.astype(np.uint8)

        bytes_per_msg = self.k // 8
        if len(padded_msg) % bytes_per_msg:
            raise ValueError(
                f"LDPC input of {len(padded_msg)} bytes is not a multiple of "
                f"k/8 = {bytes_per_msg}")

        num_blocks = len(padded_msg) // bytes_per_msg
        bits = np.unpackbits(padded_msg, bitorder='little').reshape(num_blocks, self.k)

        # Encode all blocks at once using vectorized matrix multiplication.
        # Use uint16 to safely avoid overflow before the modulo 2 operation.
        encoded_bits = (bits.astype(np.uint16) @ self.G) % 2
        encoded_bits = encoded_bits.astype(np.uint8)

        out_bits = encoded_bits.flatten()
        return np.packbits(out_bits, bitorder='little')


# =====================================================================
# Top-level encoder
# =====================================================================
class ConventionalEncoder:
    def __init__(self, cfg: TxConventionalConfig):
        self.cfg = cfg
        self.layout = compute_layout(cfg)
        self.ldpc = LDPCBatchEncoder(cfg.ldpc_n, cfg.ldpc_k)
        self._encoded_warmup: Optional[np.ndarray] = None
        self._perm_inv: Optional[np.ndarray] = None  # built lazily after first encode

    def _get_perm_inv(self, num_bits: int, num_slots: int) -> np.ndarray:
        if self._perm_inv is None:
            _, self._perm_inv = _make_interleaver(num_bits, num_slots)
        return self._perm_inv

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR ndarray -> uint8 ndarray of length layout.ldpc_output_bytes."""
        L = self.layout

        src = compress_to_budget(frame_bgr, self.cfg, L.jpeg_budget_bytes)
        src_arr = np.frombuffer(src, dtype=np.uint8)

        if len(src_arr) > L.ldpc_input_bytes:
            # Free-quality mode (or oversized JPEG) — truncate to fit.
            src_arr = src_arr[:L.ldpc_input_bytes]

        padded = np.zeros(L.ldpc_input_bytes, dtype=np.uint8)
        padded[:len(src_arr)] = src_arr

        out = self.ldpc.encode_bytes(padded)

        if len(out) != L.ldpc_output_bytes:
            raise RuntimeError(
                f"LDPC produced {len(out)} bytes, expected {L.ldpc_output_bytes}")

        if self.cfg.interleave:
            pkt_bytes = (self.cfg.packet_len * self.cfg.bits_per_symbol) // 8
            num_slots = len(out) // pkt_bytes
            num_bits  = len(out) * 8
            perm_inv  = self._get_perm_inv(num_bits, num_slots)
            bits = np.unpackbits(out, bitorder='little')
            bits = bits[perm_inv]
            out  = np.packbits(bits, bitorder='little')

        _dump = os.environ.get("DJSCC_TX_DUMP")
        if _dump and not getattr(self, "_dumped", False):
            np.save(_dump, out)
            print(f"[dbg] TX dumped LDPC-encoded bytes to {_dump} "
                  f"({len(out)} bytes, {len(out)*8} bits, "
                  f"{L.num_ldpc_blocks} blocks of n={self.cfg.ldpc_n})")
            self._dumped = True

        return out

    def encode_dummy(self) -> np.ndarray:
        if self._encoded_warmup is None:
            black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
            self._encoded_warmup = self.encode(black)
        return self._encoded_warmup


# =====================================================================
# ZMQ publish
# =====================================================================
def publish_bytes(socket: zmq.Socket, topic: bytes,
                  payload: np.ndarray, label: str = "") -> None:
    assert payload.dtype == np.uint8, f"expected uint8, got {payload.dtype}"
    raw = payload.tobytes()
    if topic:
        socket.send_multipart([topic, raw])
    else:
        socket.send(raw)
    tag = f" [{label}]" if label else ""
    print(f"  TX publish {len(payload)} bytes ({len(raw)/1024:.1f} KiB){tag}")


# Direct-ZMQ debug path: skips GR entirely. The RX expects a stream of PMT
# PDUs (one per OFDM-equivalent slot) carrying a `packet_num` long in the
# metadata dict — exactly what GR's `tagged_stream_to_pdu` produces. We
# synthesize the same shape here: chunk the LDPC byte stream into
# (packet_len * bits_per_symbol / 8)-sized slots, wrap each as a u8vector
# PDU, and PUSH it. Pair with: `image_rx_decoded_conventional.py
# --demap hard --port 5559`.
_DIRECT_PACKET_COUNTER = [0]

def publish_bytes_pdu(socket: zmq.Socket, topic: bytes,
                      payload: np.ndarray, cfg: TxConventionalConfig,
                      label: str = "") -> None:
    assert payload.dtype == np.uint8, f"expected uint8, got {payload.dtype}"
    pkt_bytes = (cfg.packet_len * cfg.bits_per_symbol) // 8
    if len(payload) % pkt_bytes:
        raise ValueError(
            f"payload {len(payload)} bytes is not a multiple of "
            f"packet_len*M/8 = {pkt_bytes}; the layout calculator should have "
            "guaranteed this — check cfg.packet_len / cfg.bits_per_symbol")
    n_pkts = len(payload) // pkt_bytes
    pn_start = _DIRECT_PACKET_COUNTER[0]

    # Pre-serialize all PDUs before sending so the hot send loop has no
    # Python/PMT overhead per packet.  Without this, PMT object creation
    # on macOS can stall the sender thread long enough (>idle_gap_s) that
    # the RX flushes a partial image mid-transmission.
    serialized = []
    for i in range(n_pkts):
        chunk = payload[i * pkt_bytes:(i + 1) * pkt_bytes]
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern("packet_num"),
                            pmt.from_long(_DIRECT_PACKET_COUNTER[0]))
        vec = pmt.init_u8vector(len(chunk), chunk.tolist())
        pdu = pmt.cons(meta, vec)
        serialized.append(pmt.serialize_str(pdu))
        _DIRECT_PACKET_COUNTER[0] += 1

    for s in serialized:
        socket.send(s)
    tag = f" [{label}]" if label else ""
    print(f"  TX direct-PDU push {n_pkts} pkts × {pkt_bytes} B "
          f"(pn {pn_start}..{_DIRECT_PACKET_COUNTER[0] - 1}, "
          f"{len(payload) / 1024:.1f} KiB){tag}")


def encode_and_send(encoder: ConventionalEncoder, socket: zmq.Socket,
                    cfg: TxConventionalConfig, frame_bgr: np.ndarray,
                    label: str = "") -> None:
    t0 = time.time()
    payload = encoder.encode(frame_bgr)
    enc_ms = (time.time() - t0) * 1000.0
    print(f"  Conventional encode {len(payload)} bytes in {enc_ms:.1f} ms"
          + (f" [{label}]" if label else ""))

    for r in range(cfg.repeat):
        rep_label = (f"{label} {r+1}/{cfg.repeat}" if label
                     else f"{r+1}/{cfg.repeat}")
        if cfg.direct_zmq:
            publish_bytes_pdu(socket, cfg.topic, payload, cfg, label=rep_label)
        else:
            publish_bytes(socket, cfg.topic, payload, label=rep_label)
        if r < cfg.repeat - 1:
            time.sleep(cfg.repeat_interval)


def send_warmup(encoder: ConventionalEncoder, socket: zmq.Socket,
                cfg: TxConventionalConfig) -> None:
    if cfg.warmup_frames <= 0:
        print("[*] Warmup skipped.")
        return
    print(f"[*] Sending {cfg.warmup_frames} warmup frames for OFDM sync...")
    payload = encoder.encode_dummy()
    for i in range(cfg.warmup_frames):
        if cfg.direct_zmq:
            publish_bytes_pdu(socket, cfg.topic, payload, cfg,
                              label=f"warmup {i+1}")
        else:
            publish_bytes(socket, cfg.topic, payload, label=f"warmup {i+1}")
        time.sleep(cfg.warmup_interval)
    print("[*] Warmup complete.")


# =====================================================================
# Interactive camera GUI
# =====================================================================
def _btn_rect(w: int, h: int) -> tuple[int, int, int, int]:
    bw, bh = 260, 54
    x1 = (w - bw) // 2
    y1 = h - bh - 20
    return x1, y1, x1 + bw, y1 + bh


def _draw_button(canvas: np.ndarray, w: int, h: int,
                 color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = _btn_rect(w, h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, canvas, 1 - OVERLAY_ALPHA, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    tw, th = cv2.getTextSize(text, FONT, 0.72, 2)[0]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, shot_count: int, status: str) -> None:
    cv2.putText(canvas, f"Shots sent: {shot_count}", (12, 34),
                FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Shots sent: {shot_count}", (12, 34),
                FONT, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, status, (12, 68),
                    FONT, 0.6, (0, 220, 220), 2, cv2.LINE_AA)


def _point_in_rect(px: int, py: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


class _SendState:
    IDLE    = "idle"
    SENDING = "sending"
    DONE    = "done"

    def __init__(self) -> None:
        self.state = self.IDLE
        self.shot_count = 0
        self._lock = threading.Lock()

    def start_send(self) -> None:
        with self._lock:
            self.state = self.SENDING

    def finish_send(self) -> None:
        with self._lock:
            self.state = self.DONE
            self.shot_count += 1

    def ack_done(self) -> None:
        with self._lock:
            if self.state == self.DONE:
                self.state = self.IDLE

    @property
    def is_sending(self) -> bool:
        with self._lock:
            return self.state == self.SENDING

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self.state, self.shot_count


def _mouse_cb(event, x, y, flags, param) -> None:
    state, w, h, trigger, mouse_pos = param
    mouse_pos[0], mouse_pos[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if _point_in_rect(x, y, _btn_rect(w, h)) and not state.is_sending:
            trigger[0] = True


def camera_interactive(encoder: ConventionalEncoder, socket: zmq.Socket,
                       cfg: TxConventionalConfig, cam_path: str) -> None:
    cam_idx = int(cam_path) if cam_path.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}")
        return

    time.sleep(0.8)
    for _ in range(5):
        cap.read()

    win = "Conventional TX (JPEG/J2K + LDPC)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cfg.width, cfg.height)
    cv2.imshow(win, np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8))
    cv2.waitKey(1)

    state = _SendState()
    trigger = [False]
    mouse_pos = [-1, -1]
    captured_frame: list[Optional[np.ndarray]] = [None]
    done_flash_end = [0.0]

    cv2.setMouseCallback(win, _mouse_cb,
                         param=(state, cfg.width, cfg.height, trigger, mouse_pos))

    send_warmup(encoder, socket, cfg)

    print(f"\n[*] Camera {cam_idx} live. Click CAPTURE & SEND or press Space.")
    print("[*] Press 'q' or Esc to quit.\n")

    def _send_worker(frame_bgr: np.ndarray) -> None:
        state.start_send()
        try:
            encode_and_send(encoder, socket, cfg, frame_bgr,
                            label=f"shot #{state.shot_count + 1}")
        finally:
            state.finish_send()
            done_flash_end[0] = time.time() + 1.2

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                print("[!] Failed to read camera frame.")
                break

            display = cv2.resize(raw, (cfg.width, cfg.height))
            now = time.time()
            cur_state, shot_count = state.snapshot()

            if cur_state == _SendState.SENDING:
                if captured_frame[0] is not None:
                    display = cv2.resize(captured_frame[0], (cfg.width, cfg.height))
                color = BTN_COLOR_ACTIVE
                text = f"  Encoding + Sending ({cfg.repeat}x)..."
                status = "Transmitting over SDR..."
            elif cur_state == _SendState.DONE or now < done_flash_end[0]:
                color = BTN_COLOR_DONE
                text = "  Sent!"
                status = f"Shot #{shot_count} transmitted ({cfg.repeat}x)"
                state.ack_done()
            else:
                hover = _point_in_rect(mouse_pos[0], mouse_pos[1],
                                       _btn_rect(cfg.width, cfg.height))
                color = BTN_COLOR_HOVER if hover else BTN_COLOR_IDLE
                text = "[ CAPTURE & SEND ]"
                status = ("Space or click to capture" if shot_count == 0
                          else f"Last: shot #{shot_count}")

            _draw_hud(display, shot_count, status)
            _draw_button(display, cfg.width, cfg.height, color, text)
            cv2.imshow(win, display)

            if trigger[0] and not state.is_sending:
                trigger[0] = False
                captured_frame[0] = raw.copy()
                threading.Thread(target=_send_worker, args=(raw.copy(),),
                                 daemon=True).start()
                print(f"[*] Shot #{shot_count + 1} captured & queued for encode+TX")

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and not state.is_sending:
                trigger[0] = True
            elif key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[*] Session ended. Total shots sent: {state.shot_count}")


# =====================================================================
# Non-interactive modes
# =====================================================================
def camera_auto(encoder: ConventionalEncoder, socket: zmq.Socket,
                cfg: TxConventionalConfig, cam_path: str,
                shots: int, interval: float) -> None:
    cam_idx = int(cam_path) if cam_path.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {cam_idx}")
        return

    print(f"[*] Camera {cam_idx} opened. Taking {shots} shot(s)...")
    time.sleep(1.0)
    for _ in range(5):
        cap.read()

    send_warmup(encoder, socket, cfg)

    try:
        for i in range(shots):
            if i > 0:
                print(f"[*] Waiting {interval}s before next shot...")
                time.sleep(interval)
            ret, frame = cap.read()
            if not ret:
                print(f"[!] Failed to capture shot {i+1}")
                continue
            print(f"\n[*] Shot {i+1}/{shots} captured")
            encode_and_send(encoder, socket, cfg, frame,
                            label=f"shot {i+1}/{shots}")
    finally:
        cap.release()


def send_from_file(encoder: ConventionalEncoder, socket: zmq.Socket,
                   cfg: TxConventionalConfig, path: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[!] Error: Could not read '{path}'")
        return
    print(f"[*] Loaded image: {path}")
    send_warmup(encoder, socket, cfg)
    encode_and_send(encoder, socket, cfg, frame, label=os.path.basename(path))


def send_from_folder(encoder: ConventionalEncoder, socket: zmq.Socket,
                     cfg: TxConventionalConfig, path: str,
                     interval: float) -> None:
    if not os.path.isdir(path):
        print(f"[!] Error: Directory '{path}' not found")
        return
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    all_files = sorted(glob.glob(os.path.join(path, "*.*")))
    images = [f for f in all_files
              if os.path.splitext(f)[1].lower() in valid_exts]
    if not images:
        print(f"[!] No valid images in '{path}'")
        return

    print(f"[*] Found {len(images)} images in '{path}'")
    send_warmup(encoder, socket, cfg)
    for i, img_path in enumerate(images):
        if i > 0:
            print(f"\n[*] Waiting {interval}s...")
            time.sleep(interval)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        name = os.path.basename(img_path)
        print(f"\n[*] Image {i+1}/{len(images)}: {name}")
        encode_and_send(encoder, socket, cfg, frame, label=name)


# =====================================================================
# Argument parsing + main
# =====================================================================
_MOD_NAME = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM"}


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Conventional (JPEG/J2K + LDPC) TX baseline for DJSCC comparison")

    # Source
    p.add_argument("--source", type=str, default="camera",
                   choices=["camera", "file", "folder"])
    p.add_argument("--path", type=str, default="0",
                   help="Camera index, file path, or folder path (default: 0)")
    p.add_argument("--shots", type=int, default=0,
                   help="Auto-capture N shots (0 = interactive, default: 0)")
    p.add_argument("--interval", type=float, default=3.0,
                   help="Seconds between auto shots (default: 3.0)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--channel", type=int, default=3)

    # Source coding
    p.add_argument("--codec", type=str, default="jpeg2000",
                   choices=["jpeg2000", "jpeg"])
    p.add_argument("--codec-quality", type=int, default=48,
                   help="JPEG: quality 1..95 (default 50). "
                        "JPEG2000: compression ratio raw/target (default 48). "
                        "Used as a *fallback* when budget targeting is off.")
    p.add_argument("--target-bytes", type=int, default=0,
                   help="Explicit JPEG byte budget; 0 = use the LDPC input "
                        "window from the layout calculator.")
    p.add_argument("--no-fit-budget", action="store_true",
                   help="Don't rate-target the codec to the byte budget; "
                        "use --codec-quality verbatim. Useful for free-quality "
                        "comparisons.")

    # Channel coding
    p.add_argument("--ldpc-n", type=int, default=960,
                   help="LDPC codeword length n (RPTU database).")
    p.add_argument("--ldpc-k", type=int, default=640,
                   help="LDPC message length k. Default rate = 2/3.")

    # Modulation hint (must match .grc; only for byte-budget arithmetic)
    p.add_argument("--bits-per-symbol", type=int, default=2,
                   choices=[1, 2, 4, 6],
                   help="Bits per modulation symbol — KEEP IN SYNC WITH .grc "
                        "`repack` and `payload_mod` variables. Default QPSK (2).")
    p.add_argument("--comp-ratio", type=int, default=6,
                   help="DJSCC inverse compression ratio used to compute "
                        "target_complex_symbols (default 6 → N_target=196608 "
                        "at 768×512×3).")
    p.add_argument("--target-symbols", type=int, default=0,
                   help="Override target_complex_symbols (0 = derive from comp-ratio).")

    # OFDM alignment
    p.add_argument("--packet-len", type=int, default=960,
                   help="Must match .grc `packet_len` (default 960).")

    # Transmission
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--repeat-interval", type=float, default=0.5)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--warmup-frames", type=int, default=3)
    p.add_argument("--warmup-interval", type=float, default=0.5)

    # ZMQ
    p.add_argument("--port", type=str, default="5557",
                   help="ZMQ PUB port (default: 5557 — distinct from "
                        "DeepJSCC's 5556).")
    p.add_argument("--topic", type=str, default="")
    p.add_argument("--bind-host", type=str, default="127.0.0.1")

    # Debug: bypass GR, push PMT PDUs straight to RX (must run RX with
    # `--demap hard --port 5559`).
    p.add_argument("--direct-zmq", action="store_true",
                   help="Skip GNU Radio: PUSH PMT PDUs (with packet_num "
                        "metadata) directly to the RX's PULL socket on port "
                        "5559. Pair with `--demap hard --port 5559` on the RX.")

    p.add_argument("--interleave", action="store_true",
                   help="Apply a block interleaver to LDPC codeword bits before "
                        "slicing into PDU slots.  Each slot then carries ~n/S "
                        "bits from every LDPC block, so a lost slot erases only "
                        "a few bits per block instead of whole blocks — enabling "
                        "true BP erasure recovery.  Must match --interleave on RX.")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> TxConventionalConfig:
    if args.target_symbols > 0:
        target_symbols = args.target_symbols
    else:
        # Same arithmetic as image_tx_encoded.py:509 to match DJSCC channel use.
        tcn = int((1 / args.comp_ratio) * 4 * 4 * 2 * 3)
        target_symbols = (tcn * (args.height // 4) * (args.width // 4)) // 2

    return TxConventionalConfig(
        width=args.width,
        height=args.height,
        channel=args.channel,
        codec=args.codec,
        codec_quality=args.codec_quality,
        target_bytes=args.target_bytes,
        fit_to_budget=not args.no_fit_budget,
        ldpc_n=args.ldpc_n,
        ldpc_k=args.ldpc_k,
        bits_per_symbol=args.bits_per_symbol,
        target_complex_symbols=target_symbols,
        packet_len=args.packet_len,
        repeat=args.repeat,
        repeat_interval=args.repeat_interval,
        warmup_frames=0 if args.no_warmup else args.warmup_frames,
        warmup_interval=args.warmup_interval,
        port=args.port,
        bind_host=args.bind_host,
        topic=args.topic.encode("utf-8") if args.topic else b"",
        direct_zmq=args.direct_zmq,
        interleave=args.interleave,
    )


def main() -> int:
    args = parse_arguments()
    cfg = build_config(args)
    encoder = ConventionalEncoder(cfg)
    L = encoder.layout

    rate = cfg.ldpc_k / cfg.ldpc_n
    mod_name = _MOD_NAME.get(cfg.bits_per_symbol, f"M={cfg.bits_per_symbol}")
    print("[*] Conventional TX:")
    print(f"    Image: {cfg.width}x{cfg.height}x{cfg.channel}")
    print(f"    Source: {cfg.codec.upper()} "
          f"(quality={cfg.codec_quality}, fit_to_budget={cfg.fit_to_budget})")
    print(f"    LDPC:   ({cfg.ldpc_n}, {cfg.ldpc_k})  rate={rate:.3f}  "
          f"blocks/image={L.num_ldpc_blocks}")
    print(f"    Modulation hint: {mod_name} (M={cfg.bits_per_symbol}) — "
          f"set .grc `repack`={cfg.bits_per_symbol} "
          f"and `payload_mod`=constellation_{mod_name.lower()}().")
    print(f"    Target: {cfg.target_complex_symbols} complex sym/image  "
          f"(actual: {L.output_complex_symbols})")
    print(f"    JPEG budget: {L.jpeg_budget_bytes} bytes  →  "
          f"LDPC out: {L.ldpc_output_bytes} bytes  →  "
          f"published per image: {L.ldpc_output_bytes} bytes")
    print(f"    Packet alignment: every {L.packet_align_bytes} bytes "
          f"(packet_len={cfg.packet_len} complex sym).")

    if cfg.direct_zmq:
        # Direct-to-RX debug: PUSH PDUs into the RX's PULL on 5559.
        zmq_address = f"tcp://{cfg.bind_host}:5559"
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(zmq_address)
        print(f"[*] [DEBUG] direct-ZMQ mode: PUSH bound to {zmq_address} "
              "(skipping GR — RX must use `--demap hard --port 5559`)")
    else:
        zmq_address = f"tcp://{cfg.bind_host}:{cfg.port}"
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(zmq_address)
        print(f"[*] ZMQ PUB bound to {zmq_address}"
              + (f" (topic='{cfg.topic.decode()}')" if cfg.topic else ""))
    print(f"[*] Repeat: {cfg.repeat}x per image, {cfg.repeat_interval}s apart")
    print("[*] Waiting 4s for ZMQ subscribers to connect...")
    time.sleep(4.0)

    try:
        if args.source == "camera":
            if args.shots == 0:
                camera_interactive(encoder, socket, cfg, args.path)
            else:
                camera_auto(encoder, socket, cfg, args.path,
                            args.shots, args.interval)
        elif args.source == "file":
            send_from_file(encoder, socket, cfg, args.path)
        elif args.source == "folder":
            send_from_folder(encoder, socket, cfg, args.path, args.interval)

        print("\n[*] All transmissions complete.")
        return 0

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
        return 130
    finally:
        socket.close()
        ctx.term()
        print("[*] Resources released. Exiting.")


if __name__ == "__main__":
    sys.exit(main())
