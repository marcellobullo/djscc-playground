#!/usr/bin/env python3
"""
Conventional (LDPC + JPEG/JPEG2000) Image Receiver — PDU-locked variant
that mirrors ``image_rx_decoded_2.py`` (the DJSCC RX) so a cold-start
packet drop on the first send becomes a *known missing slot* (zero-filled
and tolerated) instead of a stream-offset error that destroys the image.

Why PDU-based instead of raw byte stream:
- Each OFDM packet on the air carries a `packet_num` in its parsed
  header. Receiving the LDPC payload as a stream of PDUs (one PDU per
  OFDM packet, metadata + vector payload) lets us recover that
  packet_num at the python side and place the packet's contents in
  the correct slot of a per-image buffer.
- The "first packet after an idle gap" anchors the slot grid at slot 0
  for the next image, so cold-start losing the *real* first 1–3 packets
  just shows up as `seen[0..2] = False` — the rest of the image still
  lands in the right slots, missing slots are zero-filled, BP runs.
- Image boundaries are detected by `image_idx_local = (packet_num -
  anchor_pn) // PKT_PER_IMG` advancing — independent of how many bytes
  flowed at any particular moment.

Required GR-side `rx_conventional.grc` configuration:
    - Soft path:   tagged complex stream from
                   `digital_ofdm_serializer_vcc_payload`
                 → `digital_constellation_soft_decoder_cf_0`
                 → `pdu.tagged_stream_to_pdu(blocks.float_t, "packet_len")`
                 → `zeromq.push_msg_sink` on port 5559.
                   PDU payload size = packet_len * mod_order float32 LLRs
                   per OFDM packet.
    - Hard path:   tagged complex stream
                 → `digital_constellation_decoder_cb_1`
                 → `blocks_repack_bits_bb_1` (k=mod_order, l=8, GR_LSB_FIRST)
                 → `pdu.tagged_stream_to_pdu(blocks.byte_t, "packet_len")`
                 → `zeromq.push_msg_sink` on port 5559.
                   PDU payload size = packet_len * mod_order / 8 bytes
                   per OFDM packet.

Each PDU MUST carry a `packet_num` long in its metadata dict — it
propagates from `digital_packet_headerparser_b_0_0`'s parsed header
through the tagged stream and into the PDU's metadata when
`tagged_stream_to_pdu` bundles the packet.

Examples::

  python image_rx_decoded_conventional_pyldpc.py --comp-ratio 6 \\
      --codec jpeg --bits-per-symbol 4 --ldpc-n 1920 --ldpc-k 960
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pmt
import torch
import zmq
import scipy.sparse
from PIL import Image, ImageFile

from ldpc import bp_decoder

# Allow PIL to load slightly truncated streams (channel BER may corrupt the
# final J2K markers in low-SNR runs; let it try anyway).
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from utils.parity_matrix_helper import load_parity_matrix, get_generator_and_info_bits

def _pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


# =====================================================================
# Config — must mirror the TX side's encode-time knobs
# =====================================================================
@dataclass
class RxConventionalConfig:
    width: int
    height: int
    channel: int

    codec: str               # "jpeg2000" | "jpeg"  (matches TX)
    ldpc_n: int
    ldpc_k: int
    bp_iters: int

    bits_per_symbol: int     # must match .grc `mod_order`
    target_complex_symbols: int
    packet_len: int

    port: str
    connect_host: str

    demap: str               # "soft" | "hard"
    device: str              # "auto" | "cpu" | "mps" | "cuda"

    # Display / save
    window_title: str
    save: bool
    output_dir: str

    # Behaviour
    idle_gap_s: float        # reset accumulator after this much silence
    timeout_s: float         # auto-exit after this much overall silence (0 = never)

    # Diagnostics
    debug: bool              # emit [dbg] prints (LLR stats, hex dump, syndrome, accumulator timing)

    # Interleaver (must match TX --interleave)
    interleave: bool

    # Controlled-experiment mode: one PNG per transmitted image, named by
    # transmission-order index, plus a manifest.csv. tx_gain/rx_gain only tag
    # the output folder (the RX does not set the radio gains).
    exp_id_mode: bool = False
    tx_gain: Optional[str] = None
    rx_gain: Optional[str] = None


# =====================================================================
# Layout math — same as TX. Duplicated rather than imported so the RX
# is independently runnable.
# =====================================================================
@dataclass
class Layout:
    num_ldpc_blocks: int
    ldpc_input_bytes: int
    ldpc_output_bytes: int     # hard-mode wire budget per image (n bits / 8)
    ldpc_output_floats: int    # soft-mode item count per image (= num_blocks * n)
    output_complex_symbols: int


def compute_layout(cfg: RxConventionalConfig) -> Layout:
    if cfg.ldpc_n % 8 or cfg.ldpc_k % 8:
        raise ValueError(
            f"LDPC (n,k)=({cfg.ldpc_n},{cfg.ldpc_k}) must both be multiples of 8")

    bytes_per_codeword = cfg.ldpc_n // 8
    bytes_per_message  = cfg.ldpc_k // 8
    packet_align_bytes = cfg.packet_len * cfg.bits_per_symbol // 8

    target_bits_after_ldpc = cfg.target_complex_symbols * cfg.bits_per_symbol
    num_blocks = (target_bits_after_ldpc + cfg.ldpc_n - 1) // cfg.ldpc_n
    while (num_blocks * bytes_per_codeword) % packet_align_bytes != 0:
        num_blocks += 1

    return Layout(
        num_ldpc_blocks=num_blocks,
        ldpc_input_bytes=num_blocks * bytes_per_message,
        ldpc_output_bytes=num_blocks * bytes_per_codeword,
        ldpc_output_floats=num_blocks * cfg.ldpc_n,
        output_complex_symbols=(num_blocks * bytes_per_codeword * 8) // cfg.bits_per_symbol,
    )


# =====================================================================
# Block interleaver — must match image_tx_encoded_conventional_pyldpc.py
# =====================================================================
def _make_interleaver(num_bits: int, num_slots: int):
    bps = num_bits // num_slots  # bits per slot
    p = np.arange(num_bits, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_bits, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


# =====================================================================
# Channel decoder — Classic C++ PyLDPC Batch
# =====================================================================
class LDPCBatchDecoder:
    def __init__(self, ldpc_n: int, ldpc_k: int, bp_iters: int = 5,
                 device: str = "cpu"):
        self.n = ldpc_n
        self.k = ldpc_k
        self.device = device

        self.H = load_parity_matrix(ldpc_n, ldpc_k)
        
        _, self.info_bits = get_generator_and_info_bits(ldpc_n, ldpc_k)
        
        # Initialize the blazing-fast C++ Belief Propagation decoder
        self.decoder = bp_decoder(
            self.H,
            error_rate=0.01,
            max_iter=bp_iters,
            bp_method='minimum_sum'
        )

        # Convergence bookkeeping for the most recent decode_* call. BP on a
        # valid codeword sets decoder.converge=True (it found an error pattern
        # whose syndrome matches). A block that does NOT converge still returns
        # *some* message bytes, but they almost certainly contain residual bit
        # errors -- which is fatal for JPEG/J2K. These let the caller tell
        # "LDPC genuinely corrected" from "LDPC gave up but we fed JPEG garbage."
        self.last_total_blocks = 0
        self.last_failed_blocks = 0

    @property
    def last_converged(self) -> bool:
        """True iff every block in the most recent decode converged."""
        return self.last_total_blocks > 0 and self.last_failed_blocks == 0

    def decode_bytes(self, codeword_bytes: np.ndarray, *,
                     debug: bool = False) -> np.ndarray:
        """``codeword_bytes`` length = num_blocks * (n/8). Returns
        num_blocks * (k/8) message bytes."""
        if codeword_bytes.dtype != np.uint8:
            codeword_bytes = codeword_bytes.astype(np.uint8)

        bytes_per_cw = self.n // 8
        if len(codeword_bytes) % bytes_per_cw:
            raise ValueError(
                f"LDPC decode input {len(codeword_bytes)} bytes is not a "
                f"multiple of n/8={bytes_per_cw}")
        num_blocks = len(codeword_bytes) // bytes_per_cw

        bits = np.unpackbits(codeword_bytes, bitorder='little').reshape(num_blocks, self.n)
        decoded_bits_2d = np.zeros((num_blocks, self.k), dtype=np.uint8)

        self.last_total_blocks = num_blocks
        self.last_failed_blocks = 0

        for i in range(num_blocks):
            y_hard = bits[i]

            # Channel probabilities baseline for hard decoding
            probs = np.full(self.n, 0.001)

            self.decoder.update_channel_probs(probs)
            syndrome = self.H.dot(y_hard) % 2
            #syndrome = (self.H.dot(y_hard.astype(np.int32)) % 2).astype(np.uint8)
            error_vector = self.decoder.decode(syndrome)
            if not self.decoder.converge:
                self.last_failed_blocks += 1
            codeword = (y_hard + error_vector) % 2
            decoded_bits_2d[i] = codeword[self.info_bits]

        if debug and self.last_failed_blocks:
            print(f"[dbg] LDPC: {self.last_failed_blocks}/{num_blocks} blocks "
                  f"did NOT converge (residual errors -> JPEG will likely fail)")

        return np.packbits(decoded_bits_2d.reshape(-1), bitorder='little')

    def decode_llrs(self, llrs: np.ndarray, *,
                    debug: bool = False) -> np.ndarray:
        """`llrs` length = num_blocks * n."""
        if llrs.dtype != np.float32:
            llrs = llrs.astype(np.float32)
        if len(llrs) % self.n:
            raise ValueError(
                f"LDPC decode input {len(llrs)} LLRs is not a multiple of n={self.n}")
        num_blocks = len(llrs) // self.n

        if debug:
            absllrs = np.abs(llrs)
            print(f"[dbg] llrs : n={len(llrs)}  min={llrs.min():+.2f}  "
                  f"max={llrs.max():+.2f}  mean={llrs.mean():+.3f}  "
                  f"std={llrs.std():.2f}  |x|<0.1: {(absllrs<0.1).mean()*100:.2f}%")

        reshaped = llrs.reshape(num_blocks, self.n).copy()
        decoded_bits_2d = np.zeros((num_blocks, self.k), dtype=np.uint8)

        self.last_total_blocks = num_blocks
        self.last_failed_blocks = 0

        for i in range(num_blocks):
            block_llrs = reshaped[i]

            # Hard decision: LLR < 0 means bit = 1
            y_hard = (block_llrs < 0).astype(np.uint8)

            # Probability that the hard decision is INCORRECT
            probs = 1.0 / (1.0 + np.exp(np.clip(np.abs(block_llrs), 0, 50)))

            self.decoder.update_channel_probs(probs)
            syndrome = self.H.dot(y_hard) % 2
            #syndrome = (self.H.dot(y_hard.astype(np.int32)) % 2).astype(np.uint8)
            error_vector = self.decoder.decode(syndrome)
            if not self.decoder.converge:
                self.last_failed_blocks += 1
            codeword = (y_hard + error_vector) % 2
            decoded_bits_2d[i] = codeword[self.info_bits]

        if debug and self.last_failed_blocks:
            print(f"[dbg] LDPC: {self.last_failed_blocks}/{num_blocks} blocks "
                  f"did NOT converge (residual errors -> JPEG will likely fail)")

        return np.packbits(decoded_bits_2d.reshape(-1), bitorder='little')


# =====================================================================
# Source decoder — PIL JPEG / JPEG2000
# =====================================================================
def _attempt_image_recovery(data_bytes):
    try:
        img = _try_jpeg_recovery(data_bytes)
        if img is not None:
            return img
        img = _try_png_recovery(data_bytes)
        if img is not None:
            return img
        img = _try_force_image_load(data_bytes)
        if img is not None:
            return img
        return None
    except Exception as e:
        print(f"Recovery attempt failed: {e}")
        return None

def _try_jpeg_recovery(data_bytes):
    try:
        jpeg_start = data_bytes.find(b'\xff\xd8')
        if jpeg_start == -1:
            return None
        jpeg_end = data_bytes.rfind(b'\xff\xd9')
        if jpeg_end == -1:
            jpeg_data = data_bytes[jpeg_start:]
        else:
            jpeg_data = data_bytes[jpeg_start:jpeg_end+2]
        
        img_bytes = io.BytesIO(jpeg_data)
        img = Image.open(img_bytes)
        print(f"✅ JPEG recovery successful!")
        return img
    except Exception as e:
        print(f"JPEG recovery failed: {e}")
        return None

def _try_png_recovery(data_bytes):
    try:
        png_signature = b'\x89PNG\r\n\x1a\n'
        png_start = data_bytes.find(png_signature)
        if png_start == -1:
            return None
        png_data = data_bytes[png_start:]
        img_bytes = io.BytesIO(png_data)
        img = Image.open(img_bytes)
        print(f"✅ PNG recovery successful!")
        return img
    except Exception as e:
        print(f"PNG recovery failed: {e}")
        return None

def _try_force_image_load(data_bytes):
    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_bytes = io.BytesIO(data_bytes)
        img = Image.open(img_bytes)
        img.load()
        print(f"✅ Force loading successful!")
        return img
    except Exception as e:
        print(f"Force loading failed: {e}")
        return None
        
def decompress_to_bgr(payload: bytes,
                      width: int, height: int) -> Optional[np.ndarray]:
    if not payload:
        return None

    end = len(payload)
    while end > 0 and payload[end - 1] == 0:
        end -= 1
    trimmed = bytes(payload[:end]) if end < len(payload) else bytes(payload)
    trimmed_data_bytes = io.BytesIO(trimmed)

    try:
        img = Image.open(trimmed_data_bytes)
        img.load()
    except Exception as ex:
        print(f"  [!] PIL could not decode payload: {ex}")
        try:
            img = Image.open(io.BytesIO(bytes(payload)))
            img.load()
        except Exception as e:
            print(f"  [!] PIL could not decode payload: {e}")
            try:
                img = _attempt_image_recovery(trimmed)
            except Exception as e2:
                print(f"  [!] Image recovery also failed: {e2}")
                return None
            if img is None:
                return None

    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height))

    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# =====================================================================
# Receive loop — PDU lock-step 
# =====================================================================
def receive_loop(socket: zmq.Socket,
                 ldpc: LDPCBatchDecoder,
                 cfg: RxConventionalConfig,
                 layout: Layout) -> None:

    if cfg.demap == "soft":
        PKT_PAYLOAD_ITEMS = cfg.packet_len * cfg.bits_per_symbol 
        PKT_DTYPE = np.float32 
        img_total = layout.ldpc_output_floats
        _vec_extract = pmt.f32vector_elements
        wire_descr = (f"{PKT_PAYLOAD_ITEMS} float32 LLRs/PDU, "
                      f"{img_total} LLRs/image")
    else:
        PKT_PAYLOAD_ITEMS = (cfg.packet_len * cfg.bits_per_symbol) // 8
        PKT_DTYPE = np.uint8 
        img_total = layout.ldpc_output_bytes
        _vec_extract = pmt.u8vector_elements
        wire_descr = (f"{PKT_PAYLOAD_ITEMS} bytes/PDU, "
                      f"{img_total} bytes/image")

    PKT_PER_IMG = math.ceil(img_total / PKT_PAYLOAD_ITEMS)
    items_per_image = PKT_PER_IMG * PKT_PAYLOAD_ITEMS

    slot_buf = np.zeros(items_per_image, dtype=PKT_DTYPE)
    pkt_seen = [False] * PKT_PER_IMG

    anchor_pn: Optional[int] = None
    current_image_idx: Optional[int] = None
    last_valid_pdu_time = time.time()

    cv2.namedWindow(cfg.window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_title, cfg.width, cfg.height)
    if cfg.save:
        os.makedirs(cfg.output_dir, exist_ok=True)

    n_total = 0
    n_decoded_ok = 0
    last_image_time = time.time()
    exp_manifest: list[dict] = []

    perm_fwd: Optional[np.ndarray] = None
    if cfg.interleave:
        num_bits_per_img = img_total * 8 if cfg.demap == "hard" else img_total
        if num_bits_per_img % PKT_PER_IMG:
            raise ValueError(
                f"--interleave: img_total={img_total} is not divisible by "
                f"PKT_PER_IMG={PKT_PER_IMG}; cannot build a block interleaver")
        perm_fwd, _ = _make_interleaver(num_bits_per_img, PKT_PER_IMG)
        print(f"[*] Interleaver ON: {num_bits_per_img} bits × {PKT_PER_IMG} slots "
              f"→ {num_bits_per_img // PKT_PER_IMG} bits/slot/block")

    print(f"[*] PDU lock-step RX ({cfg.demap}): "
          f"{PKT_PER_IMG} packets/image, {wire_descr}")
    print(f"[*]   LDPC {cfg.ldpc_n}/{cfg.ldpc_k} × {layout.num_ldpc_blocks} "
          f"blocks → {layout.ldpc_input_bytes} message bytes")
    print(f"[*] Idle gap = {cfg.idle_gap_s}s, port = {cfg.port}")
    if cfg.debug:
        print("[*] Debug mode ON — per-image diagnostics enabled.")
    print("[*] Waiting for first burst...")

    def _flush_and_decode():
        nonlocal n_total, n_decoded_ok, last_image_time

        n_missed = sum(1 for ok in pkt_seen if not ok)
        if n_missed == PKT_PER_IMG:
            return 

        if n_missed:
            for i, ok in enumerate(pkt_seen):
                if not ok:
                    slot_buf[i*PKT_PAYLOAD_ITEMS:(i+1)*PKT_PAYLOAD_ITEMS] = 0
            print(f"  [*] image {n_total+1}: "
                  f"{n_missed}/{PKT_PER_IMG} slots missing, zero-filled")

        n_total += 1

        t0 = time.time()
        try:
            print(f"  [*] Decoding image {n_total}...")
            if cfg.interleave:
                if cfg.demap == "soft":
                    rx_llrs = -slot_buf[:img_total].astype(np.float32)
                    rx_llrs = rx_llrs.reshape(-1, cfg.bits_per_symbol)[:, ::-1].flatten()
                else:
                    raw = slot_buf[:img_total]
                    bits = np.unpackbits(raw, bitorder='little').astype(np.float32)
                    rx_llrs = (1.0 - 2.0 * bits) * 100.0
                    for slot_idx, ok in enumerate(pkt_seen):
                        if not ok:
                            lo = slot_idx * PKT_PAYLOAD_ITEMS * 8
                            hi = lo + PKT_PAYLOAD_ITEMS * 8
                            rx_llrs[lo:hi] = 0.0
                ldpc_llrs = rx_llrs[perm_fwd]
                msg_bytes = ldpc.decode_llrs(ldpc_llrs, debug=cfg.debug)
            
            elif cfg.demap == "soft":
                rx_llrs = -slot_buf[:img_total].astype(np.float32)
                rx_llrs = rx_llrs.reshape(-1, cfg.bits_per_symbol)[:, ::-1].flatten()
                msg_bytes = ldpc.decode_llrs(rx_llrs, debug=cfg.debug)
            else:
                msg_bytes = ldpc.decode_bytes(slot_buf[:img_total], debug=cfg.debug)
        except Exception as e:
            print(f"  [!] LDPC decode failed: {e}")
            return
        ldpc_ms = (time.time() - t0) * 1000.0
        ldpc_failed = ldpc.last_failed_blocks
        ldpc_blocks = ldpc.last_total_blocks
        ldpc_state = (f"converged ({ldpc_blocks} blocks)" if ldpc_failed == 0
                      else f"{ldpc_failed}/{ldpc_blocks} blocks UNCONVERGED")

        t0 = time.time()
        bgr = decompress_to_bgr(msg_bytes.tobytes(), cfg.width, cfg.height)
        jpeg_ms = (time.time() - t0) * 1000.0

        decode_ok = bgr is not None
        label = f"img {n_total}"
        if bgr is None:
            cause = ("residual LDPC errors" if ldpc_failed
                     else "source-coding corruption despite clean LDPC")
            print(f"  [{label}] LDPC {ldpc_state} ({ldpc_ms:.0f} ms) but JPEG "
                  f"decode failed [{cause}] -- displaying error placeholder")
            bgr = np.full((cfg.height, cfg.width, 3), 32, dtype=np.uint8)
            cv2.putText(bgr, "JPEG decode FAILED", (20, cfg.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            n_decoded_ok += 1
            warn = "" if ldpc_failed == 0 else f"  [!] {ldpc_state} -- image may be corrupted"
            print(f"  [{label}] LDPC {ldpc_ms:.0f} ms ({ldpc_state}) + JPEG "
                  f"{jpeg_ms:.0f} ms ({n_decoded_ok}/{n_total} ok, "
                  f"{PKT_PER_IMG - n_missed}/{PKT_PER_IMG} slots){warn}")

        last_image_time = time.time()

        if cfg.exp_id_mode:
            # n_total is incremented once per flushed image just above, in
            # transmission order. Do NOT use current_image_idx: the anchor is
            # reset per image, so it is always 0.
            img_id = n_total - 1
            fname = f"image_{img_id:03d}.png"
            if cfg.save:
                if decode_ok:
                    cv2.imwrite(os.path.join(cfg.output_dir, fname), bgr)
                else:
                    # Decode failed: save an all-black frame so every
                    # transmitted id has a file. The failure is recorded in the
                    # manifest (decode_ok=0); compute_psnr.py decides how to
                    # score it via --fail-mode, so the black pixels here are
                    # mainly for visual inspection / folder completeness.
                    black = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(cfg.output_dir, fname), black)
            exp_manifest.append({
                "image_id": img_id,
                "filename": fname,
                "image_attempt": n_total,
                "slots_seen": PKT_PER_IMG - n_missed,
                "slots_missing": n_missed,
                "decode_ok": int(decode_ok),
            })
            print(f"  [*] image_id={img_id} decode_ok={decode_ok} "
                  f"{'saved -> ' + fname if cfg.save else '(not saved)'}"
                  f"{'' if decode_ok else ' [BLACK]'} "
                  f"({PKT_PER_IMG - n_missed}/{PKT_PER_IMG} slots)")
        elif cfg.save and bgr is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fname = f"image_{n_decoded_ok:03d}_{stamp}.png"
            cv2.imwrite(os.path.join(cfg.output_dir, fname), bgr)

        display = bgr.copy()
        cv2.putText(display, f"#{n_decoded_ok} / {n_total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(cfg.window_title, display)
        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     raise KeyboardInterrupt

    def _reset_image_state(*, keep_anchor: bool):
        nonlocal pkt_seen, slot_buf, current_image_idx, anchor_pn
        pkt_seen = [False] * PKT_PER_IMG
        slot_buf.fill(0)
        current_image_idx = None
        if not keep_anchor:
            anchor_pn = None

    try:
        while True:
            if current_image_idx is not None and time.time() - last_valid_pdu_time > cfg.idle_gap_s:
                n_seen = sum(1 for ok in pkt_seen if ok)
                print(f"\n[*] {cfg.idle_gap_s}s idle gap — flushing partial "
                      f"image ({n_seen}/{PKT_PER_IMG} slots received).")
                _flush_and_decode()
                _reset_image_state(keep_anchor=False)

            if cfg.timeout_s > 0 and n_total > 0 \
                    and time.time() - last_image_time > cfg.timeout_s:
                print(f"\n[*] No image for {cfg.timeout_s}s. Exiting.")
                break

            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                if (cv2.waitKey(10) & 0xFF) == ord('q'):
                    break
                continue

            try:
                pdu = pmt.deserialize_str(raw)
                metadata = pmt.car(pdu)
                payload = pmt.cdr(pdu)
                key_pn = pmt.intern("packet_num")
                packet_num = pmt.to_long(
                    pmt.dict_ref(metadata, key_pn, pmt.from_long(-1)))
                if packet_num < 0:
                    continue
                pkt = np.array(_vec_extract(payload), dtype=PKT_DTYPE)
            except Exception as e:
                continue

            if len(pkt) != PKT_PAYLOAD_ITEMS:
                if len(pkt) < PKT_PAYLOAD_ITEMS:
                    continue
                pkt = pkt[:PKT_PAYLOAD_ITEMS]

            last_valid_pdu_time = time.time()

            if anchor_pn is None:
                anchor_pn = packet_num

            delta = packet_num - anchor_pn
            slot = delta % PKT_PER_IMG
            image_idx_local = delta // PKT_PER_IMG

            if current_image_idx is None:
                current_image_idx = image_idx_local
            elif image_idx_local != current_image_idx:
                _flush_and_decode()
                _reset_image_state(keep_anchor=True)
                current_image_idx = image_idx_local

            slot_buf[slot*PKT_PAYLOAD_ITEMS:(slot+1)*PKT_PAYLOAD_ITEMS] = pkt
            pkt_seen[slot] = True

            if slot == PKT_PER_IMG - 1:
                _flush_and_decode()
                _reset_image_state(keep_anchor=False)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[*] Receiver stopped by user.")
    finally:
        cv2.destroyAllWindows()
        if cfg.exp_id_mode and exp_manifest:
            try:
                os.makedirs(cfg.output_dir, exist_ok=True)
                man_path = os.path.join(cfg.output_dir, "manifest.csv")
                with open(man_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(exp_manifest[0].keys()))
                    w.writeheader()
                    w.writerows(exp_manifest)
                print(f"  Manifest written:         {man_path} "
                      f"({len(exp_manifest)} rows)")
            except Exception as e:
                print(f"  [!] manifest write failed: {e}")
        print(f"\n{'='*50}")
        print(f"  Total images attempted:   {n_total}")
        print(f"  Decoded successfully:     {n_decoded_ok}")
        if cfg.save and n_decoded_ok > 0:
            print(f"  Output directory:         {os.path.abspath(cfg.output_dir)}")
        print(f"{'='*50}")


# =====================================================================
# Argument parsing
# =====================================================================
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Conventional (LDPC + JPEG/J2K) RX baseline for DJSCC comparison")

    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--channel", type=int, default=3)

    p.add_argument("--codec", type=str, default="jpeg2000",
                   choices=["jpeg2000", "jpeg"])
    p.add_argument("--ldpc-n", type=int, default=960)
    p.add_argument("--ldpc-k", type=int, default=640)
    p.add_argument("--bp-iters", type=int, default=5)

    p.add_argument("--bits-per-symbol", type=int, default=2, choices=[1, 2, 4, 6])
    p.add_argument("--comp-ratio", type=int, default=6)
    p.add_argument("--target-symbols", type=int, default=0)
    p.add_argument("--packet-len", type=int, default=960)

    p.add_argument("--demap", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--device", type=str, default="cpu",
                   choices=["auto", "cpu", "mps", "cuda"])

    p.add_argument("--port", type=str, default="5559")
    p.add_argument("--connect-host", type=str, default="127.0.0.1")

    p.add_argument("--output-dir", type=str, default="./received_images_conv")
    p.add_argument("--no-save", action="store_true")

    p.add_argument("--idle-gap", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--interleave", action="store_true")

    p.add_argument("--exp-id-mode", action="store_true",
                   help="Controlled-experiment mode: save exactly one PNG per "
                        "transmitted image, named image_<order>.png by "
                        "transmission-order index, plus a manifest.csv "
                        "(records decode_ok per id). Run the TX with "
                        "--no-warmup and start this RX first so order index 0 "
                        "== first original.")
    p.add_argument("--tx-gain", type=str, default=None,
                   help="TX USRP gain for this run (folder tag only; the RX "
                        "does not set the radio gain). If both --tx-gain and "
                        "--rx-gain are given, output dir becomes "
                        "received_images/tx-<tx>_rx-<rx>.")
    p.add_argument("--rx-gain", type=str, default=None,
                   help="RX USRP gain for this run (folder tag only).")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> RxConventionalConfig:
    if args.target_symbols > 0:
        target_symbols = args.target_symbols
    else:
        tcn = int((1 / args.comp_ratio) * 4 * 4 * 2 * 3)
        target_symbols = (tcn * (args.height // 4) * (args.width // 4)) // 2

    output_dir = args.output_dir
    if args.tx_gain is not None and args.rx_gain is not None:
        output_dir = os.path.join(
            "received_images", f"tx-{args.tx_gain}_rx-{args.rx_gain}")

    return RxConventionalConfig(
        width=args.width,
        height=args.height,
        channel=args.channel,
        codec=args.codec,
        ldpc_n=args.ldpc_n,
        ldpc_k=args.ldpc_k,
        bp_iters=args.bp_iters,
        bits_per_symbol=args.bits_per_symbol,
        target_complex_symbols=target_symbols,
        packet_len=args.packet_len,
        port=args.port,
        connect_host=args.connect_host,
        demap=args.demap,
        device=_pick_device(args.device),
        window_title=f"Conventional RX ({args.demap}) — LDPC + JPEG/J2K",
        save=not args.no_save,
        output_dir=output_dir,
        idle_gap_s=args.idle_gap,
        timeout_s=args.timeout,
        debug=args.debug,
        interleave=args.interleave,
        exp_id_mode=args.exp_id_mode,
        tx_gain=args.tx_gain,
        rx_gain=args.rx_gain,
    )


_MOD_NAME = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM"}


def main() -> int:
    args = parse_arguments()

    cfg = build_config(args)
    layout = compute_layout(cfg)

    rate = cfg.ldpc_k / cfg.ldpc_n
    mod_name = _MOD_NAME.get(cfg.bits_per_symbol, f"M={cfg.bits_per_symbol}")
    print("[*] Conventional RX:")
    print(f"    Image: {cfg.width}x{cfg.height}x{cfg.channel}")
    print(f"    Codec hint: {cfg.codec.upper()} (auto-detected at decode time)")
    print(f"    LDPC:   ({cfg.ldpc_n}, {cfg.ldpc_k})  rate={rate:.3f}  "
          f"BP iters={cfg.bp_iters}  blocks/image={layout.num_ldpc_blocks}")
    print(f"    Modulation hint: {mod_name} (M={cfg.bits_per_symbol})")
    print(f"    GR will emit {layout.output_complex_symbols} complex sym/image")

    print(f"[*] Building LDPC BP decoder on device={cfg.device}...")
    t0 = time.time()
    ldpc = LDPCBatchDecoder(cfg.ldpc_n, cfg.ldpc_k, cfg.bp_iters, device=cfg.device)
    print(f"[*] Decoder ready in {time.time() - t0:.2f}s (device={cfg.device})")

    zmq_address = f"tcp://{cfg.connect_host}:{cfg.port}"
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect(zmq_address)
    print(f"[*] ZMQ PULL connected to {zmq_address}")

    if cfg.save:
        print(f"[*] Saving images to: {os.path.abspath(cfg.output_dir)}")
    else:
        print("[*] Save disabled (--no-save)")

    try:
        receive_loop(socket, ldpc, cfg, layout)
        return 0
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
        return 130
    finally:
        try:
            socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass
        socket.close()
        ctx.term()
        print("[*] Resources released. Exiting.")


if __name__ == "__main__":
    sys.exit(main())