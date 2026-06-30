#!/usr/bin/env python3
"""Unified DJSCC-playground receiver — model-agnostic.

One script for every model. The decoder is selected at runtime with ``--model``
(HF id | HF folder | alias | raw ``.pth``) and an optional ``--aligner``. It
subscribes to a ZMQ stream of GNU Radio PDUs (OFDM-demodulated cf32 symbols),
reassembles one image worth of packets in lock-step, conditions the decoder on
live SNR (if enabled and the model needs CSI), and decodes.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import pmt
import zmq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc import OutputKind, load_aligner, load_codec  # noqa: E402

# OFDM data-subcarrier layout (for the live-SNR Nulls+Taps estimate).
_OCCUPIED = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0))
             + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)))
_PILOTS = (-21, -7, 7, 21)
_DATA_IDX = np.array([k + 32 for k in _OCCUPIED if k not in _PILOTS], dtype=np.int64)


# ── interleaver (inverse of the TX block interleaver) ────────────────────────
def _make_interleaver(num_items, num_slots):
    bps = num_items // num_slots
    p = np.arange(num_items, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_items, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


class _DropPolicy:
    """``random:RATE`` | ``list:N1,N2`` | ``range:N:M`` | empty (off)."""

    def __init__(self, spec, seed):
        self.mode, self.rate, self.targets = "off", 0.0, set()
        self._rng = np.random.default_rng(seed or None)
        if not spec:
            return
        kind, _, body = spec.partition(":")
        if kind == "random":
            self.mode, self.rate = "random", float(body)
        elif kind == "list":
            self.mode = "list"
            self.targets = {int(x) for x in body.split(",") if x.strip()}
        elif kind == "range":
            lo, hi = (int(x) for x in body.split(":"))
            self.mode, self.targets = "range", set(range(lo, hi))
        else:
            raise ValueError(f"--drop-slots: unknown mode {kind!r}")

    def should_drop(self, slot):
        if self.mode == "off":
            return False
        if self.mode == "random":
            return bool(self._rng.random() < self.rate)
        return slot in self.targets


def _ncc(a, b):
    if a.shape != b.shape:
        return 0.0
    a = a.astype(np.float32); b = b.astype(np.float32)
    sa, sb = a.std(), b.std()
    if sa < 1.0 or sb < 1.0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def receive_loop(codec, socket, snr_socket, args):
    expected = codec.expected_complex_items
    PKT = args.packet_len
    PKT_PER_IMG = math.ceil(expected / PKT)
    samples = PKT_PER_IMG * PKT

    perm_fwd = None
    if args.interleave:
        perm_fwd, _ = _make_interleaver(samples, PKT_PER_IMG)

    slot_buf = np.zeros(samples, dtype=np.complex64)
    seen = [False] * PKT_PER_IMG
    pn_log: list = []
    h_by_pn: dict = {}
    raw_by_pn: dict = {}
    anchor_pn = None
    cur_img = None

    drop = _DropPolicy(args.drop_slots, args.drop_seed)
    win = "DJSCC-playground RX"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    last_saved = None
    total = unique = 0
    last_frame_t = time.time()
    last_pdu_t = time.time()

    print(f"[*] lock-step RX: {PKT_PER_IMG} pkts/image, {PKT} syms/pkt, "
          f"expected={expected}, needs_csi={codec.needs_csi}")

    def flush_and_decode():
        nonlocal total, unique, last_frame_t, last_saved
        if all(not s for s in seen):
            return
        n_missed = sum(1 for s in seen if not s)
        if n_missed:
            print(f"  [*] {n_missed}/{PKT_PER_IMG} slots missing, zero-filled")
            for i, ok in enumerate(seen):
                if not ok:
                    slot_buf[i * PKT:(i + 1) * PKT] = 0

        stream = slot_buf.copy()
        if perm_fwd is not None:
            stream = stream[perm_fwd]
        symbols = stream[:expected].copy()

        if args.clip_mag > 0:
            mags = np.abs(symbols)
            m = mags > args.clip_mag
            if m.any():
                symbols[m] = symbols[m] / mags[m] * args.clip_mag
        if args.renorm:
            pwr = float(np.mean(np.abs(symbols) ** 2))
            if pwr > 0:
                symbols *= np.complex64(np.sqrt(args.renorm_target / pwr))

        # live SNR -> decoder CSI. Build a per-packet effective-SNR vector
        # (post-EQ: 10log10(1 / mean(sigma2/|H_k|^2))) indexed by slot, and feed it
        # via set_csi_vector when the decoder consumes a per-packet map (spatial
        # CSI). Otherwise collapse to one scalar, as before.
        if snr_socket is not None and codec.needs_csi:
            use_vec = hasattr(codec, "set_csi_vector")
            per_pkt = np.full(PKT_PER_IMG, np.nan, dtype=np.float32)
            noise = []
            for pn, slot in pn_log:
                h, s2 = h_by_pn.get(pn), raw_by_pn.get(pn)
                if h is not None and s2 is not None:
                    hp = np.maximum(np.abs(h[_DATA_IDX]) ** 2, 1e-12)
                    npw = float((s2 / hp).mean())
                    noise.append(npw)
                    if npw > 0:
                        per_pkt[slot] = 10.0 * np.log10(1.0 / npw)
            if use_vec and np.isfinite(per_pkt).any():
                # seen-but-unestimated slots -> average CSI; missing slots stay
                # NaN -> the codec maps them to the drop sentinel.
                fill = float(np.nanmean(per_pkt))
                for i in range(PKT_PER_IMG):
                    if not np.isfinite(per_pkt[i]) and seen[i]:
                        per_pkt[i] = fill
                codec.set_csi_vector(per_pkt)
                nfin = int(np.isfinite(per_pkt).sum())
                print(f"  [snr] per-packet over {nfin}/{PKT_PER_IMG} pkts, "
                      f"mean={np.nanmean(per_pkt):.2f} dB")
            elif noise:
                avg = float(np.mean(noise))
                snr = 10.0 * np.log10(1.0 / avg) if avg > 0 else args.snr_db
                codec.set_csi(snr)
                print(f"  [snr] live={snr:.2f} dB over {len(noise)}/{len(pn_log)} pkts")
            for pn, _ in pn_log:
                h_by_pn.pop(pn, None); raw_by_pn.pop(pn, None)

        t0 = time.time()
        try:
            img_bytes = codec.decode(symbols)
        except Exception as e:
            print(f"  [!] decode failed: {e}")
            return
        dec_ms = (time.time() - t0) * 1000.0

        rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
            (args.height, args.width, 3))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        total += 1
        last_frame_t = time.time()

        dup = last_saved is not None and _ncc(frame, last_saved) > args.duplicate_threshold
        if not dup:
            unique += 1
            if args.no_timestamp:
                fn = f"{args.name_prefix}_{unique:03d}.png"
            else:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                fn = f"{args.name_prefix}_{unique:03d}_{stamp}.png"
            if args.save:
                cv2.imwrite(os.path.join(args.output_dir, fn), frame)
            last_saved = frame.copy()
            print(f"  [*] frame {total}: NEW #{unique} "
                  f"{'-> ' + fn if args.save else '(not saved)'} ({dec_ms:.1f} ms)")
        else:
            print(f"  [*] frame {total}: duplicate (skipped, {dec_ms:.1f} ms)")

        disp = frame.copy()
        cv2.putText(disp, f"#{unique} | frame {total}{' [DUP]' if dup else ''}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, disp)

    try:
        while True:
            if args.timeout > 0 and total > 0 and time.time() - last_frame_t > args.timeout:
                print(f"\n[*] no frames for {args.timeout}s. exiting.")
                break
            if cur_img is not None and time.time() - last_pdu_t > 0.3:
                print(f"\n[*] idle gap; flushing partial ({sum(seen)}/{PKT_PER_IMG}).")
                flush_and_decode()
                seen = [False] * PKT_PER_IMG; slot_buf.fill(0); pn_log = []
                cur_img = None; anchor_pn = None

            if snr_socket is not None:
                while True:
                    try:
                        raw = snr_socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        sp = pmt.deserialize_str(raw)
                        meta = pmt.car(sp)
                        pn = pmt.to_long(pmt.dict_ref(meta, pmt.intern('packet_num'),
                                                      pmt.from_long(-1)))
                        if pn < 0:
                            continue
                        kind = pmt.symbol_to_string(pmt.dict_ref(
                            meta, pmt.intern('kind'), pmt.intern('?')))
                        if kind == 'h':
                            h_by_pn[pn] = np.array(pmt.c32vector_elements(pmt.cdr(sp)),
                                                   dtype=np.complex64)
                        elif kind == 'raw':
                            raw_by_pn[pn] = pmt.to_double(pmt.dict_ref(
                                meta, pmt.intern('sigma2'), pmt.from_double(float('nan'))))
                    except Exception:
                        pass

            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                if (cv2.waitKey(10) & 0xFF) == ord('q'):
                    break
                continue

            try:
                pdu = pmt.deserialize_str(raw)
                meta, payload = pmt.car(pdu), pmt.cdr(pdu)
                pn = pmt.to_long(pmt.dict_ref(meta, pmt.intern("packet_num"),
                                              pmt.from_long(-1)))
                assert pn >= 0
                iq = np.array(pmt.c32vector_elements(payload), dtype=np.complex64)
            except Exception as e:
                print(f"[!] PDU parse failed: {e}")
                continue

            if len(iq) < PKT:
                continue
            iq = iq[:PKT]
            last_pdu_t = time.time()
            if anchor_pn is None:
                anchor_pn = pn
            delta = pn - anchor_pn
            slot = delta % PKT_PER_IMG
            img_idx = delta // PKT_PER_IMG

            if drop.should_drop(slot):
                print(f"[drop] slot {slot} (pn={pn})")
                continue
            if cur_img is None:
                cur_img = img_idx
            elif img_idx != cur_img:
                flush_and_decode()
                seen = [False] * PKT_PER_IMG; slot_buf.fill(0); pn_log = []
                cur_img = img_idx

            slot_buf[slot * PKT:(slot + 1) * PKT] = iq
            seen[slot] = True
            pn_log.append((pn, slot))

            if slot == PKT_PER_IMG - 1:
                flush_and_decode()
                seen = [False] * PKT_PER_IMG; slot_buf.fill(0); pn_log = []
                cur_img = None

            if args.count > 0 and unique >= args.count:
                print(f"\n[*] received {unique} unique image(s). done.")
                cv2.waitKey(2000)
                break
    except KeyboardInterrupt:
        print("\n[*] stopped.")
    finally:
        cv2.destroyAllWindows()
        print(f"\n{'='*46}\n  frames={total}  unique={unique}\n{'='*46}")


def parse_args():
    p = argparse.ArgumentParser(description="DJSCC-playground unified receiver")
    p.add_argument("--model", required=True, help="HF id | folder | alias | .pth")
    p.add_argument("--aligner", default=None)
    p.add_argument("--comp-ratio", type=float, default=6)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--snr-db", type=float, default=19.0)
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)

    p.add_argument("--port", default="5558")
    p.add_argument("--connect-host", default="127.0.0.1")
    p.add_argument("--use-live-snr", action="store_true")
    p.add_argument("--snr-port", default="5560")

    p.add_argument("--output-dir", default="./received_images")
    p.add_argument("--name-prefix", default="image",
                   help="filename prefix for saved images (default: image -> "
                        "image_001_<stamp>.png)")
    p.add_argument("--no-timestamp", action="store_true",
                   help="drop the timestamp -> <prefix>_001.png (clean sequential names)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--duplicate-threshold", type=float, default=0.92)
    p.add_argument("--timeout", type=float, default=0)
    p.add_argument("--drop-slots", default="")
    p.add_argument("--drop-seed", type=int, default=0)
    p.add_argument("--interleave", action="store_true")
    p.add_argument("--clip-mag", type=float, default=5.0)
    p.add_argument("--renorm", action="store_true")
    p.add_argument("--renorm-target", type=float, default=2.0)
    args = p.parse_args()
    args.save = not args.no_save
    return args


def main() -> int:
    args = parse_args()

    codec = load_codec(
        args.model, role="decoder", device=args.device, packet_len=args.packet_len,
        comp_ratio=args.comp_ratio, N=args.N, snr_db=args.snr_db,
        img_height=args.height, img_width=args.width,
    )
    if codec.output_kind != OutputKind.COMPLEX_SYMBOLS:
        print(f"[!] '{args.model}' is a {codec.output_kind} codec; the bytes path "
              "is not yet wired in socket_rx.py.")
        return 2
    if args.aligner:
        aligner = load_aligner(args.aligner, codec.tcn, codec.h, codec.w,
                               device=codec.device)
        codec.set_aligner(aligner)
        print(f"[*] aligner attached: kind={aligner.kind} mode={aligner.mode}")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect(f"tcp://{args.connect_host}:{args.port}")
    print(f"[*] PULL connected to tcp://{args.connect_host}:{args.port}")

    snr_socket = None
    if args.use_live_snr:
        snr_socket = ctx.socket(zmq.PULL)
        snr_socket.setsockopt(zmq.RCVHWM, 5000)
        snr_socket.connect(f"tcp://{args.connect_host}:{args.snr_port}")
        print(f"[*] live-SNR PULL connected to :{args.snr_port}")

    try:
        receive_loop(codec, socket, snr_socket, args)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        socket.setsockopt(zmq.LINGER, 0); socket.close()
        if snr_socket is not None:
            snr_socket.setsockopt(zmq.LINGER, 0); snr_socket.close()
        ctx.term()


if __name__ == "__main__":
    sys.exit(main())
