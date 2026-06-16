#!/usr/bin/env python3
"""Unified DJSCC-playground transmitter — model-agnostic.

One script for every model. The encoder is selected at runtime with ``--model``
(a HuggingFace repo id, a local HF folder, a short alias, or a raw ``.pth``), and
an optional ``--aligner`` is composed on top. The encoder runs in-process and
publishes channel symbols to a slim GNU Radio flowgraph (OFDM + USRP) over ZMQ.

Examples:
    # bundled custom-DJSCC checkpoint, 3 auto shots
    python transmitter/socket_tx.py --model checkpoints/custom_djscc/compratio-6_latest.pth \\
        --comp-ratio 6 --source camera --shots 3

    # a published HF model + a conv aligner, single file
    python transmitter/socket_tx.py --model marcellob/djscc-r6 \\
        --aligner checkpoints/aligners/aligner_conv.pth --source file --path img.png
"""

from __future__ import annotations

import argparse
import glob
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


# ── symbol interleaver (block; must match --interleave on the RX) ────────────
_INTERLEAVE_CACHE: dict = {}


def _make_interleaver(num_items: int, num_slots: int):
    bps = num_items // num_slots
    p = np.arange(num_items, dtype=np.int64)
    perm_fwd = (p % num_slots) * bps + (p // num_slots)
    perm_inv = np.empty(num_items, dtype=np.int64)
    perm_inv[perm_fwd] = p
    return perm_fwd, perm_inv


def _interleave_symbols(symbols: np.ndarray, packet_len: int) -> np.ndarray:
    n = len(symbols)
    num_slots = n // packet_len
    key = (n, num_slots)
    perm_inv = _INTERLEAVE_CACHE.get(key)
    if perm_inv is None:
        _, perm_inv = _make_interleaver(n, num_slots)
        _INTERLEAVE_CACHE[key] = perm_inv
    return symbols[perm_inv]


# ── publishing ───────────────────────────────────────────────────────────────
_PN = [0]


def publish_symbols(socket, topic, symbols, label=""):
    payload = symbols.tobytes()
    if topic:
        socket.send_multipart([topic, payload])
    else:
        socket.send(payload)
    print(f"  TX publish {len(symbols)} cf32 ({len(payload)/1024:.1f} KiB)"
          + (f" [{label}]" if label else ""))


def publish_symbols_pdu(socket, topic, symbols, packet_len, label=""):
    if len(symbols) % packet_len:
        raise ValueError(f"{len(symbols)} symbols not a multiple of {packet_len}")
    n_pkts = len(symbols) // packet_len
    start = _PN[0]
    for i in range(n_pkts):
        chunk = symbols[i * packet_len:(i + 1) * packet_len]
        meta = pmt.dict_add(pmt.make_dict(), pmt.intern("packet_num"),
                            pmt.from_long(_PN[0]))
        pdu = pmt.cons(meta, pmt.init_c32vector(len(chunk), chunk.tolist()))
        socket.send(pmt.serialize_str(pdu))
        _PN[0] += 1
    print(f"  TX direct-PDU {n_pkts}x{packet_len} (pn {start}..{_PN[0]-1})"
          + (f" [{label}]" if label else ""))


def prepare_frame(image_bgr, width, height) -> bytes:
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (width, height)), cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


def encode_and_send(codec, socket, args, frame_bytes, label=""):
    t0 = time.time()
    symbols = codec.encode(frame_bytes)
    if args.interleave:
        symbols = _interleave_symbols(symbols, codec.packet_len)
    print(f"  encode {len(symbols)} symbols in {(time.time()-t0)*1000:.1f} ms"
          + ("  [interleaved]" if args.interleave else ""))
    for r in range(args.repeat):
        rl = f"{label} {r+1}/{args.repeat}" if label else f"{r+1}/{args.repeat}"
        if args.direct_zmq:
            publish_symbols_pdu(socket, args.topic, symbols, codec.packet_len, rl)
        else:
            publish_symbols(socket, args.topic, symbols, rl)
        if r < args.repeat - 1:
            time.sleep(args.repeat_interval)


def send_warmup(codec, socket, args):
    if args.warmup_frames <= 0:
        return
    print(f"[*] {args.warmup_frames} warmup frames for OFDM sync...")
    dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8).tobytes()
    symbols = codec.encode(dummy)
    if args.interleave:
        symbols = _interleave_symbols(symbols, codec.packet_len)
    for i in range(args.warmup_frames):
        if args.direct_zmq:
            publish_symbols_pdu(socket, args.topic, symbols, codec.packet_len, f"warmup {i+1}")
        else:
            publish_symbols(socket, args.topic, symbols, f"warmup {i+1}")
        time.sleep(args.warmup_interval)


# ── capture modes ─────────────────────────────────────────────────────────────
def run_file(codec, socket, args):
    frame = cv2.imread(args.path)
    if frame is None:
        print(f"[!] cannot read {args.path}")
        return
    send_warmup(codec, socket, args)
    encode_and_send(codec, socket, args, prepare_frame(frame, args.width, args.height),
                    label=os.path.basename(args.path))


def run_folder(codec, socket, args):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    imgs = [f for f in sorted(glob.glob(os.path.join(args.path, "*.*")))
            if os.path.splitext(f)[1].lower() in exts]
    if not imgs:
        print(f"[!] no images in {args.path}")
        return
    send_warmup(codec, socket, args)
    for i, p in enumerate(imgs):
        if i:
            time.sleep(args.interval)
        frame = cv2.imread(p)
        if frame is not None:
            encode_and_send(codec, socket, args,
                            prepare_frame(frame, args.width, args.height),
                            label=os.path.basename(p))


def run_camera(codec, socket, args):
    cam = int(args.path) if args.path.isdigit() else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[!] cannot open camera {cam}")
        return
    time.sleep(1.0)
    for _ in range(5):
        cap.read()
    send_warmup(codec, socket, args)
    shots = args.shots if args.shots > 0 else 1
    try:
        for i in range(shots):
            if i:
                time.sleep(args.interval)
            ok, frame = cap.read()
            if not ok:
                continue
            print(f"\n[*] shot {i+1}/{shots}")
            encode_and_send(codec, socket, args,
                            prepare_frame(frame, args.width, args.height),
                            label=f"shot {i+1}/{shots}")
    finally:
        cap.release()


def parse_args():
    p = argparse.ArgumentParser(description="DJSCC-playground unified transmitter")
    p.add_argument("--model", required=True,
                   help="HF id | HF folder | alias | raw .pth")
    p.add_argument("--aligner", default=None, help="aligner .pth | folder | HF id")
    p.add_argument("--comp-ratio", type=float, default=6,
                   help="inverse compression ratio for raw .pth (default: 6)")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)

    p.add_argument("--source", default="camera", choices=["camera", "file", "folder"])
    p.add_argument("--path", default="0")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--repeat-interval", type=float, default=0.5)
    p.add_argument("--warmup-frames", type=int, default=0,
                   help="OFDM-sync warmup frames sent before the first image. "
                        "Default 0 (off); pass a positive value to enable.")
    p.add_argument("--warmup-interval", type=float, default=0.5)

    p.add_argument("--port", default="5556")
    p.add_argument("--topic", default="")
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--direct-zmq", action="store_true",
                   help="PUSH PDUs straight to the RX PULL on 5559 (skip GR)")
    p.add_argument("--interleave", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.topic = args.topic.encode() if args.topic else b""

    codec = load_codec(
        args.model, role="encoder", device=args.device, packet_len=args.packet_len,
        comp_ratio=args.comp_ratio, N=args.N,
        img_height=args.height, img_width=args.width,
    )
    if codec.output_kind != OutputKind.COMPLEX_SYMBOLS:
        print(f"[!] '{args.model}' is a {codec.output_kind} codec; the bytes path "
              "(conventional flowgraph) is not yet wired in socket_tx.py.")
        return 2

    if args.aligner:
        aligner = load_aligner(args.aligner, codec.tcn, codec.h, codec.w,
                               device=args.device)
        codec.set_aligner(aligner)
        print(f"[*] aligner attached: kind={aligner.kind} mode={aligner.mode}")

    print(f"[*] expected symbols/image: {codec.expected_complex_items} "
          f"({codec.expected_complex_items // codec.packet_len + 1} packets)")

    ctx = zmq.Context()
    if args.direct_zmq:
        addr = f"tcp://{args.bind_host}:5559"
        socket = ctx.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(addr)
        print(f"[*] [DEBUG] direct-ZMQ PUSH bound to {addr} (RX must use --port 5559)")
    else:
        addr = f"tcp://{args.bind_host}:{args.port}"
        socket = ctx.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 5000)
        socket.bind(addr)
        print(f"[*] ZMQ PUB bound to {addr}")

    print("[*] waiting 4s for subscribers...")
    time.sleep(4.0)

    try:
        {"file": run_file, "folder": run_folder, "camera": run_camera}[args.source](
            codec, socket, args)
        print("\n[*] all transmissions complete.")
        return 0
    except KeyboardInterrupt:
        print("\n[*] interrupted.")
        return 130
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    sys.exit(main())
