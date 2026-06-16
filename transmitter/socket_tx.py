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
import threading
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


# ── interactive camera GUI ───────────────────────────────────────────────────
BTN_COLOR_IDLE   = (34, 139, 34)
BTN_COLOR_HOVER  = (0, 200, 0)
BTN_COLOR_ACTIVE = (0, 80, 200)
BTN_COLOR_DONE   = (0, 180, 180)
OVERLAY_ALPHA    = 0.55
FONT             = cv2.FONT_HERSHEY_SIMPLEX


def _btn_rect(w, h):
    bw, bh = 260, 54
    x1 = (w - bw) // 2
    return x1, h - bh - 20, x1 + bw, h - 20


def _point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def _draw_button(canvas, w, h, color, text):
    x1, y1, x2, y2 = _btn_rect(w, h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, canvas, 1 - OVERLAY_ALPHA, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    tw, th = cv2.getTextSize(text, FONT, 0.72, 2)[0]
    cv2.putText(canvas, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(canvas, shot_count, status):
    cv2.putText(canvas, f"Shots sent: {shot_count}", (12, 34),
                FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, status, (12, 68), FONT, 0.6, (0, 220, 220), 2, cv2.LINE_AA)


class _SendState:
    IDLE, SENDING, DONE = "idle", "sending", "done"

    def __init__(self):
        self.state = self.IDLE
        self.shot_count = 0
        self._lock = threading.Lock()

    def start_send(self):
        with self._lock:
            self.state = self.SENDING

    def finish_send(self):
        with self._lock:
            self.state = self.DONE
            self.shot_count += 1

    def ack_done(self):
        with self._lock:
            if self.state == self.DONE:
                self.state = self.IDLE

    @property
    def is_sending(self):
        with self._lock:
            return self.state == self.SENDING

    def snapshot(self):
        with self._lock:
            return self.state, self.shot_count


def _mouse_cb(event, x, y, flags, param):
    state, w, h, trigger, mouse_pos = param
    mouse_pos[0], mouse_pos[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN \
            and _point_in_rect(x, y, _btn_rect(w, h)) and not state.is_sending:
        trigger[0] = True


def camera_interactive(codec, socket, args):
    cam = int(args.path) if args.path.isdigit() else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[!] cannot open camera {cam}")
        return
    time.sleep(0.8)
    for _ in range(5):
        cap.read()

    win = "DJSCC-playground TX"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    state = _SendState()
    trigger = [False]
    mouse_pos = [-1, -1]
    captured = [None]
    done_flash_end = [0.0]
    cv2.setMouseCallback(win, _mouse_cb,
                         param=(state, args.width, args.height, trigger, mouse_pos))

    send_warmup(codec, socket, args)
    print("\n[*] camera live. Click CAPTURE & SEND or press Space ('q'/Esc quits).\n")

    def _send_worker(frame_bgr):
        state.start_send()
        try:
            encode_and_send(codec, socket, args,
                            prepare_frame(frame_bgr, args.width, args.height),
                            label=f"shot #{state.shot_count + 1}")
        finally:
            state.finish_send()
            done_flash_end[0] = time.time() + 1.2

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("[!] camera read failed.")
                break
            display = cv2.resize(raw, (args.width, args.height))
            now = time.time()
            cur, shots = state.snapshot()

            if cur == _SendState.SENDING:
                if captured[0] is not None:
                    display = cv2.resize(captured[0], (args.width, args.height))
                color, text = BTN_COLOR_ACTIVE, "  Encoding + Sending..."
                status = "Transmitting over SDR..."
            elif cur == _SendState.DONE or now < done_flash_end[0]:
                color, text = BTN_COLOR_DONE, "  Sent!"
                status = f"Shot #{shots} transmitted"
                state.ack_done()
            else:
                hover = _point_in_rect(mouse_pos[0], mouse_pos[1],
                                       _btn_rect(args.width, args.height))
                color = BTN_COLOR_HOVER if hover else BTN_COLOR_IDLE
                text = "[ CAPTURE & SEND ]"
                status = "Space or click to capture" if shots == 0 \
                    else f"Last: shot #{shots}"

            _draw_hud(display, shots, status)
            _draw_button(display, args.width, args.height, color, text)
            cv2.imshow(win, display)

            if trigger[0] and not state.is_sending:
                trigger[0] = False
                captured[0] = raw.copy()
                threading.Thread(target=_send_worker, args=(raw.copy(),),
                                 daemon=True).start()
                print(f"[*] shot #{shots + 1} captured & queued")

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and not state.is_sending:
                trigger[0] = True
            elif key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[*] session ended. total shots: {state.shot_count}")


def camera_auto(codec, socket, args):
    cam = int(args.path) if args.path.isdigit() else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[!] cannot open camera {cam}")
        return
    time.sleep(1.0)
    for _ in range(5):
        cap.read()
    send_warmup(codec, socket, args)
    try:
        for i in range(args.shots):
            if i:
                time.sleep(args.interval)
            ok, frame = cap.read()
            if not ok:
                continue
            print(f"\n[*] shot {i+1}/{args.shots}")
            encode_and_send(codec, socket, args,
                            prepare_frame(frame, args.width, args.height),
                            label=f"shot {i+1}/{args.shots}")
    finally:
        cap.release()


def run_camera(codec, socket, args):
    """--shots 0 (default) -> interactive button GUI; --shots N -> auto-capture."""
    if args.shots == 0:
        camera_interactive(codec, socket, args)
    else:
        camera_auto(codec, socket, args)


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
    p.add_argument("--shots", type=int, default=0,
                   help="camera source: 0 = interactive CAPTURE & SEND button "
                        "(default), N>0 = auto-capture N shots --interval apart.")
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
