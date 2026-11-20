#!/usr/bin/env python3
"""SNR probe, receiver side: live SNR and packet loss while you move the receiver.

Pair with ``transmitter/snr_probe_tx.py`` (known payload in image-sized bursts). Runs on the
same GNU Radio RX flowgraph as socket_rx.py (payload PDUs on :5558, pilot statistics on :5560)
and reports, per burst:

  dec   decoder-side SNR  = P_x / mean|y - x|^2   -- what a DJSCC decoder actually sees
                            (noise + channel-estimate error + residual gain/phase error)
  da    data-aided SNR    = |g|^2 P_x / mean|y - g x|^2, g = one complex gain per packet
                            (pure noise floor; dec << da means a gain/phase problem, not noise)
  pil   pilot estimate    = the live-CSI estimate socket_rx.py feeds the decoder
                            (jscc/snr_estimation.py); should track `dec` within ~1 dB
  loss  packets lost (gaps in the packet numbers) and |g| (should stay ~1)

Survey mode: ``--tag <position> --duration 30`` measures for 30 s, prints a summary
(median / p10 / p90 over bursts) and appends one row to ``--log`` (CSV), so positions can be
compared afterwards. Without --duration it runs until Ctrl-C (summary on exit).

  python receiver/gnu_radio/djscc_rx.py --device-address <usrp>
  python receiver/snr_probe_rx.py --tag pos3_4m --duration 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

import numpy as np
import pmt
import zmq

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "transmitter"))
from jscc.snr_estimation import pooled_snr_eff_db  # noqa: E402
from snr_probe_tx import probe_payload  # noqa: E402


def db(x):
    return 10.0 * np.log10(max(float(x), 1e-12))


class Burst:
    def __init__(self):
        self.pns, self.err_dec, self.err_da, self.gain, self.phase = [], [], [], [], []

    def add(self, pn, y, x, px):
        g = np.vdot(x, y) / np.vdot(x, x)
        self.pns.append(pn)
        self.err_dec.append(float(np.mean(np.abs(y - x) ** 2)))
        self.err_da.append(float(np.mean(np.abs(y - g * x) ** 2)))
        self.gain.append(abs(g)); self.phase.append(np.degrees(np.angle(g)))

    def summary(self, px, pil_by_pn, tx_power, pkts):
        # Loss from the packet numbers, ignoring stray numbers: a corrupted header that still passes the
        # header CRC carries a random packet number, which would inflate max-min. Keep the packets whose
        # number is within one burst of the median, and count the missing ones against the burst length.
        pns = np.array(self.pns)
        med = np.median(pns)
        good = np.abs(pns - med) < pkts
        n = int(good.sum())
        span = int(pns[good].max() - pns[good].min() + 1)
        lost = max(0, min(span, pkts * max(1, round(span / pkts))) - n) if n else 0
        dec = db(px / np.mean(self.err_dec))
        da = db(np.mean(np.square(self.gain)) * px / np.mean(self.err_da))
        st = [pil_by_pn.pop(q) for q in self.pns if q in pil_by_pn]
        pil = pooled_snr_eff_db(st, tx_power=tx_power) if st else float("nan")
        return dict(n=n, lost=lost, stray=len(pns) - n, dec=dec, da=da, pil=pil,
                    gain=float(np.mean(self.gain)), phase_std=float(np.std(self.phase)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="5558")
    p.add_argument("--connect-host", default="127.0.0.1")
    p.add_argument("--snr-port", default="5560")
    p.add_argument("--no-pilots", action="store_true", help="do not bind the pilot-statistics port")
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--pkts", type=int, default=205, help="packets per burst, must match snr_probe_tx.py")
    p.add_argument("--min-pkts", type=int, default=20,
                   help="bursts with fewer packets are shown but left out of the summary (noise triggers)")
    p.add_argument("--seed", type=int, default=1234, help="must match snr_probe_tx.py")
    p.add_argument("--tx-power", type=float, default=2.0, help="must match snr_probe_tx.py")
    p.add_argument("--burst-gap", type=float, default=0.25, help="idle time (s) that closes a burst")
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = Ctrl-C)")
    p.add_argument("--tag", default="", help="position label, e.g. pos3_4m_LOS")
    p.add_argument("--note", default="", help="free text stored in the log (gains, antenna, ...)")
    p.add_argument("--log", default=os.path.join(_ROOT, "results", "ota", "snr_survey.csv"))
    a = p.parse_args()

    x = probe_payload(a.packet_len, a.seed, a.tx_power)
    px = float(np.mean(np.abs(x) ** 2))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL); sock.setsockopt(zmq.RCVHWM, 5000)
    sock.connect(f"tcp://{a.connect_host}:{a.port}")
    snr_sock = None
    if not a.no_pilots:
        snr_sock = ctx.socket(zmq.PULL); snr_sock.setsockopt(zmq.RCVHWM, 5000)
        snr_sock.bind(f"tcp://*:{a.snr_port}")
    print(f"[*] payload PDUs <- {a.connect_host}:{a.port}" + ("" if snr_sock is None else f", pilot stats <- :{a.snr_port}"))
    print(f"[*] waiting for probe bursts{f' (tag {a.tag})' if a.tag else ''} ...")
    print(f"{'time':>8s} {'pkts':>5s} {'lost':>5s} {'dec dB':>7s} {'da dB':>7s} {'pil dB':>7s} {'|g|':>5s} {'φ std':>6s}")

    pil_by_pn, rows = {}, []
    cur, last_t, t0 = None, time.time(), time.time()
    poller = zmq.Poller(); poller.register(sock, zmq.POLLIN)
    if snr_sock is not None:
        poller.register(snr_sock, zmq.POLLIN)

    def close():
        nonlocal cur
        if cur is not None and cur.pns:
            r = cur.summary(px, pil_by_pn, a.tx_power, a.pkts)
            rows.append(r)
            print(f"{time.time() - t0:8.1f} {r['n']:5d} {r['lost']:5d} {r['dec']:7.2f} {r['da']:7.2f} "
                  f"{r['pil']:7.2f} {r['gain']:5.2f} {r['phase_std']:6.1f}", flush=True)
        cur = None
        if len(pil_by_pn) > 10000:
            pil_by_pn.clear()

    try:
        while not a.duration or time.time() - t0 < a.duration:
            ev = dict(poller.poll(50))
            if snr_sock is not None and snr_sock in ev:
                while True:
                    try:
                        raw = snr_sock.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        sp = pmt.deserialize_str(raw); meta = pmt.car(sp)
                        g = lambda k, d: pmt.dict_ref(meta, pmt.intern(k), d)
                        if pmt.symbol_to_string(g("kind", pmt.intern("?"))) != "pilots":
                            continue
                        pn = pmt.to_long(g("packet_num", pmt.from_long(-1)))
                        f = lambda k: pmt.to_double(g(k, pmt.from_double(float("nan"))))
                        pil_by_pn[pn] = (f("slope"), f("sigma2"), f("amp2"), pmt.to_long(g("n_sym", pmt.from_long(0))))
                    except Exception:
                        pass
            if sock in ev:
                while True:
                    try:
                        raw = sock.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    pdu = pmt.deserialize_str(raw)
                    pn = pmt.to_long(pmt.dict_ref(pmt.car(pdu), pmt.intern("packet_num"), pmt.from_long(-1)))
                    y = np.array(pmt.c32vector_elements(pmt.cdr(pdu)), dtype=np.complex64)
                    if len(y) < a.packet_len:
                        continue
                    if cur is None:
                        cur = Burst()
                    cur.add(pn, y[:a.packet_len], x, px)
                    last_t = time.time()
            if cur is not None and time.time() - last_t > a.burst_gap:
                close()
    except KeyboardInterrupt:
        pass
    close()

    if not rows:
        print("\n[!] no bursts received (is the probe TX running? flowgraph synced?)")
        return 1
    shown = len(rows)
    rows = [r for r in rows if r["n"] >= a.min_pkts]
    if not rows:
        print(f"\n[!] {shown} bursts, none with >= {a.min_pkts} packets (no stable link here)")
        return 1
    dec = np.array([r["dec"] for r in rows]); da = np.array([r["da"] for r in rows])
    pil = np.array([r["pil"] for r in rows]); n = sum(r["n"] for r in rows); lost = sum(r["lost"] for r in rows)
    s = dict(time=dt.datetime.now().isoformat(timespec="seconds"), tag=a.tag, note=a.note, bursts=len(rows),
             dec_med=np.median(dec), dec_p10=np.percentile(dec, 10), dec_p90=np.percentile(dec, 90),
             da_med=np.median(da), pil_med=np.nanmedian(pil) if np.isfinite(pil).any() else float("nan"),
             loss_pct=100 * lost / max(n + lost, 1), gain=np.mean([r["gain"] for r in rows]))
    print(f"\n== {a.tag or 'summary'}: {len(rows)} bursts | decoder-side SNR median {s['dec_med']:.2f} dB "
          f"(p10 {s['dec_p10']:.2f}, p90 {s['dec_p90']:.2f}) | data-aided {s['da_med']:.2f} | pilots {s['pil_med']:.2f} "
          f"| loss {s['loss_pct']:.2f}% | |g| {s['gain']:.2f}")
    if a.tag:
        os.makedirs(os.path.dirname(a.log), exist_ok=True)
        new = not os.path.isfile(a.log)
        with open(a.log, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(s))
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.3f}" if isinstance(v, (float, np.floating)) else v) for k, v in s.items()})
        print(f"[*] appended to {a.log}")
    sock.setsockopt(zmq.LINGER, 0); sock.close()
    if snr_sock is not None:
        snr_sock.setsockopt(zmq.LINGER, 0); snr_sock.close()
    ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
