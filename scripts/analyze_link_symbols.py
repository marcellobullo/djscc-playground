#!/usr/bin/env python
"""Measure what the real link does to the DJSCC latent, symbol by symbol.

The TX latent is a deterministic function of (image, encoder), so it can be
re-created offline and compared with the symbols the RX actually handed to the
decoder (dumped by ``receiver/socket_rx.py --dump-dir``). This separates the
link's *effective* channel from the emulator's nominal SNR:

  * complex gain g (LS fit r ~ g*s) and effective SNR after removing g
  * SDR vs. the clean latent (what the decoder sees, gain error included)
  * phase / gain drift across the 20 OFDM symbols of a packet
  * per-subcarrier SNR (48 data carriers) and per-packet SNR
  * AM/AM (|r|/|s| vs |s|) -> amplifier compression / clipping
  * excess kurtosis of the error -> non-Gaussian impairments

Usage:
    python receiver/socket_rx.py --model <m> --interleave --dump-dir dumps/snr28 ...
    python scripts/analyze_link_symbols.py --dump-dir dumps/snr28 \\
        --model marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi \\
        --images /path/to/kodak [--plot link_snr28.png]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "receiver"))

from jscc import load_codec  # noqa: E402
from socket_rx import _make_interleaver  # noqa: E402

N_DATA = 48  # data subcarriers per OFDM symbol


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def analyze(r, s, seen, air, pkt_len):
    """r, s: received / clean latent (decoder order); air: on-air index per symbol."""
    slot = air // pkt_len
    ok = seen[slot]
    r, s, air = r[ok], s[ok], air[ok]
    g = np.vdot(s, r) / np.vdot(s, s)
    e = r - g * s
    ps = np.mean(np.abs(s) ** 2)
    out = {
        "gain_abs": abs(g), "gain_phase_deg": np.degrees(np.angle(g)),
        "snr_eff_db": db(abs(g) ** 2 * ps / np.mean(np.abs(e) ** 2)),
        "sdr_vs_clean_db": db(ps / np.mean(np.abs(r - s) ** 2)),
        "err_excess_kurtosis": float(np.mean(np.abs(e) ** 4) / np.mean(np.abs(e) ** 2) ** 2 - 2.0),
        "slots_missing": int((~seen).sum()),
    }
    sym = (air % pkt_len) // N_DATA                  # OFDM symbol index inside the packet
    car = (air % pkt_len) % N_DATA                   # data-carrier index
    per_sym = [np.vdot(s[sym == k], r[sym == k]) / np.vdot(s[sym == k], s[sym == k])
               for k in range(pkt_len // N_DATA)]
    out["phase_drift_deg_per_pkt"] = float(np.degrees(np.angle(per_sym[-1] / per_sym[0])))
    out["gain_drift_db_per_pkt"] = float(db(abs(per_sym[-1]) ** 2 / abs(per_sym[0]) ** 2))
    out["per_carrier_snr_db"] = [float(db(abs(g) ** 2 * np.mean(np.abs(s[car == k]) ** 2)
                                          / np.mean(np.abs(e[car == k]) ** 2))) for k in range(N_DATA)]
    pk = air // pkt_len
    out["per_packet_snr_db"] = [float(db(abs(g) ** 2 * np.mean(np.abs(s[pk == k]) ** 2)
                                         / np.mean(np.abs(e[pk == k]) ** 2)))
                                for k in np.unique(pk)]
    mag = np.abs(s)
    edges = np.quantile(mag, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(mag, edges[1:-1]), 0, 9)
    out["amam"] = [(float(mag[idx == b].mean()), float(np.mean(np.abs(r[idx == b])) / np.mean(mag[idx == b]) / abs(g)))
                   for b in range(10)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--model", required=True, help="the encoder used at the TX")
    ap.add_argument("--images", required=True, help="folder with the transmitted originals")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--packet-len", type=int, default=960)
    ap.add_argument("--plot", default=None, help="save a diagnostic figure here")
    args = ap.parse_args()

    enc = load_codec(args.model, "encoder", packet_len=args.packet_len)
    exp = enc.expected_complex_items
    clean = []
    for p in sorted(glob.glob(os.path.join(args.images, "*.png"))):
        bgr = cv2.resize(cv2.imread(p), (args.width, args.height))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        clean.append((os.path.basename(p), enc.encode(rgb.tobytes())[:exp]))

    rows = []
    for f in sorted(glob.glob(os.path.join(args.dump_dir, "frame_*.npz"))):
        d = np.load(f)
        r, seen = d["symbols"], d["seen"]
        pkt = int(d["packet_len"])
        n_slots = len(seen)
        if bool(d["interleave"]):
            perm_fwd, _ = _make_interleaver(n_slots * pkt, n_slots)
            air = np.asarray(perm_fwd[:exp])
        else:
            air = np.arange(exp)
        # match the frame to its original by normalized correlation
        cs = [abs(np.vdot(s, r)) / (np.linalg.norm(s) * np.linalg.norm(r) + 1e-12) for _, s in clean]
        i = int(np.argmax(cs))
        res = analyze(r, clean[i][1], seen, air, pkt)
        res["frame"], res["image"], res["corr"] = os.path.basename(f), clean[i][0], cs[i]
        rows.append(res)
        print(f"{res['frame']} {res['image']:<12} corr={res['corr']:.3f} |g|={res['gain_abs']:.3f} "
              f"∠g={res['gain_phase_deg']:+.1f}° SNR_eff={res['snr_eff_db']:.2f} dB "
              f"SDR={res['sdr_vs_clean_db']:.2f} dB drift={res['phase_drift_deg_per_pkt']:+.2f}° "
              f"kurt={res['err_excess_kurtosis']:+.2f} miss={res['slots_missing']}")
    if not rows:
        sys.exit(f"no frame_*.npz in {args.dump_dir}")

    med = lambda k: float(np.median([r[k] for r in rows]))
    print("\nmedian: " + "  ".join(f"{k}={med(k):.3f}" for k in
          ("gain_abs", "snr_eff_db", "sdr_vs_clean_db", "phase_drift_deg_per_pkt",
           "gain_drift_db_per_pkt", "err_excess_kurtosis")))

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        ax[0].plot(np.median([r["per_carrier_snr_db"] for r in rows], 0), "o-")
        ax[0].set(title="Effective SNR per data carrier", xlabel="data carrier", ylabel="dB")
        for r in rows[:5]:
            ax[1].plot(r["per_packet_snr_db"], lw=0.8)
        ax[1].set(title="Effective SNR per packet", xlabel="packet", ylabel="dB")
        am = np.median([[a[1] for a in r["amam"]] for r in rows], 0)
        ax[2].plot([a[0] for a in rows[0]["amam"]], am, "o-")
        ax[2].axhline(1, ls=":", c="k")
        ax[2].set(title="AM/AM (|r|/|g s| vs |s|)", xlabel="|s|", ylabel="normalized gain")
        for a in ax:
            a.grid(ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"[*] figure -> {args.plot}")


if __name__ == "__main__":
    main()
