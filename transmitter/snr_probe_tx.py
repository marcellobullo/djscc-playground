#!/usr/bin/env python3
"""SNR probe, transmitter side: send a known payload in image-sized bursts, forever.

Pair with ``receiver/snr_probe_rx.py`` to survey receiver positions before the over-the-air
experiment. Each packet carries the same known 960-symbol payload (complex Gaussian, seeded,
power ``--tx-power`` = 2.0 like the DJSCC latents, so the radio sees the same power and PAPR as
during the real runs). A burst is ``--pkts`` packets (default 205 = one 512x768 image at 1/6),
followed by ``--interval`` seconds of silence, exactly like ``socket_tx.py --source folder``.

Uses the same GNU Radio TX flowgraph and ZMQ port as socket_tx.py:

  python transmitter/gnu_radio/djscc_tx.py --device-address <usrp>
  python transmitter/snr_probe_tx.py

Loopback test without radios (PDUs straight to the probe RX, with simulated noise):

  python transmitter/snr_probe_tx.py --direct-zmq --sim-snr 15
  python receiver/snr_probe_rx.py --port 5559 --no-pilots
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pmt
import zmq


def probe_payload(packet_len: int, seed: int, tx_power: float) -> np.ndarray:
    """The known payload (shared with snr_probe_rx.py)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(packet_len) + 1j * rng.standard_normal(packet_len)
    return (x * np.sqrt(tx_power / 2.0)).astype(np.complex64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--pkts", type=int, default=205, help="packets per burst (205 = one image)")
    p.add_argument("--interval", type=float, default=1.0, help="silence between bursts (s)")
    p.add_argument("--bursts", type=int, default=0, help="number of bursts (0 = until Ctrl-C)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tx-power", type=float, default=2.0)
    p.add_argument("--port", default="5556")
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--direct-zmq", action="store_true", help="loopback test: PUSH PDUs to :5559, skip GNU Radio")
    p.add_argument("--sim-snr", type=float, default=None, help="loopback test only: add AWGN at this SNR (dB)")
    a = p.parse_args()

    x = probe_payload(a.packet_len, a.seed, a.tx_power)
    burst = np.tile(x, a.pkts)
    ctx = zmq.Context()
    if a.direct_zmq:
        sock = ctx.socket(zmq.PUSH); sock.setsockopt(zmq.SNDHWM, 5000)
        sock.bind(f"tcp://{a.bind_host}:5559")
        print(f"[*] loopback PUSH on :5559" + (f", simulated SNR {a.sim_snr} dB" if a.sim_snr is not None else ""))
    else:
        sock = ctx.socket(zmq.PUB); sock.setsockopt(zmq.SNDHWM, 5000)
        sock.bind(f"tcp://{a.bind_host}:{a.port}")
        print(f"[*] PUB on {a.bind_host}:{a.port} -> GNU Radio TX flowgraph")
    print(f"[*] burst = {a.pkts} x {a.packet_len} known symbols (seed {a.seed}, power {a.tx_power}), "
          f"every {a.interval:g} s. Ctrl-C to stop.")
    time.sleep(2.0)

    rng = np.random.default_rng()
    pn, k = 0, 0
    try:
        while a.bursts <= 0 or k < a.bursts:
            if a.direct_zmq:
                for i in range(a.pkts):
                    y = x
                    if a.sim_snr is not None:
                        s = np.sqrt(a.tx_power / 10 ** (a.sim_snr / 10) / 2)
                        y = (x + s * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))).astype(np.complex64)
                    meta = pmt.dict_add(pmt.make_dict(), pmt.intern("packet_num"), pmt.from_long(pn))
                    sock.send(pmt.serialize_str(pmt.cons(meta, pmt.init_c32vector(len(y), y.tolist()))))
                    pn += 1
            else:
                sock.send(burst.tobytes())
            k += 1
            print(f"\r  burst {k} sent", end="", flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[*] {k} bursts sent.")
        sock.setsockopt(zmq.LINGER, 0); sock.close(); ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
