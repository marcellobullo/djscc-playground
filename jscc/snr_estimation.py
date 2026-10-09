"""Receiver-side SNR estimation from the pilot tones.

Why pilots: they measure the in-band noise directly (unlike the guard-bin nulls, which a USRP's decimation filter
attenuates) and need no knowledge of the equalizer / channel-estimate scale (unlike nulls+taps, which assumed a unit
signal power and over-read the decoder-side SNR by 2.7-3.8 dB; the old pilot logger compared the pilots to +-1, but
they arrive at amplitude ~1.9 - the static payload equalizer does not touch the pilot carriers - so it was off by
10-28 dB).

Per frame, ``pilot_snr_logger`` sends three numbers computed from its T OFDM symbols x P pilot tones:
  slope : residual-CFO phase ramp (rad / OFDM symbol), from consecutive-symbol pilot products
  s2    : noise power per pilot after the ramp is removed (variance of consecutive differences / 2)
  a2    : debiased pilot-tone power  |mean over symbols|^2 - s2 / T
Channel SNR on the data carriers = a2 / s2 * tx_power / pilot_power (tx_power = 2.0 for the codecs: P = 1 per real
latent value). The decoder-side SNR is lower:
  1/SNR_eff = (1 + LOSS_C0) / SNR_ch + LOSS_K / SNR_ch^2 + ramp
- LOSS_C0, LOSS_K: receiver losses that scale with the noise (channel-estimate error, sync/CFO estimation noise);
  the second-order term makes the loss grow at low SNR. Fitted in the offline replica (seed 0) and checked on held-out
  data (seed 1): image-level error within +-0.3 dB from 4 dB up (was up to +2.6 dB with a single constant).
  The measured channel-estimate error alone explains only ~0.7 dB of the ~1.7 dB loss, so a constant model is used.
- ramp = slope^2 (T^2 - 1) / 12 when nothing downstream removes the phase ramp (no cfo_ramp_corrector).
Scripts: scripts/investigation/snr_est_eval.py (harness), snr_est_fit.py (fit / held-out evaluation).
"""
from __future__ import annotations

import numpy as np

LOSS_C0 = 0.43
LOSS_K = 1.75
MIN_DB, MAX_DB = -20.0, 45.0


def pilot_stats(pilots: np.ndarray, pilot_symbols: np.ndarray):
    """Reference implementation of what ``pilot_snr_logger`` computes. pilots: [T, P]. Returns (slope, s2, a2, T)."""
    P = np.asarray(pilots) * np.conj(np.asarray(pilot_symbols))[None, :]
    T = len(P)
    if T < 3:
        return None
    slope = float(np.angle(np.sum(P[1:] * np.conj(P[:-1]))))
    Pd = P * np.exp(-1j * slope * np.arange(T))[:, None]
    s2 = float(np.mean(np.abs(Pd[1:] - Pd[:-1]) ** 2) / 2)
    a2 = float(np.mean(np.abs(Pd.mean(axis=0)) ** 2) - s2 / T)
    return slope, s2, a2, T


def _eff_linear(s2, a2, slope, n_sym, tx_power, pilot_power, c0, k, ramp_corrected):
    snr_ch = max(float(a2), 1e-12) / max(float(s2), 1e-12) * tx_power / pilot_power
    ramp = 0.0 if ramp_corrected else float(slope) ** 2 * (n_sym * n_sym - 1) / 12.0
    return 1.0 / ((1.0 + c0) / snr_ch + k / snr_ch ** 2 + ramp)


def _db(x):
    return float(np.clip(10.0 * np.log10(max(x, 1e-12)), MIN_DB, MAX_DB))


def snr_eff_db(s2: float, a2: float, slope: float, n_sym: int, *, tx_power: float = 2.0, pilot_power: float = 1.0,
               c0: float = LOSS_C0, k: float = LOSS_K, ramp_corrected: bool = True) -> float:
    """Decoder-side (post-equalization) payload SNR in dB from one frame's pilot statistics."""
    return _db(_eff_linear(s2, a2, slope, n_sym, tx_power, pilot_power, c0, k, ramp_corrected))


def pooled_snr_eff_db(stats, erasure_frac: float = 0.0, *, tx_power: float = 2.0, pilot_power: float = 1.0,
                      c0: float = LOSS_C0, k: float = LOSS_K, ramp_corrected: bool = True) -> float:
    """One scalar for a whole image. Noise, tone power and ramp are pooled across the received packets (linear
    domain, unbiased at low SNR). ``erasure_frac`` = share of zero-filled (lost) packets: their error equals the
    signal, so the SNR the decoder sees is 1 / ((1 - p) / SNR + p)."""
    stats = [s for s in stats if s is not None and np.isfinite(s[1]) and np.isfinite(s[2]) and s[1] > 0]
    if not stats:
        return float("nan")
    st = np.array([s[:4] for s in stats], dtype=float)
    snr = _eff_linear(st[:, 1].mean(), st[:, 2].mean(), np.sqrt(np.mean(st[:, 0] ** 2)), st[:, 3].mean(),
                      tx_power, pilot_power, c0, k, ramp_corrected)
    p = float(np.clip(erasure_frac, 0.0, 1.0))
    return _db(1.0 / ((1.0 - p) / snr + p))


def per_packet_snr_db(stats_by_slot, window: int = 5, *, tx_power: float = 2.0, pilot_power: float = 1.0,
                      c0: float = LOSS_C0, k: float = LOSS_K, ramp_corrected: bool = True) -> np.ndarray:
    """Per-packet decoder-side SNR (dB), NaN where a slot has no statistics. Noise and tone power are averaged over
    the received packets within +-window//2 slots (the channel barely changes over a few 1.8 ms packets); each
    packet keeps its own phase-ramp term, which really differs from packet to packet."""
    n = len(stats_by_slot)
    out = np.full(n, np.nan, dtype=np.float32)
    ok = [s is not None and np.isfinite(s[1]) and np.isfinite(s[2]) and s[1] > 0 for s in stats_by_slot]
    h = max(int(window), 1) // 2
    for i in range(n):
        if not ok[i]:
            continue
        nb = [stats_by_slot[j] for j in range(max(0, i - h), min(n, i + h + 1)) if ok[j]]
        s2 = float(np.mean([s[1] for s in nb])); a2 = float(np.mean([s[2] for s in nb]))
        out[i] = _db(_eff_linear(s2, a2, stats_by_slot[i][0], stats_by_slot[i][3], tx_power, pilot_power, c0, k,
                                 ramp_corrected))
    return out
