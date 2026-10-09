"""Receiver-side SNR estimation from the equalized pilot tones.

Why pilots: they measure the in-band noise directly (unlike the guard-bin nulls, which a USRP's decimation filter
attenuates) and need no knowledge of the equalizer / channel-estimate scale (unlike nulls+taps, which assumed a unit
signal power and over-read the decoder-side SNR by 2.7-3.8 dB; the old pilot logger compared the equalized pilots to
+-1, but the static equalizer leaves them at amplitude ~1.9, so it was off by 10-28 dB).

Per frame, ``pilot_snr_logger`` sends three numbers computed from its T OFDM symbols x P pilot tones:
  slope : residual-CFO phase ramp (rad / OFDM symbol), from consecutive-symbol pilot products
  s2    : noise power per pilot after the ramp is removed (variance of consecutive differences / 2)
  a2    : debiased pilot-tone power  |mean over symbols|^2 - s2 / T
SNR at the pilots = a2 / s2; the data carriers are tx_power / pilot_power stronger (2.0 for the codecs: P = 1 per
real latent value). The decoder-side SNR is that channel SNR degraded by the channel-estimate noise (c_chest, a
property of the sync words) and, if nothing downstream removes it, by the phase-ramp error power slope^2 (T^2-1)/12.
Validated in the offline replica (scripts/investigation/snr_est_eval.py): bias within +-0.35 dB from 8 dB up.
"""
from __future__ import annotations

import numpy as np

C_CHEST = 0.5          # channel-estimate noise relative to the payload noise (calibrated in the replica)
MIN_DB, MAX_DB = -20.0, 45.0


def pilot_stats(pilots: np.ndarray, pilot_symbols: np.ndarray):
    """Reference implementation of what ``pilot_snr_logger`` computes. pilots: [T, P] equalized pilot tones."""
    P = np.asarray(pilots) * np.conj(np.asarray(pilot_symbols))[None, :]
    T = len(P)
    if T < 3:
        return None
    slope = float(np.angle(np.sum(P[1:] * np.conj(P[:-1]))))
    Pd = P * np.exp(-1j * slope * np.arange(T))[:, None]
    s2 = float(np.mean(np.abs(Pd[1:] - Pd[:-1]) ** 2) / 2)
    a2 = float(np.mean(np.abs(Pd.mean(axis=0)) ** 2) - s2 / T)
    return slope, s2, a2, T


def snr_eff_db(s2: float, a2: float, slope: float, n_sym: int, *, tx_power: float = 2.0, pilot_power: float = 1.0,
               c_chest: float = C_CHEST, ramp_corrected: bool = True) -> float:
    """Decoder-side (post-equalization) payload SNR in dB from one frame's (or pooled) pilot statistics."""
    s2 = max(float(s2), 1e-12)
    snr_chan = max(float(a2), 1e-12) / s2 * tx_power / pilot_power
    ramp = 0.0 if ramp_corrected else float(slope) ** 2 * (n_sym * n_sym - 1) / 12.0
    snr = 1.0 / ((1.0 + c_chest) / snr_chan + ramp)
    return float(np.clip(10.0 * np.log10(snr), MIN_DB, MAX_DB))


def pooled_snr_eff_db(stats, **kw) -> float:
    """One scalar for a whole image: pool the noise, tone power and ramp across packets (linear domain), then convert.
    Pooling first, instead of averaging per-packet dB values, keeps the estimate unbiased at low SNR."""
    stats = [s for s in stats if s is not None and np.isfinite(s[1]) and np.isfinite(s[2])]
    if not stats:
        return float("nan")
    slope2 = float(np.mean([s[0] ** 2 for s in stats]))
    return snr_eff_db(float(np.mean([s[1] for s in stats])), float(np.mean([s[2] for s in stats])),
                      slope2 ** 0.5, int(round(np.mean([s[3] for s in stats]))), **kw)
