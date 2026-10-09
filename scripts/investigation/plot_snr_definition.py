#!/usr/bin/env python
"""Visual of how SNR is defined: lab emulator (flat noise ~8x wider than the signal, SNR in the signal
bandwidth) vs the replica (noise only inside the 1 MS/s band). PSD of the real djscc_tx OFDM waveform."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e2e_link as E

rng = np.random.default_rng(0)
sym = ((rng.standard_normal(960 * 40) + 1j * rng.standard_normal(960 * 40)) / np.sqrt(2)).astype(np.complex64)
x = E.gr_tx(sym, 4.0)
N = 1024
seg = x[: (len(x) // N) * N].reshape(-1, N) * np.hanning(N)
psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2, axis=0)
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / E.SAMP_RATE)) / 1e6                      # MHz
psd = psd / psd.max()
occ = (np.abs(f) <= 26.5 * 1e6 / 64 / 1e6)
p_in = psd[occ].sum() / psd.sum()
snr = 12.0
noise_lvl = (psd.mean() * (64 / 52)) * 10 ** (-snr / 10)           # flat noise PSD for 12 dB SNR in the occupied band

fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
for ax, wide in zip(axs, (True, False)):
    xr = 4.0 if wide else 0.75
    ff = np.linspace(-xr, xr, 4000)
    ax.plot(f, 10 * np.log10(psd + 1e-12), color="C0", lw=2, label="OFDM signal (real djscc_tx waveform)")
    ax.fill_between(f, -80, 10 * np.log10(psd + 1e-12), color="C0", alpha=0.15)
    ax.hlines(10 * np.log10(noise_lvl), -xr, xr, color="C3", lw=2, label=f"noise, {snr:.0f} dB SNR in the signal band")
    ax.axvspan(-0.5, 0.5, color="0.85", zorder=0, label="1 MS/s receive band")
    ax.axvspan(-26.5 / 64, 26.5 / 64, ymin=0, ymax=0.08, color="C2", alpha=0.8, label="52 occupied carriers (0.81 MHz)")
    ax.set_xlim(-xr, xr); ax.set_ylim(-60, 5)
    ax.set_xlabel("Frequency offset from 2.45 GHz (MHz)"); ax.set_ylabel("PSD (dB, rel. to peak)")
    ax.grid(linestyle=":", alpha=0.5)
axs[0].set_title("Lab: Azimuth emulator injects flat noise ~8x wider than the signal (spectrum analyser view, +-4 MHz);\nthe USRP receiver then keeps only its 1 MHz band")
axs[1].set_title("Replica: same noise density, but only the 1 MS/s band exists (zoom, +-0.75 MHz). SNR is set in that band")
axs[0].legend(loc="upper right", fontsize=8)
fig.savefig(os.path.join(HERE, "..", "..", "results", "gr-channel", "snr_definition.png"), dpi=170)
print("fraction of signal power inside the 52 carriers: %.3f; noise ratio full band vs occupied: %.2f dB" % (p_in, 10*np.log10(64/52)))
