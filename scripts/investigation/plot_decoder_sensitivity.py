"""Latent error vs decoder sensitivity to that error, with iso-image-error diagonals.

To first order, image error ≈ (decoder gain along e) x ||e||^2, so in dB
    image error ≈ latent NMSE (dB) + relative sensitivity (dB) + const.
Points on the same dashed diagonal have the same predicted image error; down-left is better.
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("scripts/investigation/results/decoder_sensitivity.json"))
P = json.load(open("scripts/investigation/results/latent_frechet.json"))
out = sys.argv[1] if len(sys.argv) > 1 else "scripts/investigation/results/decoder_sensitivity"
label = {"misaligned": "misaligned", "wiener": "Wiener", "jwiener": "J-Wiener", "conv": "Conv", "rescnn": "ResCNN"}
base = {"misaligned": (4, 6, "left"), "wiener": (6, 4, "left"), "jwiener": (6, 4, "left")}
off = {"10.0": {**base, "conv": (6, 6, "left"), "rescnn": (-6, 6, "right")},
       "25.0": {**base, "conv": (6, -12, "left"), "rescnn": (-6, -12, "right")}}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, snr in zip(axes, ["10.0", "25.0"]):
    xs = np.linspace(-14, 9, 2)
    for c in range(-12, 14, 3):
        ax.plot(xs, c - xs, color="#bbbbbb", lw=0.6, ls="--", zorder=1)
    for n, r in S[snr].items():
        x, y = 10 * np.log10(r["latent_err"]), r["rel_db"]
        ax.scatter(x, y, s=34, color="#2a6fdb", edgecolor="white", lw=1.5, zorder=3)
        dx, dy, ha = off[snr][n]
        ax.annotate(f"{label[n]}  {P[snr][n]['psnr']:.1f} dB", (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha=ha, color="#222")
    ax.axhline(0, color="#555", lw=0.8, zorder=2)
    ax.text(-11.7, 0.25, "random direction", fontsize=6.5, ha="left", color="#555")
    ax.set_xlim(-12, 9); ax.set_ylim(-4, 10)
    ax.set_xlabel("latent error  $\\|e\\|^2/\\|z_B\\|^2$ (dB)")
    ax.set_title(f"AWGN {float(snr):g} dB", fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("decoder sensitivity to $e$\nrel. to random (dB)")
axes[0].annotate("better", xy=(-10.5, -3.3), xytext=(-7.5, -1.2), fontsize=7, color="#555",
                 arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(f"{out}.{ext}", dpi=200)
print(out)
