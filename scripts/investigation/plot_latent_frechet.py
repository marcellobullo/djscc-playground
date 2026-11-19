"""Scatter: latent Fréchet distance (3x3 patches, d=144) to p_B vs decoded PSNR, from latent_frechet.json."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scripts/investigation/results/latent_frechet.json"))
out = sys.argv[2] if len(sys.argv) > 2 else "scripts/investigation/results/latent_frechet"
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
for ax, snr in zip(axes, ["10.0", "25.0"]):
    rows = {k: v for k, v in R[snr].items() if "psnr" in v}
    floor = R[snr]["z_B (DIV2K)"]["fd144"]
    ax.axvspan(0.5, floor, color="#e8e8e8", lw=0)
    ax.text(floor * 0.92, 0.04, "content\nfloor", transform=ax.get_xaxis_transform(), ha="right", fontsize=7, color="#555")
    for n, v in rows.items():
        ax.scatter(v["fd144"], v["psnr"], s=30, color="#2a6fdb", edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(n, (v["fd144"], v["psnr"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#222")
    ax.set_xscale("log"); ax.set_xlim(0.5, 250)
    ax.set_xlabel(r"latent FD to $p_B$ (3$\times$3 patches)")
    ax.set_title(f"AWGN {float(snr):g} dB", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("PSNR (dB)")
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(f"{out}.{ext}", dpi=200)
print(out)
