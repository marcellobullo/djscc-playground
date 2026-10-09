#!/usr/bin/env python
"""New ADJSCC fine-tune (matched SNR input) in the GNU Radio replica vs the old models and SSCC.
Writes results/gr-channel/6_new_model_labmatched.png, 6_new_model_fixed.png and 7_new_model_csi_grid.png."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_gr_vs_emulator as P

G = P.GR
def load(f):
    d = json.load(open(f"{G}/{f}"))["results"]
    return {s["snr"]: ((np.mean([x["psnr"] for x in s["per_image"]]), np.std([x["psnr"] for x in s["per_image"]]), len(s["per_image"])),
                       (np.mean([x["ssim"] for x in s["per_image"]]), np.std([x["ssim"] for x in s["per_image"]]), len(s["per_image"]))) for s in d}
ss = {k: P.curve(v) for k, v in P.load_gr_sscc("sscc_gap16").items()}
variants = {
  "labmatched": ("lab-like chain (2-sample frame gap)", {
     "old: deployed (CSI 19)": ("labmatched/deployed_asis_gap2.json" if os.path.exists(f"{G}/labmatched/deployed_asis_gap2.json") else None, "C0"),
     "old: ConvNeXt AWGN (CSI 19)": ("labmatched/convnext-awgn_asis_gap2.json", "C3"),
     "old: ADJSCC (fixed design SNR 5)": ("labmatched/adjscc-awgn_asis_gap2.json", "C1")},
     "matched_model/matched_ep0_labmatched.json"),
  "fixed": ("fixed chain (frame gap + CFO-ramp corrector)", {
     "old: deployed (CSI 19)": ("deployed_fixed/deployed_fixed.json", "C0"),
     "old: ConvNeXt AWGN (CSI 19)": ("convnext-awgn_fixed/convnext-awgn_fixed.json", "C3"),
     "old: ADJSCC (fixed design SNR 5)": ("adjscc-awgn_fixed/adjscc-awgn_fixed.json", "C1")},
     "matched_model/matched_ep0_fixed.json"),
}
for key, (title, olds, newf) in variants.items():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ents = []
    for k, c in sorted(ss.items()):
        col, ls, mk, lab = P.sscc_style(k)
        ents.append((c, lab, dict(color=col, linestyle=ls, marker=mk, ms=4, lw=1.0, alpha=0.8)))
    for lab, (f, col) in olds.items():
        if f and os.path.exists(f"{G}/{f}"):
            ents.append((load(f), lab, dict(color=col, marker="o", lw=1.6, ms=5, ls="--", alpha=0.85)))
    ents.append((load(newf), "NEW: ADJSCC fine-tune, CSI = SNR (1 epoch)", dict(color="k", marker="o", lw=3.0, ms=7, zorder=10)))
    P.panel(axs, ents, "")
    axs[0].legend(fontsize=7, loc="lower right"); axs[0].set_title("PSNR vs SNR"); axs[1].set_title("SSIM vs SNR")
    fig.suptitle(f"New ADJSCC model vs old models and SSCC - GNU Radio AWGN, {title}, 24 Kodak images")
    fig.savefig(f"{G}/6_new_model_{key}.png", dpi=190); plt.close(fig)

# CSI grid heatmaps of the new model in the replica
S = [0, 4, 8, 12, 16, 20, 24, 28]
fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
for ax, f, t in zip(axs, ("grid_psnr.npy", "gridfixed_psnr.npy"), ("lab-like chain", "fixed chain")):
    M = np.load(f"{G}/matched_model/{f}"); ax.imshow(M, cmap="viridis", origin="lower")
    for i in range(8):
        for j in range(8): ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < M.mean() else "k", fontweight="bold" if j == M[i].argmax() else "normal")
        ax.add_patch(plt.Rectangle((M[i].argmax() - .5, i - .5), 1, 1, fill=False, ec="red", lw=2)); ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.2, ls="--"))
    ax.set_xticks(range(8), S); ax.set_yticks(range(8), S); ax.set_xlabel("CSI fed to encoder + decoder (dB)"); ax.set_ylabel("Nominal channel SNR (dB)"); ax.set_title(f"New model, replica PSNR (dB), {t}")
fig.suptitle("Red = best CSI per row, dashed white = matched. In the replica the decoder sees about 4 dB less than the nominal SNR.")
fig.savefig(f"{G}/7_new_model_csi_grid.png", dpi=170)
