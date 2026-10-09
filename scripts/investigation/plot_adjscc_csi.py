#!/usr/bin/env python
"""ADJSCC (adjscc-cr6-awgn) PSNR / SSIM vs channel SNR for different encoder+decoder CSI inputs
(lab-like chain, 24 Kodak images, GNU Radio AWGN replica). Writes results/gr-channel/adjscc_csi_sweep.png/.csv"""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "gr-channel")
runs = {  # label -> (json, style)
    "Fixed 0 dB": (f"{G}/csi_scan/adjscc_csi0.json", dict(color="C2", marker="s")),
    "Fixed 5 dB (used so far)": (f"{G}/labmatched/adjscc-awgn_asis_gap2.json", dict(color="C0", marker="o", lw=2.6)),
    "Fixed 10 dB": (f"{G}/csi_scan/adjscc_csi10.json", dict(color="C1", marker="^")),
    "Fixed 15 dB": (f"{G}/csi_scan/adjscc_csi15.json", dict(color="C4", marker="v")),
    "Fixed 20 dB": (f"{G}/csi_scan/adjscc_csi20.json", dict(color="C5", marker="D")),
    "Actual channel SNR": (f"{G}/csi_follow/adjscc-awgn_asis_gap2_csifollow.json", dict(color="C3", marker="o", ls="--", lw=2.6)),
}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
rows = []
for lab, (f, st) in runs.items():
    d = json.load(open(f))["results"]
    snr = [r["snr"] for r in d]
    for ax, k in zip(axs, ("psnr", "ssim")):
        ax.plot(snr, [r[k] for r in d], label=lab, ms=5, **st)
    rows += [dict(csi=lab, snr_db=r["snr"], psnr=r["psnr"], ssim=r["ssim"]) for r in d]
for ax, yl in zip(axs, ("PSNR (dB)", "SSIM")):
    ax.set_xlabel("Channel SNR (dB)"); ax.set_ylabel(yl); ax.grid(linestyle=":", alpha=0.5)
axs[0].set_title("ADJSCC: PSNR vs SNR for different CSI inputs"); axs[1].set_title("ADJSCC: SSIM vs SNR for different CSI inputs")
axs[0].legend(fontsize=9, loc="lower right")
fig.suptitle("adjscc-cr6-awgn, lab-like chain (2-sample frame gap), GNU Radio AWGN, 24 Kodak images")
fig.savefig(f"{G}/adjscc_csi_sweep.png", dpi=180)
with open(f"{G}/adjscc_csi_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
