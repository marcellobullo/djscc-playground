#!/usr/bin/env python
"""Fully trained matched-SNR ADJSCC (20 epochs) in the GNU Radio replica vs the 1-epoch checkpoint, the old models
and SSCC, plus its CSI grids. Writes results/gr-channel/8_final_model_{labmatched,fixed}.png, 9_final_model_csi_grid.png
and 8_final_model_summary.csv."""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_gr_vs_emulator as P

G = P.GR; F = f"{G}/matched_model_final"; S = [0, 4, 8, 12, 16, 20, 24, 28]
def load(f):
    d = json.load(open(f))["results"]
    return {s["snr"]: tuple((float(np.mean([x[k] for x in s["per_image"]])), float(np.std([x[k] for x in s["per_image"]])), len(s["per_image"]))
                             for k in ("psnr", "ssim")) for s in d}
ss = {k: P.curve(v) for k, v in P.load_gr_sscc("sscc_gap16").items()}
rows = []
for key, title, olds, ep0, fin in (
    ("labmatched", "lab-like chain (2-sample frame gap)",
     {"old: deployed (CSI 19)": ("labmatched/deployed_asis_gap2.json", "C0"), "old: ConvNeXt AWGN (CSI 19)": ("labmatched/convnext-awgn_asis_gap2.json", "C3"),
      "old: ADJSCC (fixed design SNR 5)": ("labmatched/adjscc-awgn_asis_gap2.json", "C1")},
     "matched_model/matched_ep0_labmatched.json", "matched_model_final/final_labmatched.json"),
    ("fixed", "fixed chain (frame gap + CFO-ramp corrector)",
     {"old: deployed (CSI 19)": ("deployed_fixed/deployed_fixed.json", "C0"), "old: ConvNeXt AWGN (CSI 19)": ("convnext-awgn_fixed/convnext-awgn_fixed.json", "C3"),
      "old: ADJSCC (fixed design SNR 5)": ("adjscc-awgn_fixed/adjscc-awgn_fixed.json", "C1")},
     "matched_model/matched_ep0_fixed.json", "matched_model_final/final_fixed.json")):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ents = []
    for k, c in sorted(ss.items()):
        col, ls, mk, lab = P.sscc_style(k)
        ents.append((c, lab, dict(color=col, linestyle=ls, marker=mk, ms=4, lw=1.0, alpha=0.75)))
        rows += [dict(chain=key, system="SSCC " + lab, snr_db=s, psnr=v[0][0], ssim=v[1][0]) for s, v in c.items()]
    for lab, (f, col) in olds.items():
        c = load(f"{G}/{f}"); ents.append((c, lab, dict(color=col, marker="o", lw=1.5, ms=4, ls="--", alpha=0.8)))
        rows += [dict(chain=key, system=lab, snr_db=s, psnr=v[0][0], ssim=v[1][0]) for s, v in c.items()]
    for lab, f, kw in (("NEW ADJSCC, CSI = SNR, 1 epoch", ep0, dict(color="0.55", marker="o", lw=2.0, ms=5)),
                       ("NEW ADJSCC, CSI = SNR, 20 epochs", fin, dict(color="k", marker="o", lw=3.0, ms=7, zorder=10))):
        c = load(f"{G}/{f}"); ents.append((c, lab, kw))
        rows += [dict(chain=key, system=lab, snr_db=s, psnr=v[0][0], ssim=v[1][0]) for s, v in c.items()]
    P.panel(axs, ents, "")
    axs[0].legend(fontsize=7, loc="lower right"); axs[0].set_title("PSNR vs SNR"); axs[1].set_title("SSIM vs SNR")
    fig.suptitle(f"Matched-SNR ADJSCC after 20 epochs - GNU Radio AWGN, {title}, 24 Kodak images")
    fig.savefig(f"{G}/8_final_model_{key}.png", dpi=190); plt.close(fig)
with open(f"{G}/8_final_model_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def grid(d, prefix):
    M = np.zeros((8, 8))
    for j, c in enumerate(S):
        r = json.load(open(f"{d}/{prefix}{c}.json"))["results"]
        for i in range(8): M[i, j] = r[i]["psnr"]
    return M
ideal = json.load(open(f"{F}/ideal_awgn_grid.json"))["psnr"]
mats = [(np.array([[ideal[str(c)][i] for c in S] for i in range(8)]), "ideal AWGN channel"),
        (grid(f"{F}/grid", "grid_csi"), "replica, lab-like chain"), (grid(f"{F}/grid_fixed", "gridfixed_csi"), "replica, fixed chain")]
fig, axs = plt.subplots(1, 3, figsize=(20, 6.3), constrained_layout=True)
for ax, (M, t) in zip(axs, mats):
    ax.imshow(M, cmap="viridis", origin="lower")
    for i in range(8):
        for j in range(8): ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < M.mean() else "k", fontweight="bold" if j == M[i].argmax() else "normal")
        ax.add_patch(plt.Rectangle((M[i].argmax() - .5, i - .5), 1, 1, fill=False, ec="red", lw=2)); ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.2, ls="--"))
    ax.set_xticks(range(8), S); ax.set_yticks(range(8), S); ax.set_xlabel("CSI fed to encoder + decoder (dB)"); ax.set_ylabel("Channel SNR (dB)"); ax.set_title(f"PSNR (dB), {t}")
    np.save(f"{F}/grid_{t.replace(', ', '_').replace(' ', '_')}.npy", M)
fig.suptitle("Matched-SNR ADJSCC, 20 epochs: red = best CSI per row, dashed white = matched (diagonal)")
fig.savefig(f"{G}/9_final_model_csi_grid.png", dpi=170)
for M, t in mats: print(f"{t:26s} best CSI per row:", [S[int(M[i].argmax())] for i in range(8)], " diagonal within 0.2 dB of best:", sum(int(M[i, i] >= M[i].max() - 0.2) for i in range(8)), "/8")
