#!/usr/bin/env python
"""Plot the GNU Radio replica alignment runs (scripts/investigation/run_gr_alignment.sh).

(a) PSNR vs channel SNR for matched / aligned / misaligned / target, with the model gap
    (matched - target) and the alignment gap (target - aligned) marked at one SNR;
(b) Alignment Recovery Ratio ARR = (aligned - misaligned) / (target - misaligned).

  python scripts/investigation/plot_gr_alignment.py [--dir results/gr-align] [--gap-snr 25]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = [  # tag, label, colour (categorical slots 1-4), marker, line style
    ("matched", "Matched (ADJSCC → ADJSCC)", "#2a78d6", "o", "-"),
    ("target", "Target (ConvNeXt → ConvNeXt)", "#eb6834", "s", "-"),
    ("aligned", "Aligned (ADJSCC → aligner → ConvNeXt)", "#1baf7a", "D", "-"),
    ("misaligned", "Misaligned (ADJSCC → ConvNeXt)", "#eda100", "v", "--"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(d, tag):
    r = json.load(open(os.path.join(d, f"{tag}.json")))["results"]
    return np.array([x["snr"] for x in r]), np.array([x["psnr"] for x in r]), np.array([x["ssim"] for x in r])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bracket(ax, x, y0, y1, text):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(x + 0.6, (y0 + y1) / 2, text, va="center", ha="left", fontsize=7.5, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "gr-align"))
    ap.add_argument("--gap-snr", type=float, default=25.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = {t: load(a.dir, t) for t, *_ in RUNS}
    snr = data["matched"][0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw=dict(width_ratios=[1.7, 1]))

    for tag, label, c, m, ls in RUNS:
        s, p, _ = data[tag]
        ax.plot(s, p, ls, color=c, lw=2, marker=m, ms=5, mec="white", mew=1, label=label, zorder=3)
    k = int(np.argmin(np.abs(snr - a.gap_snr)))
    pm, pt, pa = data["matched"][1][k], data["target"][1][k], data["aligned"][1][k]
    bracket(ax, snr[k] + 0.9, pt, pm, f"model gap\n{pm - pt:.1f} dB")
    bracket(ax, snr[k] + 0.9, pa, pt, f"alignment gap\n{pt - pa:.1f} dB")
    ax.set_xlabel("Channel SNR (dB)", color=INK)
    ax.set_ylabel("PSNR (dB)", color=INK)
    ax.set_xlim(snr.min() - 1, snr.max() + 6)
    ax.set_ylim(5, 42)
    ax.set_xticks(snr)
    ax.set_title("(a) Reconstruction quality, GNU Radio replica", fontsize=9, color=INK, loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=7, frameon=False, labelcolor=INK)
    style(ax)

    arr = (data["aligned"][1] - data["misaligned"][1]) / (data["target"][1] - data["misaligned"][1])
    bx.plot(snr, arr, "-", color="#1baf7a", lw=2, marker="D", ms=5, mec="white", mew=1, zorder=3)
    for s, v in zip(snr, arr):
        bx.text(s, v + 0.04, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    bx.axhline(1, color=MUTED, lw=1, ls=":")
    bx.text(snr.min(), 1.02, "target", fontsize=7, color=MUTED, va="bottom")
    bx.axhline(0, color=MUTED, lw=1, ls=":")
    bx.text(snr.min(), 0.02, "no aligner", fontsize=7, color=MUTED, va="bottom")
    bx.set_ylim(-0.05, 1.12)
    bx.set_xticks(snr)
    bx.set_xlabel("Channel SNR (dB)", color=INK)
    bx.set_ylabel("ARR", color=INK)
    bx.set_title("(b) Alignment Recovery Ratio", fontsize=9, color=INK, loc="left")
    style(bx)

    fig.tight_layout()
    out = a.out or os.path.join(a.dir, "gr_alignment")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
