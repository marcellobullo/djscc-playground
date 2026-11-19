#!/usr/bin/env python
"""Compare aligners in the GNU Radio replica (run_gr_alignment.sh + run_gr_aligner_sweep.sh).

(a) PSNR vs channel SNR per aligner, with target and misaligned as reference lines;
(b) ARR vs channel SNR per aligner. Also prints a table (PSNR / SSIM / ARR at --at dB,
parameter count) and writes it as CSV next to the figure.

  python scripts/investigation/plot_gr_aligner_comparison.py name=label:folder ...
"""
import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "D", "^", "P", "X"]
INK, MUTED, GRID, REF = "#1f1f1e", "#6b6a64", "#e4e3de", "#8a8983"


def load(d, tag):
    r = json.load(open(os.path.join(d, f"{tag}.json")))["results"]
    return (np.array([x["snr"] for x in r]), np.array([x["psnr"] for x in r]),
            np.array([x["ssim"] for x in r]))


def n_params(folder):
    w = glob.glob(os.path.join(folder, "*.pth"))
    if not w:
        return 0
    sd = torch.load(w[0], map_location="cpu")
    return int(sum(v.numel() for v in sd.values()))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aligners", nargs="+", help="name=label:folder (results in aligned_<name>.json)")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "gr-align"))
    ap.add_argument("--at", type=float, default=25.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    snr, p_t, s_t = load(a.dir, "target")
    _, p_m, s_m = load(a.dir, "misaligned")
    k = int(np.argmin(np.abs(snr - a.at)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax.plot(snr, p_t, "-", color=REF, lw=1.5, zorder=2)
    ax.text(snr[-1] + 0.6, p_t[-1], "target", fontsize=7, color=MUTED, va="center")
    ax.plot(snr, p_m, "--", color=REF, lw=1.5, zorder=2)
    ax.text(snr[-1] + 0.6, p_m[-1], "misaligned", fontsize=7, color=MUTED, va="center")

    rows = []
    for i, spec in enumerate(a.aligners):
        name, rest = spec.split("=", 1)
        label, folder = rest.split(":", 1)
        s, p, ss = load(a.dir, f"aligned_{name}")
        arr = (p - p_m) / (p_t - p_m)
        c, m = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        ax.plot(s, p, "-", color=c, lw=2, marker=m, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
        bx.plot(s, arr, "-", color=c, lw=2, marker=m, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
        rows.append(dict(aligner=label, params=n_params(folder), psnr=round(float(p[k]), 2),
                         ssim=round(float(ss[k]), 3), arr=round(float(arr[k]), 3),
                         psnr_peak=round(float(p.max()), 2)))

    ax.set_xlabel("Channel SNR (dB)", color=INK); ax.set_ylabel("PSNR (dB)", color=INK)
    ax.set_xticks(snr); ax.set_xlim(snr.min() - 1, snr.max() + 6)
    ax.set_title("(a) Aligned PSNR, GNU Radio replica", fontsize=9, color=INK, loc="left")
    style(ax)
    bx.axhline(1, color=MUTED, lw=1, ls=":"); bx.axhline(0, color=MUTED, lw=1, ls=":")
    bx.set_ylim(-0.05, 1.08); bx.set_xticks(snr)
    bx.set_xlabel("Channel SNR (dB)", color=INK); bx.set_ylabel("ARR", color=INK)
    bx.set_title("(b) Alignment Recovery Ratio", fontsize=9, color=INK, loc="left")
    style(bx)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), fontsize=7.5, frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1 if len(l) <= 3 else 0.14, 1, 1))

    out = a.out or os.path.join(a.dir, "gr_aligner_comparison")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    with open(f"{out}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"@ {snr[k]:.0f} dB: target {p_t[k]:.2f} / misaligned {p_m[k]:.2f}")
    print(f"{'aligner':34s} {'params':>8s} {'PSNR':>6s} {'SSIM':>6s} {'ARR':>6s} {'peak':>6s}")
    for r in rows:
        print(f"{r['aligner']:34s} {r['params']:8d} {r['psnr']:6.2f} {r['ssim']:6.3f} {r['arr']:6.3f} {r['psnr_peak']:6.2f}")
    print(f"saved {out}.pdf / .png / .csv")


if __name__ == "__main__":
    main()
