#!/usr/bin/env python
"""Collect PSNR / SSIM vs channel SNR for (a) the lab channel-emulator runs (results/djscc-psnr-vs-snr,
results/conventional, scored from the saved images exactly like the plotting notebooks) and (b) the
GNU Radio AWGN replica runs (results/gr-channel/*), save everything as CSV/JSON, and draw:

  1_djscc_vs_sscc_gr_asis.png    DJSCC (3 models, chain as deployed) vs SSCC (9 combos), GR channel
  2_djscc_vs_sscc_gr_fixed.png   DJSCC with frame-gap + CFO-ramp fixes vs SSCC with the frame gap
  3_emulator_vs_gr.png           emulator vs GR channel: deployed DJSCC (left), SSCC combos (right)
  4_slide_replica_gr.png         the slide layout (SSCC + deployed DJSCC) for the GR channel

SSCC decode failures count as PSNR 0 / SSIM 0 (USE_FAIL_FLOOR, as in the notebook); DJSCC curves average the
decoded images only (as in the notebook; lost images are reported in the CSV).
Run in the `demo` env: python scripts/investigation/plot_gr_vs_emulator.py
"""
import csv
import glob
import json
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
GR = os.path.join(RES, "gr-channel")
KODAK = "/Users/marcellobullo/Downloads/kodak_dataset"
W, H = 768, 512
SNRS = [0, 4, 8, 12, 16, 20, 24, 28]
MODN = {1: "BPSK", 2: "QPSK", 4: "16QAM"}
DJSCC_RUNS = {                      # key -> (label, colour)
    "deployed": ("DJSCC deployed (ofdm-spatialcsi)", "C0"),
    "convnext-awgn": ("DJSCC ConvNeXt (AWGN)", "C3"),
    "adjscc-awgn": ("ADJSCC (AWGN)", "C1"),
}


def load_orig(i):
    if not os.path.exists(f"{KODAK}/kodim{i:02d}.png"):
        return None          # stray index without an original (skipped, as in the notebooks)
    return cv2.resize(np.array(Image.open(f"{KODAK}/kodim{i:02d}.png"))[..., :3], (W, H))


_SSIM = None


def metrics(rec_rgb, orig):
    global _SSIM
    import torch
    from kaira.metrics.image import SSIM
    if _SSIM is None:
        _SSIM = SSIM(data_range=1.0).eval()
    t = lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    mse = np.mean((rec_rgb.astype(np.float64) - orig.astype(np.float64)) ** 2)
    p = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    with torch.no_grad():
        s = float(_SSIM(t(rec_rgb), t(orig)).mean())
    return p, s


def agg(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std()), len(a)


def load_emulator(cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {"djscc": {}, "sscc": {}}
    for s in SNRS:
        rows = []
        for f in sorted(glob.glob(f"{RES}/djscc-psnr-vs-snr/snr_{s}dB/*.png")):
            idx = int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
            o = load_orig(idx)
            if o is None:
                continue
            rec = cv2.resize(np.array(Image.open(f))[..., :3], (W, H))
            rows.append(metrics(rec, o))
        out["djscc"][str(s)] = rows
        for d in sorted(glob.glob(f"{RES}/conventional/snr_{s}dB/ldpc_n*_k*_mod*")):
            n, k, m = map(int, re.search(r"ldpc_n(\d+)_k(\d+)_mod(\d+)", d).groups())
            rows = []
            for f in sorted(glob.glob(f"{d}/image_*.png")):
                mm = re.search(r"image_(\d+)(_fail)?\.png", f)
                o = load_orig(int(mm.group(1)) + 1)
                if o is None:
                    continue
                if mm.group(2):
                    rows.append((0.0, 0.0))                       # decode failure floor
                    continue
                rec = cv2.resize(np.array(Image.open(f))[..., :3], (W, H))
                rows.append(metrics(rec, o))
            out["sscc"].setdefault(f"{n}_{k}_{m}", {})[str(s)] = rows
    json.dump(out, open(cache, "w"))
    return out


def curve(rows_by_snr):
    ms = {}
    for s in SNRS:
        r = rows_by_snr.get(str(s)) or []
        if r:
            ms[s] = (agg([x[0] for x in r]), agg([x[1] for x in r]))
    return ms        # {snr: ((mean,std,n) psnr, (mean,std,n) ssim)}


def load_gr_djscc(key, state):
    f = f"{GR}/{key}_{state}/{key}_{state}.json"
    if state == "labmatched":      # DJSCC chain as deployed + 2-sample frame gap (no spurious frame drops, like the lab)
        f = f"{GR}/labmatched/{key}_asis_gap2.json"
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    return {str(int(r["snr"])): [(x["psnr"], x["ssim"]) for x in r["per_image"]] for r in d["results"]}, \
           {str(int(r["snr"])): r["images_lost"] for r in d["results"]}


def load_gr_sscc(tag):
    f = f"{GR}/{tag}/data.json"
    out = {}
    if os.path.exists(f):
        for r in json.load(open(f))["results"]:
            out.setdefault(f"{r['ldpc_n']}_{r['ldpc_k']}_{r['mod']}", {})[str(int(r["snr"]))] = \
                [(x["psnr"], x["ssim"]) for x in r["per_image"]]
    return out


def sscc_style(key):
    n, k, m = map(int, key.split("_"))
    rates = {1152: 0, 1728: 1, 1920: 2}
    col = plt.cm.viridis(np.linspace(0.15, 0.85, 3))[rates.get(k, 0)]
    return col, {1: "-", 2: "--", 4: ":"}[m], {1: "o", 2: "s", 4: "^"}[m], f"LDPC {k}/{n} ({k / n:.2f}) - {MODN[m]}"


def plot_curve(ax, c, which, **kw):
    xs = sorted(c)
    mu = np.array([c[s][which][0] for s in xs])
    sd = np.array([c[s][which][1] for s in xs])
    ln, = ax.plot(xs, mu, **kw)
    ax.fill_between(xs, mu - sd, mu + sd, color=kw.get("color"), alpha=0.08)
    return ln


def panel(fig_axes, entries, title):
    """entries: list of (curve, label, plot-kwargs)."""
    for ax, which, yl in zip(fig_axes, (0, 1), ("PSNR (dB)", "SSIM (higher = better)")):
        for c, label, kw in entries:
            if c:
                plot_curve(ax, c, which, label=label, **kw)
        ax.set_xlabel("Channel SNR (dB)")
        ax.set_ylabel(yl)
        ax.grid(linestyle=":", alpha=0.5)
    fig_axes[0].set_title(title + ": PSNR vs SNR")
    fig_axes[1].set_title(title + ": SSIM vs SNR")


def main():
    emu = load_emulator(f"{GR}/emulator_metrics.json")
    emu_dj = curve(emu["djscc"])
    emu_ss = {k: curve(v) for k, v in emu["sscc"].items()}
    gr_dj, lost = {}, {}
    for key in DJSCC_RUNS:
        for st in ("asis", "fixed", "labmatched"):
            r = load_gr_djscc(key, st)
            if r:
                gr_dj[(key, st)] = curve(r[0])
                lost[(key, st)] = r[1]
    gr_ss = {t: {k: curve(v) for k, v in load_gr_sscc(t).items()} for t in ("sscc_asis", "sscc_gap16")}

    # ---- tables ----------------------------------------------------------------------------------
    rows = []
    def add(source, system, c):
        for s, ((p, ps, n), (q, qs, _)) in sorted(c.items()):
            rows.append(dict(source=source, system=system, snr_db=s, psnr_mean=p, psnr_std=ps,
                             ssim_mean=q, ssim_std=qs, n_images=n))
    add("emulator", "DJSCC deployed", emu_dj)
    for k, c in emu_ss.items():
        add("emulator", "SSCC " + sscc_style(k)[3], c)
    for (key, st), c in gr_dj.items():
        add(f"gnuradio_{st}", DJSCC_RUNS[key][0], c)
    for t, d in gr_ss.items():
        for k, c in d.items():
            add("gnuradio_" + t.split("_")[1], "SSCC " + sscc_style(k)[3], c)
    with open(f"{GR}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({f"{k[0]}_{k[1]}": v for k, v in lost.items()}, open(f"{GR}/djscc_lost_images.json", "w"), indent=1)
    print(f"[*] summary.csv: {len(rows)} rows")

    # ---- 1 / 2: DJSCC vs SSCC on the GR channel -------------------------------------------------------
    for st, ss_tag, name, ttl in (("labmatched", "sscc_gap16", "1_djscc_vs_sscc_gr_labmatched.png", "GNU Radio AWGN, lab-like chain (2-sample gap)"),
                                  ("asis", "sscc_asis", "1_djscc_vs_sscc_gr_asis.png", "GNU Radio AWGN, chains as deployed"),
                                  ("fixed", "sscc_gap16", "2_djscc_vs_sscc_gr_fixed.png", "GNU Radio AWGN, fixed DJSCC chain (SSCC + frame gap)")):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        ents = []
        for k, c in sorted(gr_ss[ss_tag].items()):
            col, ls, mk, lab = sscc_style(k)
            ents.append((c, lab, dict(color=col, linestyle=ls, marker=mk, ms=4, lw=1.2)))
        for key, (lab, col) in DJSCC_RUNS.items():
            ents.append((gr_dj.get((key, st)), lab, dict(color=col, marker="o", lw=2.6, ms=6, zorder=5)))
        panel(axs, ents, "DJSCC vs SSCC")
        axs[0].legend(fontsize=7, loc="lower right")
        fig.suptitle("DJSCC vs conventional SSCC - " + ttl)
        fig.savefig(f"{GR}/{name}", dpi=200); plt.close(fig)

    # ---- 3: emulator vs GR ------------------------------------------------------------------------
    for st, ss_tag, suf, ttl in (("labmatched", "sscc_gap16", "_labmatched", "lab-like chain (2-sample gap)"), ("asis", "sscc_asis", "", "chain as deployed"), ("fixed", "sscc_gap16", "_fixed", "fixed DJSCC chain, SSCC + frame gap")):
      fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
      for j, which in enumerate((0, 1)):
          ax = axs[j, 0]
          plot_curve(ax, emu_dj, which, label="Emulator (lab)", color="k", marker="o", lw=2)
          for key, (lab, col) in DJSCC_RUNS.items():
              if (key, st) in gr_dj:
                  plot_curve(ax, gr_dj[(key, st)], which, label=f"GNU Radio: {lab}", color=col, marker="s", ls="--", lw=1.8)
          ax2 = axs[j, 1]
          for k in sorted(emu_ss):
              col, ls, mk, lab = sscc_style(k)
              plot_curve(ax2, emu_ss[k], which, color=col, marker=mk, ls="-", lw=1.2, ms=4, label=f"emulator: {lab}")
              if k in gr_ss[ss_tag]:
                  plot_curve(ax2, gr_ss[ss_tag][k], which, color=col, marker=mk, ls=":", lw=1.2, ms=4, alpha=0.9, mfc="none")
          for a_ in (ax, ax2):
              a_.set_xlabel("Channel SNR (dB)"); a_.grid(linestyle=":", alpha=0.5)
              a_.set_ylabel("PSNR (dB)" if which == 0 else "SSIM")
      axs[0, 0].set_title("DJSCC: lab emulator vs GNU Radio channel (" + ttl + ")")
      axs[0, 1].set_title("SSCC: emulator (solid) vs GNU Radio channel (dotted, hollow markers)")
      axs[0, 0].legend(fontsize=8, loc="lower right"); axs[0, 1].legend(fontsize=6, loc="lower right")
      fig.savefig(f"{GR}/3_emulator_vs_gr" + suf + ".png", dpi=200); plt.close(fig)

    # ---- 4: slide replica for the GR channel (SSCC + deployed DJSCC) ----------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ents = []
    for k, c in sorted(gr_ss["sscc_asis"].items()):
        col, ls, mk, lab = sscc_style(k)
        ents.append((c, lab, dict(color=col, linestyle=ls, marker=mk, ms=4, lw=1.2)))
    ents.append((gr_dj.get(("deployed", "asis")), "DJSCC (deployed)", dict(color="C0", marker="o", lw=2.4, zorder=5)))
    panel(axs, ents, "GNU Radio AWGN, conventional SSCC")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.savefig(f"{GR}/4_slide_replica_gr.png", dpi=200); plt.close(fig)
    print("[*] plots written to", GR)


if __name__ == "__main__":
    main()
