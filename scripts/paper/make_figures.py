#!/usr/bin/env python
"""Figures for the WCNC WS15 paper (paper/wcnc-ws15/figs/).

  fig_replica.pdf  (a) PSNR vs channel SNR in the GNU Radio replica: matched / target /
                   misaligned references and the selected aligners; (b) ARR vs SNR.
  fig_alpha.pdf    validation PSNR of the latent-anchored Jacobian-weighted Wiener aligner
                   vs alpha (alpha = 1 is plain Wiener), per-position and 16x16-tile metric.

  python scripts/paper/make_figures.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results", "gr-align")
OUT = os.path.join(ROOT, "paper", "wcnc-ws15", "figs")
INK, MUTED, GRID, REF = "#1f1f1e", "#6b6a64", "#e4e3de", "#8a8983"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "P"]
ALIGNERS = [  # (results tag, label)
    ("rescnn", "ResCNN (trained)"),
    ("jw_t1_best", "Anchored J-Wiener (closed form)"),
    ("conv", "Conv 5x5 (trained)"),
    ("wiener5_100", "Wiener 5x5 (closed form)"),
    ("otpilot1_p1", "Gaussian OT + 1 pilot"),
]


def load(tag):
    r = json.load(open(os.path.join(RES, f"{tag}.json")))["results"]
    return np.array([x["snr"] for x in r]), np.array([x["psnr"] for x in r])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def fig_replica():
    snr, p_mat = load("matched")
    _, p_t = load("target")
    _, p_m = load("misaligned")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax.plot(snr, p_mat, "-", color=INK, lw=1.3, zorder=2)
    ax.plot(snr, p_t, "-", color=REF, lw=1.3, zorder=2)
    ax.plot(snr, p_m, "--", color=REF, lw=1.3, zorder=2)
    for y, lab in ((p_mat[-1], "matched"), (p_t[-1], "target"), (p_m[-1], "misaligned")):
        ax.text(snr[-1] + 0.6, y, lab, fontsize=6.5, color=MUTED, va="center")
    for i, (tag, lab) in enumerate(ALIGNERS):
        s, p = load(f"aligned_{tag}")
        arr = (p - p_m) / (p_t - p_m)
        kw = dict(color=COLORS[i], lw=1.6, marker=MARKERS[i], ms=3.8, mec="white", mew=0.6, zorder=3)
        ax.plot(s, p, "-", label=lab, **kw)
        bx.plot(s, arr, "-", **kw)
    ss, ps = sscc_envelope()
    if ss is not None:
        ax.plot(ss, ps, ":", color=INK, lw=1.4, drawstyle="default", zorder=2,
                label="JPEG + LDPC (best MCS per SNR)")
    ax.set_xlabel("Channel SNR (dB)", color=INK, fontsize=8)
    ax.set_ylabel("PSNR (dB)", color=INK, fontsize=8)
    ax.set_xticks(snr); ax.set_xlim(snr.min() - 1, snr.max() + 6.5)
    ax.set_title("(a) Reconstruction quality", fontsize=8.5, color=INK, loc="left")
    style(ax)
    bx.axhline(1, color=MUTED, lw=0.8, ls=":"); bx.axhline(0, color=MUTED, lw=0.8, ls=":")
    bx.set_ylim(-0.05, 1.05); bx.set_xticks(snr)
    bx.set_xlabel("Channel SNR (dB)", color=INK, fontsize=8)
    bx.set_ylabel("ARR", color=INK, fontsize=8)
    bx.set_title("(b) Alignment recovery ratio", fontsize=8.5, color=INK, loc="left")
    style(bx)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.5)
    fig.savefig(os.path.join(OUT, "fig_replica.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_replica.png"), dpi=200, bbox_inches="tight")


def fig_alpha():
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    labels = None
    for i, (folder, lab) in enumerate((("jw_t1_best", "per-position metric"), ("jw_t16_best", "16x16-tile metric"))):
        cfg = json.load(open(os.path.join(ROOT, "ckpts", "aligners", folder, "aligner_config.json")))
        sw = cfg["sweep"]
        al = sorted(sw, key=float)
        labels = al
        x = np.arange(len(al))
        ax.plot(x, [sw[a] for a in al], "-", color=COLORS[i], lw=1.6, marker=MARKERS[i], ms=3.8,
                mec="white", mew=0.6, label=lab, zorder=3)
    ax.axhline(sw["1.0"], color=REF, lw=1, ls="--", zorder=2)
    ax.text(len(labels) - 1, sw["1.0"] - 1.6, "plain Wiener", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([str(float(a)).rstrip("0").rstrip(".") for a in labels], fontsize=6.5)
    ax.set_xlabel(r"Anchor weight $\alpha$", color=INK, fontsize=8)
    ax.set_ylabel("Val. PSNR (dB)", color=INK, fontsize=8)
    ax.set_ylim(12, 28)
    style(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_alpha.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_alpha.png"), dpi=200, bbox_inches="tight")


def sscc_envelope():
    """Best (LDPC rate, modulation) per SNR from e2e_sscc.py; failed images count as 0 dB."""
    f = os.path.join(RES, "sscc", "data.json")
    if not os.path.exists(f):
        return None, None
    best = {}
    for r in json.load(open(f))["results"]:
        best[r["snr"]] = max(best.get(r["snr"], 0.0), r["psnr"])
    s = np.array(sorted(best))
    return s, np.array([best[x] for x in s])


def per_image(path, snr):
    r = [x for x in json.load(open(path))["results"] if x["snr"] == snr][0]
    return {p["img"]: p["psnr"] for p in r["per_image"]}


QUAL_RUNS = [("matched", "Matched"), ("target", "Target"), ("misaligned", "Misaligned"),
             ("conv", "Conv (trained)"), ("jw", "J-Wiener (closed f.)"), ("rescnn", "ResCNN (trained)")]


def fig_qual(images=("kodim23.png", "kodim05.png"), snr=25, crop=(176, 96, 352, 224)):
    """Crops of the reconstructions (replica) for each run; PSNR of the full image in the corner."""
    from PIL import Image
    kodak = os.path.expanduser("~/Downloads/kodak_dataset")
    qd = os.path.join(RES, "qual")
    x0, y0, w, h = crop
    rows = []
    for name in images:
        idx = int(name[5:7])
        row = [("Original", np.asarray(Image.open(os.path.join(kodak, name)).convert("RGB").resize((768, 512))), None)]
        for tag, lab in QUAL_RUNS:
            im = np.asarray(Image.open(os.path.join(qd, "img", tag, f"snr_{snr}dB", f"snr{snr}_{idx:03d}.png")).convert("RGB"))
            row.append((lab, im, per_image(os.path.join(qd, f"{tag}.json"), snr)[name]))
        rows.append(row)
    n = len(rows[0])
    fig, axs = plt.subplots(len(rows), n, figsize=(7.16, 0.80 * len(rows) + 0.2))
    for r, row in enumerate(rows):
        for c, (lab, im, p) in enumerate(row):
            a = axs[r, c]
            a.imshow(im[y0:y0 + h, x0:x0 + w]); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if r == 0:
                a.set_title(lab, fontsize=6.8, color=INK, pad=3)
            if p is not None:
                a.text(0.03, 0.04, f"{p:.1f} dB", transform=a.transAxes, fontsize=6, color="white",
                       bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_qual.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT, "fig_qual.png"), bbox_inches="tight", dpi=200)


def fig_budget():
    """(a,b) PSNR vs number of pilot / training images at 25 and 10 dB; (c) PSNR vs SNR for
    closed-form aligners fitted at 5 / 15 / 25 dB."""
    bd = os.path.join(RES, "budget")

    def series(prefix, extra=None):
        pts = {}
        for f in glob.glob(os.path.join(bd, f"{prefix}_n*.json")):
            n = int(os.path.basename(f)[len(prefix) + 2:-5])
            pts[n] = {r["snr"]: r["psnr"] for r in json.load(open(f))["results"]}
        for n, f in (extra or {}).items():
            if os.path.exists(f):
                pts[n] = {r["snr"]: r["psnr"] for r in json.load(open(f))["results"]}
        ns = sorted(pts)
        return ns, pts

    _, p_t = load("target"); snr_ref, p_m = load("misaligned")
    ref = lambda arr, s: arr[list(snr_ref).index(s)]
    S = [
        ("wiener", "Wiener (closed form)", None),
        ("jw", "J-Wiener (closed form)", None),
        ("otp", "Gaussian OT + Procrustes", None),
        ("conv", "Conv (trained)", {800: os.path.join(RES, "aligned_conv.json")}),
        ("rescnn", "ResCNN (trained)", {800: os.path.join(RES, "aligned_rescnn.json")}),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for k, snr in enumerate((25, 10)):
        ax = axs[k]
        for i, (pre, lab, extra) in enumerate(S):
            ns, pts = series(pre, extra)
            ns = [n for n in ns if snr in pts[n]]
            if not ns:
                continue
            ax.plot(ns, [pts[n][snr] for n in ns], "-", color=COLORS[i], lw=1.5, marker=MARKERS[i], ms=3.5,
                    mec="white", mew=0.5, label=lab if k == 0 else None, zorder=3)
        ax.axhline(ref(p_t, snr), color=REF, lw=1.1); ax.axhline(ref(p_m, snr), color=REF, lw=1.1, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("Pilot / training images", color=INK, fontsize=7.5)
        ax.set_title(f"({'ab'[k]}) Data budget, {snr} dB", fontsize=8, color=INK, loc="left")
        style(ax)
    axs[0].set_ylabel("PSNR (dB)", color=INK, fontsize=7.5)
    ax = axs[2]
    fd = os.path.join(RES, "fitsnr")
    for i, (meth, lab) in enumerate((("wiener", "Wiener"), ("jw", "J-Wiener"))):
        for j, (s_fit, f) in enumerate(((5, os.path.join(fd, f"{meth}_s5.json")), (15, os.path.join(fd, f"{meth}_s15.json")),
                                         (25, os.path.join(RES, "aligned_wiener5_100.json" if meth == "wiener" else "aligned_jw_t1_best.json")))):
            if not os.path.exists(f):
                continue
            r = json.load(open(f))["results"]
            ax.plot([x["snr"] for x in r], [x["psnr"] for x in r], ["-", "--", ":"][j], color=COLORS[i], lw=1.3,
                    label=f"{lab}, fit @ {s_fit} dB")
    s, p = load("aligned_conv")
    ax.plot(s, p, "-", color=COLORS[3], lw=1.3, marker=MARKERS[3], ms=3, label="Conv (trained @ 25 dB)")
    ax.set_xlabel("Channel SNR (dB)", color=INK, fontsize=7.5)
    ax.set_title("(c) Fitting SNR", fontsize=8, color=INK, loc="left")
    style(ax); ax.legend(fontsize=5.4, frameon=False, labelcolor=INK, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    ax.set_ylim(15.5, 35.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.4, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_budget.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_budget.png"), dpi=200, bbox_inches="tight")


ROBUST = [("", "AWGN (reference)"), ("_mp3", "Multipath, 3 taps"), ("_mp8", "Multipath, 8 taps"),
          ("_cfo", "CFO 500 Hz"), ("_pn", "Phase noise 100 Hz"), ("_drop05", "5% packets lost"),
          ("_drop10", "10% packets lost"), ("_drop20", "20% packets lost")]


def table_robust(snr=25):
    """LaTeX rows: PSNR of matched / target / misaligned / aligners and ARR, per impairment."""
    rd = os.path.join(RES, "robust")
    base = {"matched": "matched", "target": "target", "misaligned": "misaligned", "conv": "aligned_conv",
            "rescnn": "aligned_rescnn", "jw": "aligned_jw_t1_best"}
    lines = []
    for sfx, lab in ROBUST:
        v = {}
        for k, b in base.items():
            f = os.path.join(RES, b + ".json") if not sfx else os.path.join(rd, f"{k}{sfx}.json")
            v[k] = [r["psnr"] for r in json.load(open(f))["results"] if r["snr"] == snr][0]
        arr = lambda k: (v[k] - v["misaligned"]) / (v["target"] - v["misaligned"])
        lines.append(f"{lab} & {v['matched']:.1f} & {v['target']:.1f} & {v['misaligned']:.1f} & "
                     f"{v['conv']:.1f} & {v['jw']:.1f} & {v['rescnn']:.1f} & {arr('rescnn'):.2f} \\\\")
    open(os.path.join(OUT, f"table_robust_{snr}.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def table_pairs():
    """LaTeX rows for every source model scored into ConvNeXt-AWGN (results/gr-align/pair_<name>)."""
    _, p_t = load("target")
    k = 4  # 25 dB
    rows = [("ADJSCC", RES, dict(matched="matched", misaligned="misaligned", conv="aligned_conv",
                                  wiener="aligned_wiener5_100", jw="aligned_jw_t1_best", rescnn="aligned_rescnn"))]
    for name, lab in (("spatialcsi", "ConvNeXt spatial-CSI"), ("b2019", "Bourtsoulatze")):
        d = os.path.join(RES, f"pair_{name}")
        if os.path.isdir(d):
            rows.append((lab, d, dict(matched="matched", misaligned="misaligned", conv="conv",
                                      wiener="wiener5", jw="jw_best", rescnn="rescnn")))
    out = []
    for lab, d, m in rows:
        v = {}
        for key, tag in m.items():
            f = os.path.join(d, tag + ".json")
            v[key] = json.load(open(f))["results"][k]["psnr"] if os.path.exists(f) else float("nan")
        arr = (v["rescnn"] - v["misaligned"]) / (p_t[k] - v["misaligned"])
        out.append(f"{lab} & {v['matched']:.1f} & {v['misaligned']:.1f} & {v['conv']:.1f} & {v['wiener']:.1f} & "
                   f"{v['jw']:.1f} & {v['rescnn']:.1f} & {arr:.2f} \\\\")
    open(os.path.join(OUT, "table_pairs.tex"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":

    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig_replica()
    fig_alpha()
    fig_qual()
    fig_budget()
    table_robust(25)
    table_robust(15)
    table_pairs()
    print(f"saved figures in {OUT}")
