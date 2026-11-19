#!/usr/bin/env python
"""Lab emulator vs GNU Radio replica on ONE SNR axis: the SNR in the 1 MHz band (what the analyzer's channel
power measures and what the replica's nominal SNR means). The lab dial is mapped to that axis with
x = dial - DELTA, DELTA = 1.2 dB (best fit of the analyzer channel-power readings at dial 12/8/4 dB; consistent
range 0.3..2.0 dB). Replica = lab-like chain (2-sample frame gap), SSCC with the frame gap (no drops)."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_gr_vs_emulator as P

DELTA, DLO, DHI = 1.2, 0.3, 2.0
emu = P.load_emulator(f"{P.GR}/emulator_metrics.json")
emu_dj = P.curve(emu["djscc"]); emu_ss = {k: P.curve(v) for k, v in emu["sscc"].items()}
gr_dj = {k: P.curve(P.load_gr_djscc(k, "labmatched")[0]) for k in P.DJSCC_RUNS}
gr_ss = {k: P.curve(v) for k, v in P.load_gr_sscc("sscc_gap16").items()}

def shifted(c):  # dial -> SNR in the 1 MHz band
    return {s - DELTA: v for s, v in c.items()}
def xs(c): return sorted(c)
def line(ax, c, which, **kw):
    x = xs(c); m = np.array([c[s][which][0] for s in x]); sd = np.array([c[s][which][1] for s in x])
    ax.plot(x, m, **kw); ax.fill_between(x, m - sd, m + sd, color=kw.get("color"), alpha=0.07)

fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
for j, which in enumerate((0, 1)):
    ax, ax2 = axs[j, 0], axs[j, 1]
    lab = shifted(emu_dj)
    line(ax, lab, which, label=f"Lab emulator, dial - {DELTA} dB", color="k", marker="o", lw=2.2)
    xl = np.array(xs(lab)); m = np.array([lab[s][which][0] for s in xl])
    ax.errorbar(xl, m, xerr=[np.full_like(xl, DELTA - DLO + 0.0) * 0 + (DHI - DELTA), np.full_like(xl, DELTA - DLO)], fmt="none", ecolor="0.5", elinewidth=1, capsize=2, label=f"offset range {DLO}-{DHI} dB")
    for k, (lab_, col) in P.DJSCC_RUNS.items():
        line(ax, gr_dj[k], which, label=f"Replica: {lab_}", color=col, marker="s", ls="--", lw=1.6)
    for k in sorted(emu_ss):
        col, ls, mk, lb = P.sscc_style(k)
        line(ax2, shifted(emu_ss[k]), which, color=col, marker=mk, ls="-", lw=1.2, ms=4, label=f"lab: {lb}")
        if k in gr_ss: line(ax2, gr_ss[k], which, color=col, marker=mk, ls=":", lw=1.2, ms=4, mfc="none")
    for a_ in (ax, ax2):
        a_.set_xlabel("SNR in the 1 MHz band (dB)"); a_.grid(linestyle=":", alpha=0.5); a_.set_ylabel("PSNR (dB)" if which == 0 else "SSIM")
axs[0, 0].set_title("DJSCC: lab (dial shifted to the true SNR) vs replica"); axs[0, 1].set_title("SSCC: lab (solid, shifted) vs replica (dotted, hollow)")
axs[0, 0].legend(fontsize=8, loc="lower right"); axs[0, 1].legend(fontsize=6, loc="lower right")
fig.suptitle(f"Common SNR axis = SNR in the 1 MHz band; lab dial shifted by -{DELTA} dB (analyzer-derived, uncertain {DLO}-{DHI} dB)")
fig.savefig(f"{P.GR}/5_common_axis_lab_vs_replica.png", dpi=190)

# agreement numbers for the deployed model: lab at (dial - delta) vs replica interpolated there
rows = []
rx = xs(gr_dj["deployed"]); ry = [gr_dj["deployed"][s][0][0] for s in rx]
for d in (0, 4, 8, 12, 16, 20, 24, 28):
    x = d - DELTA
    for dl in (0.0, DELTA):
        xx = d - dl; rp = float(np.interp(xx, rx, ry)); lb = emu_dj[d][0][0]
        rows.append(dict(dial=d, shift_db=dl, lab_psnr=lb, replica_psnr=rp, diff=rp - lb))
with open(f"{P.GR}/5_common_axis_deployed_agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for dl in (0.0, DELTA):
    d = np.array([r["diff"] for r in rows if r["shift_db"] == dl]); print(f"shift {dl}: replica-lab mean {d.mean():+.2f} dB, rms {np.sqrt((d**2).mean()):.2f}, low-SNR(0-12) mean {d[:4].mean():+.2f}")
