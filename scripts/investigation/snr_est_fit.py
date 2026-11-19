#!/usr/bin/env python
"""Offline fit/evaluation of the pilot-based SNR estimator variants on the per-packet logger statistics saved by
snr_est_eval.py (results/gr-channel/snr_estimation/fit/*.json). Fit on seed 0, evaluate on the held-out seed 1."""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
D = os.path.join(HERE, "..", "..", "results", "gr-channel", "snr_estimation", "fit")
TX = 2.0

def load(seed, cpe):
    return json.load(open(f"{D}/snr_est_gap2_cfo0{'_cpe' if cpe else ''}_seed{seed}.json"))["results"]

def pooled(rows):
    st = np.array([r["logger"] for r in rows])                      # slope, s2, a2, T
    return np.sqrt(np.mean(st[:, 0] ** 2)), st[:, 1].mean(), st[:, 2].mean(), st[:, 3].mean()

def eff(slope, s2, a2, T, c0, k, ramp_corr):
    snr = max(a2, 1e-12) / max(s2, 1e-12) * TX
    ramp = 0.0 if ramp_corr else slope ** 2 * (T * T - 1) / 12.0
    return 1.0 / ((1 + c0) / snr + k / snr ** 2 + ramp)

def truths(res):
    rows = [r for r in res["rows"] if "logger" in r and r["logger"][1] > 0]
    sig = np.mean([r["g2"] * r["ps"] for r in rows]); mse = np.mean([r["mse"] for r in rows])
    n_lost = res["frames_tx"] - len(rows)
    t_rx = 10 * np.log10(sig / mse)
    t_er = 10 * np.log10(sig / ((len(rows) * mse + n_lost * sig) / res["frames_tx"]))
    return rows, t_rx, t_er, n_lost / res["frames_tx"]

def img_err(results, c0, k, cpe, snrs=(8, 12, 16, 20, 24, 28)):
    e = []
    for res in results:
        if res["snr"] not in snrs: continue
        rows, t_rx, _, _ = truths(res)
        e.append(10 * np.log10(eff(*pooled(rows), c0, k, cpe)) - t_rx)
    return np.array(e)

# ---- fit (c0, k) on seed 0, both chains jointly, image-level dB error over 4..28 dB
fitset = [(load(0, False), False), (load(0, True), True)]
best = None
for c0 in np.arange(0.0, 1.01, 0.01):
    for k in np.arange(0.0, 6.01, 0.05):
        err = np.concatenate([img_err(r, c0, k, cpe, snrs=(4, 8, 12, 16, 20, 24, 28)) for r, cpe in fitset])
        cost = float(np.mean(err ** 2))
        if best is None or cost < best[0]: best = (cost, c0, k)
_, C0, K = best
print(f"fitted on seed 0: c0 = {C0:.2f}, k = {K:.2f}  (current model: c0 = 0.50, k = 0)\n")

for cpe in (False, True):
    res = load(1, cpe)
    print(f"=== held-out seed 1, {'fixed chain (ramp corrected)' if cpe else 'as-is chain'}")
    print(" nominal | loss p  | image-level err: current  two-term | with erasures: truth  two-term+eras  err | per-packet std of (est-true): w=1  3  5  9")
    for r in res:
        rows, t_rx, t_er, p = truths(r)
        if len(rows) < 20: continue
        cur = 10 * np.log10(eff(*pooled(rows), 0.5, 0.0, cpe)) - t_rx
        e2 = eff(*pooled(rows), C0, K, cpe); two = 10 * np.log10(e2) - t_rx
        er = 10 * np.log10(1 / ((1 - p) / e2 + p))
        rows = sorted(rows, key=lambda x: x["pn"])
        stds = []
        for w in (1, 3, 5, 9):
            h = w // 2; d = []
            for i, x in enumerate(rows):
                nb = [y for y in rows[max(0, i - h): i + h + 1] if abs(y["pn"] - x["pn"]) <= h]
                _, s2w, a2w, Tw = pooled(nb)                           # smooth noise and tone power only,
                d.append(10 * np.log10(eff(abs(x["logger"][0]), s2w, a2w, Tw, C0, K, cpe)) - x["true"])   # keep own ramp
            stds.append((np.mean(d), np.std(d)))
        print(f"   {r['snr']:4.0f}  | {p:5.3f}  |          {cur:+6.2f}   {two:+6.2f}   |       {t_er:6.2f}   {er:6.2f}   {er - t_er:+6.2f} |   "
              + "  ".join(f"{s:.2f}" for _, s in stds) + f"   (bias w=5 {stds[2][0]:+.2f})")
json.dump(dict(c0=float(C0), k=float(K)), open(f"{D}/fitted_loss_model.json", "w"))
