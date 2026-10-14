#!/usr/bin/env python
"""Score over-the-air alignment runs: PSNR / SSIM of each received folder against Kodak, plus the Alignment Recovery
Ratio ARR = (aligned - misaligned) / (matched - misaligned).

Each received image is matched to its Kodak original by content (normalized correlation), so lost or skipped images
do not shift the pairing. Metrics are computed like the plotting notebooks (768x512, RGB PSNR, Kaira SSIM).

  python scripts/eval_ota_alignment.py --kodak /path/to/kodak \
      --run aligned=received_images/ota_align/aligned --run misaligned=received_images/ota_align/misaligned \
      --run matched=received_images/ota_align/matched [--run target=received_images/ota_align/target]
"""
import argparse, csv, glob, os
import cv2
import numpy as np
import torch
from PIL import Image
from kaira.metrics.image import SSIM

W, H = 768, 512


def load(p):
    return cv2.resize(np.array(Image.open(p).convert("RGB")), (W, H))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kodak", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--out", default=None, help="CSV with per-image scores (default: <first run dir>/../alignment_scores.csv)")
    a = ap.parse_args()
    origs = {os.path.splitext(os.path.basename(p))[0]: load(p) for p in sorted(glob.glob(f"{a.kodak}/kodim*.png"))}
    keys = list(origs)
    small = np.stack([cv2.resize(origs[k], (96, 64)).astype(np.float32).ravel() for k in keys])
    small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6)
    ssim = SSIM(data_range=1.0).eval()
    t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    rows, summary = [], {}
    for spec in a.run:
        label, d = spec.split("=", 1)
        files = sorted(glob.glob(f"{d}/*.png"))
        ps, ss = [], []
        for f in files:
            rec = load(f)
            v = cv2.resize(rec, (96, 64)).astype(np.float32).ravel(); v = (v - v.mean()) / (v.std() + 1e-6)
            k = keys[int(np.argmax(small @ v))]
            mse = np.mean((rec.astype(float) - origs[k].astype(float)) ** 2)
            p = 10 * np.log10(255 ** 2 / mse)
            with torch.no_grad():
                s = float(ssim(t(rec), t(origs[k])).mean())
            ps.append(p); ss.append(s)
            rows.append(dict(run=label, file=os.path.basename(f), original=k, psnr=p, ssim=s))
        summary[label] = (np.mean(ps) if ps else np.nan, np.std(ps) if ps else np.nan, np.mean(ss) if ss else np.nan, len(ps),
                          len({r["original"] for r in rows if r["run"] == label}))
    print(f"{'run':12s} {'n':>3s} {'unique':>6s}  {'PSNR (dB)':>14s}  {'SSIM':>6s}")
    for label, (p, sd, s, n, u) in summary.items():
        print(f"{label:12s} {n:3d} {u:6d}  {p:7.2f} +- {sd:4.2f}  {s:6.3f}")
    if all(k in summary for k in ("aligned", "misaligned", "matched")):
        for i, name in ((0, "PSNR"), (2, "SSIM")):
            gap = summary["matched"][i] - summary["misaligned"][i]
            print(f"ARR ({name}) = {(summary['aligned'][i] - summary['misaligned'][i]) / gap:.2f}" if gap else f"ARR ({name}): no gap")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.run[0].split('=', 1)[1])), "alignment_scores.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("per-image scores ->", out)


if __name__ == "__main__":
    main()
