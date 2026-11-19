#!/usr/bin/env python
"""Qualitative grid: YOLO-seg detections (boxes, labels, masks) on the same COCO images
before and after alignment. Rows = images, columns = original / runs.

Images are picked automatically (few, large, non-crowd objects) unless --ids is given.

  python scripts/task/show_yolo.py --subset data/coco_subset500 \
      --run "misaligned=out/coco_sim/misaligned/snr_25dB" --run "ResCNN=out/coco_sim/rescnn/snr_25dB" \
      --out scripts/task/results/yolo_examples_25dB
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_yolo import DEFAULT_WEIGHTS, unletterbox  # noqa: E402


def pick(subset, n, seed):
    coco = json.load(open(os.path.join(subset, "instances.json")))
    sizes = {im["id"]: im["width"] * im["height"] for im in coco["images"]}
    objs = {}
    for an in coco["annotations"]:
        objs.setdefault(an["image_id"], []).append(an)
    good = [iid for iid, a in objs.items()
            if 2 <= len(a) <= 6 and not any(x["iscrowd"] for x in a)
            and sum(x["area"] > 0.04 * sizes[iid] for x in a) >= 2]
    random.Random(seed).shuffle(good)
    return [f"{i:012d}" for i in sorted(good[:n])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    p.add_argument("--ids", nargs="*", default=[], help="image stems (default: auto-pick)")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    meta = json.load(open(os.path.join(a.subset, "meta.json")))
    stems = a.ids or pick(a.subset, a.n, a.seed)
    cols = [("original", None)] + [tuple(r.split("=", 1)) for r in a.run]

    fig, axes = plt.subplots(len(stems), len(cols), figsize=(3.0 * len(cols), 2.35 * len(stems)), squeeze=False)
    for i, s in enumerate(stems):
        for j, (lab, d) in enumerate(cols):
            if d is None:
                img = cv2.imread(os.path.join(a.subset, "orig", s + ".jpg"))
            else:
                img = unletterbox(cv2.imread(os.path.join(d, s + ".png")), meta[s])
            r = model.predict(img, imgsz=640, conf=a.conf, retina_masks=True, verbose=False)[0]
            lw = max(2, round(max(img.shape[:2]) / 250))
            vis = r.plot(line_width=lw, font_size=max(12, lw * 6))
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(lab, fontsize=11)
            if j == 0:
                ax.set_ylabel(s.lstrip("0"), fontsize=8, color="#555")
    if a.title:
        fig.suptitle(a.title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=150)
    print(f"{a.out}.png  ids: {' '.join(stems)}")


if __name__ == "__main__":
    main()
