#!/usr/bin/env python
"""Build a fixed COCO val2017 subset for the goal-oriented (YOLO) evaluation.

The DJSCC codecs take a fixed 512x768 (HxW) input, while COCO images vary in size and
orientation. Each selected image is letterboxed: scaled to fit inside 512x768 with its aspect
ratio kept, then centred on a gray (114) canvas. The geometry is stored in meta.json so the
evaluation can crop the padding off the *received* image and resize it back to the original
size, so COCO ground truth (boxes, masks) applies unchanged.

Outputs in --out:
  tx/<image_id:012d>.png     512x768 letterboxed images to transmit (feed this folder to
                             e2e_link.py / the OTA transmitter with --pattern '*.png')
  orig/<image_id:012d>.jpg   untouched originals (reference "no channel" evaluation)
  instances.json             COCO annotations restricted to the subset
  meta.json                  {stem: {id, orig_w, orig_h, scale, left, top, new_w, new_h}}

Images come either from a local val2017 folder (--images-dir) or are fetched one by one from
the coco_url in the annotation file (only the subset is downloaded).

  python scripts/task/build_coco_subset.py --ann data/coco/annotations/instances_val2017.json \
      --n 500 --out data/coco_subset500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request

import cv2
import numpy as np

H, W = 512, 768


def letterbox(img, pad=114):
    h, w = img.shape[:2]
    s = min(H / h, W / w)
    nh, nw = round(h * s), round(w * s)
    top, left = (H - nh) // 2, (W - nw) // 2
    out = np.full((H, W, 3), pad, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, dict(scale=s, left=left, top=top, new_w=nw, new_h=nh)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="instances_val2017.json")
    p.add_argument("--images-dir", default="", help="local val2017 folder (else download each image)")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    coco = json.load(open(a.ann))
    has_obj = {an["image_id"] for an in coco["annotations"] if not an.get("iscrowd", 0)}
    imgs = sorted((im for im in coco["images"] if im["id"] in has_obj), key=lambda im: im["id"])
    random.Random(a.seed).shuffle(imgs)
    imgs = sorted(imgs[: a.n], key=lambda im: im["id"])
    ids = {im["id"] for im in imgs}

    for d in ("tx", "orig"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    meta = {}
    for k, im in enumerate(imgs):
        stem = f"{im['id']:012d}"
        dst = os.path.join(a.out, "orig", stem + ".jpg")
        if not os.path.isfile(dst):
            if a.images_dir:
                shutil.copy(os.path.join(a.images_dir, im["file_name"]), dst)
            else:
                urllib.request.urlretrieve(im["coco_url"], dst)
        bgr = cv2.imread(dst)
        assert bgr.shape[:2] == (im["height"], im["width"]), stem
        lb, g = letterbox(bgr)
        cv2.imwrite(os.path.join(a.out, "tx", stem + ".png"), lb)
        meta[stem] = dict(id=im["id"], orig_w=im["width"], orig_h=im["height"], **g)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(imgs)}", flush=True)

    sub = dict(info=coco.get("info", {}), licenses=coco.get("licenses", []), categories=coco["categories"],
               images=imgs, annotations=[an for an in coco["annotations"] if an["image_id"] in ids])
    json.dump(sub, open(os.path.join(a.out, "instances.json"), "w"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    n_ann = len(sub["annotations"])
    portrait = sum(m["orig_h"] > m["orig_w"] for m in meta.values())
    print(f"{len(imgs)} images ({portrait} portrait), {n_ann} annotations -> {a.out}")


if __name__ == "__main__":
    main()
