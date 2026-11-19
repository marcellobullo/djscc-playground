#!/usr/bin/env python
"""Goal-oriented evaluation of received images with YOLO-seg on the COCO subset.

Every run is a folder of received 512x768 images (letterboxed, from build_coco_subset.py). Each
image has its padding cropped off and is resized back to the original COCO size, then YOLO-seg
runs on it. Reported per run:

  box_map, box_map50     detection mAP@[.5:.95] / mAP@.5 (localization + class)
  box_map_agn            class-agnostic box mAP@[.5:.95] (localization only)
  mask_map, mask_map50   instance-segmentation mAP
  found                  share of GT objects hit by a detection (conf>=0.25, IoU>=0.5, any class)
  label_acc              share of those found objects whose predicted class is right

against two references (--gt):
  coco    COCO ground truth (absolute task performance)
  pseudo  YOLO's own detections on the untouched originals (conf>=0.5): isolates what the
          transmission costs, independent of YOLO's own mistakes

The reference run "original" (YOLO on orig/) is always evaluated first.

Received-image folders may name files <image_id:012d>.png (decode_ota_dumps.py) or
snr<S>_<NNN>.png (e2e_link.py --save-dir; NNN = 1-based index in sorted subset order).

  python scripts/task/eval_yolo.py --subset data/coco_subset500 \
      --run misaligned=out/e2e_coco/img/misaligned/snr_25dB --run rescnn=out/.../rescnn/snr_25dB \
      --out scripts/task/results/yolo_awgn25.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import io
import json
import os
import re

import cv2
import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ckpts", "yolo", "yolo11m-seg.pt")


def load_run(folder, meta):
    stems = sorted(meta)
    out = {}
    for f in glob.glob(os.path.join(folder, "*.png")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in meta:
            m = re.search(r"_(\d+)$", stem)
            if not m:
                continue
            stem = stems[int(m.group(1)) - 1]
        out[stem] = f
    return out


def unletterbox(img, g):
    crop = img[g["top"]:g["top"] + g["new_h"], g["left"]:g["left"] + g["new_w"]]
    return cv2.resize(crop, (g["orig_w"], g["orig_h"]), interpolation=cv2.INTER_CUBIC)


def recon_metrics(images, meta, subset):
    """PSNR / SSIM of the received image against the transmitted one, on the image region only
    (letterbox padding excluded), at 512x768 scale."""
    import torch
    from kaira.metrics.image import SSIM
    ssim = SSIM(data_range=1.0).eval()
    tt = lambda im: torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1)[None].float().div(255.0)
    ps, ss = [], []
    for stem, path in sorted(images.items()):
        g = meta[stem]
        sl = (slice(g["top"], g["top"] + g["new_h"]), slice(g["left"], g["left"] + g["new_w"]))
        rx = cv2.imread(path)[sl].astype(np.float64)
        tx = cv2.imread(os.path.join(subset, "tx", stem + ".png"))[sl].astype(np.float64)
        ps.append(10 * np.log10(255 ** 2 / max(np.mean((rx - tx) ** 2), 1e-10)))
        with torch.no_grad():
            ss.append(float(ssim(tt(rx), tt(tx)).mean()))
    return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), psnr_per_image=[float(x) for x in ps])


def predict(model, images, meta, coco91, imgsz, letterboxed):
    dets = []
    for stem, path in sorted(images.items()):
        g = meta[stem]
        img = cv2.imread(path)
        if letterboxed:
            img = unletterbox(img, g)
        r = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, retina_masks=True, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy().astype(np.uint8) if r.masks is not None else None
        for j in range(len(conf)):
            x0, y0, x1, y1 = xyxy[j].tolist()
            d = dict(image_id=g["id"], category_id=coco91[cls[j]], score=float(conf[j]),
                     bbox=[x0, y0, x1 - x0, y1 - y0])
            if masks is not None:
                rle = mask_util.encode(np.asfortranarray(masks[j]))
                rle["counts"] = rle["counts"].decode()
                d["segmentation"] = rle
            dets.append(d)
    return dets


def quiet_eval(gt, dets, iou_type, use_cats=True):
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
        ev = COCOeval(gt, dt, iou_type)
        ev.params.imgIds = sorted(gt.getImgIds())
        ev.params.useCats = int(use_cats)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def box_iou(a, b):
    """a [N,4], b [M,4] in xywh -> [N,M] IoU."""
    a = np.c_[a[:, :2], a[:, :2] + a[:, 2:]]
    b = np.c_[b[:, :2], b[:, :2] + b[:, 2:]]
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def found_and_label(gt, dets, conf=0.25, thr=0.5):
    by_img = {}
    for d in dets:
        if d["score"] >= conf:
            by_img.setdefault(d["image_id"], []).append(d)
    n = found = correct = 0
    for iid in gt.getImgIds():
        anns = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=iid)) if not a.get("iscrowd", 0)]
        if not anns:
            continue
        n += len(anns)
        ds = by_img.get(iid, [])
        if not ds:
            continue
        iou = box_iou(np.array([a["bbox"] for a in anns], float), np.array([d["bbox"] for d in ds], float))
        for i, a in enumerate(anns):
            j = int(iou[i].argmax())
            if iou[i, j] >= thr:
                found += 1
                correct += ds[j]["category_id"] == a["category_id"]
    return found / max(n, 1), correct / max(found, 1)


def pseudo_gt(coco_gt, dets, conf=0.5):
    ds = dict(images=coco_gt.dataset["images"], categories=coco_gt.dataset["categories"], annotations=[])
    for k, d in enumerate(x for x in dets if x["score"] >= conf):
        a = dict(id=k + 1, image_id=d["image_id"], category_id=d["category_id"], bbox=d["bbox"], iscrowd=0,
                 area=float(mask_util.area(d["segmentation"])) if "segmentation" in d else d["bbox"][2] * d["bbox"][3])
        if "segmentation" in d:
            a["segmentation"] = d["segmentation"]
        ds["annotations"].append(a)
    g = COCO()
    g.dataset = ds
    with contextlib.redirect_stdout(io.StringIO()):
        g.createIndex()
    return g


def score(gt, dets):
    bm, bm50 = quiet_eval(gt, dets, "bbox")
    agn, _ = quiet_eval(gt, dets, "bbox", use_cats=False)
    mm, mm50 = quiet_eval(gt, dets, "segm")
    found, lab = found_and_label(gt, dets)
    return dict(box_map=bm, box_map50=bm50, box_map_agn=agn, mask_map=mm, mask_map50=mm50,
                found=found, label_acc=lab, n_images=len({d["image_id"] for d in dets}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", required=True)
    p.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--gt", nargs="+", default=["coco", "pseudo"], choices=["coco", "pseudo"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.converter import coco80_to_coco91_class
    model = YOLO(a.weights)
    coco91 = coco80_to_coco91_class()
    meta = json.load(open(os.path.join(a.subset, "meta.json")))
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco = COCO(os.path.join(a.subset, "instances.json"))

    orig = {s: os.path.join(a.subset, "orig", s + ".jpg") for s in meta}
    runs = [("original", orig, False)] + [(lab, load_run(d, meta), True) for lab, d in (r.split("=", 1) for r in a.run)]
    dets = {}
    for lab, imgs, lb in runs:
        if len(imgs) < len(meta):
            print(f"[!] {lab}: {len(imgs)}/{len(meta)} images found (missing ones count as no detections)")
        dets[lab] = predict(model, imgs, meta, coco91, a.imgsz, lb)

    recon = {lab: recon_metrics(imgs, meta, a.subset) for lab, imgs, lb in runs if lb}
    refs = {}
    if "coco" in a.gt:
        refs["coco"] = gt_coco
    if "pseudo" in a.gt:
        refs["pseudo"] = pseudo_gt(gt_coco, dets["original"])
    res = {ref: {lab: score(g, dets[lab]) for lab, _, _ in runs} for ref, g in refs.items()}
    res["recon"] = recon
    res["_config"] = dict(weights=os.path.basename(a.weights), imgsz=a.imgsz, subset=a.subset,
                          n_images=len(meta), runs={lab: d for lab, d in (r.split("=", 1) for r in a.run)})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    cols = ["box_map", "box_map50", "box_map_agn", "mask_map", "found", "label_acc"]
    for ref in refs:
        print(f"\n== reference: {ref}")
        print(f"{'':14s}" + "".join(f"{c:>12s}" for c in cols))
        for lab, r in res[ref].items():
            print(f"{lab:14s}" + "".join(f"{r[c]:12.3f}" for c in cols))
    print("\n== reconstruction (image region only)")
    for lab, r in recon.items():
        print(f"{lab:14s}  PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.3f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
