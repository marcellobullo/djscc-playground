#!/usr/bin/env python
"""Decode over-the-air latent dumps (socket_rx.py --dump-dir) offline with any decoder and
any set of aligners, so one over-the-air recording scores every aligner on the *same*
received symbols (same channel realization, same lost packets).

Each dump frame holds the de-interleaved received latent (before clipping) and the live
CSI the receiver measured. Every decoded image is matched to its Kodak original by content
(as in eval_ota_alignment.py); if an original was received more than once, the first copy
is kept. Writes <out>/<label>/<kodim>.png and <out>/scores.json / scores.csv.

  python scripts/decode_ota_dumps.py --dump received_images/ota/pos1/dump_adjscc \
      --kodak ~/Downloads/kodak_dataset --model marcellobullo/djscc-convnext-cr6-awgn \
      --run misaligned=none --run conv=ckpts/adjscc-matched-snr-final/aligner-convnext-awgn \
      --run rescnn=ckpts/aligners/rescnn --out results/ota/pos1

Frames are mapped to originals once (by the dump's "stem" field when present, else by content on the
reference run, --match-run, default the first run) and every run is scored on that same mapping. List
the best-quality run first, e.g. --run matched=none@ckpts/adjscc-matched-snr-final/best.
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from jscc import load_aligner, load_codec  # noqa: E402

W, H = 768, 512


def load_img(p):
    return cv2.resize(np.array(Image.open(p).convert("RGB")), (W, H))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="folder of frame_*.npz from socket_rx.py --dump-dir")
    ap.add_argument("--kodak", required=True)
    ap.add_argument("--pattern", default="kodim*.png", help="image glob inside --kodak (e.g. '*.png' for the COCO subset)")
    ap.add_argument("--model", required=True, help="decoder (HF id | folder | alias | .pth)")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=ALIGNER[@MODEL]",
                    help="aligner folder or 'none', optionally @decoder model (default --model)")
    ap.add_argument("--match-run", default="", help="run whose decodes map frames to originals (default: first run; "
                    "use the best-quality run, e.g. matched)")
    ap.add_argument("--csi", type=float, default=None, help="override the recorded CSI (dB)")
    ap.add_argument("--clip-mag", type=float, default=5.0, help="same clipping as socket_rx.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from kaira.metrics.image import SSIM
    ssim = SSIM(data_range=1.0).eval()
    tt = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    origs = {os.path.splitext(os.path.basename(p))[0]: load_img(p) for p in sorted(glob.glob(f"{a.kodak}/{a.pattern}"))}
    keys = list(origs)
    small = np.stack([cv2.resize(origs[k], (96, 64)).astype(np.float32).ravel() for k in keys])
    small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6)

    frames = sorted(glob.glob(os.path.join(a.dump, "*.npz")))
    if not frames:
        sys.exit(f"no .npz dumps in {a.dump}")
    codecs = {}

    def get_dec(model):
        if model not in codecs:
            codecs[model] = load_codec(model, "decoder", packet_len=960, snr_db=19.0)
        return codecs[model]

    def decode(dec, f):
        d = np.load(f)
        sym = d["symbols"].astype(np.complex64)
        if a.clip_mag > 0:
            m = np.abs(sym)
            big = m > a.clip_mag
            sym[big] = sym[big] / m[big] * a.clip_mag
        csi = a.csi if a.csi is not None else float(d["csi_db"]) if "csi_db" in d else 19.0
        if hasattr(dec, "set_csi_vector") and "csi_vec_db" in d and a.csi is None:
            dec.set_csi_vector(d["csi_vec_db"])
        rec = np.frombuffer(dec.decode(sym, csi=csi), dtype=np.uint8).reshape(H, W, 3)
        lost = int((~d["seen"]).sum()) if "seen" in d else 0
        stem = str(d["stem"]) if "stem" in d else None
        return rec, csi, lost, stem

    runs = []
    for spec in a.run:
        label, rest = spec.split("=", 1)
        al, _, model = rest.partition("@")
        runs.append((label, al, model or a.model))

    def attach(dec, al):
        dec.set_aligner(None if al == "none" else load_aligner(al, dec.tcn, dec.h, dec.w, device=dec.device))

    # frame -> original, decided ONCE (dump field "stem" if present, else content match on the reference run,
    # first copy kept) and reused for every run, so badly decoded runs (misaligned) are scored on the right images
    ref = next((r for r in runs if r[0] == a.match_run), runs[0])
    mapping, cache = {}, {}
    dec = get_dec(ref[2]); attach(dec, ref[1])
    for f in frames:
        rec, csi, lost, stem = decode(dec, f)
        if stem is not None and stem in origs:
            k = stem
        else:
            v = cv2.resize(rec, (96, 64)).astype(np.float32).ravel()
            v = (v - v.mean()) / (v.std() + 1e-6)
            k = keys[int(np.argmax(small @ v))]
        if k in mapping.values():
            continue
        mapping[f] = k
        cache[f] = (rec, csi, lost)
    print(f"[*] {len(mapping)}/{len(frames)} frames mapped to distinct originals (reference run: {ref[0]})")

    rows, summary = [], {}
    for label, al, model in runs:
        dec = get_dec(model); attach(dec, al)
        od = os.path.join(a.out, label)
        os.makedirs(od, exist_ok=True)
        ps, ss = [], []
        for f, k in mapping.items():
            rec, csi, lost = cache[f] if (label, al, model) == ref else decode(dec, f)[:3]
            mse = np.mean((rec.astype(float) - origs[k].astype(float)) ** 2)
            p = 10 * np.log10(255 ** 2 / max(mse, 1e-10))
            with torch.no_grad():
                sv = float(ssim(tt(rec), tt(origs[k])).mean())
            ps.append(p); ss.append(sv)
            cv2.imwrite(os.path.join(od, f"{k}.png"), cv2.cvtColor(rec, cv2.COLOR_RGB2BGR))
            rows.append(dict(run=label, frame=os.path.basename(f), original=k, csi_db=round(csi, 2),
                             lost_pkts=lost, psnr=round(p, 3), ssim=round(sv, 4)))
        summary[label] = dict(n=len(ps), psnr=float(np.mean(ps)), psnr_std=float(np.std(ps)), ssim=float(np.mean(ss)),
                              aligner=al, model=model)
        print(f"{label:14s} n={len(ps):3d}  PSNR {np.mean(ps):6.2f} +- {np.std(ps):4.2f}  SSIM {np.mean(ss):.3f}", flush=True)
    csis = [r["csi_db"] for r in rows if r["run"] == rows[0]["run"]]
    summary["_csi_db_mean"] = float(np.mean(csis))
    summary["_lost_pkts_mean"] = float(np.mean([r["lost_pkts"] for r in rows if r["run"] == rows[0]["run"]]))
    json.dump(summary, open(os.path.join(a.out, "scores.json"), "w"), indent=2)
    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"mean live CSI {summary['_csi_db_mean']:.2f} dB, lost pkts/image {summary['_lost_pkts_mean']:.1f}")
    print("->", os.path.join(a.out, "scores.json"))


if __name__ == "__main__":
    main()
