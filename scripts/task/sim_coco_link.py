#!/usr/bin/env python
"""Fast latent-level AWGN simulation of the four-run protocol on the COCO subset (dry run for
the over-the-air task evaluation). Writes <out>/<config>/snr_<S>dB/<image_id>.png, which
eval_yolo.py reads directly.

Configs: matched (A->A), target (B->B), misaligned (A->B), and A->B with each RX aligner.

  python scripts/task/sim_coco_link.py --subset data/coco_subset500 --snr 10 25 --out out/coco_sim
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from jscc.aligners import load_aligner  # noqa: E402
from training.train import get_device, make_channel  # noqa: E402
from training.train_aligner import decode_tgt, encode_src, freeze  # noqa: E402

ALIGNERS = {
    "conv": "ckpts/adjscc-matched-snr-final/aligner-convnext-awgn",
    "wiener": "ckpts/aligners/wiener5",
    "jwiener": "ckpts/aligners/jw_t1_best",
    "rescnn": "ckpts/aligners/rescnn",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--src-model", default="ckpts/adjscc-matched-snr-final/best")
    p.add_argument("--tgt-model", default="marcellobullo/djscc-convnext-cr6-awgn")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 25.0])
    p.add_argument("--configs", nargs="+", default=["matched", "target", "misaligned", *ALIGNERS])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = get_device("auto")
    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(a.src_model, trust_remote_code=True).to(dev))
    tgt = freeze(AutoModel.from_pretrained(a.tgt_model, trust_remote_code=True).to(dev))
    st, tt = src.config.model_type, tgt.config.model_type
    C = tgt.config.M
    paths = sorted(glob.glob(os.path.join(a.subset, "tx", "*.png")))
    als = {}

    for snr in a.snr:
        torch.manual_seed(a.seed)
        ch = make_channel("awgn", snr)
        for i in range(0, len(paths), a.batch):
            ps = paths[i:i + a.batch]
            x = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(q), cv2.COLOR_BGR2RGB) for q in ps]))
            x = x.permute(0, 3, 1, 2).float().div(255).to(dev)
            with torch.no_grad():
                za, zb = encode_src(src, st, x, snr), encode_src(tgt, tt, x, snr)
                ya, yb = ch(za)[0], ch(zb)[0]
                h, w = zb.shape[2:]
                n_pkts = math.ceil((C * h * w // 2) / 960)
                csi = torch.full((len(ps), 1), snr, device=dev)
                for cfg in a.configs:
                    if cfg == "matched":
                        img = decode_tgt(src, st, ya, csi, None, n_pkts, 960, -20.0)
                    elif cfg == "target":
                        img = decode_tgt(tgt, tt, yb, csi, None, n_pkts, 960, -20.0)
                    else:
                        z = ya
                        if cfg != "misaligned":
                            if cfg not in als:
                                als[cfg] = load_aligner(ALIGNERS[cfg], C, h, w, device=dev)
                            z = als[cfg].apply_rx(ya)
                        img = decode_tgt(tgt, tt, z, csi, None, n_pkts, 960, -20.0)
                    img = (img.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                    od = os.path.join(a.out, cfg, f"snr_{int(snr)}dB")
                    os.makedirs(od, exist_ok=True)
                    for q, im in zip(ps, img):
                        cv2.imwrite(os.path.join(od, os.path.basename(q)), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            if (i // a.batch) % 25 == 0:
                print(f"  {snr:g} dB  {i + len(ps)}/{len(paths)}", flush=True)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
