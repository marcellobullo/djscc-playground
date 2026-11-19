#!/usr/bin/env python
"""Sweep alpha of the latent-anchored Jacobian-weighted Wiener aligner from saved statistics
(training/fit_closed_form_aligner.py --method jwiener --save-stats ...), scoring each solution on
held-out images (DIV2K valid) through simulated AWGN at --snr-db, decoding with the target
decoder. alpha = 1 is plain Wiener (latent MSE); alpha = 0 is the image-space metric only.
Saves the best one as a conv aligner folder.

  python scripts/investigation/sweep_jwiener_alpha.py ckpts/aligners/jw_t1_stats.pt --out ckpts/aligners/jw_t1_best
"""
import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from jscc.data import make_loader  # noqa: E402
from training.fit_closed_form_aligner import patches, to_conv5  # noqa: E402
from training.train import get_device, make_channel  # noqa: E402
from training.train_aligner import decode_tgt, encode_src, freeze  # noqa: E402


def solve(st, alpha, ridge):
    C, d = st["C"], st["d"]
    K = torch.kron(st["XtX"], torch.eye(C, dtype=st["XtX"].dtype))
    kv = st["XtY"].reshape(-1)
    U, u = st["UtU"], st["Utt"]
    A = (1 - alpha) * U / U.trace() + alpha * K / K.trace()
    rhs = (1 - alpha) * u / U.trace() + alpha * kv / K.trace()
    A = A + ridge * A.diagonal().mean() * torch.eye(len(A), dtype=A.dtype)
    B = torch.linalg.solve(A, rhs).reshape(d + 1, C)
    return B[:d].T, B[d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--src-model", default=os.path.join(ROOT, "ckpts/adjscc-matched-snr-final/best"))
    ap.add_argument("--tgt-model", default="marcellobullo/djscc-convnext-cr6-awgn")
    ap.add_argument("--val-dir", default=os.path.expanduser("~/research/djscc-demo-alignment/data/DIV2K_valid_HR"))
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--snr-db", type=float, default=25.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = get_device("auto")
    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(a.src_model, trust_remote_code=True).to(dev))
    tgt = freeze(AutoModel.from_pretrained(a.tgt_model, trust_remote_code=True).to(dev))
    st_, tt = src.config.model_type, tgt.config.model_type
    H, W = tgt.config.img_height, tgt.config.img_width
    stats = torch.load(a.stats)
    C, d = stats["C"], stats["d"]
    k = int(round(math.sqrt(d // C)))

    # fixed held-out set: received source latents (one noise draw) + originals
    data, n = [], 0
    with torch.no_grad():
        for x, _ in make_loader([a.val_dir], H, W, 1, train=False, num_workers=2):
            x = x.to(dev)
            z = encode_src(src, st_, x, a.snr_db)
            y = make_channel("awgn", a.snr_db)(z)[0]
            data.append((x, y)); n += 1
            if n >= a.images:
                break
    n_pkts = math.ceil((C * y.shape[2] * y.shape[3] // 2) / 960)
    csi = torch.full((1, 1), a.snr_db, device=dev)

    res = []
    for al in a.alphas:
        Wm, b = solve(stats, al, a.ridge)
        sd = to_conv5(Wm, b, C, k)
        conv = torch.nn.Conv2d(C, C, 5, padding=2).to(dev)
        conv.weight.data, conv.bias.data = sd["conv.weight"].to(dev), sd["conv.bias"].to(dev)
        ps = []
        with torch.no_grad():
            for x, y in data:
                out = decode_tgt(tgt, tt, conv(y), csi, None, n_pkts, 960, -20.0).clamp(0, 1)
                ps.append(float(10 * torch.log10(1 / ((out - x) ** 2).mean())))
        p = sum(ps) / len(ps)
        res.append((al, p, sd))
        print(f"alpha {al:5.3f}  val PSNR {p:6.2f} dB  |W| {Wm.norm():.3f}", flush=True)
    best = max(res, key=lambda r: r[1])
    print(f"[*] best alpha {best[0]} -> {best[1]:.2f} dB (alpha=1 is plain Wiener: {res[-1][1]:.2f} dB)")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        torch.save(best[2], os.path.join(a.out, "aligner.pth"))
        json.dump(dict(kind="conv", objective="closed_form", method="jwiener", alpha=best[0], kernel=k,
                       stats=os.path.basename(a.stats), selected_on=f"{a.images} DIV2K valid @ {a.snr_db} dB",
                       sweep={str(r[0]): round(r[1], 3) for r in res}, tcn=C, applied_at="receiver"),
                  open(os.path.join(a.out, "aligner_config.json"), "w"), indent=2)
        print(f"[*] saved {a.out}")


if __name__ == "__main__":
    main()
