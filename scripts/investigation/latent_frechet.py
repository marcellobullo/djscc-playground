"""Latent Fréchet distance of aligner outputs to the target encoder's latent distribution.

For every aligner T, the received latent y = AWGN(E_A(x)) is mapped to T(y) and compared with
the real target latents z_B = E_B(x) on the same Kodak images, in two ways:

  * distribution  : Fréchet distance between the Gaussian fits of {T(y)} and {z_B}, using the
                    16-d channel vector at each latent position (d=16) and the 3x3 neighbourhood
                    (d=144, captures local spatial structure). Unpaired: ignores which image is which.
  * correspondence: paired latent NMSE  ||T(y) - z_B||^2 / ||z_B||^2   (dB).
  * end to end    : PSNR of D_B(T(y)) against x.

Extra row "wiener+noise": Wiener output plus Gaussian noise drawn from the Wiener residual
covariance (estimated on DIV2K-valid pilots, not Kodak), i.e. the simplest posterior sampler.

Floor row "z_B (DIV2K)": target latents of a different image set, to show the FD scale that
comes from image content alone.

    python scripts/investigation/latent_frechet.py --snr 10 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from jscc.aligners import load_aligner  # noqa: E402
from jscc.data import make_loader  # noqa: E402
from training.train import get_device, make_channel  # noqa: E402
from training.train_aligner import decode_tgt, encode_src, freeze  # noqa: E402

A = "ckpts/aligners"
ALIGNERS = {
    "misaligned": None,
    "conv": "ckpts/adjscc-matched-snr-final/aligner-convnext-awgn",
    "wiener": f"{A}/wiener5",
    "jwiener": f"{A}/jw_t1_best",
    "rescnn": f"{A}/rescnn",
    "gauss-OT (blind)": f"{A}/ot1",
    "OT + 8 pilots": f"{A}/otpilot1_p8",
    "parseval": f"{A}/parseval5",
}


def vecs(z, k):
    """[B, C, h, w] -> [B*h*w, C*k*k] rows (k x k neighbourhood, zero padded)."""
    if k == 1:
        return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    u = F.unfold(z, k, padding=k // 2)
    return u.transpose(1, 2).reshape(-1, z.shape[1] * k * k)


def gauss(X):
    X = X.double()
    mu = X.mean(0)
    Xc = X - mu
    return mu, Xc.T @ Xc / (len(X) - 1)


def frechet(a, b):
    (m1, S1), (m2, S2) = a, b
    w, V = torch.linalg.eigh(S1)
    r1 = (V * w.clamp_min(0).sqrt()) @ V.T
    M = r1 @ S2 @ r1
    tr_sqrt = torch.linalg.eigvalsh((M + M.T) / 2).clamp_min(0).sqrt().sum()
    return float(((m1 - m2) ** 2).sum() + torch.trace(S1) + torch.trace(S2) - 2 * tr_sqrt)


def encode_set(src, st, tgt, tt, loader, snr, dev):
    xs, za, zb = [], [], []
    for images, _ in loader:
        images = images.to(dev)
        with torch.no_grad():
            za.append(encode_src(src, st, images, snr))
            zb.append(encode_src(tgt, tt, images, snr))
        xs.append(images)
    return torch.cat(xs), torch.cat(za), torch.cat(zb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-model", default="ckpts/adjscc-matched-snr-final/best")
    p.add_argument("--tgt-model", default="marcellobullo/djscc-convnext-cr6-awgn")
    p.add_argument("--kodak", default=os.path.expanduser("~/Downloads/kodak_dataset"))
    p.add_argument("--div2k-valid", default=os.path.expanduser(
        "~/research/djscc-demo-alignment/data/DIV2K_valid_HR"))
    p.add_argument("--ref-images", type=int, default=24)
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 25.0])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "scripts/investigation/results/latent_frechet.json"))
    a = p.parse_args()

    dev = get_device(a.device)
    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(a.src_model, trust_remote_code=True).to(dev))
    tgt = freeze(AutoModel.from_pretrained(a.tgt_model, trust_remote_code=True).to(dev))
    st, tt = src.config.model_type, tgt.config.model_type
    C, H, W = tgt.config.M, tgt.config.img_height, tgt.config.img_width

    kodak = make_loader([a.kodak], H, W, 4, train=False, num_workers=0)
    ref_ds = make_loader([a.div2k_valid], H, W, 4, train=False, num_workers=0)

    results = {}
    for snr in a.snr:
        torch.manual_seed(a.seed)
        x, za, zb = encode_set(src, st, tgt, tt, kodak, snr, dev)
        h, w = zb.shape[2:]
        n_pkts = math.ceil((C * h * w // 2) / 960)
        ch = make_channel("awgn", snr)
        with torch.no_grad():
            y = ch(za)[0]

        # DIV2K-valid pilots: floor reference + Wiener residual covariance (never Kodak)
        xr, zar, zbr = [], [], []
        for i, (images, _) in enumerate(ref_ds):
            if 4 * i >= a.ref_images:
                break
            images = images.to(dev)
            with torch.no_grad():
                zar.append(encode_src(src, st, images, snr))
                zbr.append(encode_src(tgt, tt, images, snr))
        zar, zbr = torch.cat(zar), torch.cat(zbr)

        wien = load_aligner(ALIGNERS["wiener"], C, h, w, device=dev)
        with torch.no_grad():
            res = vecs(zbr - wien.apply_rx(ch(zar)[0]), 1).cpu().double()
        Lres = torch.linalg.cholesky(torch.cov(res.T) + 1e-9 * torch.eye(C, dtype=torch.float64))

        outs = {}
        for name, path in ALIGNERS.items():
            if path is None:
                outs[name] = y
                continue
            al = load_aligner(path, C, h, w, device=dev)
            with torch.no_grad():
                outs[name] = torch.cat([al.apply_rx(y[i:i + 4]) for i in range(0, len(y), 4)])
        g = torch.Generator().manual_seed(a.seed + 1)
        noise = (torch.randn(zb.numel() // C, C, generator=g, dtype=torch.float64) @ Lres.T).float()
        noise = noise.reshape(zb.shape[0], h, w, C).permute(0, 3, 1, 2).to(dev)
        outs["wiener+noise"] = outs["wiener"] + noise

        ref = {k: gauss(vecs(zb, k).cpu()) for k in (1, 3)}
        rows = {}
        for name, z in list(outs.items()) + [("z_B (DIV2K)", zbr)]:
            r = {f"fd{k*k*C}": frechet(gauss(vecs(z, k).cpu()), ref[k]) for k in (1, 3)}
            if name != "z_B (DIV2K)":
                r["nmse_db"] = float(10 * torch.log10(((z - zb) ** 2).sum() / (zb ** 2).sum()))
                ps = []
                with torch.no_grad():
                    for i in range(0, len(z), 4):
                        img = decode_tgt(tgt, tt, z[i:i + 4], torch.full((len(z[i:i + 4]), 1), snr, device=dev),
                                         None, n_pkts, 960, -20.0).clamp(0, 1)
                        mse = ((img - x[i:i + 4]) ** 2).flatten(1).mean(1)
                        ps += (10 * torch.log10(1 / mse.clamp_min(1e-10))).cpu().tolist()
                r["psnr"] = float(np.mean(ps))
            rows[name] = r
        rows["z_B (Kodak)"] = {"fd16": 0.0, "fd144": 0.0, "nmse_db": float("-inf")}
        results[str(snr)] = rows

        print(f"\n== {snr:g} dB   (trace of Σ_B, d=16: {float(torch.trace(ref[1][1])):.3f})")
        print(f"{'':18s}{'FD16':>9s}{'FD144':>9s}{'NMSE dB':>9s}{'PSNR':>8s}")
        for name, r in rows.items():
            print(f"{name:18s}{r['fd16']:9.4f}{r['fd144']:9.3f}"
                  f"{r.get('nmse_db', float('nan')):9.2f}{r.get('psnr', float('nan')):8.2f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
