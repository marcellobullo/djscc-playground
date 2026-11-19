"""Where does each aligner put its latent error, as seen by the decoder?

For each aligner T and Kodak image, e = T(y) - z_B. The decoder's response to e at z_B is
measured with a central finite difference (exact JVP up to O(eps^2)):

    J e  ~=  [D_B(z_B + eps e) - D_B(z_B - eps e)] / (2 eps)

Reported (averaged over images):
  * gain(e)   = ||J e||^2 / ||e||^2                  decoder sensitivity along the error
  * gain(rnd) = same for an isotropic random direction (mean sensitivity of D_B at z_B)
  * rel       = gain(e) / gain(rnd)    < 1 -> error placed in directions the decoder ignores
  * scale%    = share of ||e||^2 along z_B itself (pure rescaling of the latent)
  * rel_perp  = rel after removing the scale component from e
  * lin       = ||D(T(y)) - D(z_B)|| / ||J e||       (≈1 when the linearization holds)

    python scripts/investigation/decoder_sensitivity.py --snr 10 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import latent_frechet as L  # noqa: E402

NAMES = ["misaligned", "wiener", "jwiener", "conv", "rescnn"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-model", default="ckpts/adjscc-matched-snr-final/best")
    p.add_argument("--tgt-model", default="marcellobullo/djscc-convnext-cr6-awgn")
    p.add_argument("--kodak", default=os.path.expanduser("~/Downloads/kodak_dataset"))
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 25.0])
    p.add_argument("--eps", type=float, default=1e-2, help="finite-difference step, relative to ||z_B||/||e||")
    p.add_argument("--probes", type=int, default=4, help="random directions per image")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "decoder_sensitivity.json"))
    a = p.parse_args()

    dev = L.get_device("auto")
    from transformers import AutoModel
    src = L.freeze(AutoModel.from_pretrained(a.src_model, trust_remote_code=True).to(dev))
    tgt = L.freeze(AutoModel.from_pretrained(a.tgt_model, trust_remote_code=True).to(dev))
    st, tt = src.config.model_type, tgt.config.model_type
    C, H, W = tgt.config.M, tgt.config.img_height, tgt.config.img_width
    kodak = L.make_loader([a.kodak], H, W, 4, train=False, num_workers=0)

    results = {}
    for snr in a.snr:
        torch.manual_seed(a.seed)
        x, za, zb = L.encode_set(src, st, tgt, tt, kodak, snr, dev)
        h, w = zb.shape[2:]
        n_pkts = math.ceil((C * h * w // 2) / 960)
        y = L.make_channel("awgn", snr)(za)[0]

        def D(z):
            with torch.no_grad():
                return L.decode_tgt(tgt, tt, z, torch.full((len(z), 1), snr, device=dev), None, n_pkts, 960, -20.0)

        def Jv(z0, v):
            """J v at z0 (per sample), step scaled so eps*||v|| = a.eps*||z0||."""
            s = (a.eps * z0.flatten(1).norm(dim=1) / v.flatten(1).norm(dim=1).clamp_min(1e-12)).view(-1, 1, 1, 1)
            return (D(z0 + s * v) - D(z0 - s * v)) / (2 * s)

        def gain(z0, v):
            return (Jv(z0, v).flatten(1).pow(2).sum(1) / v.flatten(1).pow(2).sum(1)).cpu()

        outs = {}
        for n in NAMES:
            path = L.ALIGNERS[n]
            if path is None:
                outs[n] = y
            else:
                al = L.load_aligner(path, C, h, w, device=dev)
                with torch.no_grad():
                    outs[n] = torch.cat([al.apply_rx(y[i:i + 4]) for i in range(0, len(y), 4)])

        g = torch.Generator(device="cpu").manual_seed(a.seed + 1)
        rows = {}
        rnd_all = []
        for i in range(0, len(zb), 4):
            z0 = zb[i:i + 4]
            for _ in range(a.probes):
                u = torch.randn(z0.shape, generator=g).to(dev)
                rnd_all.append(gain(z0, u))
        # per-image random gain, averaged over probes (rnd_all is batch-major, probe-minor)
        per_img = torch.stack([torch.cat(rnd_all[k::a.probes]) for k in range(a.probes)]).mean(0)

        for n, z in outs.items():
            ge, gp, sc, lin = [], [], [], []
            for i in range(0, len(zb), 4):
                z0, e = zb[i:i + 4], (z - zb)[i:i + 4]
                ge.append(gain(z0, e))
                dir_ = z0.flatten(1) / z0.flatten(1).norm(dim=1, keepdim=True)
                par = (e.flatten(1) * dir_).sum(1, keepdim=True) * dir_
                sc.append((par.pow(2).sum(1) / e.flatten(1).pow(2).sum(1)).cpu())
                ep = (e.flatten(1) - par).view_as(e)
                gp.append(gain(z0, ep))
                actual = (D(z0 + e) - D(z0)).flatten(1).norm(dim=1)
                lin.append((actual / Jv(z0, e).flatten(1).norm(dim=1)).cpu())
            ge, gp, sc, lin = map(torch.cat, (ge, gp, sc, lin))
            rows[n] = {
                "gain_e": float(ge.mean()), "gain_rnd": float(per_img.mean()),
                "rel": float((ge / per_img).mean()), "rel_db": float(10 * torch.log10((ge / per_img).mean())),
                "scale_pct": float(100 * sc.mean()), "rel_perp": float((gp / per_img).mean()),
                "lin": float(lin.mean()),
                "latent_err": float((z - zb).pow(2).sum() / zb.pow(2).sum()),
            }
        results[str(snr)] = rows
        print(f"\n== {snr:g} dB   mean decoder gain on random directions: {float(per_img.mean()):.4g}")
        print(f"{'':12s}{'rel':>8s}{'rel dB':>8s}{'scale%':>8s}{'rel_perp':>9s}{'lin':>7s}")
        for n, r in rows.items():
            print(f"{n:12s}{r['rel']:8.3f}{r['rel_db']:8.2f}{r['scale_pct']:8.1f}{r['rel_perp']:9.3f}{r['lin']:7.2f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
