#!/usr/bin/env python3
"""Training-free (closed-form) receiver-side aligners, fitted from paired latents.

No gradient training: one pass over a set of "semantic pilot" images collects the
source-encoder latent (through the channel) and the target model's own encoder latent
for the same image, then the aligner is computed in closed form. Both methods act on
local latent patches (k x k neighbourhood of the 16-channel latent), so they stay
receiver-only and keep the transmitted dimension unchanged, unlike the full-latent
zero-shot equalizer of Pannacci et al., which compresses the whole latent at the TX to
at most N (number of anchors) dimensions.

  --method wiener    linear MMSE / ridge regression from the received k x k patch to
                     the target latent vector (with bias): the best *linear* conv
                     aligner in the latent-MSE sense. Noise enters through the
                     channel-corrupted inputs.
  --method jwiener   decoder-Jacobian-weighted Wiener: minimizes the *image-space* error
                     linearized through the target decoder,
                       sum_i (W x_i + b - z_i)^T J_i^T J_i (W x_i + b - z_i),
                     where J_i is the block of the target decoder's Jacobian at latent
                     position i (evaluated at the clean target latent). The metric
                     J^T J is estimated with random image-space probes v ~ N(0, I):
                     g = J^T v gives E[g g^T] = J^T J. A position-independent metric
                     would leave the unconstrained least-squares solution unchanged,
                     so the gain can only come from how the metric varies over
                     positions and images.
                     With --tile T > 1 the probe residual g^T r is summed inside T x T
                     latent tiles before squaring, so the cross-position terms of J^T J
                     inside a tile are kept (the full image error is the limit T = h).
  --method ot        blind (pilot-free) Gaussian optimal transport, 1x1: the closed-form
                     Monge map between N(mu_y, S_y) of the received source latents and
                     N(mu_B, S_B) of the target encoder's latents, computed from
                     *different* images (unpaired):
                       T(y) = mu_B + S_y^-1/2 (S_y^1/2 S_B S_y^1/2)^1/2 S_y^-1/2 (y - mu_y).
                     Second-order statistics fix the map only up to a rotation; OT
                     picks the minimum-displacement (symmetric) one.
  --method otpilot   semi-blind: blind whitening / colouring as in `ot`, plus the
                     rotation from `--pilot-images` paired pilot images (orthogonal
                     Procrustes on the whitened pilot vectors):
                       T(y) = mu_B + S_B^1/2 R S_y^-1/2 (y - mu_y).
  --method parseval  Parseval-frame equalizer (Pannacci et al., zero-shot recipe applied
                     patch-wise): pick `--anchors` pilot patches, build the Parseval
                     frames F~ = polar(A_in) and G~ = polar(A_out)^T from the anchor
                     matrices and map x -> G~ F~ x (whitening cancels at the receiver).

The result is saved as a ``conv`` aligner (Conv2d(16, 16, 5); a 1x1 or 3x3 fit is
zero-padded into the 5x5 kernel), so socket_rx / e2e_link load it unchanged.

  python training/fit_closed_form_aligner.py --method wiener --kernel 5 \\
      --src-model ckpts/adjscc-matched-snr-final/best --tgt-model marcellobullo/djscc-convnext-cr6-awgn \\
      --data-dir data/DIV2K_train_HR --snr-db 25 --out ckpts/aligners/wiener5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from jscc.data import make_loader  # noqa: E402
from training.train import get_device, make_channel  # noqa: E402
from training.train_aligner import decode_tgt, encode_src, freeze  # noqa: E402


def patches(z, k):
    """[B, C, h, w] -> [B*h*w, C*k*k] (zero-padded k x k neighbourhoods)."""
    B, C, h, w = z.shape
    u = F.unfold(z, k, padding=k // 2)            # [B, C*k*k, h*w]
    return u.transpose(1, 2).reshape(-1, C * k * k)


def polar(a):
    """Orthogonal (Parseval) factor U V^T of a."""
    U, _, Vt = torch.linalg.svd(a, full_matrices=False)
    return U @ Vt


def sqrtm_sym(S, inv=False, eps=1e-9):
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(eps)
    return (V * (w.rsqrt() if inv else w.sqrt())) @ V.T


class Moments:
    """Running mean / covariance of row vectors (float64, CPU)."""

    def __init__(self, d):
        self.n, self.s, self.ss = 0, torch.zeros(d, dtype=torch.float64), torch.zeros(d, d, dtype=torch.float64)

    def add(self, X):
        self.n += len(X); self.s += X.sum(0); self.ss += X.T @ X

    def mean_cov(self):
        mu = self.s / self.n
        return mu, self.ss / self.n - torch.outer(mu, mu)


def to_conv5(W, b, C, k):
    """[C, C*k*k] weights (+bias) -> Conv2d(C, C, 5) state dict."""
    w = torch.zeros(C, C, 5, 5, dtype=torch.float32)
    o = (5 - k) // 2
    w[:, :, o:o + k, o:o + k] = W.reshape(C, C, k, k).float()
    return {"conv.weight": w, "conv.bias": (b if b is not None else torch.zeros(C)).float()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["wiener", "jwiener", "ot", "otpilot", "parseval"], required=True)
    p.add_argument("--kernel", type=int, default=5, choices=[1, 3, 5])
    p.add_argument("--src-model", required=True)
    p.add_argument("--tgt-model", required=True)
    p.add_argument("--data-dir", nargs="+", required=True)
    p.add_argument("--images", type=int, default=0, help="number of images used (0 = all)")
    p.add_argument("--snr-db", type=float, default=25.0, help="channel SNR of the pilots (and ADJSCC conditioning)")
    p.add_argument("--anchors", type=int, default=0, help="[parseval] number of anchor patches (0 = C*k*k)")
    p.add_argument("--ridge", type=float, default=1e-3, help="[wiener/jwiener] ridge, relative to the mean diagonal")
    p.add_argument("--probes", type=int, default=4, help="[jwiener] random image-space probes per image")
    p.add_argument("--positions", type=int, default=4096, help="[jwiener, tile 1] latent positions sampled per image")
    p.add_argument("--tile", type=int, default=1,
                   help="[jwiener] sum the probe residual inside T x T latent tiles before squaring, which keeps "
                        "the cross-position terms of J^T J inside a tile (1 = per-position blocks only)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="[jwiener] weight of the latent-space Wiener term (0 = image metric only, 1 = plain Wiener)")
    p.add_argument("--save-stats", default="", help="[jwiener] save the normal-equation statistics (.pt) for alpha sweeps")
    p.add_argument("--pilot-images", type=int, default=1, help="[otpilot] paired pilot images for the rotation")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = get_device(a.device)
    if a.method in ("ot", "otpilot"):
        a.kernel = 1                                   # same-dimension maps only
    if a.method == "jwiener":
        a.batch_size = 1                               # per-image Jacobian probes

    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(a.src_model, trust_remote_code=True).to(dev))
    tgt = freeze(AutoModel.from_pretrained(a.tgt_model, trust_remote_code=True).to(dev))
    st, tt = src.config.model_type, tgt.config.model_type
    C, H, W = tgt.config.M, tgt.config.img_height, tgt.config.img_width
    k = a.kernel
    d = C * k * k
    n_anchor = a.anchors or d
    loader = make_loader(a.data_dir, H, W, a.batch_size, train=False, num_workers=4)

    XtX = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    XtY = torch.zeros(d + 1, C, dtype=torch.float64)
    D1 = (d + 1) * C
    UtU = torch.zeros(D1, D1, dtype=torch.float64) if a.method == "jwiener" else None
    Utt = torch.zeros(D1, dtype=torch.float64) if a.method == "jwiener" else None
    mom_y, mom_b = Moments(C), Moments(C)
    pil_y, pil_b = [], []
    pool_in, pool_out, n_img, t0 = [], [], 0, time.time()
    for bi, (images, _) in enumerate(loader):
        images = images.to(dev)
        with torch.no_grad():
            z_in = encode_src(src, st, images, a.snr_db)
            z_out = encode_src(tgt, tt, images, a.snr_db)
            ch = make_channel("awgn", a.snr_db)
            y_in = ch(z_in)[0] if ch is not None else z_in
        X = patches(y_in, k).cpu().double()
        Y = z_out.permute(0, 2, 3, 1).reshape(-1, C).cpu().double()
        if a.method == "wiener":
            X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
            XtX += X1.T @ X1
            XtY += X1.T @ Y
        elif a.method == "jwiener":
            h, w = z_out.shape[2:]
            n_pkts = math.ceil((C * h * w // 2) / 960)
            z = z_out.detach().clone().requires_grad_(True)
            img = decode_tgt(tgt, tt, z, torch.full((1, 1), a.snr_db, device=dev), None, n_pkts, 960, -20.0)
            T = a.tile
            if T <= 1:   # per-position metric blocks only (no cross-position terms)
                sel = torch.randperm(len(X))[: a.positions]
                X1 = torch.cat([X[sel], torch.ones(len(sel), 1, dtype=X.dtype)], 1).float().to(dev)   # [S, d+1]
                Ys = Y[sel].float().to(dev)                                                           # [S, C]
            else:        # all positions, summed inside T x T latent tiles before squaring
                X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1).float().to(dev)
                Ys = Y.float().to(dev)
            Xa = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
            XtX += Xa.T @ Xa
            XtY += Xa.T @ Y
            for q in range(a.probes):
                v = torch.randn_like(img)
                g, = torch.autograd.grad(img, z, v, retain_graph=q < a.probes - 1)
                Gall = g.permute(0, 2, 3, 1).reshape(-1, C)                                           # [h*w, C]
                if T <= 1:
                    G = Gall[sel.to(dev)]
                    U = (X1[:, :, None] * G[:, None, :]).reshape(len(sel), D1)                        # u = x (x) g
                    t = (G * Ys).sum(1)                                                               # g . z_B
                    UtU += (U.T @ U).cpu().double()
                    Utt += (U.T @ t).cpu().double()
                    continue
                for r0 in range(0, h, T):                                                             # one row of tiles at a time
                    rows = slice(r0 * w, min(r0 + T, h) * w)
                    G, Xr, Yr = Gall[rows], X1[rows], Ys[rows]
                    nr = G.shape[0] // w
                    U = (Xr[:, :, None] * G[:, None, :]).reshape(nr, w // T, T, D1).sum((0, 2))        # [w/T, D1]
                    t = (G * Yr).sum(1).reshape(nr, w // T, T).sum((0, 2))
                    UtU += (U.T @ U).cpu().double()
                    Utt += (U.T @ t).cpu().double()
        elif a.method in ("ot", "otpilot"):
            # unpaired: source statistics from even batches, target statistics from odd ones
            (mom_y if bi % 2 == 0 else mom_b).add(X if bi % 2 == 0 else Y)
            if a.method == "otpilot" and n_img < a.pilot_images:
                m = min(a.pilot_images - n_img, images.shape[0]) * (len(X) // images.shape[0])
                pil_y.append(X[:m]); pil_b.append(Y[:m])
        else:   # parseval: keep a random sample of pilot patches to draw anchors from
            sel = torch.randperm(len(X))[: max(4 * n_anchor // max(1, len(loader)), 64)]
            pool_in.append(X[sel]); pool_out.append(Y[sel])
        n_img += images.shape[0]
        if n_img % 40 == 0:
            print(f"  [*] {n_img} images, {time.time()-t0:.0f}s", flush=True)
        if a.images and n_img >= a.images:
            break
    print(f"[*] {n_img} images, {st} -> {tt}, method={a.method}, k={k}, d={d}, {time.time()-t0:.0f}s")

    info = {}
    if a.method == "wiener":
        lam = a.ridge * XtX.diagonal()[:d].mean()
        reg = lam * torch.eye(d + 1, dtype=XtX.dtype)
        reg[d, d] = 0
        B = torch.linalg.solve(XtX + reg, XtY)        # [d+1, C]
        Wm, bias = B[:d].T, B[d]
        info = dict(ridge=a.ridge)
    elif a.method == "jwiener":
        # alpha mixes the (trace-normalized) image-space metric with the latent-space
        # Wiener objective: the decoder is nearly blind to the latent's scale
        # (|J z| << |J r|), so alpha = 0 is ill-posed along that direction.
        K = torch.kron(XtX, torch.eye(C, dtype=XtX.dtype))
        kv = XtY.reshape(-1)
        if a.save_stats:
            torch.save(dict(UtU=UtU, Utt=Utt, XtX=XtX, XtY=XtY, C=C, d=d), a.save_stats)
        A = (1 - a.alpha) * UtU / UtU.trace() + a.alpha * K / K.trace()
        rhs = (1 - a.alpha) * Utt / UtU.trace() + a.alpha * kv / K.trace()
        lam = a.ridge * A.diagonal().mean()
        theta = torch.linalg.solve(A + lam * torch.eye(D1, dtype=A.dtype), rhs)
        Bm = theta.reshape(d + 1, C)                   # [a, c] = W[c, a]
        Wm, bias = Bm[:d].T, Bm[d]
        info = dict(ridge=a.ridge, probes=a.probes, positions=a.positions, tile=a.tile, alpha=a.alpha)
    elif a.method in ("ot", "otpilot"):
        mu_y, S_y = mom_y.mean_cov()
        mu_b, S_b = mom_b.mean_cov()
        Sy_h, Sy_ih, Sb_h = sqrtm_sym(S_y), sqrtm_sym(S_y, inv=True), sqrtm_sym(S_b)
        if a.method == "ot":
            Wm = Sy_ih @ sqrtm_sym(Sy_h @ S_b @ Sy_h) @ Sy_ih
        else:
            Yw = (torch.cat(pil_y) - mu_y) @ Sy_ih     # whitened received pilots
            Bw = (torch.cat(pil_b) - mu_b) @ sqrtm_sym(S_b, inv=True)
            R = polar(Bw.T @ Yw)                       # orthogonal Procrustes: Bw ~ Yw R^T
            Wm = Sb_h @ R @ Sy_ih
            info = dict(pilot_images=a.pilot_images, pilot_vectors=int(len(Yw)))
        bias = mu_b - Wm @ mu_y
        info.update(unpaired_split="even/odd batches", src_vectors=mom_y.n, tgt_vectors=mom_b.n)
    else:
        Ain, Aout = torch.cat(pool_in), torch.cat(pool_out)
        idx = torch.randperm(len(Ain))[:n_anchor]
        Ft = polar(Ain[idx])                           # [n, d]
        Gt = polar(Aout[idx]).T                        # [C, n]
        Wm, bias = (Gt @ Ft), None                     # [C, d]
        info = dict(anchors=int(len(idx)))

    os.makedirs(a.out, exist_ok=True)
    torch.save(to_conv5(Wm, bias, C, k), os.path.join(a.out, "aligner.pth"))
    cfg = dict(kind="conv", objective="closed_form", method=a.method, kernel=k,
               src_type=st, tgt_type=tt, src_model=a.src_model, tgt_model=a.tgt_model,
               channel="awgn", snr_db=a.snr_db, images=n_img, tcn=C,
               applied_at="receiver", **info)
    json.dump(cfg, open(os.path.join(a.out, "aligner_config.json"), "w"), indent=2)
    print(f"[*] saved {a.out}")


if __name__ == "__main__":
    main()
