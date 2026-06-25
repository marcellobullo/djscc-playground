#!/usr/bin/env python3
"""Train a semantic aligner that bridges a TX encoder and an RX decoder from two
independently-trained models (any family x any family, same latent geometry).

The aligner is the shape-preserving ``conv`` family from ``jscc.aligners`` (a
``Conv2d(tcn, tcn, 5)`` applied to the *received* latent at the RX, exactly as the
deployed codec applies it). Both models are frozen; only the aligner is trained:

    x_hat = tgt.decoder( aligner( channel(+drops)( constraint(src.encoder(x)) ) ), csi )
    minimize  loss(x_hat, x)

Two training objectives (``--objective``):

  * ``end2end`` (default) — the above: image-space loss back-propagated through the
    frozen target decoder. Directly optimizes deployed reconstruction; supports any
    loss (incl. perceptual) and the full OFDM + spatial-CSI path.
  * ``latent`` — decoder-free latent matching (the reference recipe). Caches paired
    latents ``z_in = src.encoder(x)`` and ``z_out = tgt.encoder(x)`` (the target
    decoder's *matching* encoder), then fits the aligner so ``aligner(channel(z_in))
    ≈ z_out``. This unlocks the training-free **closed-form** producers the end2end
    mode cannot make: ``--kind linear`` (Wiener) and ``--kind zeroshot`` (SVD).

This reuses ``training/train.py``'s channel / loss / SNR / drop machinery, so OFDM,
mse_lpips, --snr-min/--snr-max and --drop-min/--drop-max all work the same way.

The result is saved as a folder (``aligner.pth`` + ``aligner_config.json``) that
``jscc.load_aligner`` reads directly:

    python receiver/socket_rx.py --model <tgt-model> --aligner <out-folder>

Example (bridge an ADJSCC TX encoder to your ConvNeXt spatial-CSI RX decoder over
the OFDM channel):
    python training/train_aligner.py \\
        --src-model marcellobullo/adjscc-cr6-awgn \\
        --tgt-model ckpts/djscc-convnext-cr6-ofdm-spatialcsi/best \\
        --kind conv --channel ofdm --snr-min 0 --snr-max 20 --drop-min 0 --drop-max 0.3 \\
        --loss mse_lpips --train-data-dir /data/DIV2K_train_HR \\
        --val-data-dir /data/DIV2K_valid_HR --epochs 30 --out ckpts/aligner-adjscc2spatialcsi
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc.data import make_loader  # noqa: E402
from jscc.channels import reconcile_csi  # noqa: E402
from jscc.adjscc.nn import powerConstraint  # noqa: E402
from jscc.djscc_spatialcsi.nn import packetwise_to_element_map  # noqa: E402
from jscc.aligners.aligner import _build_by_kind  # noqa: E402
from jscc.aligners import modules as _m  # noqa: E402
# Reuse the proven recipe helpers so this trainer can't drift from train.py.
from training.train import (  # noqa: E402
    get_device, make_channel, build_loss, sample_snr, sample_drop)


def encode_src(src_model, src_type, images, snr):
    """Encoder + power constraint, dispatched on the *source* family (== train.py)."""
    B = images.shape[0]
    snr_t = torch.full((B, 1), float(snr), dtype=torch.float32, device=images.device)
    if src_type == "adjscc":
        z = src_model.encoder(images, snr_t)
        return powerConstraint(z.flatten(), P=src_model.power).reshape(z.shape)
    z = src_model.encoder(images)
    return src_model.constraint(z)


def decode_tgt(tgt_model, tgt_type, z, csi, drop_mask, n_pkts, packet_len, sentinel_db):
    """Decoder, dispatched on the *target* family (== train.py's decode half)."""
    tcn, h, w = z.shape[1], z.shape[2], z.shape[3]
    if tgt_type == "djscc_spatialcsi":
        per_pkt = reconcile_csi(csi, "vector", n_pkts)
        if drop_mask is not None:
            per_pkt = per_pkt.masked_fill(drop_mask, sentinel_db)
        csi_map = packetwise_to_element_map(
            per_pkt, M=tcn, H=h, W=w, pkt_len_complex=packet_len)
        return tgt_model.decoder(z, csi_map)
    return tgt_model.decoder(z, reconcile_csi(csi, "scalar", n_pkts))


def aligner_forward(src_model, src_type, tgt_model, tgt_type, aligner, images, snr,
                    channel, drop_prob, packet_len, sentinel_db):
    """src encode + channel (+drops) under no_grad (frozen), then the trainable
    aligner, then the frozen tgt decoder (grad flows through it to the aligner)."""
    B = images.shape[0]
    dev = images.device
    with torch.no_grad():
        z = encode_src(src_model, src_type, images, snr)
        if channel is not None:
            z, csi = channel(z)
        else:
            csi = torch.full((B, 1), float(snr), device=dev)
        tcn, h, w = z.shape[1], z.shape[2], z.shape[3]
        n_pkts = math.ceil((tcn * h * w // 2) / packet_len)
        drop_mask = None
        if drop_prob > 0:
            drop_mask = torch.rand(B, n_pkts, device=dev) < drop_prob
            drop_map = packetwise_to_element_map(
                drop_mask.float(), M=tcn, H=h, W=w, pkt_len_complex=packet_len)
            z = z * (1.0 - drop_map)
    z = aligner(z.detach())                     # only this carries gradients
    return decode_tgt(tgt_model, tgt_type, z, csi, drop_mask, n_pkts,
                      packet_len, sentinel_db)


@torch.no_grad()
def validate(src_model, src_type, tgt_model, tgt_type, aligner, loader, device,
             args, snr_val):
    aligner.eval()
    psnrs = []
    for images, _ in loader:
        images = images.to(device)
        ch = make_channel(args.channel, snr_val, args.coherence_time, args.k_factor,
                          args.ofdm_taps, args.ofdm_decay, args.ofdm_subcarriers,
                          args.ofdm_eq, args.packet_len)
        out = aligner_forward(src_model, src_type, tgt_model, tgt_type, aligner,
                              images, snr_val, ch, 0.0, args.packet_len,
                              args.sentinel_db).clamp(0.0, 1.0)
        mse = F.mse_loss(out, images, reduction="none").mean(dim=(1, 2, 3))
        psnrs.extend((10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu().numpy())
    return float(np.mean(psnrs)) if psnrs else 0.0


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── latent-matching objective (decoder-free) ─────────────────────────────────
# Cache paired clean latents once (src encoder -> tgt matching encoder), then fit
# the aligner to map src-space -> tgt-space. Mirrors the reference recipe, adapted
# to HF AutoModels and the playground channels.

@torch.no_grad()
def build_paired_latents(src, src_type, tgt, tgt_type, loader, device, args, max_cache):
    """[N, tcn, h, w] x2: z_in from the src encoder, z_out from the tgt's matching
    encoder. One SNR per batch conditions BOTH (matters for adjscc; ignored by the
    channel-blind encoders)."""
    z_in, z_out, n = [], [], 0
    for images, _ in loader:
        images = images.to(device)
        snr = sample_snr(args)
        z_in.append(encode_src(src, src_type, images, snr).cpu())
        z_out.append(encode_src(tgt, tgt_type, images, snr).cpu())
        n += images.shape[0]
        if n % 40 < args.batch_size:
            print(f"  [*] cached {n} latents")
        if max_cache and n >= max_cache:
            break
    return torch.cat(z_in, dim=0), torch.cat(z_out, dim=0)


def fit_linear(Z_in, Z_out, snr_db, channel_type):
    """Closed-form regularized Wiener solution (ported). NB: allocates an m x m
    matrix (m = tcn*h*w), so it only fits at small latent sizes."""
    X = Z_in.flatten(start_dim=1).H               # [m, N]
    Y = Z_out.flatten(start_dim=1).H
    sigma2 = 1.0 / (10 ** (snr_db / 10))
    noise_cov = sigma2 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    reg = (1000 * 10 ** (-snr_db / 30)) if channel_type == "AWGN" else 1000
    reg_matrix = reg * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    Fm = Y @ X.H @ torch.linalg.inv(X @ X.H + noise_cov + reg_matrix)
    return _m._LinearAlignment(align_matrix=Fm.T)


def fit_zeroshot(Z_in, Z_out, snr_db, channel_usage, device):
    """Training-free SVD/Parseval + whitening aligner (ported). Compresses to
    ``channel_usage`` dims at the TX and expands at the RX."""
    inp = Z_in.flatten(start_dim=1).to(device)    # [N, m]
    out = Z_out.flatten(start_dim=1).to(device)
    idx = torch.randperm(inp.size(0), device=device)[:channel_usage]
    U, _, Vt = torch.linalg.svd(inp[idx], full_matrices=False)
    F_tilde = (U @ Vt).to(device)                 # [cu, m]
    U, _, Vt = torch.linalg.svd(out[idx], full_matrices=False)
    G_tilde = (U @ Vt).H.to(device)               # [m, cu]

    proj = F_tilde @ inp.T                         # [cu, N]
    C = torch.cov(proj)
    try:
        L = torch.linalg.cholesky(C)
    except RuntimeError as e:
        if "must have at least 2 dimensions" in str(e):
            L = torch.sqrt(C).unsqueeze(0).unsqueeze(1)
        else:
            L = None
            for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
                try:
                    L = torch.linalg.cholesky(
                        C + eps * torch.eye(C.shape[0], device=device, dtype=C.dtype))
                    break
                except RuntimeError:
                    continue
            if L is None:
                raise RuntimeError("Cholesky failed even after regularization.")
    mean = proj.mean(axis=1, keepdim=True)         # [cu, 1]
    if snr_db is not None:
        reg = 1.0 / (10 ** (snr_db / 10))
        G = torch.linalg.inv(torch.Tensor([1 + reg]).unsqueeze(0))
    else:
        G = torch.Tensor([1]).unsqueeze(0)
    return _m._ZeroShotAlignment(F_tilde.cpu(), G_tilde.cpu(), G.cpu(),
                                 L.cpu(), mean.cpu())


def _inject(z, args, snr, drop, packet_len):
    """Apply the playground channel (+ packet drops) to a cached latent batch."""
    ch = make_channel(args.channel, snr, args.coherence_time, args.k_factor,
                      args.ofdm_taps, args.ofdm_decay, args.ofdm_subcarriers,
                      args.ofdm_eq, packet_len)
    y = ch(z)[0] if ch is not None else z
    if drop > 0:
        B, tcn, h, w = y.shape
        n_pkts = math.ceil((tcn * h * w // 2) / packet_len)
        dm = torch.rand(B, n_pkts, device=y.device) < drop
        dmap = packetwise_to_element_map(
            dm.float(), M=tcn, H=h, W=w, pkt_len_complex=packet_len)
        y = y * (1.0 - dmap)
    return y


def fit_neural_latent(aligner, Z_in, Z_out, args, device, rep_snr):
    """Adam fit on cached latents with early stopping: minimize MSE(aligner(noisy
    z_in), clean z_out). Channel noise (+drops) is re-sampled per batch; validation
    uses the representative SNR and no drops for stable stopping."""
    N = Z_in.shape[0]
    use_val = N >= 10
    vs = max(1, int(0.1 * N)) if use_val else 0
    tr_in, tr_out = Z_in[:N - vs], Z_out[:N - vs]
    va_in, va_out = (Z_in[N - vs:], Z_out[N - vs:]) if use_val else (Z_in, Z_out)

    aligner = aligner.to(device)
    crit = nn.MSELoss()
    optimizer = optim.Adam(aligner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state, stale, epoch = float("inf"), None, 0, 0
    while True:
        aligner.train()
        perm = torch.randperm(tr_in.shape[0])
        for s in range(0, tr_in.shape[0], args.batch_size):
            sel = perm[s:s + args.batch_size]
            inp = _inject(tr_in[sel].to(device), args, sample_snr(args),
                          sample_drop(args), args.packet_len)
            optimizer.zero_grad()
            loss = crit(aligner(inp), tr_out[sel].to(device))
            loss.backward()
            optimizer.step()
        aligner.eval()
        with torch.no_grad():
            inp = _inject(va_in.to(device), args, rep_snr, 0.0, args.packet_len)
            cur = crit(aligner(inp), va_out.to(device)).item()
        epoch += 1
        if best - cur > 1e-5:
            best, best_state, stale = cur, copy.deepcopy(aligner.state_dict()), 0
        else:
            stale += 1
        if epoch % 25 == 0 or stale >= args.patience:
            print(f"  [*] epoch {epoch}: val_mse={cur:.6f} best={best:.6f} stale={stale}")
        if stale >= args.patience or epoch > 10000:
            break
    if best_state is not None:
        aligner.load_state_dict(best_state)
    print(f"  [*] latent fit done: {epoch} epochs, best val_mse {best:.6f}")
    return aligner.cpu()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-model", required=True, help="TX-encoder model (HF id | folder)")
    p.add_argument("--tgt-model", required=True, help="RX-decoder model (HF id | folder)")
    p.add_argument("--objective", default="end2end", choices=["end2end", "latent"],
                   help="end2end = image loss through frozen tgt decoder (default); "
                        "latent = decoder-free latent matching (enables closed-form)")
    p.add_argument("--kind", default="conv",
                   choices=["conv", "twoconv", "linear", "mlp", "zeroshot"],
                   help="aligner family (conv = deployed default; zeroshot/closed-form "
                        "linear require --objective latent)")
    p.add_argument("--train-data-dir", nargs="+", required=True)
    p.add_argument("--val-data-dir", nargs="+", default=None)
    # latent-objective only
    p.add_argument("--channel-usage", type=int, default=0,
                   help="[latent/zeroshot] transmitted dim (0 = #cached latents)")
    p.add_argument("--max-cache", type=int, default=0,
                   help="[latent] cap on cached calibration images (0 = all)")
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="[latent neural] Adam weight decay")
    p.add_argument("--patience", type=int, default=10,
                   help="[latent neural] early-stop patience (epochs)")

    # channel / SNR / drops (identical surface to train.py)
    p.add_argument("--channel", default="awgn",
                   choices=["awgn", "rayleigh", "rician", "ofdm", "none"])
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--snr-min", type=float, default=None)
    p.add_argument("--snr-max", type=float, default=None)
    p.add_argument("--drop-prob", type=float, default=0.0,
                   help="fixed per-packet erasure probability")
    p.add_argument("--drop-min", type=float, default=None,
                   help="with --drop-max, sample drop prob ~ U[min,max] per batch")
    p.add_argument("--drop-max", type=float, default=None)
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--sentinel-db", type=float, default=-20.0)
    p.add_argument("--coherence-time", type=int, default=1)
    p.add_argument("--k-factor", type=float, default=4.0)
    p.add_argument("--ofdm-taps", type=int, default=8)
    p.add_argument("--ofdm-decay", type=float, default=4.0)
    p.add_argument("--ofdm-subcarriers", type=int, default=64)
    p.add_argument("--ofdm-eq", default="zf", choices=["zf", "mmse"])

    p.add_argument("--loss", default="mse",
                   choices=["mse", "l1", "msssim", "ssim", "lpips", "vgg", "mse_lpips"])
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="ckpts/aligner", help="output aligner folder")
    p.add_argument("--push-to-hub", default=None,
                   help="optional HF repo id (uploads the aligner folder)")
    return p.parse_args()


def save_aligner(state_dict, args, tcn, src_type, tgt_type, n_samples=None):
    """Write aligner.pth + aligner_config.json (the format jscc.load_aligner reads)."""
    torch.save(state_dict, os.path.join(args.out, "aligner.pth"))
    cfg = {"kind": args.kind, "objective": args.objective,
           "src_type": src_type, "tgt_type": tgt_type,
           "src_model": args.src_model, "tgt_model": args.tgt_model,
           "channel": args.channel, "tcn": tcn, "applied_at": "receiver"}
    if n_samples is not None:                       # zeroshot needs this to rebuild
        cfg["n_samples"] = int(n_samples)
    with open(os.path.join(args.out, "aligner_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def run_end2end(args, src, src_type, tgt, tgt_type, device, tcn, h, w, H, W):
    """Image-space loss back-propagated through the frozen target decoder."""
    aligner = _build_by_kind(args.kind, m=tcn * h * w, tcn=tcn, n_samples=None).to(device)
    n_params = sum(p.numel() for p in aligner.parameters())
    print(f"[*] objective=end2end kind={args.kind} params={n_params} tcn={tcn} {W}x{H}")

    train_loader = make_loader(args.train_data_dir, H, W, args.batch_size,
                               train=True, num_workers=args.num_workers)
    val_loader = (make_loader(args.val_data_dir, H, W, args.batch_size,
                              train=False, num_workers=args.num_workers)
                  if args.val_data_dir else None)

    optimizer = torch.optim.Adam(aligner.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = build_loss(args.loss, device, args.mse_weight, args.lpips_weight)

    snr_desc = (f"U[{args.snr_min},{args.snr_max}]"
                if args.snr_min is not None else f"{args.snr_db} dB")
    drop_desc = (f"U[{args.drop_min},{args.drop_max}]"
                 if args.drop_min is not None else f"{args.drop_prob}")
    print(f"[*] loss={args.loss} channel={args.channel} snr={snr_desc} drop={drop_desc} "
          f"epochs={args.epochs} bs={args.batch_size}")

    best_psnr = 0.0
    for epoch in range(args.epochs):
        aligner.train()
        running, n = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            snr = sample_snr(args)
            drop = sample_drop(args)
            channel = make_channel(args.channel, snr, args.coherence_time,
                                   args.k_factor, args.ofdm_taps, args.ofdm_decay,
                                   args.ofdm_subcarriers, args.ofdm_eq, args.packet_len)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = aligner_forward(src, src_type, tgt, tgt_type, aligner, images,
                                      snr, channel, drop, args.packet_len, args.sentinel_db)
                loss = criterion(out, images) / args.accum_steps
            scaler.scale(loss).backward() if use_amp else loss.backward()
            if (i + 1) % args.accum_steps == 0:
                (scaler.step(optimizer), scaler.update()) if use_amp else optimizer.step()
                optimizer.zero_grad()
            running += loss.item() * args.accum_steps
            n += 1
            if (i + 1) % 50 == 0:
                print(f"  ep{epoch} [{i+1}/{len(train_loader)}] loss={running/n:.6f}")
        scheduler.step()

        val_snr = (0.5 * (args.snr_min + args.snr_max)
                   if args.snr_min is not None else args.snr_db)
        if val_loader:
            psnr = validate(src, src_type, tgt, tgt_type, aligner, val_loader,
                            device, args, val_snr)
        else:
            psnr = -10 * np.log10(running / max(n, 1) + 1e-10)
        print(f"[*] epoch {epoch} done in {time.time()-t0:.0f}s  val@{val_snr:.0f}dB  psnr={psnr:.3f}")

        if psnr >= best_psnr:
            best_psnr = psnr
            save_aligner(aligner.state_dict(), args, tcn, src_type, tgt_type)
            print(f"  [*] new best ({psnr:.2f} dB) -> {args.out}")
    print(f"[*] done (end2end). best val PSNR {best_psnr:.2f} dB.")
    return best_psnr


def run_latent(args, src, src_type, tgt, tgt_type, device, tcn, h, w, H, W):
    """Decoder-free latent matching; enables closed-form linear / zeroshot."""
    loader = make_loader(args.train_data_dir, H, W, args.batch_size,
                         train=False, num_workers=args.num_workers)
    print(f"[*] objective=latent kind={args.kind} — caching paired latents "
          f"(src encoder -> tgt matching encoder) ...")
    Z_in, Z_out = build_paired_latents(src, src_type, tgt, tgt_type, loader,
                                       device, args, args.max_cache)
    print(f"[*] cached Z_in={tuple(Z_in.shape)} Z_out={tuple(Z_out.shape)}")

    rep_snr = (0.5 * (args.snr_min + args.snr_max)
               if args.snr_min is not None else args.snr_db)
    chan_type = "AWGN" if args.channel == "awgn" else "other"
    n_samples = None

    if args.kind == "linear":
        print(f"[*] closed-form Wiener linear @ {rep_snr:.1f} dB ...")
        aligner = fit_linear(Z_in, Z_out, rep_snr, chan_type)
    elif args.kind == "zeroshot":
        cu = min(args.channel_usage or Z_in.shape[0], Z_in.shape[0])
        n_samples = cu
        print(f"[*] zero-shot SVD aligner (channel_usage={cu}) @ {rep_snr:.1f} dB ...")
        aligner = fit_zeroshot(Z_in, Z_out, rep_snr, cu, device)
    else:                                            # conv / twoconv / mlp
        module = _build_by_kind(args.kind, m=tcn * h * w, tcn=tcn, n_samples=None)
        print(f"[*] Adam latent fit ({args.kind}) over channel={args.channel} "
              f"@ snr={rep_snr:.1f} dB ...")
        aligner = fit_neural_latent(module, Z_in, Z_out, args, device, rep_snr)

    save_aligner(aligner.state_dict(), args, tcn, src_type, tgt_type, n_samples)
    extra = f" (transmitted dim {n_samples} reals)" if n_samples is not None else ""
    print(f"[*] done (latent){extra}. Deploy with --aligner {args.out}")
    return 0.0


def main() -> int:
    args = parse_args()
    if args.kind == "zeroshot" and args.objective != "latent":
        raise SystemExit("--kind zeroshot requires --objective latent (training-free)")
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(args.src_model, trust_remote_code=True).to(device))
    tgt = freeze(AutoModel.from_pretrained(args.tgt_model, trust_remote_code=True).to(device))
    src_type, tgt_type = src.config.model_type, tgt.config.model_type

    # The aligner is a same-shape bridge, so latent geometry must match.
    if src.config.M != tgt.config.M:
        raise ValueError(f"latent channels differ: src M={src.config.M} tgt M={tgt.config.M}")
    if (src.config.img_height, src.config.img_width) != \
       (tgt.config.img_height, tgt.config.img_width):
        raise ValueError("src/tgt training resolution must match")

    tcn = tgt.config.M
    H, W = tgt.config.img_height, tgt.config.img_width
    h, w = tgt.config.latent_hw
    args.sentinel_db = getattr(tgt.config, "sentinel_drop_db", args.sentinel_db)
    print(f"[*] src={src_type}({args.src_model})  ->  tgt={tgt_type}({args.tgt_model})  device={device}")

    if args.objective == "latent":
        run_latent(args, src, src_type, tgt, tgt_type, device, tcn, h, w, H, W)
    else:
        run_end2end(args, src, src_type, tgt, tgt_type, device, tcn, h, w, H, W)

    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_to_hub, repo_type="model", exist_ok=True, private=True)
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push_to_hub,
                              repo_type="model")
        print(f"[*] pushed -> {args.push_to_hub} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
