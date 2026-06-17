#!/usr/bin/env python3
"""Single training script for any djscc-playground model.

Loads the model straight from the HuggingFace hub (or a local HF folder) with
``AutoModel.from_pretrained`` — the loaded object is a Kaira-native model, so we
drive the proven recipe directly: encoder -> power constraint -> Kaira channel
(+ optional packet drop) -> decoder, MSE loss, random or fixed SNR. Because the
model is a ``PreTrainedModel``, the result is saved with ``save_pretrained`` (and
optionally pushed) — no separate export step.

The per-family conditioning is dispatched on ``config.model_type``:
  * djscc            : channel-blind encoder; decoder takes a scalar SNR.
  * djscc_spatialcsi : decoder takes a per-element CSI map (drops -> sentinel SNR).
  * adjscc           : SNR-adaptive encoder AND decoder (attention).

Examples:
  # fine-tune a published model over SNR ~ U[0,20] with 10% packet drops
  python training/train.py --model marcellobullo/djscc-convnext-cr6-awgn \\
      --train-data-dir /data/DIV2K_train_HR --val-data-dir /data/DIV2K_valid_HR \\
      --channel awgn --snr-min 0 --snr-max 20 --drop-prob 0.10 \\
      --epochs 50 --batch-size 4 --out ckpts/my-run

  # train from scratch (reinit weights) at a fixed SNR
  python training/train.py --model marcellobullo/adjscc-cr6-awgn --reinit \\
      --train-data-dir /data/train --channel awgn --snr-db 10 --epochs 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc.data import make_loader  # noqa: E402
from jscc.adjscc.nn import powerConstraint  # noqa: E402
from jscc.djscc_spatialcsi.nn import packetwise_to_element_map  # noqa: E402


def get_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def make_channel(name: str, snr_db: float, coherence_time: int = 1,
                 k_factor: float = 4.0):
    """Build a Kaira channel for the given (per-batch) SNR. None = no channel."""
    name = name.lower()
    if name in ("none", "identity"):
        return None
    from kaira.channels import AWGNChannel, FlatFadingChannel
    if name == "awgn":
        return AWGNChannel(snr_db=snr_db)
    if name in ("rayleigh", "rician"):
        kw = {"k_factor": k_factor} if name == "rician" else {}
        return FlatFadingChannel(fading_type=name, coherence_time=coherence_time,
                                 snr_db=snr_db, **kw)
    raise ValueError(f"unknown --channel {name!r} (awgn|rayleigh|rician|none)")


def forward_train(model, model_type, images, snr, channel, drop_prob,
                  packet_len, sentinel_db):
    """Encoder -> constraint -> channel (+drop) -> decoder, per-family. Returns
    the reconstruction. Mirrors the reference recipe, model-agnostic via dispatch."""
    B = images.shape[0]
    dev = images.device
    snr_t = torch.full((B, 1), float(snr), dtype=torch.float32, device=dev)

    # --- encode + power constraint -----------------------------------------
    if model_type == "adjscc":
        z = model.encoder(images, snr_t)
        z = powerConstraint(z.flatten(), P=model.power).reshape(z.shape)
    else:
        z = model.encoder(images)
        z = model.constraint(z)

    # --- channel noise ------------------------------------------------------
    if channel is not None:
        z = channel(z)

    # --- packet drops (symbol-accurate, shared across families) ------------
    tcn, h, w = z.shape[1], z.shape[2], z.shape[3]
    n_pkts = math.ceil((tcn * h * w // 2) / packet_len)
    drop_mask = None
    if drop_prob > 0:
        drop_mask = torch.rand(B, n_pkts, device=dev) < drop_prob
        drop_map = packetwise_to_element_map(
            drop_mask.float(), M=tcn, H=h, W=w, pkt_len_complex=packet_len)
        z = z * (1.0 - drop_map)

    # --- decode (per-family conditioning) ----------------------------------
    if model_type == "djscc_spatialcsi":
        per_pkt = torch.full((B, n_pkts), float(snr), device=dev)
        if drop_mask is not None:
            per_pkt = per_pkt.masked_fill(drop_mask, sentinel_db)
        csi_map = packetwise_to_element_map(
            per_pkt, M=tcn, H=h, W=w, pkt_len_complex=packet_len)
        return model.decoder(z, csi_map)
    return model.decoder(z, snr_t)


def sample_snr(args) -> float:
    if args.snr_min is not None and args.snr_max is not None:
        return float(np.random.uniform(args.snr_min, args.snr_max))
    return float(args.snr_db)


@torch.no_grad()
def validate(model, model_type, loader, device, args, snr_val):
    model.eval()
    psnrs = []
    for images, _ in loader:
        images = images.to(device)
        ch = make_channel(args.channel, snr_val, args.coherence_time, args.k_factor)
        out = forward_train(model, model_type, images, snr_val, ch,
                            0.0, args.packet_len, args.sentinel_db)
        mse = F.mse_loss(out, images, reduction="none").mean(dim=(1, 2, 3))
        psnrs.extend((10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu().numpy())
    return float(np.mean(psnrs)) if psnrs else 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local HF folder")
    p.add_argument("--reinit", action="store_true",
                   help="reinitialize weights (train from scratch, keep architecture)")
    p.add_argument("--train-data-dir", nargs="+", required=True)
    p.add_argument("--val-data-dir", nargs="+", default=None)

    # channel / SNR / drops
    p.add_argument("--channel", default="awgn",
                   choices=["awgn", "rayleigh", "rician", "none"])
    p.add_argument("--snr-db", type=float, default=10.0, help="fixed SNR (dB)")
    p.add_argument("--snr-min", type=float, default=None,
                   help="with --snr-max, sample SNR ~ U[min,max] per batch")
    p.add_argument("--snr-max", type=float, default=None)
    p.add_argument("--drop-prob", type=float, default=0.0)
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--sentinel-db", type=float, default=-20.0,
                   help="SNR assigned to dropped packets in the spatial-CSI map")
    p.add_argument("--coherence-time", type=int, default=1)
    p.add_argument("--k-factor", type=float, default=4.0)

    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="ckpts/run", help="output dir (best/ + last.pth)")
    p.add_argument("--resume", default=None, help="path to a last.pth to resume")
    p.add_argument("--push-to-hub", default=None, help="push best model to this repo id")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    if args.reinit:
        print("[*] reinitializing weights (training from scratch)")
        model = model.__class__(model.config)
    model_type = model.config.model_type
    H, W = model.config.img_height, model.config.img_width
    sentinel = getattr(model.config, "sentinel_drop_db", args.sentinel_db)
    args.sentinel_db = sentinel
    model.to(device)
    print(f"[*] model={model.__class__.__name__} type={model_type} {W}x{H} "
          f"device={device}")

    train_loader = make_loader(args.train_data_dir, H, W, args.batch_size,
                               train=True, num_workers=args.num_workers)
    val_loader = (make_loader(args.val_data_dir, H, W, args.batch_size,
                              train=False, num_workers=args.num_workers)
                  if args.val_data_dir else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()

    start_epoch, best_psnr = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["op"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("Ave_PSNR", 0.0)
        print(f"[*] resumed from epoch {start_epoch} (best PSNR {best_psnr:.2f})")

    snr_desc = (f"U[{args.snr_min},{args.snr_max}]"
                if args.snr_min is not None else f"{args.snr_db} dB")
    print(f"[*] channel={args.channel} snr={snr_desc} drop={args.drop_prob} "
          f"epochs={args.epochs} bs={args.batch_size}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            snr = sample_snr(args)
            channel = make_channel(args.channel, snr, args.coherence_time, args.k_factor)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = forward_train(model, model_type, images, snr, channel,
                                    args.drop_prob, args.packet_len, args.sentinel_db)
                loss = criterion(out, images) / args.accum_steps
            scaler.scale(loss).backward() if use_amp else loss.backward()
            if (i + 1) % args.accum_steps == 0:
                (scaler.step(optimizer), scaler.update()) if use_amp else optimizer.step()
                optimizer.zero_grad()
            running += loss.item() * args.accum_steps
            n += 1
            if (i + 1) % 50 == 0:
                avg = running / n
                print(f"  ep{epoch} [{i+1}/{len(train_loader)}] loss={avg:.6f} "
                      f"psnr~{-10*np.log10(avg+1e-10):.2f}dB")
        scheduler.step()

        val_snr = (0.5 * (args.snr_min + args.snr_max)
                   if args.snr_min is not None else args.snr_db)
        psnr = (validate(model, model_type, val_loader, device, args, val_snr)
                if val_loader else -10 * np.log10(running / max(n, 1) + 1e-10))
        print(f"[*] epoch {epoch} done in {time.time()-t0:.0f}s  "
              f"val_psnr@{val_snr:.0f}dB={psnr:.2f}")

        torch.save({"net": model.state_dict(), "op": optimizer.state_dict(),
                    "epoch": epoch, "Ave_PSNR": max(best_psnr, psnr)},
                   os.path.join(args.out, "last.pth"))
        if psnr >= best_psnr:
            best_psnr = psnr
            model.save_pretrained(os.path.join(args.out, "best"))
            print(f"  [*] new best ({psnr:.2f} dB) -> {args.out}/best (save_pretrained)")

    if args.push_to_hub:
        model.save_pretrained(os.path.join(args.out, "best"))
        model.push_to_hub(args.push_to_hub)
        print(f"[*] pushed best -> {args.push_to_hub}")
    print(f"[*] done. best val PSNR {best_psnr:.2f} dB. Deploy with "
          f"--model {args.out}/best")
    return 0


if __name__ == "__main__":
    sys.exit(main())
