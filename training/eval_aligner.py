#!/usr/bin/env python3
"""Evaluate a semantic aligner on a held-out test set — playground-native.

Runs the real encode -> channel -> (aligner) -> decode pipeline for three conditions:

  misaligned : src encoder -> tgt decoder, NO aligner       (the broken cross pair)
  aligned    : src encoder -> tgt decoder, WITH aligner      (--aligner)
  matched    : tgt encoder -> tgt decoder                    (your ceiling)

``misaligned`` and ``aligned`` share the *same* received latent (one channel + drop
realization per image), so the aligner is the only difference between them; ``matched``
is scored at the same operating point with its own realization.

Metrics: PSNR + MS-SSIM + LPIPS (the same set as ``train.py`` validation). For each
metric we also report the Alignment Recovery Ratio (ARR) — how much of the
matched-vs-misaligned gap the aligner recovers:

    ARR = (aligned - misaligned) / (matched - misaligned)        (LPIPS: signs flipped)

Unlike the reference (clean-only), this is channel-aware: pass ``--channel none`` for a
clean alignment-only measurement, or ``--channel ofdm``/``awgn``/... to score over the
link (optionally sweeping ``--snr-list`` and ``--drop-prob``). Per-image and summary
metrics are written as CSV under ``--out-dir``.

Example:
    python training/eval_aligner.py \\
        --src-model marcellobullo/adjscc-cr6-awgn \\
        --tgt-model ckpts/djscc-convnext-cr6-ofdm-spatialcsi/best \\
        --aligner ckpts/aligner-adjscc2spatialcsi \\
        --test-dir /data/kodak --channel ofdm --snr-list 0 10 20 --out-dir eval_out
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc import load_aligner  # noqa: E402
from jscc.djscc_spatialcsi.nn import packetwise_to_element_map  # noqa: E402
from training.train import get_device, make_channel, build_val_metrics  # noqa: E402
from training.train_aligner import encode_src, decode_tgt, freeze  # noqa: E402

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.webp")
HIGHER_BETTER = {"psnr": True, "msssim": True, "lpips": False}


def read_image(path, H, W, device):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC)
    t = torch.from_numpy(np.array(img)).float().div(255.0)   # np.array -> writable
    return t.permute(2, 0, 1).unsqueeze(0).to(device)        # [1, 3, H, W]


@torch.no_grad()
def channel_pass(z, args, snr, drop_prob):
    """One channel (+ drop) realization on a latent: returns the received latent and
    the side-info the decoder needs (csi, drop_mask, n_pkts)."""
    B, dev = z.shape[0], z.device
    ch = make_channel(args.channel, snr, args.coherence_time, args.k_factor,
                      args.ofdm_taps, args.ofdm_decay, args.ofdm_subcarriers,
                      args.ofdm_eq, args.packet_len)
    if ch is not None:
        z, csi = ch(z)
    else:
        csi = torch.full((B, 1), float(snr), device=dev)
    tcn, h, w = z.shape[1], z.shape[2], z.shape[3]
    n_pkts = math.ceil((tcn * h * w // 2) / args.packet_len)
    drop_mask = None
    if drop_prob > 0:
        drop_mask = torch.rand(B, n_pkts, device=dev) < drop_prob
        dmap = packetwise_to_element_map(
            drop_mask.float(), M=tcn, H=h, W=w, pkt_len_complex=args.packet_len)
        z = z * (1.0 - dmap)
    return z, csi, drop_mask, n_pkts


@torch.no_grad()
def reconstruct(src, src_type, tgt, tgt_type, aligner, image, args, snr):
    """All conditions for one image. misaligned/aligned share one received latent."""
    out = {}
    # src encode -> ONE channel realization -> decode with/without aligner.
    z = encode_src(src, src_type, image, snr)
    z_rx, csi, dm, n_pkts = channel_pass(z, args, snr, args.drop_prob)
    out["misaligned"] = decode_tgt(tgt, tgt_type, z_rx, csi, dm, n_pkts,
                                   args.packet_len, args.sentinel_db).clamp(0, 1)
    z_al = aligner.apply_rx(z_rx)
    out["aligned"] = decode_tgt(tgt, tgt_type, z_al, csi, dm, n_pkts,
                                args.packet_len, args.sentinel_db).clamp(0, 1)
    if not args.skip_matched:
        z2 = encode_src(tgt, tgt_type, image, snr)
        z2_rx, csi2, dm2, n2 = channel_pass(z2, args, snr, args.drop_prob)
        out["matched"] = decode_tgt(tgt, tgt_type, z2_rx, csi2, dm2, n2,
                                    args.packet_len, args.sentinel_db).clamp(0, 1)
    return out


@torch.no_grad()
def score(out, ref, metrics):
    mse = F.mse_loss(out, ref).item()
    res = {"psnr": 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)}
    for k, m in metrics.items():
        try:
            res[k] = float(m(out, ref).mean())
        except Exception:
            pass
    return res


def arr(mis, aln, mat, higher_better):
    denom = (mat - mis) if higher_better else (mis - mat)
    num = (aln - mis) if higher_better else (mis - aln)
    return float("nan") if abs(denom) < 1e-9 else num / denom


def save_png(t, path):
    arr8 = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-model", required=True, help="TX-encoder model (HF id | folder)")
    p.add_argument("--tgt-model", required=True, help="RX-decoder model (HF id | folder)")
    p.add_argument("--aligner", required=True, help="aligner .pth | folder | HF id")
    p.add_argument("--test-dir", nargs="+", required=True, help="held-out test image folder(s)")
    p.add_argument("--channel", default="none",
                   choices=["none", "awgn", "rayleigh", "rician", "ofdm"],
                   help="none = clean (isolate alignment); else score over the link")
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--snr-list", type=float, nargs="+", default=None,
                   help="evaluate at several SNRs (overrides --snr-db)")
    p.add_argument("--drop-prob", type=float, default=0.0,
                   help="per-packet erasure prob (same realization across conditions/image)")
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--sentinel-db", type=float, default=-20.0)
    p.add_argument("--coherence-time", type=int, default=1)
    p.add_argument("--k-factor", type=float, default=4.0)
    p.add_argument("--ofdm-taps", type=int, default=8)
    p.add_argument("--ofdm-decay", type=float, default=4.0)
    p.add_argument("--ofdm-subcarriers", type=int, default=64)
    p.add_argument("--ofdm-eq", default="zf", choices=["zf", "mmse"])
    p.add_argument("--skip-matched", action="store_true", help="skip the matched ceiling")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--max-images", type=int, default=0, help="cap on test images (0 = all)")
    p.add_argument("--out-dir", default="aligner_eval", help="output dir for CSVs/recons")
    p.add_argument("--no-save-images", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible channel/drops)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoModel
    src = freeze(AutoModel.from_pretrained(args.src_model, trust_remote_code=True).to(device))
    tgt = freeze(AutoModel.from_pretrained(args.tgt_model, trust_remote_code=True).to(device))
    src_type, tgt_type = src.config.model_type, tgt.config.model_type
    if src.config.M != tgt.config.M or \
       (src.config.img_height, src.config.img_width) != (tgt.config.img_height, tgt.config.img_width):
        raise SystemExit("src/tgt latent geometry (M, resolution) must match")

    tcn = tgt.config.M
    H, W = tgt.config.img_height, tgt.config.img_width
    h, w = tgt.config.latent_hw
    args.sentinel_db = getattr(tgt.config, "sentinel_drop_db", args.sentinel_db)

    aligner = load_aligner(args.aligner, tcn, h, w, device=device)
    if aligner.mode == "compressing":
        raise SystemExit(
            f"aligner kind={aligner.kind} is 'compressing' (zero-shot): it changes the "
            "transmitted dimension and is not supported by this eval; evaluate residual "
            "aligners (conv/twoconv/linear/mlp) or measure zero-shot via the socket path.")
    metrics = build_val_metrics(device, args.lpips_net) if not args.no_lpips \
        else {k: v for k, v in build_val_metrics(device, args.lpips_net).items() if k != "lpips"}
    metric_names = ["psnr"] + list(metrics.keys())

    paths = sorted(f for d in args.test_dir for ext in IMG_EXTS
                   for f in glob.glob(os.path.join(d, ext)))
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"no images under {args.test_dir}")

    snr_list = args.snr_list if args.snr_list else [args.snr_db]
    conditions = ["misaligned", "aligned"] + ([] if args.skip_matched else ["matched"])
    print(f"[*] {src_type} -> {tgt_type} | aligner={aligner.kind} | channel={args.channel} "
          f"drop={args.drop_prob} | SNRs={snr_list} | {len(paths)} images | metrics={metric_names}")

    if not args.no_save_images:
        for c in conditions:
            os.makedirs(os.path.join(args.out_dir, c), exist_ok=True)

    # agg[snr][cond][metric] = [values]
    agg = {s: {c: {m: [] for m in metric_names} for c in conditions} for s in snr_list}
    per_image = []

    for snr in snr_list:
        for idx, path in enumerate(paths):
            torch.manual_seed(args.seed + idx)      # same channel/drops across conditions
            ref = read_image(path, H, W, device)
            recon = reconstruct(src, src_type, tgt, tgt_type, aligner, ref, args, snr)
            row = {"image": os.path.basename(path), "snr_db": snr}
            for c in conditions:
                vals = score(recon[c], ref, metrics)
                for k, v in vals.items():
                    agg[snr][c][k].append(v)
                    row[f"{k}_{c}"] = v
            per_image.append(row)
            if not args.no_save_images and snr == snr_list[0]:
                stem = os.path.splitext(os.path.basename(path))[0] + ".png"
                for c in conditions:
                    save_png(recon[c], os.path.join(args.out_dir, c, stem))
            if (idx + 1) % 20 == 0:
                print(f"  [snr {snr:g}] {idx+1}/{len(paths)} "
                      f"aligned PSNR {np.mean(agg[snr]['aligned']['psnr']):.2f} dB")

    write_csvs(args, snr_list, conditions, metric_names, agg, per_image)
    print_summary(snr_list, conditions, metric_names, agg)
    return 0


def write_csvs(args, snr_list, conditions, metric_names, agg, per_image):
    per_csv = os.path.join(args.out_dir, "metrics_per_image.csv")
    cols = ["image", "snr_db"] + [f"{m}_{c}" for c in conditions for m in metric_names]
    with open(per_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(per_image)

    sum_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    sum_cols = (["snr_db", "metric"]
                + [f"{c}_{s}" for c in conditions for s in ("mean", "std")] + ["ARR"])
    rows = []
    for snr in snr_list:
        for m in metric_names:
            r = {"snr_db": snr, "metric": m}
            means = {}
            for c in conditions:
                v = np.array(agg[snr][c][m], dtype=np.float64)
                r[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
                r[f"{c}_std"] = float(v.std()) if v.size else float("nan")
                means[c] = r[f"{c}_mean"]
            r["ARR"] = (arr(means["misaligned"], means["aligned"], means["matched"],
                            HIGHER_BETTER[m]) if "matched" in conditions else float("nan"))
            rows.append(r)
    with open(sum_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sum_cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[*] per-image CSV -> {per_csv}")
    print(f"[*] summary  CSV -> {sum_csv}")


def print_summary(snr_list, conditions, metric_names, agg):
    print("\n=== mean metrics ===")
    for snr in snr_list:
        print(f"-- SNR {snr:g} dB --")
        for m in metric_names:
            cells = "  ".join(f"{c}={np.mean(agg[snr][c][m]):.3f}" for c in conditions)
            extra = ""
            if "matched" in conditions:
                a = arr(np.mean(agg[snr]["misaligned"][m]), np.mean(agg[snr]["aligned"][m]),
                        np.mean(agg[snr]["matched"][m]), HIGHER_BETTER[m])
                extra = f"   ARR={a*100:.1f}%"
            print(f"  {m:7s}: {cells}{extra}")


if __name__ == "__main__":
    sys.exit(main())
