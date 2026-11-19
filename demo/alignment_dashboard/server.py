#!/usr/bin/env python3
"""Live alignment dashboard: ADJSCC matched-SNR encoder -> channel -> one of four receivers.

Runs the same simulated pipeline as ``training/eval_aligner.py`` and serves a web UI to
switch the receiver live while recording:

  matched    : ADJSCC encoder -> ADJSCC decoder                    (source model on this link)
  misaligned : ADJSCC encoder -> ConvNeXt-AWGN decoder, no aligner (lower bound)
  aligned    : ADJSCC encoder -> aligner -> ConvNeXt-AWGN decoder  (the experiment)
  target     : ConvNeXt encoder -> ConvNeXt-AWGN decoder           (ceiling for aligned)

matched / misaligned / aligned share ONE received latent (same channel noise and lost
packets), so the receiver is the only thing that changes when you switch.

  python demo/alignment_dashboard/server.py --images data/kodak --port 8765
  # then open http://localhost:8765 (from a laptop: ssh -L 8765:localhost:8765 <gpu-host>)
"""

import argparse
import base64
import glob
import io
import math
import os
import sys
import threading
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageOps

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc import load_aligner  # noqa: E402
from jscc.djscc_spatialcsi.nn import packetwise_to_element_map  # noqa: E402
from training.train import get_device, make_channel  # noqa: E402
from training.train_aligner import decode_tgt, encode_src, freeze  # noqa: E402

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
MODES = ("matched", "misaligned", "aligned", "target")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-model", default="ckpts/adjscc-matched-snr-final/best")
    p.add_argument("--tgt-model", default="marcellobullo/djscc-convnext-cr6-awgn")
    p.add_argument("--aligner", default="ckpts/adjscc-matched-snr-final/aligner-convnext-awgn")
    p.add_argument("--images", default="data/kodak", help="folder of test images")
    p.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh", "rician", "ofdm"])
    p.add_argument("--packet-len", type=int, default=960)
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


# ---------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, args):
        from transformers import AutoModel
        self.args = args
        self.dev = get_device(args.device)
        os.chdir(_ROOT)  # model paths in the defaults are repo-relative
        self.src = freeze(AutoModel.from_pretrained(args.src_model, trust_remote_code=True).to(self.dev))
        self.tgt = freeze(AutoModel.from_pretrained(args.tgt_model, trust_remote_code=True).to(self.dev))
        self.src_type, self.tgt_type = self.src.config.model_type, self.tgt.config.model_type
        self.H, self.W = self.tgt.config.img_height, self.tgt.config.img_width
        self.tcn = self.tgt.config.M
        self.h, self.w = self.tgt.config.latent_hw
        self.sentinel = getattr(self.tgt.config, "sentinel_drop_db", -20.0)
        self.aligner = load_aligner(args.aligner, self.tcn, self.h, self.w, device=self.dev)
        self.n_pkts = math.ceil((self.tcn * self.h * self.w // 2) / args.packet_len)
        from kaira.metrics.image import SSIM
        self.ssim = SSIM(data_range=1.0).to(self.dev).eval()
        self.lock = threading.Lock()
        self.paths = sorted(f for ext in IMG_EXTS for f in glob.glob(os.path.join(args.images, ext)))
        self.uploads: dict[str, Image.Image] = {}
        print(f"[*] {len(self.paths)} images from {args.images} | device {self.dev} | "
              f"latent {self.tcn}x{self.h}x{self.w} ({self.n_pkts} packets) | aligner {self.aligner.kind}")

    # images -----------------------------------------------------------------
    def pil(self, image_id: str) -> Image.Image:
        if image_id in self.uploads:
            img = self.uploads[image_id]
        else:
            img = Image.open(self.paths[int(image_id)]).convert("RGB")
        if img.height > img.width:  # keep portrait Kodak images undistorted
            img = img.rotate(90, expand=True)
        return ImageOps.fit(img, (self.W, self.H), Image.BICUBIC)

    def tensor(self, image_id: str) -> torch.Tensor:
        a = np.array(self.pil(image_id), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.dev)

    # channel ----------------------------------------------------------------
    def channel(self, z, snr, drop_mask):
        a = self.args
        ch = make_channel(a.channel, snr, packet_len=a.packet_len)
        z, csi = ch(z)
        if drop_mask is not None:
            dmap = packetwise_to_element_map(drop_mask.float(), M=self.tcn, H=self.h, W=self.w,
                                             pkt_len_complex=a.packet_len)
            z = z * (1.0 - dmap)
        return z, csi

    def decode(self, model, mtype, z, csi, dm):
        return decode_tgt(model, mtype, z, csi, dm, self.n_pkts, self.args.packet_len,
                          self.sentinel).clamp(0, 1)

    def score(self, out, ref):
        mse = torch.mean((out - ref) ** 2).item()
        psnr = 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)
        return psnr, float(self.ssim(out, ref).mean())

    @torch.no_grad()
    def run(self, image_id: str, snr: float, drop: float, seed: int, with_target: bool = True):
        with self.lock:
            img = self.tensor(image_id)
            g = torch.Generator(device="cpu").manual_seed(seed)
            dm = (torch.rand(1, self.n_pkts, generator=g) < drop).to(self.dev) if drop > 0 else None

            torch.manual_seed(seed)
            z = encode_src(self.src, self.src_type, img, snr)
            z_rx, csi = self.channel(z, snr, dm)
            z_al = self.aligner.apply_rx(z_rx)
            rec = {
                "matched": self.decode(self.src, self.src_type, z_rx, csi, dm),
                "misaligned": self.decode(self.tgt, self.tgt_type, z_rx, csi, dm),
                "aligned": self.decode(self.tgt, self.tgt_type, z_al, csi, dm),
            }
            # Target: ConvNeXt's own encoder, same noise seed and same lost packets.
            torch.manual_seed(seed)
            z_t = encode_src(self.tgt, self.tgt_type, img, snr)
            z_trx, csi_t = self.channel(z_t, snr, dm)
            if with_target:
                rec["target"] = self.decode(self.tgt, self.tgt_type, z_trx, csi_t, dm)

            metrics = {k: dict(zip(("psnr", "ssim"), self.score(v, img))) for k, v in rec.items()}

            # How far each latent is from the language the ConvNeXt decoder speaks
            # (NMSE to ConvNeXt's clean latent of the same image).
            def nmse(a):
                return 10 * math.log10(float(((a - z_t) ** 2).sum() / (z_t ** 2).sum()))
            latent = {"rx_db": nmse(z_rx), "aligned_db": nmse(z_al), "target_rx_db": nmse(z_trx)}

            return {
                "images": {k: to_b64(v) for k, v in rec.items()},
                "original": to_b64(img),
                "metrics": metrics,
                "latent": latent,
                "latent_maps": {"rx": latent_map(z_rx), "aligned": latent_map(z_al),
                                "target": latent_map(z_t)},
                "iq": iq_points(z_rx, z, seed),
                "lost": int(dm.sum()) if dm is not None else 0,
                "n_pkts": self.n_pkts,
            }

    @torch.no_grad()
    def sweep(self, image_id: str, snrs, drop: float, seed: int):
        """PSNR vs SNR for the four receivers on one image (batched over SNR)."""
        with self.lock:
            img = self.tensor(image_id)
            curves = {m: [] for m in MODES}
            for snr in snrs:
                g = torch.Generator(device="cpu").manual_seed(seed)
                dm = (torch.rand(1, self.n_pkts, generator=g) < drop).to(self.dev) if drop > 0 else None
                torch.manual_seed(seed)
                z_rx, csi = self.channel(encode_src(self.src, self.src_type, img, snr), snr, dm)
                torch.manual_seed(seed)
                z_trx, csi_t = self.channel(encode_src(self.tgt, self.tgt_type, img, snr), snr, dm)
                outs = {
                    "matched": self.decode(self.src, self.src_type, z_rx, csi, dm),
                    "misaligned": self.decode(self.tgt, self.tgt_type, z_rx, csi, dm),
                    "aligned": self.decode(self.tgt, self.tgt_type, self.aligner.apply_rx(z_rx), csi, dm),
                    "target": self.decode(self.tgt, self.tgt_type, z_trx, csi_t, dm),
                }
                for m, o in outs.items():
                    curves[m].append(round(self.score(o, img)[0], 2))
            return {"snr": list(snrs), "psnr": curves}


# ---------------------------------------------------------------------------- encoding helpers

def to_b64(t: torch.Tensor, fmt="JPEG") -> str:
    a = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return pil_b64(Image.fromarray(a), fmt)


def pil_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=93) if fmt == "JPEG" else img.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# Diverging blue <-> gray <-> red, for signed latent values.
_NEG, _MID, _POS = np.array([57, 135, 229]), np.array([40, 40, 38]), np.array([230, 103, 103])


def latent_map(z: torch.Tensor, cols=4, n=8) -> str:
    """First ``n`` latent channels as a tiled diverging heatmap."""
    a = z[0, :n].float().cpu().numpy()
    s = np.percentile(np.abs(a), 99) + 1e-6
    a = np.clip(a / s, -1, 1)[..., None]
    rgb = np.where(a < 0, _MID + (-a) * (_NEG - _MID), _MID + a * (_POS - _MID)).astype(np.uint8)
    c, h, w, _ = rgb.shape
    rows = math.ceil(c / cols)
    gap = 3
    mosaic = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 0, np.uint8)
    for i in range(c):
        r, k = divmod(i, cols)
        mosaic[r * (h + gap): r * (h + gap) + h, k * (w + gap): k * (w + gap) + w] = rgb[i]
    return pil_b64(Image.fromarray(mosaic), "PNG")


def iq_points(z_rx, z_tx, seed, n=1500):
    """A random subset of received complex symbols (and their clean TX positions)."""
    rx = z_rx.flatten().view(-1, 2).cpu().numpy()
    tx = z_tx.flatten().view(-1, 2).cpu().numpy()
    idx = np.random.default_rng(seed).choice(len(rx), size=min(n, len(rx)), replace=False)
    return {"rx": np.round(rx[idx], 3).tolist(), "tx": np.round(tx[idx], 3).tolist()}


# ---------------------------------------------------------------------------- web app

def build_app(engine: Engine):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, Response
    from pydantic import BaseModel

    app = FastAPI()

    class RunReq(BaseModel):
        image: str
        snr: float = 20.0
        drop: float = 0.0
        seed: int = 0

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_HERE, "index.html"))

    @app.get("/api/info")
    def info():
        a = engine.args
        return {
            "images": [{"id": str(i), "name": os.path.splitext(os.path.basename(p))[0]}
                       for i, p in enumerate(engine.paths)],
            "src": a.src_model, "tgt": a.tgt_model, "aligner": a.aligner,
            "aligner_kind": engine.aligner.kind, "channel": a.channel,
            "latent": [engine.tcn, engine.h, engine.w], "n_pkts": engine.n_pkts,
            "device": str(engine.dev), "res": [engine.W, engine.H],
        }

    @lru_cache(maxsize=512)
    def _thumb(image_id: str) -> bytes:
        img = engine.pil(image_id).resize((192, 128), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    @app.get("/api/thumb/{image_id}")
    def thumb(image_id: str):
        try:
            return Response(_thumb(image_id), media_type="image/jpeg")
        except (IndexError, ValueError, KeyError):
            raise HTTPException(404)

    @app.post("/api/run")
    def run(req: RunReq):
        return engine.run(req.image, req.snr, req.drop, req.seed)

    @app.post("/api/sweep")
    def sweep(req: RunReq):
        return engine.sweep(req.image, list(range(0, 31, 2)), req.drop, req.seed)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), slot: str = ""):
        # ``slot`` overwrites a fixed id (the live camera); otherwise each upload is new.
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_id = f"u{slot}" if slot else f"u{len(engine.uploads)}"
        engine.uploads[image_id] = img
        return {"id": image_id, "name": file.filename or image_id}

    return app


def main():
    args = parse_args()
    args.images = os.path.abspath(args.images) if os.path.isabs(args.images) or os.path.isdir(args.images) \
        else os.path.join(_ROOT, args.images)
    engine = Engine(args)
    if engine.paths:
        engine.run("0", 20.0, 0.0, 0)  # warm-up (cuDNN autotune, first-call latency)
    import uvicorn
    print(f"[*] open http://{args.host}:{args.port}")
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
