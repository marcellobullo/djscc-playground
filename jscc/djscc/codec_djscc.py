"""DSP-glue adapters that turn a loaded DJSCCModel into BaseCodec halves.

This is the deployment layer: it owns everything Kaira/HF deliberately don't —
real/complex packing, the AveragePower constraint application, packet_len padding,
the CSI tensor, and the optional aligner hooks. The encode/decode bodies are a
faithful port of the djscc-demo ``custom_djscc/codec.py`` inference path.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from ..base import BaseDecoderCodec, BaseEncoderCodec, OutputKind
from .configuration_djscc import DJSCCConfig
from .modeling_djscc import DJSCCModel


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _load_split_state(model: DJSCCModel, ckpt_path: str) -> None:
    """Load a raw djscc-demo checkpoint (keys prefixed ``encoder.``/``decoder.``)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    print(f"[codec] loaded {loaded}/{len(state)} tensors from {ckpt_path} "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DJSCCEncoderCodec(BaseEncoderCodec):
    output_kind = OutputKind.COMPLEX_SYMBOLS
    needs_csi = False

    def __init__(self, model: DJSCCModel, config: DJSCCConfig, *,
                 device: str = "auto", packet_len: int = 960,
                 padding_zeros: int = 0, warmup: bool = False, **_) -> None:
        super().__init__()
        self.cfg = config
        self.device = _pick_device(device)
        self.tcn = config.M
        self.h, self.w = config.latent_hw
        self.packet_len = packet_len
        self.padding_zeros = padding_zeros

        self.encoder = model.encoder.to(self.device).eval()
        self.constraint = model.constraint

        if warmup and self.device.type in ("mps", "cuda"):
            dummy = torch.randn(1, 3, config.img_height, config.img_width,
                                device=self.device)
            with torch.no_grad():
                for _ in range(3):
                    self.encoder(dummy)

    @property
    def expected_complex_items(self) -> int:
        if self.aligner is not None and self.aligner.mode == "compressing":
            return math.ceil(self.aligner.transmitted_dim / 2)
        return (self.tcn * self.h * self.w) // 2

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        img = (
            torch.from_numpy(
                np.frombuffer(frame_bytes, dtype=np.uint8)
                .reshape((self.cfg.img_height, self.cfg.img_width, 3)).copy())
            .float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(self.device)
        )

        compressing = self.aligner is not None and self.aligner.mode == "compressing"
        with torch.no_grad():
            latent_flat = self.encoder(img).flatten()
            if compressing:
                latent_flat = self.aligner.compress(latent_flat.unsqueeze(0)).flatten()
            constrained = self.constraint(latent_flat.unsqueeze(0)).squeeze(0)

        chn = constrained.detach().cpu().numpy().astype(np.float32)
        if len(chn) % 2:
            chn = np.append(chn, 0.0)
        dim_z = len(chn) // 2
        symbols = (chn[:dim_z] + 1j * chn[dim_z:]).astype(np.complex64)

        if self.padding_zeros > 0 and not compressing:
            symbols = np.concatenate(
                [symbols, np.zeros(self.padding_zeros, dtype=np.complex64)])
        remainder = len(symbols) % self.packet_len
        if remainder:
            symbols = np.concatenate(
                [symbols, np.zeros(self.packet_len - remainder, dtype=np.complex64)])
        return symbols


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DJSCCDecoderCodec(BaseDecoderCodec):
    output_kind = OutputKind.COMPLEX_SYMBOLS

    def __init__(self, model: DJSCCModel, config: DJSCCConfig, *,
                 device: str = "auto", packet_len: int = 960,
                 snr_db: float = 19.0, warmup: bool = False, **_) -> None:
        super().__init__()
        self.cfg = config
        self.device = _pick_device(device)
        self.tcn = config.M
        self.h, self.w = config.latent_hw
        self.packet_len = packet_len
        self.needs_csi = config.needs_csi
        self.snr_db = snr_db

        self.decoder = model.decoder.to(self.device).eval()

        if warmup and self.device.type in ("mps", "cuda"):
            dummy = torch.randn(1, self.tcn, self.h, self.w, device=self.device)
            csi = torch.tensor([[snr_db]], device=self.device)
            with torch.no_grad():
                for _ in range(3):
                    self.decoder(dummy, csi)

    @property
    def expected_complex_items(self) -> int:
        if self.aligner is not None and self.aligner.mode == "compressing":
            return math.ceil(self.aligner.transmitted_dim / 2)
        return (self.tcn * self.h * self.w) // 2

    def set_csi(self, value: float) -> None:
        self.snr_db = float(value)

    def decode(self, payload: np.ndarray, csi: Optional[float] = None) -> bytes:
        if csi is not None:
            self.set_csi(csi)

        expected = self.expected_complex_items
        if len(payload) < expected:
            raise ValueError(f"expected >= {expected} symbols, got {len(payload)}")
        chn_in = np.asarray(payload[:expected], dtype=np.complex64)

        t = torch.from_numpy(chn_in)
        reals = torch.cat([t.real, t.imag]).to(dtype=torch.float32, device=self.device)
        csi_tensor = torch.tensor([[self.snr_db]], dtype=torch.float32,
                                  device=self.device)

        compressing = self.aligner is not None and self.aligner.mode == "compressing"
        with torch.no_grad():
            if compressing:
                cu = self.aligner.transmitted_dim
                z = reals[:cu].reshape(cu, 1)
                ct = self.aligner.decompress(z).reshape(1, self.tcn, self.h, self.w)
            else:
                ct = reals.reshape(1, self.tcn, self.h, self.w)
                if self.aligner is not None:
                    ct = self.aligner.apply_rx(ct)
            decoded = self.decoder(ct, csi_tensor)

        img = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        return img.permute(1, 2, 0).cpu().numpy().tobytes()


# ---------------------------------------------------------------------------
# Factory entry points (used by jscc.loader)
# ---------------------------------------------------------------------------

ENCODER_CODEC = DJSCCEncoderCodec
DECODER_CODEC = DJSCCDecoderCodec


def _comp_ratio_to_M(comp_ratio: float) -> int:
    # matches djscc-demo: tcn = (1/comp_ratio) * 4*4*2*3
    return int((1.0 / comp_ratio) * 4 * 4 * 2 * 3)


def build(role: str, *, ckpt: str, comp_ratio: float = 6, N: int = 256,
          img_height: int = 512, img_width: int = 768, csi_length: int = 1,
          csi_embed_dim: int = 64, device: str = "auto", packet_len: int = 960,
          snr_db: float = 19.0, warmup: bool = False, **_):
    """Build a DJSCC codec half from a raw ``.pth`` checkpoint (legacy path)."""
    config = DJSCCConfig(
        N=N, M=_comp_ratio_to_M(comp_ratio),
        img_height=img_height, img_width=img_width,
        csi_length=csi_length, csi_embed_dim=csi_embed_dim, packet_len=packet_len,
    )
    model = DJSCCModel(config)
    _load_split_state(model, ckpt)
    if role == "encoder":
        return DJSCCEncoderCodec(model, config, device=device,
                                 packet_len=packet_len, warmup=warmup)
    return DJSCCDecoderCodec(model, config, device=device, packet_len=packet_len,
                             snr_db=snr_db, warmup=warmup)
