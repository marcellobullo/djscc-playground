"""DSP-glue adapters for AD-JSCC.

Both halves are SNR-adaptive: the encoder takes the design SNR as a side input
(``needs_csi=True`` on the encoder too, unlike the channel-blind ConvNeXt model),
and the power constraint is the ADJSCC ``powerConstraint`` function rather than a
Kaira module. Lives in the installed ``jscc`` package (``config.codec_module``);
not copied to the Hub.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from ..base import BaseDecoderCodec, BaseEncoderCodec, OutputKind
from ..djscc.codec_djscc import _comp_ratio_to_M, _pick_device
from .configuration_adjscc import AdjsccConfig
from .modeling_adjscc import AdjsccModel
from .nn import powerConstraint


def _load_adjscc_state(model: AdjsccModel, ckpt_path: str) -> None:
    """Load a raw ADJSCC checkpoint (keys prefixed ``attention_encoder.`` /
    ``attention_decoder.``) into the model's encoder/decoder submodules."""
    model.encoder.load_pretrained_weights(ckpt_path)
    model.decoder.load_pretrained_weights(ckpt_path)


# ── encoder (SNR-adaptive) ───────────────────────────────────────────────────

class AdjsccEncoderCodec(BaseEncoderCodec):
    output_kind = OutputKind.COMPLEX_SYMBOLS
    needs_csi = True

    def __init__(self, model: AdjsccModel, config: AdjsccConfig, *,
                 device: str = "auto", packet_len: int = 960, snr_db: float = 19.0,
                 padding_zeros: int = 0, warmup: bool = False, **_) -> None:
        super().__init__()
        self.cfg = config
        self.device = _pick_device(device)
        self.tcn = config.M
        self.h, self.w = config.latent_hw
        self.packet_len = packet_len
        self.padding_zeros = padding_zeros
        self.power = float(config.power)
        self.snr_db = float(snr_db)
        self.encoder = model.encoder.to(self.device).eval()

        if warmup and self.device.type in ("mps", "cuda"):
            dummy = torch.randn(1, 3, config.img_height, config.img_width, device=self.device)
            attn = torch.tensor([[self.snr_db]], device=self.device)
            with torch.no_grad():
                self.encoder(dummy, attn)

    @property
    def expected_complex_items(self) -> int:
        if self.aligner is not None and self.aligner.mode == "compressing":
            return math.ceil(self.aligner.transmitted_dim / 2)
        return (self.tcn * self.h * self.w) // 2

    def set_csi(self, value: float) -> None:
        """Design SNR (dB) fed to the attention encoder."""
        self.snr_db = float(value)

    def encode(self, frame_bytes: bytes) -> np.ndarray:
        img = (
            torch.from_numpy(
                np.frombuffer(frame_bytes, dtype=np.uint8)
                .reshape((self.cfg.img_height, self.cfg.img_width, 3)).copy())
            .float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(self.device)
        )
        attn = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)
        compressing = self.aligner is not None and self.aligner.mode == "compressing"
        with torch.no_grad():
            flat = self.encoder(img, attn).flatten()
            if compressing:
                flat = self.aligner.compress(flat.unsqueeze(0)).flatten()
            norm = powerConstraint(flat, P=self.power)

        chn = norm.detach().cpu().numpy().astype(np.float32)
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


# ── decoder (SNR-adaptive) ───────────────────────────────────────────────────

class AdjsccDecoderCodec(BaseDecoderCodec):
    output_kind = OutputKind.COMPLEX_SYMBOLS

    def __init__(self, model: AdjsccModel, config: AdjsccConfig, *,
                 device: str = "auto", packet_len: int = 960, snr_db: float = 19.0,
                 warmup: bool = False, **_) -> None:
        super().__init__()
        self.cfg = config
        self.device = _pick_device(device)
        self.tcn = config.M
        self.h, self.w = config.latent_hw
        self.packet_len = packet_len
        self.needs_csi = config.needs_csi
        self.snr_db = float(snr_db)
        self.decoder = model.decoder.to(self.device).eval()

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
        attn = torch.tensor([[self.snr_db]], dtype=torch.float32, device=self.device)

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
            decoded = self.decoder(ct, attn)

        img = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        return img.permute(1, 2, 0).cpu().numpy().tobytes()


ENCODER_CODEC = AdjsccEncoderCodec
DECODER_CODEC = AdjsccDecoderCodec


def build(role: str, *, ckpt: str, comp_ratio: float = 6, img_height: int = 512,
          img_width: int = 768, power: float = 1.0, device: str = "auto",
          packet_len: int = 960, snr_db: float = 19.0, warmup: bool = False, **_):
    """Build an AD-JSCC codec half from a raw ADJSCC ``.pth`` checkpoint."""
    config = AdjsccConfig(
        M=_comp_ratio_to_M(comp_ratio), img_height=img_height,
        img_width=img_width, power=power, packet_len=packet_len,
    )
    model = AdjsccModel(config)
    _load_adjscc_state(model, ckpt)
    if role == "encoder":
        return AdjsccEncoderCodec(model, config, device=device,
                                  packet_len=packet_len, snr_db=snr_db, warmup=warmup)
    return AdjsccDecoderCodec(model, config, device=device,
                              packet_len=packet_len, snr_db=snr_db, warmup=warmup)
