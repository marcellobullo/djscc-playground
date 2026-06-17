"""DSP-glue adapters for the spatial-CSI ("no-band") DJSCC variant.

The encoder half is identical to the standard variant, so it is reused verbatim.
The decoder half differs: it builds a per-element CSI map (from a per-packet SNR
vector) and feeds it to ``DJSCCDecoderSpatialCSI``. Lives in the installed ``jscc``
package (referenced via ``config.codec_module``); not copied to the Hub.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..base import BaseDecoderCodec, OutputKind
# Reuse the channel-blind encoder adapter and helpers from the standard variant.
from ..djscc.codec_djscc import (
    DJSCCEncoderCodec,
    _comp_ratio_to_M,
    _load_split_state,
    _pick_device,
)
from .configuration_djscc_spatialcsi import DJSCCSpatialCSIConfig
from .modeling_djscc_spatialcsi import DJSCCSpatialCSIModel
from .nn import packetwise_to_element_map


class DJSCCSpatialCSIDecoderCodec(BaseDecoderCodec):
    """Spatial-CSI decoder: scalar or per-packet SNR -> per-element CSI map."""

    output_kind = OutputKind.COMPLEX_SYMBOLS

    def __init__(self, model: DJSCCSpatialCSIModel, config: DJSCCSpatialCSIConfig, *,
                 device: str = "auto", packet_len: int = 960,
                 snr_db: float = 19.0, warmup: bool = False, **_) -> None:
        super().__init__()
        self.cfg = config
        self.device = _pick_device(device)
        self.tcn = config.M
        self.h, self.w = config.latent_hw
        self.packet_len = packet_len
        self.needs_csi = config.needs_csi
        self.csi_db_scale = config.csi_db_scale
        self.sentinel_drop_db = config.sentinel_drop_db
        self.snr_db = float(snr_db)

        self.decoder = model.decoder.to(self.device).eval()
        self._expected = (self.tcn * self.h * self.w) // 2
        self.n_pkts = int(np.ceil(self._expected / packet_len))
        self._snr_per_pkt = np.full(self.n_pkts, self.snr_db, dtype=np.float32)
        self._element_map: Optional[torch.Tensor] = None

        if warmup and self.device.type in ("mps", "cuda"):
            dummy = torch.randn(1, self.tcn, self.h, self.w, device=self.device)
            with torch.no_grad():
                self.decoder(dummy, self._build_map())

    @property
    def expected_complex_items(self) -> int:
        return self._expected

    def set_aligner(self, aligner) -> None:
        if aligner is not None and getattr(aligner, "mode", "") == "compressing":
            raise ValueError(
                "zero-shot (compressing) aligner is unsupported by the spatial-CSI "
                "decoder; use a conv/twoconv/linear/mlp aligner.")
        self.aligner = aligner

    # ── CSI state ────────────────────────────────────────────────────────────
    def set_csi(self, value: float) -> None:
        """Uniform per-packet SNR (dB) across the image."""
        self.snr_db = float(value)
        self._snr_per_pkt = np.full(self.n_pkts, self.snr_db, dtype=np.float32)
        self._element_map = None

    def set_csi_vector(self, snr_db_per_pkt) -> None:
        """Per-packet SNR (dB); non-finite entries become the drop sentinel."""
        vec = np.asarray(snr_db_per_pkt, dtype=np.float32).reshape(-1)
        if vec.size != self.n_pkts:
            raise ValueError(f"expected {self.n_pkts} per-packet SNRs, got {vec.size}")
        vec = vec.copy()
        vec[~np.isfinite(vec)] = self.sentinel_drop_db
        self._snr_per_pkt = vec
        self._element_map = None

    def _build_map(self) -> torch.Tensor:
        t = torch.from_numpy(self._snr_per_pkt).unsqueeze(0).to(self.device)
        return packetwise_to_element_map(
            t, M=self.tcn, H=self.h, W=self.w, pkt_len_complex=self.packet_len)

    # ── inference ──────────────────────────────────────────────────────────--
    def decode(self, payload: np.ndarray, csi: Optional[float] = None) -> bytes:
        if csi is not None:
            self.set_csi(csi)
        if len(payload) < self._expected:
            raise ValueError(f"expected >= {self._expected} symbols, got {len(payload)}")
        chn_in = np.asarray(payload[:self._expected], dtype=np.complex64)

        t = torch.from_numpy(chn_in)
        ct = torch.cat([t.real, t.imag]).to(
            dtype=torch.float32, device=self.device).reshape(1, self.tcn, self.h, self.w)
        if self.aligner is not None:
            ct = self.aligner.apply_rx(ct)

        csi_map = self._element_map if self._element_map is not None else self._build_map()
        with torch.no_grad():
            decoded = self.decoder(ct, csi_map)

        img = torch.clamp(decoded.squeeze(0) * 255.0, 0, 255).byte()
        return img.permute(1, 2, 0).cpu().numpy().tobytes()


ENCODER_CODEC = DJSCCEncoderCodec          # channel-blind encoder, reused as-is
DECODER_CODEC = DJSCCSpatialCSIDecoderCodec


def build(role: str, *, ckpt: str, comp_ratio: float = 6, N: int = 256,
          img_height: int = 512, img_width: int = 768, csi_embed_dim: int = 64,
          csi_db_scale: float = 20.0, device: str = "auto", packet_len: int = 960,
          snr_db: float = 19.0, warmup: bool = False, **_):
    """Build a spatial-CSI codec half from a raw ``no_band`` ``.pth`` checkpoint."""
    config = DJSCCSpatialCSIConfig(
        N=N, M=_comp_ratio_to_M(comp_ratio),
        img_height=img_height, img_width=img_width,
        csi_embed_dim=csi_embed_dim, csi_db_scale=csi_db_scale, packet_len=packet_len,
    )
    model = DJSCCSpatialCSIModel(config)
    _load_split_state(model, ckpt)
    if role == "encoder":
        return DJSCCEncoderCodec(model, config, device=device,
                                 packet_len=packet_len, warmup=warmup)
    return DJSCCSpatialCSIDecoderCodec(model, config, device=device,
                                       packet_len=packet_len, snr_db=snr_db, warmup=warmup)
