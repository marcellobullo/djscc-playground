"""DSP-glue for the Bourtsoulatze-2019 baseline.

Geometry and packing are identical to the custom ConvNeXt model, and the decoder
tolerates an ignored CSI arg, so the standard DJSCC encoder/decoder codec adapters
are reused verbatim. ``needs_csi=False`` propagates from the config.
"""

from __future__ import annotations

from ..djscc.codec_djscc import (
    DJSCCDecoderCodec,
    DJSCCEncoderCodec,
    _comp_ratio_to_M,
    _load_split_state,
)
from .configuration_bourtsoulatze import BourtsoulatzeConfig
from .modeling_bourtsoulatze import BourtsoulatzeModel

ENCODER_CODEC = DJSCCEncoderCodec
DECODER_CODEC = DJSCCDecoderCodec


def build(role: str, *, ckpt: str = None, comp_ratio: float = 6, img_height: int = 512,
          img_width: int = 768, average_power: float = 1.0, device: str = "auto",
          packet_len: int = 960, snr_db: float = 19.0, warmup: bool = False, **_):
    """Build a Bourtsoulatze codec half. ``ckpt`` optional (trained .pth); without
    it the model is randomly initialized (for save_pretrained + train --reinit)."""
    config = BourtsoulatzeConfig(
        M=_comp_ratio_to_M(comp_ratio), img_height=img_height,
        img_width=img_width, average_power=average_power, packet_len=packet_len,
    )
    model = BourtsoulatzeModel(config)
    if ckpt:
        _load_split_state(model, ckpt)
    if role == "encoder":
        return DJSCCEncoderCodec(model, config, device=device,
                                 packet_len=packet_len, warmup=warmup)
    return DJSCCDecoderCodec(model, config, device=device, packet_len=packet_len,
                             snr_db=snr_db, warmup=warmup)
