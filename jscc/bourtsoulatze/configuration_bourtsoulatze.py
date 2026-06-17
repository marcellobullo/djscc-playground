"""HuggingFace config for the Bourtsoulatze-2019 DeepJSCC baseline (from Kaira).

The canonical analog DeepJSCC (Bourtsoulatze et al., 2019): a fixed-SNR,
non-adaptive CNN. Same latent geometry as the other families ([B, M, H/4, W/4]),
so it slots into the existing codec adapters unchanged.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class BourtsoulatzeConfig(PretrainedConfig):
    model_type = "deepjscc_b2019"

    def __init__(
        self,
        M: int = 16,                  # num_transmitted_filters (latent channels)
        img_height: int = 512,
        img_width: int = 768,
        average_power: float = 1.0,
        needs_csi: bool = False,      # not SNR-adaptive
        output_kind: str = "complex_symbols",
        packet_len: int = 960,
        codec_module: str = "jscc.bourtsoulatze.codec_bourtsoulatze",
        **kwargs,
    ) -> None:
        self.M = M
        self.img_height = img_height
        self.img_width = img_width
        self.average_power = average_power
        self.needs_csi = needs_csi
        self.output_kind = output_kind
        self.packet_len = packet_len
        self.codec_module = codec_module
        super().__init__(**kwargs)

    @property
    def latent_hw(self):
        return self.img_height // 4, self.img_width // 4
