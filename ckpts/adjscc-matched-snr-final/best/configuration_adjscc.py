"""HuggingFace config for AD-JSCC (Attention-based Deep JSCC)."""

from __future__ import annotations

from transformers import PretrainedConfig


class AdjsccConfig(PretrainedConfig):
    model_type = "adjscc"

    def __init__(
        self,
        M: int = 16,                  # tcn / "rate" — latent channel count
        img_height: int = 512,
        img_width: int = 768,
        power: float = 1.0,           # P in the per-symbol power constraint
        needs_csi: bool = True,       # SNR-adaptive at BOTH encoder and decoder
        output_kind: str = "complex_symbols",
        packet_len: int = 960,
        codec_module: str = "jscc.adjscc.codec_adjscc",
        **kwargs,
    ) -> None:
        self.M = M
        self.img_height = img_height
        self.img_width = img_width
        self.power = power
        self.needs_csi = needs_csi
        self.output_kind = output_kind
        self.packet_len = packet_len
        self.codec_module = codec_module
        super().__init__(**kwargs)

    @property
    def latent_hw(self):
        return self.img_height // 4, self.img_width // 4
