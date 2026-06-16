"""HuggingFace config for the custom DJSCC model.

The config is the single source of truth for the build recipe — ``from_pretrained``
reconstructs the network from these fields, so the socket scripts no longer carry
``--tcn/--N/--comp-ratio`` hyper-parameter math.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class DJSCCConfig(PretrainedConfig):
    model_type = "djscc"

    def __init__(
        self,
        N: int = 256,
        M: int = 16,                  # tcn — real+imag channel planes
        img_height: int = 512,
        img_width: int = 768,
        csi_length: int = 1,
        csi_embed_dim: int = 64,
        average_power: float = 1.0,
        needs_csi: bool = True,       # decoder is CSI-adaptive
        output_kind: str = "complex_symbols",
        packet_len: int = 960,
        # dotted path to the module exposing ENCODER_CODEC / DECODER_CODEC
        codec_module: str = "jscc.djscc.codec_djscc",
        **kwargs,
    ) -> None:
        self.N = N
        self.M = M
        self.img_height = img_height
        self.img_width = img_width
        self.csi_length = csi_length
        self.csi_embed_dim = csi_embed_dim
        self.average_power = average_power
        self.needs_csi = needs_csi
        self.output_kind = output_kind
        self.packet_len = packet_len
        self.codec_module = codec_module
        super().__init__(**kwargs)

    @property
    def latent_hw(self):
        """Spatial size of the latent (4x downsampled)."""
        return self.img_height // 4, self.img_width // 4
