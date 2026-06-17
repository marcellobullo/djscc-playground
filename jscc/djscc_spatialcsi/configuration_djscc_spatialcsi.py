"""HuggingFace config for the spatial-CSI ("no-band") DJSCC variant."""

from __future__ import annotations

from transformers import PretrainedConfig


class DJSCCSpatialCSIConfig(PretrainedConfig):
    model_type = "djscc_spatialcsi"

    def __init__(
        self,
        N: int = 256,
        M: int = 16,
        img_height: int = 512,
        img_width: int = 768,
        csi_embed_dim: int = 64,
        csi_db_scale: float = 20.0,
        sentinel_drop_db: float = -20.0,
        average_power: float = 1.0,
        needs_csi: bool = True,
        output_kind: str = "complex_symbols",
        packet_len: int = 960,
        codec_module: str = "jscc.djscc_spatialcsi.codec_djscc_spatialcsi",
        **kwargs,
    ) -> None:
        self.N = N
        self.M = M
        self.img_height = img_height
        self.img_width = img_width
        self.csi_embed_dim = csi_embed_dim
        self.csi_db_scale = csi_db_scale
        self.sentinel_drop_db = sentinel_drop_db
        self.average_power = average_power
        self.needs_csi = needs_csi
        self.output_kind = output_kind
        self.packet_len = packet_len
        self.codec_module = codec_module
        super().__init__(**kwargs)

    @property
    def latent_hw(self):
        return self.img_height // 4, self.img_width // 4
