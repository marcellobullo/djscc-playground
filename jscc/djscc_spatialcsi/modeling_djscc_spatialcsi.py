"""HuggingFace PreTrainedModel wrapper for the spatial-CSI ("no-band") DJSCC.

Same template as ``jscc.djscc.modeling_djscc``: wraps the Kaira-native encoder +
spatial-CSI decoder + power constraint, exposes them as plain submodules, and
delegates the DSP glue to ``jscc.djscc_spatialcsi.codec_djscc_spatialcsi``.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from kaira.constraints.power import AveragePowerConstraint

from .configuration_djscc_spatialcsi import DJSCCSpatialCSIConfig
from .nn import DJSCCDecoderSpatialCSI, DJSCCEncoder


class DJSCCSpatialCSIModel(PreTrainedModel):
    """End-to-end no-band model: encoder + power constraint + spatial-CSI decoder."""

    config_class = DJSCCSpatialCSIConfig
    main_input_name = "image"
    _tied_weights_keys: list = []

    def __init__(self, config: DJSCCSpatialCSIConfig) -> None:
        super().__init__(config)
        self.encoder = DJSCCEncoder(N=config.N, M=config.M)
        self.decoder = DJSCCDecoderSpatialCSI(
            N=config.N, M=config.M,
            csi_embed_dim=config.csi_embed_dim, csi_db_scale=config.csi_db_scale,
        )
        self.constraint = AveragePowerConstraint(average_power=config.average_power)
        self.post_init()

    def _init_weights(self, module) -> None:  # preserve loaded / module-init weights
        pass

    def forward(self, image: torch.Tensor, csi_map_db: torch.Tensor,
                channel: Optional[torch.nn.Module] = None) -> torch.Tensor:
        z = self.constraint(self.encoder(image))
        if channel is not None:
            z = channel(z)
        return self.decoder(z, csi_map_db)


DJSCCSpatialCSIConfig.register_for_auto_class()
DJSCCSpatialCSIModel.register_for_auto_class("AutoModel")
