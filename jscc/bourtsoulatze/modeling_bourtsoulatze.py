"""HuggingFace wrapper for the Bourtsoulatze-2019 DeepJSCC baseline.

Imports the encoder/decoder straight from ``kaira.models.image`` (no vendored
nn.py) — the HF repo's modeling code just requires ``kaira`` installed, like the
other Kaira-native models. Not SNR-adaptive: ``forward`` ignores ``csi``.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from kaira.constraints.power import AveragePowerConstraint
from kaira.models.image import (
    Bourtsoulatze2019DeepJSCCDecoder,
    Bourtsoulatze2019DeepJSCCEncoder,
)

from .configuration_bourtsoulatze import BourtsoulatzeConfig


class BourtsoulatzeModel(PreTrainedModel):
    config_class = BourtsoulatzeConfig
    main_input_name = "image"
    _tied_weights_keys: list = []

    def __init__(self, config: BourtsoulatzeConfig) -> None:
        super().__init__(config)
        # Kaira image models may use compressai GDN (real math in __init__);
        # a nested CPU context overrides HF's meta-device init (see adjscc).
        with torch.device("cpu"):
            self.encoder = Bourtsoulatze2019DeepJSCCEncoder(config.M)
            self.decoder = Bourtsoulatze2019DeepJSCCDecoder(config.M)
        self.constraint = AveragePowerConstraint(average_power=config.average_power)
        self.post_init()

    def _init_weights(self, module) -> None:  # preserve loaded / module-init weights
        pass

    def forward(self, image: torch.Tensor, csi: Optional[torch.Tensor] = None,
                channel: Optional[torch.nn.Module] = None) -> torch.Tensor:
        z = self.constraint(self.encoder(image))
        if channel is not None:
            z = channel(z)
        return self.decoder(z)  # csi ignored (non-adaptive)


BourtsoulatzeConfig.register_for_auto_class()
BourtsoulatzeModel.register_for_auto_class("AutoModel")
