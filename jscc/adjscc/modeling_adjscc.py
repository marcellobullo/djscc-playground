"""HuggingFace PreTrainedModel wrapper for AD-JSCC.

Same template as the other variants: wraps the SNR-adaptive attention encoder +
decoder, exposes them as plain submodules, and delegates the DSP glue to
``jscc.adjscc.codec_adjscc``. Plain-torch + compressai (GDN), not Kaira-native;
loading a pushed repo requires ``compressai`` installed.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from .configuration_adjscc import AdjsccConfig
from .nn import Args, Attention_Decoder, Attention_Encoder, powerConstraint


class AdjsccModel(PreTrainedModel):
    """End-to-end AD-JSCC: SNR-adaptive encoder + power constraint + decoder."""

    config_class = AdjsccConfig
    main_input_name = "image"
    _tied_weights_keys: list = []

    def __init__(self, config: AdjsccConfig) -> None:
        super().__init__(config)
        # compressai's GDN runs real tensor math in __init__, which clashes with
        # transformers' meta-device fast init (cpu-vs-meta mismatch). A nested
        # CPU device context overrides the outer meta context, so the submodules
        # build materialized on CPU; HF then loads the real weights over them.
        with torch.device("cpu"):
            self.encoder = Attention_Encoder(Args(tcn=config.M))
            self.decoder = Attention_Decoder(Args(tcn=config.M))
        self.power = float(config.power)
        self.post_init()

    def _init_weights(self, module) -> None:  # preserve loaded / module-init weights
        pass

    def forward(self, image: torch.Tensor, snr: torch.Tensor,
                channel: Optional[torch.nn.Module] = None) -> torch.Tensor:
        z = self.encoder(image, snr)
        shape = z.shape
        z = powerConstraint(z.flatten(), P=self.power).reshape(shape)
        if channel is not None:
            z = channel(z)
        return self.decoder(z, snr)


AdjsccConfig.register_for_auto_class()
AdjsccModel.register_for_auto_class("AutoModel")
