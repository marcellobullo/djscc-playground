"""HuggingFace PreTrainedModel wrapper around the Kaira-native DJSCC modules.

This is the "package" layer: it carries weights + config + (via push_to_hub) the
custom code, and exposes the trained ``encoder`` / ``decoder`` / ``constraint`` as
plain submodules. It deliberately does NOT do any complex-symbol packing or CSI
plumbing — that lives in :mod:`jscc.djscc.codec_djscc`, so the same model can be
driven by the split TX/RX deployment over a real RF channel.

Kaira-native: the building blocks in :mod:`jscc.djscc.nn` subclass
``kaira.models.base.BaseModel`` and the constraint is ``kaira.constraints``.
Loading an HF repo that ships this file therefore requires ``kaira`` installed.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from kaira.constraints.power import AveragePowerConstraint

from .configuration_djscc import DJSCCConfig
from .nn import DJSCCDecoder, DJSCCEncoder


class DJSCCModel(PreTrainedModel):
    """End-to-end DJSCC model (encoder + power constraint + CSI-adaptive decoder).

    ``forward`` runs the full chain for training / offline simulation, optionally
    through a Kaira channel. At deployment the codec adapter calls ``.encoder`` and
    ``.decoder`` directly and substitutes the real USRP link for ``channel``.
    """

    config_class = DJSCCConfig
    main_input_name = "image"

    def __init__(self, config: DJSCCConfig) -> None:
        super().__init__(config)
        self.encoder = DJSCCEncoder(N=config.N, M=config.M)
        self.decoder = DJSCCDecoder(
            N=config.N, M=config.M,
            csi_length=config.csi_length, csi_embed_dim=config.csi_embed_dim,
        )
        self.constraint = AveragePowerConstraint(average_power=config.average_power)
        # Sets up HF bookkeeping (all_tied_weights_keys etc.); _init_weights is a
        # no-op so the modules' own initialisation / loaded weights are preserved.
        self.post_init()

    _tied_weights_keys: list = []

    # weights come from from_pretrained / a loaded checkpoint; don't re-init.
    def _init_weights(self, module) -> None:  # noqa: D401
        pass

    def forward(self, image: torch.Tensor, csi: torch.Tensor,
                channel: Optional[torch.nn.Module] = None) -> torch.Tensor:
        z = self.encoder(image)
        z = self.constraint(z)
        if channel is not None:
            z = channel(z)
        return self.decoder(z, csi)


# Register so push_to_hub writes the auto_map and AutoModel.from_pretrained works.
DJSCCConfig.register_for_auto_class()
DJSCCModel.register_for_auto_class("AutoModel")
