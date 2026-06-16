"""Custom DJSCC model with channel-blind encoder and CSI-adaptive decoder.

Encoder: ConvNeXt blocks — depthwise 7x7 + pointwise 1x1, LayerNorm, GELU.
  No CSI (sigma^2 unknown at TX). 4x spatial downsampling.

Decoder: ConvNeXt blocks + FiLM conditioning (scale + shift from CSI).
  CSI-adaptive via learned affine modulation at every block.
"""

from typing import Any, Optional

import torch
from torch import nn

from kaira.constraints.base import BaseConstraint
from kaira.constraints.power import AveragePowerConstraint
from kaira.models.base import BaseModel, ChannelAwareBaseModel


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for NCHW tensors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt v1 block: depthwise 7x7 → LayerNorm → pointwise expand →
    GELU → pointwise contract → residual add."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class Downsample2x(nn.Module):
    """Spatial 2x downsample: LayerNorm → stride-2 conv."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class Upsample2x(nn.Module):
    """Spatial 2x upsample: Conv → PixelShuffle."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: affine transform conditioned on CSI.

    Produces per-channel scale and shift from the CSI embedding.
    Initialized near identity: output ≈ x when weights are small.
    """

    def __init__(self, feature_dim: int, csi_embed_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(csi_embed_dim, feature_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, csi_embed: torch.Tensor) -> torch.Tensor:
        params = self.fc(csi_embed)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DJSCCEncoder(BaseModel):
    """Channel-blind DJSCC encoder using ConvNeXt blocks.

    No CSI — learns robust representations across all channel conditions.
    4x spatial downsampling: [B, 3, H, W] → [B, M, H/4, W/4].
    """

    def __init__(self, N: int = 256, M: int = 16, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M

        self.stem = nn.Sequential(
            nn.Conv2d(3, N, 7, stride=2, padding=3),
            LayerNorm2d(N),
        )

        self.stage1 = nn.Sequential(
            ConvNeXtBlock(N),
            ConvNeXtBlock(N),
        )

        self.down = Downsample2x(N, N)

        self.stage2 = nn.Sequential(
            ConvNeXtBlock(N),
            ConvNeXtBlock(N),
        )

        self.head = nn.Conv2d(N, M, 1)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DJSCCDecoder(ChannelAwareBaseModel):
    """CSI-adaptive DJSCC decoder using ConvNeXt blocks + FiLM conditioning.

    Each ConvNeXt block is followed by a FiLM layer that applies learned
    per-channel scale and shift based on a CSI embedding (e.g. estimated SNR).
    4x spatial upsampling: [B, M, H/4, W/4] → [B, 3, H, W].
    """

    def __init__(
        self, N: int = 256, M: int = 16, csi_length: int = 1,
        csi_embed_dim: int = 64, *args: Any, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.csi_length = csi_length

        self.csi_embed = nn.Sequential(
            nn.Linear(csi_length, csi_embed_dim),
            nn.GELU(),
            nn.Linear(csi_embed_dim, csi_embed_dim),
        )

        self.head = nn.Sequential(
            nn.Conv2d(M, N, 1),
            LayerNorm2d(N),
        )

        self.stage1_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage1_film = nn.ModuleList([FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)])

        self.up1 = Upsample2x(N, N)

        self.stage2_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage2_film = nn.ModuleList([FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)])

        self.up2 = Upsample2x(N, 3)

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if csi.dim() == 1:
            csi = csi.unsqueeze(-1)
        if csi.dim() > 2:
            csi = csi.flatten(start_dim=1)

        emb = self.csi_embed(csi)
        x = self.head(x)

        for block, film in zip(self.stage1_blocks, self.stage1_film):
            x = film(block(x), emb)

        x = self.up1(x)

        for block, film in zip(self.stage2_blocks, self.stage2_film):
            x = film(block(x), emb)

        x = self.up2(x)
        return x


# ---------------------------------------------------------------------------
# End-to-end model
# ---------------------------------------------------------------------------

class DJSCCModel(BaseModel):
    """End-to-end DJSCC model for training.

    Chains: encoder → constraint → channel → decoder.
    The encoder is channel-blind; the decoder is CSI-adaptive.
    """

    def __init__(
        self,
        encoder: DJSCCEncoder,
        decoder: DJSCCDecoder,
        constraint: Optional[BaseConstraint] = None,
        channel: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.constraint = constraint or AveragePowerConstraint(average_power=1.0)
        self.channel = channel

    def forward(
        self, source: torch.Tensor, csi: Optional[torch.Tensor] = None,
        *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        if csi is None:
            raise ValueError("CSI must be provided for DJSCCModel forward pass")

        encoded = self.encoder(source)
        constrained = self.constraint(encoded)

        if self.channel is not None:
            received = self.channel(constrained, *args, **kwargs)
        else:
            received = constrained

        decoded = self.decoder(received, csi)
        return decoded
