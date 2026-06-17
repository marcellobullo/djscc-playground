"""Self-contained nn modules for the spatial-CSI ("no-band") DJSCC variant.

Same channel-blind ConvNeXt encoder as ``jscc.djscc.nn``, but the decoder
(``DJSCCDecoderSpatialCSI``) consumes a per-element CSI map (same spatial shape
as the received latent) instead of a scalar SNR. The map is built from a
per-packet SNR vector via ``packetwise_to_element_map``. This removes the
horizontal banding that a scalar-CSI decoder shows on OTA per-packet SNR
variation.

Kept self-contained (no cross-package imports) so HuggingFace ``push_to_hub``
copies a standalone repo. Kaira-native: blocks subclass ``kaira.models.BaseModel``.
"""

from typing import Any

import torch
from torch import nn

from kaira.models.base import BaseModel, ChannelAwareBaseModel


# ── building blocks (shared with jscc.djscc.nn) ──────────────────────────────

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for NCHW tensors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt v1 block: depthwise 7x7 -> LayerNorm -> pointwise expand ->
    GELU -> pointwise contract -> residual add."""

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
    """Spatial 2x downsample: LayerNorm -> stride-2 conv."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class Upsample2x(nn.Module):
    """Spatial 2x upsample: Conv -> PixelShuffle."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on a CSI embedding."""

    def __init__(self, feature_dim: int, csi_embed_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(csi_embed_dim, feature_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, csi_embed: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(csi_embed).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


# ── channel-blind encoder (identical to the standard variant) ────────────────

class DJSCCEncoder(BaseModel):
    """Channel-blind DJSCC encoder. [B, 3, H, W] -> [B, M, H/4, W/4]."""

    def __init__(self, N: int = 256, M: int = 16, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.stem = nn.Sequential(
            nn.Conv2d(3, N, 7, stride=2, padding=3),
            LayerNorm2d(N),
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock(N), ConvNeXtBlock(N))
        self.down = Downsample2x(N, N)
        self.stage2 = nn.Sequential(ConvNeXtBlock(N), ConvNeXtBlock(N))
        self.head = nn.Conv2d(N, M, 1)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        return self.head(x)


# ── packet -> element CSI layout helper ──────────────────────────────────────

def packetwise_to_element_map(per_packet: torch.Tensor, M: int, H: int, W: int,
                              pkt_len_complex: int) -> torch.Tensor:
    """Expand a [B, n_pkts] vector to a [B, M, H, W] map using the same
    flatten/serialize layout the codec uses (C,H,W order; real half then imag
    half; both halves of a complex symbol share its packet's value)."""
    B, n_pkts = per_packet.shape
    if M % 2 != 0:
        raise ValueError(f"M must be even (got {M})")
    half_M = M // 2
    half_size = half_M * H * W
    expanded = per_packet.repeat_interleave(pkt_len_complex, dim=1)
    if expanded.shape[1] >= half_size:
        real_half = expanded[:, :half_size]
    else:
        pad = expanded[:, -1:].expand(B, half_size - expanded.shape[1])
        real_half = torch.cat([expanded, pad], dim=1)
    real_half_map = real_half.reshape(B, half_M, H, W)
    return torch.cat([real_half_map, real_half_map], dim=1)


# ── spatial-CSI decoder ──────────────────────────────────────────────────────

class DJSCCDecoderSpatialCSI(ChannelAwareBaseModel):
    """CSI-adaptive decoder consuming a per-element reliability map.

    Head conv takes concat(received_latent, csi_map_norm) -> 2M channels; FiLM
    still uses a scalar (per-image mean of the CSI map). [B, M, H/4, W/4] +
    csi_map [B, M, H/4, W/4] -> [B, 3, H, W].
    """

    def __init__(self, N: int = 256, M: int = 16, csi_embed_dim: int = 64,
                 csi_db_scale: float = 20.0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.csi_embed_dim = csi_embed_dim
        self.csi_db_scale = csi_db_scale

        self.csi_embed = nn.Sequential(
            nn.Linear(1, csi_embed_dim),
            nn.GELU(),
            nn.Linear(csi_embed_dim, csi_embed_dim),
        )
        self.head = nn.Sequential(nn.Conv2d(2 * M, N, 1), LayerNorm2d(N))
        self.stage1_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage1_film = nn.ModuleList(
            [FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)])
        self.up1 = Upsample2x(N, N)
        self.stage2_blocks = nn.ModuleList([ConvNeXtBlock(N), ConvNeXtBlock(N)])
        self.stage2_film = nn.ModuleList(
            [FiLMLayer(N, csi_embed_dim), FiLMLayer(N, csi_embed_dim)])
        self.up2 = Upsample2x(N, 3)

    def forward(self, x: torch.Tensor, csi_map_db: torch.Tensor,
                *args: Any, **kwargs: Any) -> torch.Tensor:
        if csi_map_db.dim() != 4:
            raise ValueError(
                "csi_map_db must be 4-D [B, M, H, W], got "
                f"{tuple(csi_map_db.shape)}; build it with "
                "packetwise_to_element_map().")
        if csi_map_db.shape != x.shape:
            csi_map_db = csi_map_db.expand_as(x)

        csi_scalar = csi_map_db.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        emb = self.csi_embed(csi_scalar)
        csi_map_norm = csi_map_db / self.csi_db_scale

        x = self.head(torch.cat([x, csi_map_norm], dim=1))
        for block, film in zip(self.stage1_blocks, self.stage1_film):
            x = film(block(x), emb)
        x = self.up1(x)
        for block, film in zip(self.stage2_blocks, self.stage2_film):
            x = film(block(x), emb)
        return self.up2(x)
