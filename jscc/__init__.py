"""djscc-playground codec framework.

Public API:
    load_codec(model, role, ...)   -> BaseEncoderCodec | BaseDecoderCodec
    load_aligner(spec, tcn, h, w)  -> BaseAligner | None
"""

from .base import (
    BaseAligner,
    BaseCodec,
    BaseDecoderCodec,
    BaseEncoderCodec,
    OutputKind,
    Role,
)
from .loader import load_aligner, load_codec

__all__ = [
    "BaseAligner",
    "BaseCodec",
    "BaseDecoderCodec",
    "BaseEncoderCodec",
    "OutputKind",
    "Role",
    "load_codec",
    "load_aligner",
]
