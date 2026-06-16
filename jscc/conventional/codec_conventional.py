"""Conventional separated source-channel coding (SSCC) baseline.

This codec is NOT a neural network — it is JPEG/BPG-style source compression plus
channel coding, emitting packed *bytes* (``output_kind == "bytes"``) that the
``conventional_{tx,rx}.grc`` flowgraph maps to QAM symbols. It lives in the alias
registry (``conventional``) rather than as an HF repo for that reason.

TODO: port the encode/decode bodies from djscc-demo's
``transmitter/socket_conventional_tx.py`` and ``receiver/socket_conventional_rx.py``.
The interface and wiring (output_kind=bytes -> conventional flowgraph, byte ZMQ
serialization) are in place so the socket scripts can already route to it.
"""

from __future__ import annotations

import numpy as np

from ..base import BaseDecoderCodec, BaseEncoderCodec, OutputKind


class ConventionalEncoderCodec(BaseEncoderCodec):
    output_kind = OutputKind.BYTES
    needs_csi = False

    def __init__(self, *, device: str = "auto", packet_len: int = 960, **kw) -> None:
        super().__init__()
        self.packet_len = packet_len
        self._kw = kw

    @property
    def expected_complex_items(self) -> int:
        # Conventional is byte-paced, not symbol-paced; the socket treats the
        # bytes path separately. Returned for interface completeness.
        return 0

    def encode(self, frame_bytes: bytes) -> bytes:
        raise NotImplementedError(
            "Conventional SSCC encode not yet ported — see module docstring.")


class ConventionalDecoderCodec(BaseDecoderCodec):
    output_kind = OutputKind.BYTES

    def __init__(self, *, device: str = "auto", packet_len: int = 960, **kw) -> None:
        super().__init__()
        self.packet_len = packet_len
        self._kw = kw

    @property
    def expected_complex_items(self) -> int:
        return 0

    def decode(self, payload, csi=None) -> bytes:
        raise NotImplementedError(
            "Conventional SSCC decode not yet ported — see module docstring.")


ENCODER_CODEC = ConventionalEncoderCodec
DECODER_CODEC = ConventionalDecoderCodec


def build(role: str, *, device: str = "auto", packet_len: int = 960, **kw):
    if role == "encoder":
        return ConventionalEncoderCodec(device=device, packet_len=packet_len, **kw)
    return ConventionalDecoderCodec(device=device, packet_len=packet_len, **kw)
