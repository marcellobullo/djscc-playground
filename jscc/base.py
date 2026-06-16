"""Stable codec/aligner contract for djscc-playground.

The transmit and receive scripts (`transmitter/socket_tx.py`,
`receiver/socket_rx.py`) only ever talk to these abstractions, so adding a new
model means shipping a new ``BaseEncoderCodec`` / ``BaseDecoderCodec`` pair
(usually loaded from a HuggingFace repo via :func:`jscc.loader.load_codec`) —
never editing a socket script.

Three things make a codec model-agnostic at the wire level:

* ``output_kind``           — ``"complex_symbols"`` routes to the DJSCC GNU Radio
                              flowgraph (cf32 over the air); ``"bytes"`` routes to
                              the conventional SSCC flowgraph (packed bytes).
* ``expected_complex_items`` — how many cf32 symbols make up one image. The
                              socket uses this for lock-step packetisation; an
                              attached compressing aligner may change it.
* ``needs_csi``             — whether the decoder consumes channel state (SNR).
                              Channel-blind codecs simply report ``False`` and
                              ``set_csi`` is a no-op.
"""

from __future__ import annotations

import abc
from typing import Optional


class OutputKind:
    """Wire domain a codec emits/consumes — selects the GNU Radio flowgraph."""

    COMPLEX_SYMBOLS = "complex_symbols"   # -> djscc_{tx,rx}.grc
    BYTES = "bytes"                       # -> conventional_{tx,rx}.grc


class Role:
    ENCODER = "encoder"
    DECODER = "decoder"


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

class BaseAligner(abc.ABC):
    """Optional semantic aligner composed onto a codec (any model x any aligner).

    Two modes, mirroring the two families in the alignment literature:

    * ``"residual"``    — shape-preserving map applied to the *received* latent at
                          the RX: ``apply_rx(x4d) -> x4d``.
    * ``"compressing"`` — changes the transmitted dimensionality (zero-shot
                          aligner): ``compress(flat)`` at the TX and
                          ``decompress(z)`` at the RX. ``transmitted_dim`` is the
                          number of real values actually sent.

    ``stage`` is informational (``"tx_latent" | "rx_latent" | "both"``); the codec
    branches on ``mode``.
    """

    kind: str = "identity"
    mode: str = "residual"            # "residual" | "compressing"
    stage: str = "rx_latent"          # "tx_latent" | "rx_latent" | "both"
    transmitted_dim: Optional[int] = None

    # residual mode -----------------------------------------------------------
    def apply_rx(self, latent_4d):
        """Map the received [1, tcn, h, w] latent. Default: identity."""
        return latent_4d

    # compressing mode --------------------------------------------------------
    def compress(self, flat_1xm):
        raise NotImplementedError("compress() only defined for compressing aligners")

    def decompress(self, z):
        raise NotImplementedError("decompress() only defined for compressing aligners")


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

class BaseCodec(abc.ABC):
    """Common state shared by the encoder and decoder halves of a codec."""

    role: str = ""
    output_kind: str = OutputKind.COMPLEX_SYMBOLS
    needs_csi: bool = False
    packet_len: int = 960

    def __init__(self) -> None:
        self.aligner: Optional[BaseAligner] = None

    @property
    @abc.abstractmethod
    def expected_complex_items(self) -> int:
        """Number of cf32 symbols that make up one image (post-aligner)."""

    def set_aligner(self, aligner: Optional[BaseAligner]) -> None:
        self.aligner = aligner


class BaseEncoderCodec(BaseCodec):
    role = Role.ENCODER

    @abc.abstractmethod
    def encode(self, frame_bytes: bytes):
        """Raw HWC uint8 image bytes -> np.ndarray (complex64) or bytes.

        Return dtype must match ``output_kind``: complex64 for COMPLEX_SYMBOLS,
        a ``bytes``/uint8 buffer for BYTES.
        """


class BaseDecoderCodec(BaseCodec):
    role = Role.DECODER

    @abc.abstractmethod
    def decode(self, payload, csi=None) -> bytes:
        """Received symbols/bytes -> raw HWC uint8 image bytes."""

    def set_csi(self, value: float) -> None:
        """Update channel state (e.g. live SNR in dB). No-op if not needs_csi."""
