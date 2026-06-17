"""Self-contained training channels that need no Kaira (pure torch).

The Kaira ``AWGNChannel`` / ``FlatFadingChannel`` used in ``training/train.py``
cover the *flat* cases. They do NOT model the frequency-selective multipath that
the real OFDM link (``transmitter``/``receiver`` GRC: fft_len=64, cp_len=16,
52 occupied carriers, 4 pilots) actually exhibits.

``OFDMMultipathChannel`` fills that gap. It reuses the channel-sampling core of
the OFDM-guided-DJSCC reference (exponential power-delay profile, ``L`` complex
taps, FFT to the per-subcarrier response ``H_k``) but is wired to model the
**post-equalization effective channel**, because the deployed receiver already
equalizes per subcarrier (``ofdm_chanest_vcvc`` + ``ofdm_frame_equalizer_vcvc``).
Concretely, per complex latent symbol on subcarrier ``k``:

    y_k = H_k * s_k + n_k,   n_k ~ CN(0, N0)
    s_hat_k = y_k / H_k                       (ZF)     = s_k + n_k / H_k
    s_hat_k = conj(H_k)/(|H_k|^2 + N0/Es) y_k (MMSE)

so the decoder sees AWGN whose variance is amplified by ``1/|H_k|^2`` on faded
subcarriers — exactly the per-subcarrier-varying SNR that the spatial-CSI decoder
and the RX SNR loggers were designed around. Training on the *raw* multipath
channel (no equalization) would instead force the network to undo distortion the
GNU Radio receiver already removes, so that is intentionally not what this does.

Contract matches the Kaira channels: ``channel(z) -> z`` on a real NCHW latent
``[B, M, H, W]``, where the M (even) feature channels pack one complex symbol per
spatial location as ``[real_half | imag_half]`` in (C, H, W) flatten order — the
same layout as ``packetwise_to_element_map`` and the codec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ── CSI contract ─────────────────────────────────────────────────────────────
# Every channel call returns ``(y, csi)``: the received latent plus its own CSI,
# expressed as *effective SNR in dB*. ``csi`` is either a scalar ``[B, 1]`` (one
# value per image — AWGN / flat fading) or a per-packet vector ``[B, n_pkts]``
# (OFDM multipath). The model declares what it needs and ``reconcile_csi`` adapts:
# vector -> scalar averages; scalar -> vector broadcasts (held constant).

def reconcile_csi(csi: torch.Tensor, need: str, n_pkts: int) -> torch.Tensor:
    """Adapt a channel's CSI to the granularity a decoder consumes.

    Args:
        csi: ``[B, 1]`` scalar or ``[B, K]`` vector, effective SNR (dB).
        need: ``"scalar"`` (one value per image) or ``"vector"`` (per packet).
        n_pkts: target vector length when ``need == "vector"``.
    """
    B = csi.shape[0]
    is_vector = csi.dim() == 2 and csi.shape[1] > 1
    if need == "scalar":
        return csi.mean(dim=1, keepdim=True) if is_vector else csi.reshape(B, 1)
    if need != "vector":
        raise ValueError(f"need must be 'scalar' or 'vector' (got {need!r})")
    if not is_vector:                                  # hold scalar constant
        return csi.reshape(B, 1).expand(B, n_pkts)
    K = csi.shape[1]                                   # already a vector
    if K == n_pkts:
        return csi
    if K > n_pkts:
        return csi[:, :n_pkts]
    return torch.cat([csi, csi[:, -1:].expand(B, n_pkts - K)], dim=1)


class ScalarCSIChannel(nn.Module):
    """Wrap a base channel (or ``None``) so it obeys the ``(y, csi)`` contract,
    reporting the nominal commanded SNR (dB) as a scalar CSI. Used for the Kaira
    AWGN / flat-fading channels, which don't expose a realized per-symbol gain."""

    def __init__(self, base: nn.Module | None, snr_db: float) -> None:
        super().__init__()
        self.base = base
        self.snr_db = float(snr_db)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = z if self.base is None else self.base(z)
        csi = torch.full((z.shape[0], 1), self.snr_db,
                         device=z.device, dtype=torch.float32)
        return y, csi


class FlatFadingChannel(nn.Module):
    """Coherent frequency-flat fading on the real DJSCC latent.

    Kaira's ``FlatFadingChannel`` multiplies by a complex gain and returns a
    complex tensor, which the real-valued decoders can't consume. This version is
    aware of the real-plane packing: it maps ``z`` to complex symbols (first M/2
    planes = real, second M/2 = imag), applies one complex gain ``h`` per image
    (block fading) — Rayleigh ``h ~ CN(0,1)`` or Rician with the given K-factor
    (``E[|h|^2] = 1`` either way) — adds AWGN, then equalizes (``s_hat = s + n/h``)
    as a coherent receiver does, and re-stacks to the real layout. The flat-fading
    analogue of ``OFDMMultipathChannel``: obeys the ``(y, csi)`` contract,
    reporting the realized per-image effective SNR (dB), ``snr + 10*log10|h|^2``,
    as a scalar ``[B, 1]``.
    """

    def __init__(self, snr_db: float, fading_type: str = "rayleigh",
                 k_factor: float = 4.0, per_image_channel: bool = True) -> None:
        super().__init__()
        fading_type = fading_type.lower()
        if fading_type not in ("rayleigh", "rician"):
            raise ValueError(
                f"fading_type must be 'rayleigh' or 'rician' (got {fading_type!r})")
        self.snr_db = float(snr_db)
        self.fading_type = fading_type
        self.k_factor = float(k_factor)
        self.per_image_channel = bool(per_image_channel)

    def _sample_gain(self, B: int, device: torch.device) -> torch.Tensor:
        n = B if self.per_image_channel else 1
        scat = (torch.randn(n, 1, device=device)
                + 1j * torch.randn(n, 1, device=device)) / (2.0 ** 0.5)  # CN(0,1)
        if self.fading_type == "rayleigh":
            h = scat
        else:                                          # rician, E|h|^2 == 1
            k = self.k_factor
            h = (k / (k + 1.0)) ** 0.5 + (1.0 / (k + 1.0)) ** 0.5 * scat
        return h.expand(B, 1) if not self.per_image_channel else h     # [B, 1]

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        in_dtype = z.dtype
        zf = z.float()
        B, M, H, W = zf.shape
        if M % 2 != 0:
            raise ValueError(f"latent channels M must be even (got {M})")

        flat = zf.reshape(B, -1)
        half = flat.shape[1] // 2
        s = torch.complex(flat[:, :half], flat[:, half:2 * half])      # [B, half]

        h = self._sample_gain(B, zf.device)                            # [B, 1]
        es = s.abs().pow(2).mean().clamp_min(1e-12)
        n0 = es * (10.0 ** (-self.snr_db / 10.0))
        noise = (torch.sqrt(n0 / 2.0)
                 * (torch.randn(B, half, device=zf.device)
                    + 1j * torch.randn(B, half, device=zf.device)))
        y = h * s + noise

        h_safe = torch.where(h.abs() < 1e-6, torch.full_like(h, 1e-6), h)
        s_hat = y / h_safe                                             # coherent EQ

        out = torch.cat([s_hat.real, s_hat.imag], dim=1)
        csi = self.snr_db + 10.0 * torch.log10(h.abs().pow(2).clamp_min(1e-12))
        return out.reshape(B, M, H, W).to(in_dtype), csi               # csi: [B,1]


class OFDMMultipathChannel(nn.Module):
    """Frequency-selective multipath channel + per-subcarrier equalization.

    Args:
        snr_db: average SNR (dB), defined per complex symbol (signal power
            measured empirically, as in the OFDM reference), constant per call.
        taps: number of channel taps ``L`` (delay spread). 1 == flat fading.
        decay: exponential power-delay-profile decay constant.
        n_subcarriers: OFDM FFT size used to spread the frequency-selective
            profile across the latent stream (matches the link's ``fft_len``).
        equalizer: ``"zf"`` (zero-forcing) or ``"mmse"``.
        packet_len: complex symbols per packet, used to aggregate the returned
            CSI to per-packet granularity (the finest the spatial-CSI decoder and
            the real receiver can express).
        per_image_channel: sample an independent channel realization per batch
            element (block fading, one coherence block per image). If False, one
            shared realization for the whole batch.
    """

    def __init__(self, snr_db: float, taps: int = 8, decay: float = 4.0,
                 n_subcarriers: int = 64, equalizer: str = "zf",
                 packet_len: int = 960, per_image_channel: bool = True) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError(f"taps must be >= 1 (got {taps})")
        equalizer = equalizer.lower()
        if equalizer not in ("zf", "mmse"):
            raise ValueError(f"equalizer must be 'zf' or 'mmse' (got {equalizer!r})")
        if taps > n_subcarriers:
            raise ValueError(f"taps ({taps}) must be <= n_subcarriers ({n_subcarriers})")
        self.snr_db = float(snr_db)
        self.taps = int(taps)
        self.decay = float(decay)
        self.n_sub = int(n_subcarriers)
        self.equalizer = equalizer
        self.packet_len = int(packet_len)
        self.per_image_channel = bool(per_image_channel)

        # Exponential power-delay profile, normalized to unit total power so the
        # average subcarrier gain E[|H_k|^2] == 1 (no mean-SNR shift, only
        # frequency-selective variation around it).
        pdp = torch.exp(-torch.arange(self.taps).float() / self.decay)
        self.register_buffer("pdp", pdp / pdp.sum())

    def _sample_response(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample ``H_k`` per subcarrier: [n, n_sub] complex frequency response."""
        pdp = self.pdp.to(device)
        taps = (torch.sqrt(pdp / 2.0)
                * (torch.randn(n, self.taps, device=device)
                   + 1j * torch.randn(n, self.taps, device=device)))
        taps_zp = torch.cat(
            [taps, torch.zeros(n, self.n_sub - self.taps, device=device,
                               dtype=taps.dtype)], dim=-1)  # match complex dtype (MPS)
        return torch.fft.fft(taps_zp, dim=-1)  # [n, n_sub]

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y, csi)``: equalized latent (same shape/dtype as ``z``) and
        a per-packet effective-SNR vector ``[B, n_pkts]`` in dB."""
        in_dtype = z.dtype
        zf = z.float()
        B, M, H, W = zf.shape
        if M % 2 != 0:
            raise ValueError(f"latent channels M must be even (got {M})")

        flat = zf.reshape(B, -1)              # [B, M*H*W]
        half = flat.shape[1] // 2             # M/2 * H * W complex symbols
        s = torch.complex(flat[:, :half], flat[:, half:2 * half])  # [B, half]

        # Per-subcarrier gain for every symbol: symbol j sits on carrier j % n_sub,
        # so the frequency-selective profile tiles across the latent stream.
        n_real = B if self.per_image_channel else 1
        Hf = self._sample_response(n_real, zf.device)             # [n_real, n_sub]
        carrier = torch.arange(half, device=zf.device) % self.n_sub
        Hk = Hf[:, carrier]                                        # [n_real, half]
        if not self.per_image_channel:
            Hk = Hk.expand(B, -1)

        # AWGN with power set from the measured complex-symbol power (OFDM-ref
        # convention): N0 = Es * 10^(-snr/10).
        es = s.abs().pow(2).mean().clamp_min(1e-12)
        n0 = es * (10.0 ** (-self.snr_db / 10.0))
        noise = (torch.sqrt(n0 / 2.0)
                 * (torch.randn(B, half, device=zf.device)
                    + 1j * torch.randn(B, half, device=zf.device)))
        y = Hk * s + noise

        # Per-subcarrier equalization (what the GR receiver does).
        if self.equalizer == "zf":
            Hk_safe = torch.where(Hk.abs() < 1e-6, torch.full_like(Hk, 1e-6), Hk)
            s_hat = y / Hk_safe
        else:  # mmse
            s_hat = (Hk.conj() / (Hk.abs().pow(2) + n0 / es)) * y

        out = torch.cat([s_hat.real, s_hat.imag], dim=1)          # [B, 2*half]
        return out.reshape(B, M, H, W).to(in_dtype), self._csi(Hk, es, n0, half)

    def _csi(self, Hk: torch.Tensor, es: torch.Tensor, n0: torch.Tensor,
             half: int) -> torch.Tensor:
        """Per-packet effective SNR (dB) from the realized fade. After per-
        subcarrier equalization the post-EQ SINR is ``|H_k|^2 * Es / N0`` (same
        for ZF and MMSE), i.e. noise ``N0/|H_k|^2``; aggregate it per packet and
        invert. Matches the receiver's ``10*log10(1 / mean(sigma2/|H_k|^2))``."""
        per_sym_noise = n0 / Hk.abs().pow(2).clamp_min(1e-12)      # [B, half]
        n_pkts = (half + self.packet_len - 1) // self.packet_len
        pad = n_pkts * self.packet_len - half
        if pad > 0:                                  # zero-pad partial last packet
            per_sym_noise = F.pad(per_sym_noise, (0, pad), value=0.0)
        blocks = per_sym_noise.reshape(-1, n_pkts, self.packet_len)
        # Per-packet mean over the *valid* symbols (the last packet holds `pad`
        # fewer). Done as sum / count rather than NaN-padding + ``nanmean``, which
        # returns 0 for a partially-padded block on MPS (-> spurious +inf SNR).
        counts = torch.full((n_pkts,), float(self.packet_len), device=Hk.device)
        if pad > 0:
            counts[-1] = self.packet_len - pad
        pp_noise = blocks.sum(dim=2) / counts                     # [B, n_pkts]
        return 10.0 * torch.log10((es / pp_noise).clamp_min(1e-12))   # [B, n_pkts]
