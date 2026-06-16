"""Pluggable semantic aligner: wraps the modules in :mod:`jscc.aligners.modules`
into the :class:`jscc.base.BaseAligner` contract and loads them by path / folder /
HuggingFace id.

The codec composes an aligner at the adapter layer (``codec.set_aligner(...)``), so
any model can be paired with any aligner. ``kind`` -> ``mode`` mapping:

    zeroshot -> "compressing"  (compress at TX, decompress at RX; changes dim)
    others   -> "residual"     (shape-preserving map applied at the RX)

Resolution of ``spec``:
  * a ``.pth`` file  -> kind inferred from the filename (legacy convention).
  * a folder / HF id -> ``aligner_config.json`` ({"kind": ..., "n_samples": ...})
                        if present, else filename inference on the contained .pth.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import torch

from ..base import BaseAligner
from . import modules as _m


class Aligner(BaseAligner):
    """Thin wrapper exposing a loaded aligner module through the BaseAligner API."""

    def __init__(self, module: torch.nn.Module, kind: str) -> None:
        self.module = module
        self.kind = kind
        self.mode = "compressing" if kind == "zeroshot" else "residual"
        self.stage = "both" if kind == "zeroshot" else "rx_latent"
        self.transmitted_dim = getattr(module, "transmitted_dim", None)

    def apply_rx(self, latent_4d):
        return self.module(latent_4d)

    def compress(self, flat_1xm):
        return self.module.compression(flat_1xm)

    def decompress(self, z):
        return self.module.decompression(z)


def _build_by_kind(kind: str, m: int, tcn: int, n_samples: Optional[int]) -> torch.nn.Module:
    if kind == "linear":
        return _m._LinearAlignment(size=m)
    if kind == "mlp":
        return _m._MLPAlignment(input_dim=m, hidden_dims=[m])
    if kind == "twoconv":
        return _m._TwoConvAlignment(in_channels=tcn, hidden_channels=tcn,
                                    out_channels=tcn, kernel_size=5)
    if kind == "conv":
        return _m._ConvolutionalAlignment(in_channels=tcn, out_channels=tcn,
                                          kernel_size=5)
    if kind == "zeroshot":
        if not n_samples:
            raise ValueError("zero-shot aligner needs n_samples (channel usage)")
        return _m._ZeroShotAlignment(
            F_tilde=torch.zeros(n_samples, m), G_tilde=torch.zeros(m, n_samples),
            G=torch.zeros(1, 1), L=torch.zeros(n_samples, n_samples),
            mean=torch.zeros(n_samples, 1))
    raise ValueError(f"unknown aligner kind {kind!r}")


def _from_folder(folder: str, tcn: int, h: int, w: int, device: str):
    m = tcn * h * w
    cfg_path = os.path.join(folder, "aligner_config.json")
    weights = sorted(glob.glob(os.path.join(folder, "*.pth")))
    if not weights:
        raise FileNotFoundError(f"no .pth weights in aligner folder {folder}")
    weight_path = weights[0]

    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))
        kind = cfg["kind"]
        module = _build_by_kind(kind, m, tcn, cfg.get("n_samples"))
        module.load_state_dict(torch.load(weight_path, map_location=device))
        module = module.to(device).eval()
        if kind == "zeroshot":
            module.transmitted_dim = int(module.F_tilde.shape[0])
    else:
        # fall back to the filename-inference factory
        module = _m.load_aligner(weight_path, tcn, h, w, device=device)
        kind = module.kind
    return Aligner(module, kind)


def load_aligner(spec: str, tcn: int, h: int, w: int,
                 device: str = "cpu", n_samples: Optional[int] = None) -> Aligner:
    # local .pth file -> legacy filename-inferred factory
    if os.path.isfile(spec) and spec.endswith(".pth"):
        module = _m.load_aligner(spec, tcn, h, w, device=device, n_samples=n_samples)
        return Aligner(module, module.kind)

    # local folder
    if os.path.isdir(spec):
        return _from_folder(spec, tcn, h, w, device)

    # otherwise: HuggingFace repo id
    from huggingface_hub import snapshot_download
    folder = snapshot_download(spec)
    return _from_folder(folder, tcn, h, w, device)
