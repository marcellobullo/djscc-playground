"""Resolve a model spec into a BaseCodec, and an aligner spec into a BaseAligner.

Resolution order for :func:`load_codec` (``model`` argument):

1. Ends with ``.pth``  -> legacy raw checkpoint. Routed to the model family's
   ``build(role, ckpt=..., **overrides)`` (defaults to custom DJSCC). Lets the
   checkpoints copied from djscc-demo run before they are exported to HF format.
2. Known alias        -> looked up in :mod:`jscc.registry` (e.g. ``conventional``).
3. Otherwise          -> a local folder or HuggingFace repo id, loaded with
   ``AutoModel.from_pretrained(..., trust_remote_code=True)``. The config carries
   ``codec_module``, whose ``ENCODER_CODEC`` / ``DECODER_CODEC`` classes wrap the
   returned ``nn.Module`` with the DSP glue.

The HF ``auto_map`` is the registry for neural models; we add nothing on top of it
except the tiny alias table for the non-neural baseline.
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from .base import BaseAligner, BaseCodec, Role
from .registry import resolve_alias

# Default model family used for bare ``.pth`` checkpoints.
_LEGACY_MODULE = "jscc.djscc.codec_djscc"
# Spatial-CSI ("no-band") variant, selected when the filename signals it.
_SPATIALCSI_MODULE = "jscc.djscc_spatialcsi.codec_djscc_spatialcsi"
# AD-JSCC (attention) variant, selected when the filename signals it.
_ADJSCC_MODULE = "jscc.adjscc.codec_adjscc"


def load_codec(model: str, role: str, *, device: str = "auto",
               packet_len: int = 960, aligner: Optional[BaseAligner] = None,
               **overrides) -> BaseCodec:
    if role not in (Role.ENCODER, Role.DECODER):
        raise ValueError(f"role must be 'encoder' or 'decoder', got {role!r}")

    # 1) legacy raw checkpoint -------------------------------------------------
    if model.endswith(".pth"):
        name = os.path.basename(model).lower()
        if any(t in name for t in ("no_band", "noband", "spatialcsi")):
            legacy = _SPATIALCSI_MODULE
        elif any(t in name for t in ("adjscc", "ad_jscc")):
            legacy = _ADJSCC_MODULE
        else:
            legacy = _LEGACY_MODULE
        mod = importlib.import_module(legacy)
        codec = mod.build(role, ckpt=model, device=device,
                          packet_len=packet_len, **overrides)

    else:
        spec = resolve_alias(model)
        # 2) registered alias --------------------------------------------------
        if spec is not None and spec.get("builtin"):
            mod = importlib.import_module(spec["module"])
            kwargs = dict(spec.get("kwargs", {}))
            kwargs.update(overrides)
            codec = mod.build(role, device=device, packet_len=packet_len, **kwargs)
        else:
            # 3) HuggingFace repo id or local folder ---------------------------
            path_or_id = spec["hf"] if spec else model
            codec = _load_hf(path_or_id, role, device=device,
                             packet_len=packet_len, **overrides)

    if aligner is not None:
        codec.set_aligner(aligner)
    return codec


def _load_hf(path_or_id: str, role: str, *, device: str, packet_len: int,
             **overrides) -> BaseCodec:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(path_or_id, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(path_or_id, trust_remote_code=True)

    codec_module = getattr(config, "codec_module", None)
    if codec_module is None:
        raise ValueError(
            f"{path_or_id}: config has no 'codec_module' field — cannot pick a "
            "BaseCodec adapter. Re-export with scripts/export_djscc_to_hf.py.")
    mod = importlib.import_module(codec_module)
    cls = mod.ENCODER_CODEC if role == Role.ENCODER else mod.DECODER_CODEC
    return cls(hf_model, config, device=device, packet_len=packet_len, **overrides)


def load_aligner(spec: Optional[str], tcn: int, h: int, w: int,
                 device: str = "cpu", n_samples: Optional[int] = None):
    """Load a semantic aligner from a ``.pth`` path, a folder/HF repo, or ``None``.

    Returns a :class:`jscc.base.BaseAligner` (or ``None``). See
    :mod:`jscc.aligners.aligner` for the resolution details.
    """
    if not spec:
        return None
    from .aligners.aligner import load_aligner as _impl
    return _impl(spec, tcn=tcn, h=h, w=w, device=device, n_samples=n_samples)
