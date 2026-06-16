"""Short-name aliases for codecs that are not HuggingFace repos.

This is intentionally tiny — it is NOT a competing model registry. Neural models
are resolved through HuggingFace ``auto_map`` (see :func:`jscc.loader.load_codec`).
The registry exists for:

* the conventional SSCC baseline, which is not a neural network and therefore
  cannot live in an HF repo, and
* convenience aliases that point at a bundled checkpoint or a canonical HF id.

An alias maps to one of:
  * ``{"builtin": True, "module": "<dotted path>", "kwargs": {...}}``  — a module
    exposing ``build(role, **kwargs)`` that returns a BaseCodec, or
  * ``{"hf": "<repo id or local folder>"}``                          — resolved
    via AutoModel.
"""

from __future__ import annotations

from typing import Any, Dict

_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Classical separated source-channel coding baseline (bytes over the air).
    "conventional": {
        "builtin": True,
        "module": "jscc.conventional.codec_conventional",
        "kwargs": {},
    },
    # Convenience alias for the bundled custom-DJSCC checkpoint (legacy .pth).
    # Override hyper-params (comp_ratio, resolution) from the CLI as needed.
    "djscc-r6": {
        "builtin": True,
        "module": "jscc.djscc.codec_djscc",
        "kwargs": {"ckpt": "checkpoints/custom_djscc/compratio-6_latest.pth",
                   "comp_ratio": 6},
    },
}


def register(name: str, spec: Dict[str, Any]) -> None:
    _REGISTRY[name] = spec


def resolve_alias(name: str):
    """Return the spec dict for a known alias, else ``None`` (treat as path/HF id)."""
    return _REGISTRY.get(name)


def aliases():
    return sorted(_REGISTRY)
