"""Pluggable semantic aligners (residual + zero-shot)."""

from .aligner import Aligner, load_aligner

__all__ = ["Aligner", "load_aligner"]
