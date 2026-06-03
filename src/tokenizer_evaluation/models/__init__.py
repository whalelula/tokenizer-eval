from __future__ import annotations

from .base import EmbeddingExtractor, resolve_device
from .mert import MERTExtractor
from .same import SAMEExtractor

EXTRACTORS = {
    "mert": MERTExtractor,
    "same": SAMEExtractor,
}

__all__ = ["EmbeddingExtractor", "EXTRACTORS", "MERTExtractor", "SAMEExtractor", "resolve_device"]
