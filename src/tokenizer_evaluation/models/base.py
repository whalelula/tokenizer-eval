from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


def resolve_device(device: str):
    import torch

    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Use --device cpu.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def should_use_half(device, dtype: str) -> bool:
    if dtype == "float16":
        return str(device).startswith("cuda")
    return False


class EmbeddingExtractor(Protocol):
    name: str
    embedding_dim: int | None

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        ...

    def extract_batch(self, audio_paths: list[str | Path]) -> np.ndarray:
        ...


class LoopingExtractor:
    name: str
    embedding_dim: int | None = None

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        raise NotImplementedError

    def extract_batch(self, audio_paths: list[str | Path]) -> np.ndarray:
        vectors = [self.extract_one(path) for path in audio_paths]
        return np.stack(vectors, axis=0).astype("float32")
