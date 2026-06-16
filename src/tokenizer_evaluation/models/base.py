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
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false. Use --device cpu."
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def should_use_half(device, dtype: str) -> bool:
    if dtype == "float16":
        return str(device).startswith("cuda")
    return False


def ensure_channel_count(waveform, channels: int):
    """Return audio with the requested number of channels."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, samples], got {waveform.shape}.")
    if channels <= 0:
        raise ValueError("channels must be positive.")
    if waveform.shape[0] == channels:
        return waveform
    if channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    if waveform.shape[0] == 1:
        return waveform.repeat(channels, 1)
    if waveform.shape[0] > channels:
        return waveform[:channels]

    repeats = channels - waveform.shape[0]
    return torch_cat([waveform, waveform[-1:].repeat(repeats, 1)], dim=0)


def pool_sequence_tensor(values, pooling: str = "mean", time_axis: int = -1):
    """Pool a tensor or ndarray over time, then flatten non-batch dimensions."""
    import torch

    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.ndim == 0:
        raise ValueError("Cannot pool a scalar latent.")
    if tensor.ndim == 1:
        return tensor

    axis = time_axis + tensor.ndim if time_axis < 0 else time_axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"time_axis {time_axis} is out of bounds for latent shape {tensor.shape}.")

    if pooling == "flatten":
        return tensor.reshape(-1)
    if pooling == "mean":
        pooled = tensor.mean(dim=axis)
    elif pooling == "mean_std":
        pooled = torch_cat([tensor.mean(dim=axis), tensor.std(dim=axis, unbiased=False)], dim=-1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")
    return pooled.reshape(-1)


def tensor_to_numpy(values) -> np.ndarray:
    import torch

    if torch.is_tensor(values):
        return values.detach().float().cpu().numpy().reshape(-1).astype("float32")
    return np.asarray(values, dtype="float32").reshape(-1)


def torch_cat(values, dim: int):
    import torch

    return torch.cat(values, dim=dim)


def require_dependency(package: str, install_hint: str):
    raise ImportError(
        f"Missing dependency '{package}'. Install it before using this extractor. "
        f"Hint: {install_hint}"
    )


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
