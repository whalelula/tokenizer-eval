from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
    mono: bool = True,
    max_duration_seconds: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Load an audio file as a tensor shaped [channels, samples]."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.float()

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sr is not None and sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr

    if max_duration_seconds is not None:
        max_samples = int(max_duration_seconds * sample_rate)
        waveform = waveform[:, :max_samples]

    return waveform.contiguous(), sample_rate


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Return [2, samples] audio for models that expect stereo input."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, samples], got {waveform.shape}.")
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    if waveform.shape[0] > 2:
        return waveform[:2]
    return waveform
