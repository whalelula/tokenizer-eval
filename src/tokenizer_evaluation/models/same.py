from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer_evaluation.audio import ensure_stereo, load_audio
from tokenizer_evaluation.models.base import LoopingExtractor, resolve_device


class SAMEExtractor(LoopingExtractor):
    """SAME latent extractor based on stable-audio-3 AutoencoderModel.encode."""

    name = "same"

    def __init__(
        self,
        model_name: str = "same-s",
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 32,
        chunk_size_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> None:
        import torch
        from stable_audio_3 import AutoencoderModel

        self.model_name = model_name
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.chunked = chunked
        self.chunk_size = int(chunk_size_seconds) if chunk_size_seconds is not None else chunk_size
        self.overlap = int(overlap_seconds) if overlap_seconds is not None else overlap

        self.model = AutoencoderModel.from_pretrained(model_name)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self._torch = torch

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, sample_rate = load_audio(
            audio_path,
            target_sr=None,
            mono=False,
            max_duration_seconds=self.max_duration_seconds,
        )
        waveform = ensure_stereo(waveform).to(self.device)

        with self._torch.inference_mode():
            latents = self._encode(waveform, sample_rate)

        vector = self._pool(latents)
        return vector.detach().float().cpu().numpy().reshape(-1)

    def _encode(self, waveform, sample_rate: int):
        if not self.chunked:
            return self.model.encode(waveform, sample_rate)

        try:
            return self.model.encode(
                waveform,
                sample_rate,
                chunked=True,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
        except TypeError:
            return self.model.encode(waveform, sample_rate)

    def _pool(self, latents):
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3:
            raise ValueError(f"Expected SAME latents shaped [batch, channels, frames], got {latents.shape}.")

        if self.pooling == "mean":
            return latents.mean(dim=-1).squeeze(0)
        if self.pooling == "mean_std":
            mean = latents.mean(dim=-1).squeeze(0)
            std = latents.std(dim=-1).squeeze(0)
            return self._torch.cat([mean, std], dim=0)
        if self.pooling == "flatten":
            return latents.reshape(latents.shape[0], -1).squeeze(0)
        raise ValueError(f"Unknown SAME pooling mode: {self.pooling}")
