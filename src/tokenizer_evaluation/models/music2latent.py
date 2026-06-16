from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer_evaluation.audio import load_audio
from tokenizer_evaluation.models.base import (
    LoopingExtractor,
    ensure_channel_count,
    pool_sequence_tensor,
    require_dependency,
    resolve_device,
    tensor_to_numpy,
)


class Music2LatentsExtractor(LoopingExtractor):
    """Music2Latents extractor using SonyCSLParis/music2latent EncoderDecoder."""

    name = "music2latents"

    def __init__(
        self,
        model_name: str | None = None,
        load_path_inference: str | Path | None = None,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int = 44100,
        channels: int = 2,
        extract_features: bool = True,
    ) -> None:
        try:
            from music2latent import EncoderDecoder
        except ImportError as exc:
            require_dependency(
                "music2latent",
                "install the official SonyCSLParis/music2latent package.",
            )
            raise exc

        self.model_name = model_name
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.extract_features = extract_features

        kwargs = {"device": str(self.device)}
        if load_path_inference is not None:
            kwargs["load_path_inference"] = str(load_path_inference)
        elif model_name is not None:
            kwargs["load_path_inference"] = str(model_name)
        self.model = EncoderDecoder(**kwargs)

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=False,
            max_duration_seconds=self.max_duration_seconds,
        )
        waveform = ensure_channel_count(waveform, self.channels)
        audio = waveform.cpu().numpy()
        latents = self.model.encode(audio, extract_features=self.extract_features)
        vector = pool_sequence_tensor(latents, pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)
