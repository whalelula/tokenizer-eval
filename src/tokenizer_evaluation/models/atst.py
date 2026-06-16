from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer_evaluation.audio import load_audio
from tokenizer_evaluation.models.base import (
    LoopingExtractor,
    pool_sequence_tensor,
    require_dependency,
    resolve_device,
    tensor_to_numpy,
)


class ATSTExtractor(LoopingExtractor):
    """ATST/ATST-Frame extractor using Audio-WestlakeU/audiossl embedding helpers."""

    name = "atst"

    def __init__(
        self,
        checkpoint_path: str | Path,
        representation: str = "scene",
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
    ) -> None:
        import torch

        try:
            from audiossl.methods.atstframe.embedding import (
                get_scene_embedding,
                get_timestamp_embedding,
                load_model,
            )
        except ImportError as exc:
            require_dependency(
                "audiossl",
                "install the official Audio-WestlakeU/audiossl package, "
                "then pass an ATST checkpoint.",
            )
            raise exc

        self.checkpoint_path = Path(checkpoint_path)
        self.representation = representation
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = 16000
        self._torch = torch
        self._get_scene_embedding = get_scene_embedding
        self._get_timestamp_embedding = get_timestamp_embedding

        self.model = load_model(str(self.checkpoint_path))
        self.model.eval().to(self.device)

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.to(self.device)

        with self._torch.inference_mode():
            if self.representation == "scene":
                features = self._get_scene_embedding(audio, self.model).squeeze(0)
            elif self.representation in {"timestamp", "frame"}:
                features = self._get_timestamp_embedding(audio, self.model).squeeze(0)
                features = pool_sequence_tensor(features, pooling=self.pooling, time_axis=0)
            else:
                raise ValueError("ATST representation must be 'scene' or 'timestamp'.")

        return tensor_to_numpy(features)
