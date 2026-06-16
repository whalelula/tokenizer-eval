from __future__ import annotations

import sys
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


class BEATsExtractor(LoopingExtractor):
    """BEATs extractor based on the official microsoft/unilm BEATs.py code."""

    name = "beats"

    def __init__(
        self,
        checkpoint_path: str | Path,
        repo_path: str | Path | None = None,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
    ) -> None:
        import torch

        self.checkpoint_path = Path(checkpoint_path)
        self.repo_path = Path(repo_path) if repo_path is not None else None
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = 16000
        self._torch = torch

        self._prepare_import_path()
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError as exc:
            require_dependency(
                "BEATs",
                "clone https://github.com/microsoft/unilm and pass "
                "--repo-path path/to/unilm/beats, "
                "or install that folder on PYTHONPATH.",
            )
            raise exc

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(config)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval().to(self.device)

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.squeeze(0).unsqueeze(0).to(self.device)
        padding_mask = self._torch.zeros(audio.shape, dtype=self._torch.bool, device=self.device)

        with self._torch.inference_mode():
            features, _ = self.model.extract_features(audio, padding_mask=padding_mask)

        if features.ndim == 3:
            vector = pool_sequence_tensor(features.squeeze(0), pooling=self.pooling, time_axis=0)
        else:
            vector = features.squeeze(0)
        return tensor_to_numpy(vector)

    def _prepare_import_path(self) -> None:
        if self.repo_path is None:
            return
        path = str(self.repo_path.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
