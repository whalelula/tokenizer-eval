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


class WavCubeExtractor(LoopingExtractor):
    """WavCube feature extractor based on the official Vocos feature_extractor.infer path."""

    name = "wavcube"

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        repo_path: str | Path | None = None,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int = 16000,
    ) -> None:
        import torch

        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.repo_path = Path(repo_path) if repo_path is not None else None
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = int(sample_rate)
        self._torch = torch

        self._prepare_import_path()
        try:
            from vocos import Vocos
        except ImportError as exc:
            require_dependency(
                "vocos from WavCube",
                "clone https://github.com/yanghaha0908/WavCube and pass "
                "--repo-path path/to/WavCube.",
            )
            raise exc

        self.model = Vocos.from_config(str(self.config_path))
        state = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        self.model.load_state_dict(state_dict, strict=False)
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
            features = self.model.feature_extractor.infer(audio).squeeze(0)
        vector = pool_sequence_tensor(features, pooling=self.pooling, time_axis=0)
        return tensor_to_numpy(vector)

    def _prepare_import_path(self) -> None:
        if self.repo_path is None:
            return
        path = str(self.repo_path.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
