from __future__ import annotations

import sys
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


class MuCodecExtractor(LoopingExtractor):
    """MuCodec code extractor using the official generate.MuCodec helper."""

    name = "mucodec"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        layer_num: int = 1,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int = 48000,
        channels: int = 2,
        load_main_model: bool = False,
    ) -> None:
        selected_model_path = model_path or checkpoint_path
        self.model_path = Path(selected_model_path) if selected_model_path else None
        if self.model_path is None:
            raise ValueError("MuCodecExtractor requires --checkpoint-path or --model-path.")
        self.repo_path = Path(repo_path) if repo_path is not None else None
        self.layer_num = int(layer_num)
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

        self._prepare_import_path()
        try:
            from generate import MuCodec
        except ImportError as exc:
            require_dependency(
                "MuCodec",
                "clone https://github.com/tencent-ailab/MuCodec and pass "
                "--repo-path path/to/MuCodec.",
            )
            raise exc

        self.model = MuCodec(
            str(self.model_path),
            layer_num=self.layer_num,
            load_main_model=load_main_model,
            device=str(self.device),
        )

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=False,
            max_duration_seconds=self.max_duration_seconds,
        )
        waveform = ensure_channel_count(waveform, self.channels)
        codes = self.model.sound2code(waveform)
        vector = pool_sequence_tensor(codes.float().squeeze(0), pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _prepare_import_path(self) -> None:
        if self.repo_path is None:
            return
        path = str(self.repo_path.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
