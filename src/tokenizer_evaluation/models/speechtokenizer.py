from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from tokenizer_evaluation.audio import load_audio
from tokenizer_evaluation.models.base import (
    LoopingExtractor,
    pool_sequence_tensor,
    require_dependency,
    resolve_device,
    tensor_to_numpy,
)


class SpeechTokenizerExtractor(LoopingExtractor):
    """SpeechTokenizer extractor using the official load_from_checkpoint API.

    The default quantized representation is the first post-RVQ continuous
    feature layer, matching the other codec-style tokenizers.
    """

    name = "speechtokenizer"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        model_name: str = "fnlp/SpeechTokenizer",
        model_subdir: str = "speechtokenizer_hubert_avg",
        representation: str = "quantized",
        layers: Sequence[int] | None = None,
        n_q: int | None = None,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
    ) -> None:
        import torch

        try:
            from speechtokenizer import SpeechTokenizer
        except ImportError as exc:
            require_dependency(
                "speechtokenizer",
                "install the official ZhangXInFD/SpeechTokenizer package.",
            )
            raise exc

        self.representation = representation
        self.layers = tuple(layers) if layers is not None else None
        self.n_q = n_q
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self._torch = torch

        if checkpoint_path is None or config_path is None:
            checkpoint_path, config_path = self._download_default_checkpoint(
                model_name=model_name,
                model_subdir=model_subdir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
            )

        self.model = SpeechTokenizer.load_from_checkpoint(
            str(config_path),
            str(checkpoint_path),
        )
        self.model.eval().to(self.device)
        self.sample_rate = int(getattr(self.model, "sample_rate", 16000))

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            features = self._extract_features(audio)

        vector = pool_sequence_tensor(features.squeeze(0), pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _extract_features(self, audio):
        if self.representation == "codes":
            codes = self.model.encode(audio, n_q=self.n_q)
            if codes.ndim == 3:
                codes = codes.permute(1, 0, 2)
            return codes.float()

        layers = self.layers
        if layers is None:
            layers = (0,)

        if self.representation == "quantized":
            quantized = self.model.forward_feature(audio, layers=list(layers))
            if isinstance(quantized, (list, tuple)):
                quantized = self._torch.cat(list(quantized), dim=1)
            return quantized

        if self.representation == "semantic":
            _, _, semantic = self.model(audio, n_q=self.n_q, layers=[int(layers[0])])
            return semantic.transpose(1, 2)

        raise ValueError(
            "SpeechTokenizer representation must be 'quantized', 'semantic', or 'codes'."
        )

    def _download_default_checkpoint(
        self,
        *,
        model_name: str,
        model_subdir: str,
        checkpoint_path: str | Path | None,
        config_path: str | Path | None,
    ) -> tuple[Path, Path]:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            require_dependency(
                "huggingface_hub",
                "install huggingface_hub or pass --config-path and --checkpoint-path explicitly.",
            )
            raise exc

        root = Path(snapshot_download(model_name))
        default_dir = root / model_subdir
        checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else default_dir / "SpeechTokenizer.pt"
        )
        config = Path(config_path) if config_path is not None else default_dir / "config.json"
        return checkpoint, config
