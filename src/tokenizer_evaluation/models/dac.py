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


class DACExtractor(LoopingExtractor):
    """Descript Audio Codec extractor.

    The default representation is DAC's first post-RVQ continuous layer:
    ``model.encode(..., n_quantizers=1)`` returns ``z`` after the first
    quantizer rather than pre-quantized latents or discrete code IDs.
    """

    name = "dac"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_path: str | Path | None = None,
        model_name: str = "44khz",
        model_bitrate: str = "8kbps",
        tag: str = "latest",
        representation: str = "quantized",
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int | None = None,
    ) -> None:
        import torch

        try:
            import dac
        except ImportError as exc:
            require_dependency(
                "descript-audio-codec",
                "install descript-audio-codec or pip install -e '.[dac]'.",
            )
            raise exc

        self.representation = representation
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self._torch = torch

        selected_model_path = model_path or checkpoint_path
        if selected_model_path is not None:
            load_path = Path(selected_model_path)
        else:
            load_path = dac.utils.download(
                model_type=model_name,
                model_bitrate=model_bitrate,
                tag=tag,
            )

        self.model = dac.DAC.load(load_path)
        self.model.eval().to(self.device)
        self.sample_rate = int(sample_rate or getattr(self.model, "sample_rate", 44100))

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, sample_rate = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            audio = self.model.preprocess(audio, sample_rate)
            features = self._extract_features(audio)

        vector = pool_sequence_tensor(features.squeeze(0), pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _extract_features(self, audio):
        if self.representation == "quantized":
            z, _, _, _, _ = self.model.encode(audio, n_quantizers=1)
            return z

        _, codes, latents, _, _ = self.model.encode(audio)
        if self.representation == "pre_quantized":
            return latents
        if self.representation == "codes":
            return codes.float()

        raise ValueError("DAC representation must be 'quantized', 'pre_quantized', or 'codes'.")
