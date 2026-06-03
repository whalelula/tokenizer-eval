from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer_evaluation.audio import load_audio
from tokenizer_evaluation.models.base import LoopingExtractor, resolve_device, should_use_half


class MERTExtractor(LoopingExtractor):
    """MERT hidden-state extractor based on the official Hugging Face usage."""

    name = "mert"

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        layer: int = -1,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.model_name = model_name
        self.layer = layer
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.sample_rate = int(self.processor.sampling_rate)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval().to(self.device)
        if should_use_half(self.device, dtype):
            self.model.half()
        self._torch = torch

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.squeeze(0).cpu().numpy()
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        selected = hidden_states[self.layer]
        vector = self._pool(selected)
        return vector.detach().float().cpu().numpy().reshape(-1)

    def _pool(self, hidden):
        if self.pooling == "mean":
            return hidden.mean(dim=1).squeeze(0)
        if self.pooling == "mean_std":
            mean = hidden.mean(dim=1).squeeze(0)
            std = hidden.std(dim=1).squeeze(0)
            return self._torch.cat([mean, std], dim=-1)
        if self.pooling == "first":
            return hidden[:, 0, :].squeeze(0)
        raise ValueError(f"Unknown MERT pooling mode: {self.pooling}")
