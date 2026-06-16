from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer_evaluation.audio import load_audio
from tokenizer_evaluation.models.base import LoopingExtractor, resolve_device, tensor_to_numpy


class HuBERTExtractor(LoopingExtractor):
    """HuBERT hidden-state extractor.

    The original HuBERT implementation lives in fairseq. This wrapper uses the
    Hugging Face checkpoint API so it can share the same extraction path as MERT.
    """

    name = "hubert"

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        layer: int = -1,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name
        self.layer = layer
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.sample_rate = int(getattr(self.processor, "sampling_rate", 16000))
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval().to(self.device)
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

        if outputs.hidden_states is not None:
            hidden = outputs.hidden_states[self.layer]
        else:
            hidden = outputs.last_hidden_state
        vector = self._pool(hidden)
        return tensor_to_numpy(vector)

    def _pool(self, hidden):
        if self.pooling == "mean":
            return hidden.mean(dim=1).squeeze(0)
        if self.pooling == "mean_std":
            mean = hidden.mean(dim=1).squeeze(0)
            std = hidden.std(dim=1).squeeze(0)
            return self._torch.cat([mean, std], dim=-1)
        if self.pooling == "first":
            return hidden[:, 0, :].squeeze(0)
        if self.pooling == "flatten":
            return hidden.reshape(-1)
        raise ValueError(f"Unknown HuBERT pooling mode: {self.pooling}")
