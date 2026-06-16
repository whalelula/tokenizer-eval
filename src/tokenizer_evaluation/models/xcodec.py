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


class XCodecExtractor(LoopingExtractor):
    """X-Codec extractor using the Hugging Face/official model wrapper."""

    name = "x-codec"

    def __init__(
        self,
        model_name: str = "hf-audio/xcodec-hubert-general",
        representation: str = "pre_quantized",
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        trust_remote_code: bool = False,
    ) -> None:
        import torch

        try:
            from transformers import AutoFeatureExtractor, XcodecModel
        except ImportError as exc:
            require_dependency(
                "transformers.XcodecModel",
                "install a recent transformers release with X-Codec support.",
            )
            raise exc

        self.model_name = model_name
        self.representation = representation
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self._torch = torch

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.sample_rate = int(getattr(self.processor, "sampling_rate", 16000))
        self.model = XcodecModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval().to(self.device)

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            max_duration_seconds=self.max_duration_seconds,
        )
        audio = waveform.squeeze(0).cpu().numpy()
        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        with self._torch.inference_mode():
            features = self._extract_features(input_values)

        vector = pool_sequence_tensor(features.squeeze(0), pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _extract_features(self, input_values):
        if self.representation == "codes":
            encoded = self.model.encode(input_values)
            fallback_codes = encoded[0] if isinstance(encoded, tuple) else encoded
            codes = getattr(encoded, "audio_codes", fallback_codes)
            return codes.float()

        if self.representation != "pre_quantized":
            raise ValueError("X-Codec representation must be 'pre_quantized' or 'codes'.")

        try:
            return self._pre_quantized_features(input_values)
        except AttributeError:
            encoded = self.model.encode(input_values)
            fallback_codes = encoded[0] if isinstance(encoded, tuple) else encoded
            codes = getattr(encoded, "audio_codes", fallback_codes)
            return codes.float()

    def _pre_quantized_features(self, input_values):
        import torch.nn.functional as functional

        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)

        semantic_input = self.model._extract_semantic_features(input_values).detach()
        semantic = self.model.encoder_semantic(semantic_input.transpose(1, 2))
        acoustic = self.model.acoustic_encoder(input_values)

        if acoustic.shape[-1] != semantic.shape[-1]:
            diff = acoustic.shape[-1] - semantic.shape[-1]
            if diff > 0:
                semantic = functional.pad(semantic, (0, diff))
            else:
                acoustic = functional.pad(acoustic, (0, -diff))

        return self.model.fc(self._torch.cat([acoustic, semantic], dim=1))
