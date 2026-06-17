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
    """X-Codec extractor using the Hugging Face/official model wrapper.

    The default representation is the first post-quantizer continuous codebook
    embedding rather than pre-quantized encoder states or discrete code IDs.
    SSL and VAE branch latents are also exposed for representation evaluation
    before the two branches are projected into the shared quantizer input.
    """

    name = "x-codec"

    def __init__(
        self,
        model_name: str = "hf-audio/xcodec-hubert-general",
        representation: str = "quantized",
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

        if self.representation == "pre_quantized":
            return self._pre_quantized_features(input_values)

        if self.representation == "ssl_latent":
            return self._ssl_encoder_last_hidden_state(input_values)

        if self.representation == "vae_latent":
            return self._vae_encoder_last_hidden_state(input_values)

        if self.representation == "quantized":
            return self._quantized_features(input_values)

        raise ValueError(
            "X-Codec representation must be 'quantized', 'pre_quantized', "
            "'ssl_latent', 'vae_latent', or 'codes'."
        )

    def _quantized_features(self, input_values):
        pre_quantized = self._pre_quantized_features(input_values)
        quantizer = getattr(self.model, "quantizer", None)
        if quantizer is None:
            raise RuntimeError("X-Codec model does not expose a quantizer module.")

        quantized = self._decode_first_quantizer(quantizer, pre_quantized)
        if quantized is None:
            raise RuntimeError(
                "Could not extract the first continuous post-quantizer layer from "
                "this X-Codec model. Use --representation pre_quantized or "
                "--representation codes for an explicit fallback."
            )
        return quantized

    def _pre_quantized_features(self, input_values):
        import torch.nn.functional as functional

        input_values = self._ensure_channel_axis(input_values)

        semantic_input = self.model._extract_semantic_features(input_values).detach()
        semantic = self.model.encoder_semantic(semantic_input.transpose(1, 2))
        acoustic = self.model.acoustic_encoder(input_values)

        if acoustic.shape[-1] != semantic.shape[-1]:
            diff = acoustic.shape[-1] - semantic.shape[-1]
            if diff > 0:
                semantic = functional.pad(semantic, (0, diff))
            else:
                acoustic = functional.pad(acoustic, (0, -diff))

        features = self._torch.cat([acoustic, semantic], dim=1)
        try:
            return self.model.fc(features.transpose(1, 2)).transpose(1, 2)
        except RuntimeError:
            return self.model.fc(features)

    def _ssl_encoder_last_hidden_state(self, input_values):
        import torch.nn.functional as functional

        input_values = self._ensure_channel_axis(input_values)
        semantic_input = input_values[:, 0, :]
        pad = int(getattr(self.model, "pad", 0))
        if pad:
            semantic_input = functional.pad(semantic_input, (pad, pad))

        semantic_outputs = self.model.semantic_model(
            semantic_input,
            output_hidden_states=True,
        )
        hidden_states = getattr(semantic_outputs, "hidden_states", None)
        if hidden_states:
            last_hidden_state = hidden_states[-1]
        else:
            last_hidden_state = getattr(semantic_outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("X-Codec semantic_model did not return hidden states.")
        if last_hidden_state.ndim != 3:
            raise RuntimeError(
                "Expected X-Codec SSL latent with shape [batch, time, channels], "
                f"got {last_hidden_state.shape}."
            )
        return last_hidden_state.transpose(1, 2).float()

    def _vae_encoder_last_hidden_state(self, input_values):
        input_values = self._ensure_channel_axis(input_values)
        return self.model.acoustic_encoder(input_values).float()

    def _ensure_channel_axis(self, input_values):
        if input_values.ndim == 2:
            return input_values.unsqueeze(1)
        if input_values.ndim != 3:
            raise ValueError(f"Expected X-Codec input with 2 or 3 dims, got {input_values.shape}.")
        return input_values

    def _decode_first_quantizer(self, quantizer, pre_quantized):
        quantizers = getattr(quantizer, "quantizers", None)
        if quantizers is not None and len(quantizers) > 0:
            first = quantizers[0]
            if hasattr(first, "encode") and hasattr(first, "decode"):
                codes = first.encode(pre_quantized)
                return first.decode(codes)
            result = first(pre_quantized)
            return result[0] if isinstance(result, tuple) else result

        if hasattr(quantizer, "encode") and hasattr(quantizer, "decode"):
            codes = quantizer.encode(pre_quantized)
            if codes.ndim == 3:
                codes = codes[:1]
            return quantizer.decode(codes)

        return None


class XCodecSSLLatentExtractor(XCodecExtractor):
    """X-Codec SSL encoder last-layer latent before quantization."""

    name = "x-codec-ssl-latent"

    def __init__(self, *args, representation: str = "ssl_latent", **kwargs) -> None:
        super().__init__(*args, representation=representation, **kwargs)


class XCodecVAELatentExtractor(XCodecExtractor):
    """X-Codec VAE/acoustic encoder last-layer latent before quantization."""

    name = "x-codec-vae-latent"

    def __init__(self, *args, representation: str = "vae_latent", **kwargs) -> None:
        super().__init__(*args, representation=representation, **kwargs)


class XCodecPreQuantizedExtractor(XCodecExtractor):
    """X-Codec fused acoustic+semantic latent before quantization."""

    name = "x-codec-pre-quantized"

    def __init__(self, *args, representation: str = "pre_quantized", **kwargs) -> None:
        super().__init__(*args, representation=representation, **kwargs)


class XCodecPostQuantizedExtractor(XCodecExtractor):
    """X-Codec first continuous post-quantizer latent."""

    name = "x-codec-post-quantized"

    def __init__(self, *args, representation: str = "quantized", **kwargs) -> None:
        super().__init__(*args, representation=representation, **kwargs)
