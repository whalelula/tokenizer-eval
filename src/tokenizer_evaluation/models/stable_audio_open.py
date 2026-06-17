from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class StableAudioOpenVAEExtractor(LoopingExtractor):
    """Stable Audio Open VAE/pretransform latent extractor.

    The official stable-audio-tools checkpoint exposes the autoencoder either
    directly or as a pretransform attached to the generation model.
    """

    name = "stable-audio-open-vae"

    def __init__(
        self,
        model_name: str = "stabilityai/stable-audio-open-1.0",
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int | None = None,
        channels: int | None = None,
        use_vae_checkpoint: bool = True,
        **encode_kwargs: Any,
    ) -> None:
        import torch

        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError as exc:
            require_dependency(
                "stable_audio_tools",
                "install the official Stability-AI/stable-audio-tools package.",
            )
            raise exc

        self.model_name = model_name
        self.pooling = pooling
        self.device = resolve_device(device)
        self.dtype = dtype
        self.max_duration_seconds = max_duration_seconds
        self.encode_kwargs = encode_kwargs
        self._torch = torch

        if use_vae_checkpoint:
            try:
                model, model_config = self._load_vae_checkpoint(model_name)
            except self._vae_checkpoint_missing_errors():
                model, model_config = get_pretrained_model(model_name)
        else:
            model, model_config = get_pretrained_model(model_name)
        self.model_config = model_config
        self.model = model
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.encoder = self._resolve_encoder(self.model)
        self.sample_rate = int(
            sample_rate or self._config_value(model_config, "sample_rate", 44100)
        )
        self.channels = int(
            channels
            or self._config_value(model_config, "audio_channels", 0)
            or self._config_value(model_config, "io_channels", 2)
        )

    def extract_one(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=False,
            max_duration_seconds=self.max_duration_seconds,
        )
        waveform = ensure_channel_count(waveform, self.channels).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            latents = self._encode(waveform)

        latents = self._unwrap_latents(latents).squeeze(0)
        vector = pool_sequence_tensor(latents, pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _encode(self, waveform):
        try:
            return self.encoder.encode(waveform, **self.encode_kwargs)
        except TypeError:
            return self.encoder.encode(waveform)

    def _resolve_encoder(self, model):
        candidates = [model]
        for attr in ("pretransform", "autoencoder", "vae"):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                candidates.append(candidate)
                nested = getattr(candidate, "model", None)
                if nested is not None:
                    candidates.append(nested)
        for candidate in candidates:
            if hasattr(candidate, "encode"):
                if hasattr(candidate, "eval"):
                    candidate.eval()
                if hasattr(candidate, "to"):
                    candidate.to(self.device)
                return candidate
        raise ValueError(
            "Could not find an encode-capable VAE/pretransform on the stable-audio-tools model."
        )

    def _load_vae_checkpoint(self, model_name: str):
        """Load Stable Audio Open's VAE-only checkpoint instead of the full diffusion model."""
        from huggingface_hub import hf_hub_download
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        config_path = hf_hub_download(
            model_name,
            filename="vae_model_config.json",
            repo_type="model",
        )
        checkpoint_path = hf_hub_download(
            model_name,
            filename="vae_model.ckpt",
            repo_type="model",
        )
        with Path(config_path).open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(checkpoint_path), strict=False)
        return model, model_config

    def _vae_checkpoint_missing_errors(self):
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        return (EntryNotFoundError, LocalEntryNotFoundError)

    def _config_value(self, config: dict[str, Any], key: str, default: int) -> int:
        if key in config:
            return int(config[key])
        model_config = config.get("model", {})
        if isinstance(model_config, dict) and key in model_config:
            return int(model_config[key])
        pretransform = model_config.get("pretransform") if isinstance(model_config, dict) else {}
        if isinstance(pretransform, dict) and key in pretransform:
            return int(pretransform[key])
        return default

    def _unwrap_latents(self, value):
        if isinstance(value, tuple):
            return value[0]
        if isinstance(value, dict):
            for key in ("latents", "z", "embeddings"):
                if key in value:
                    return value[key]
        return value
