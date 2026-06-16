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
    """MuCodec first post-quantizer continuous MuEncoder latent extractor.

    MuCodec's public helper exposes ``sound2code`` for compressed discrete
    tokens. For representation evaluation we read the MuEncoder embeddings
    returned by ``fetch_codes_batch`` and pass them through the first RVQ
    quantizer, preserving the post-quantization continuous representation.
    """

    name = "mucodec"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        layer_num: int = 7,
        pooling: str = "mean",
        device: str = "cuda",
        dtype: str = "auto",
        max_duration_seconds: float | None = 4.0,
        sample_rate: int = 48000,
        channels: int = 2,
        load_main_model: bool = False,
        chunk_seconds: float = 40.96,
        batch_size: int = 3,
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
        self.chunk_seconds = float(chunk_seconds)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("MuCodec batch_size must be positive.")

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
        latents = self._extract_quantized_latents(waveform)
        vector = pool_sequence_tensor(latents.squeeze(0), pooling=self.pooling, time_axis=-1)
        return tensor_to_numpy(vector)

    def _extract_quantized_latents(self, waveform):
        import torch

        if waveform.ndim == 2:
            audios = waveform.unsqueeze(0).to(self.device)
        elif waveform.ndim == 3:
            audios = waveform.to(self.device)
        else:
            raise ValueError(f"Expected MuCodec waveform with 2 or 3 dims, got {waveform.shape}.")

        with torch.inference_mode():
            audios = self.model.preprocess_audio(audios)
            audios = audios.squeeze(0)
            original_length = audios.shape[-1]
            min_samples = int(self.chunk_seconds * self.sample_rate)
            output_len = int(original_length / float(self.sample_rate) * 25) + 1

            while audios.shape[-1] < min_samples + 480:
                audios = torch.cat([audios, audios], dim=-1)

            chunk_count = audios.shape[-1] // min_samples + 1
            audios = torch.cat([audios, audios], dim=-1)
            audios = audios[:, : int(chunk_count * (min_samples + 480))]
            audio_input = (
                audios.reshape(self.channels, -1, min_samples + 480)
                .permute(1, 0, 2)
                .reshape(-1, self.channels, min_samples + 480)
            )

            latent_chunks = []
            for start in range(0, audio_input.shape[0], self.batch_size):
                _, embeds, _ = self.model.model.fetch_codes_batch(
                    audio_input[start : start + self.batch_size],
                    additional_feats=[],
                    layer=self._encoder_layer_index(),
                )
                if not embeds:
                    raise RuntimeError("MuCodec fetch_codes_batch returned no encoder embeddings.")
                pre_quantized = torch.cat(embeds, dim=1)
                latent_chunks.append(self._first_quantized_layer(pre_quantized))

            latents = torch.cat(latent_chunks, dim=0)
            latents = latents.permute(1, 0, 2).reshape(1, latents.shape[1], -1)
            return latents[:, :, :output_len].float()

    def _encoder_layer_index(self) -> int:
        return int(getattr(self.model, "layer_num", self.layer_num - 1))

    def _first_quantized_layer(self, pre_quantized):
        quantizer = getattr(self.model.model, "rvq_muencoder_emb", None)
        if quantizer is None:
            raise RuntimeError("MuCodec model does not expose rvq_muencoder_emb.")

        quantizers = getattr(quantizer, "quantizers", None)
        if quantizers is not None and len(quantizers) > 0:
            result = quantizers[0](pre_quantized)
            return result[0] if isinstance(result, tuple) else result

        try:
            result = quantizer(pre_quantized, n_quantizers=0)
        except TypeError as exc:
            raise RuntimeError(
                "Could not isolate MuCodec's first RVQ continuous layer from "
                "this checkpoint wrapper."
            ) from exc
        return result[0] if isinstance(result, tuple) else result

    def _prepare_import_path(self) -> None:
        if self.repo_path is None:
            return
        path = str(self.repo_path.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
