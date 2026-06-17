from __future__ import annotations

from .atst import ATSTExtractor
from .base import EmbeddingExtractor, resolve_device
from .beats import BEATsExtractor
from .dac import DACExtractor
from .hubert import HuBERTExtractor
from .mert import MERTExtractor
from .mucodec import MuCodecExtractor
from .music2latent import Music2LatentsExtractor
from .same import SAMEExtractor
from .speechtokenizer import SpeechTokenizerExtractor
from .stable_audio_open import StableAudioOpenVAEExtractor
from .wavcube import WavCubeExtractor
from .xcodec import (
    XCodecExtractor,
    XCodecPostQuantizedExtractor,
    XCodecPreQuantizedExtractor,
    XCodecSSLLatentExtractor,
    XCodecVAELatentExtractor,
)

EXTRACTORS = {
    "atst": ATSTExtractor,
    "beats": BEATsExtractor,
    "dac": DACExtractor,
    "hubert": HuBERTExtractor,
    "mert": MERTExtractor,
    "mucodec": MuCodecExtractor,
    "music2latents": Music2LatentsExtractor,
    "same": SAMEExtractor,
    "speechtokenizer": SpeechTokenizerExtractor,
    "stable-audio-open-vae": StableAudioOpenVAEExtractor,
    "stable_audio_open_vae": StableAudioOpenVAEExtractor,
    "wavcube": WavCubeExtractor,
    "x-codec": XCodecExtractor,
    "xcodec": XCodecExtractor,
    "x-codec-pre-quantized": XCodecPreQuantizedExtractor,
    "xcodec-pre-quantized": XCodecPreQuantizedExtractor,
    "x-codec-concat-latent": XCodecPreQuantizedExtractor,
    "xcodec-concat-latent": XCodecPreQuantizedExtractor,
    "x-codec-post-quantized": XCodecPostQuantizedExtractor,
    "xcodec-post-quantized": XCodecPostQuantizedExtractor,
    "x-codec-quantized": XCodecPostQuantizedExtractor,
    "xcodec-quantized": XCodecPostQuantizedExtractor,
    "x-codec-ssl-latent": XCodecSSLLatentExtractor,
    "xcodec-ssl-latent": XCodecSSLLatentExtractor,
    "x-codec-ssl": XCodecSSLLatentExtractor,
    "xcodec-ssl": XCodecSSLLatentExtractor,
    "x-codec-vae-latent": XCodecVAELatentExtractor,
    "xcodec-vae-latent": XCodecVAELatentExtractor,
    "x-codec-vae": XCodecVAELatentExtractor,
    "xcodec-vae": XCodecVAELatentExtractor,
}

__all__ = [
    "ATSTExtractor",
    "BEATsExtractor",
    "DACExtractor",
    "EmbeddingExtractor",
    "EXTRACTORS",
    "HuBERTExtractor",
    "MERTExtractor",
    "MuCodecExtractor",
    "Music2LatentsExtractor",
    "SAMEExtractor",
    "SpeechTokenizerExtractor",
    "StableAudioOpenVAEExtractor",
    "WavCubeExtractor",
    "XCodecExtractor",
    "XCodecPostQuantizedExtractor",
    "XCodecPreQuantizedExtractor",
    "XCodecSSLLatentExtractor",
    "XCodecVAELatentExtractor",
    "resolve_device",
]
