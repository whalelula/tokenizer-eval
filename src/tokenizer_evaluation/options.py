from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedOption:
    key: str
    display_name: str
    aliases: tuple[str, ...] = ()


SUPPORTED_DATASETS: tuple[NamedOption, ...] = (
    NamedOption("nsynth", "NSynth"),
    NamedOption("medley-solo-db", "medley-solo-db", aliases=("medley-solos-db",)),
    NamedOption("mtg-jamendo", "MTG-Jamendo", aliases=("jamendo",)),
)

SUPPORTED_TOKENIZERS: tuple[NamedOption, ...] = (
    NamedOption("mert", "MERT"),
    NamedOption("hubert", "HuBERT"),
    NamedOption("beats", "BEATs"),
    NamedOption("atst", "ATST"),
    NamedOption(
        "stable-audio-open-vae",
        "stable audio open VAE (Stability-AI/stable-audio-tools)",
        aliases=("stable-audio-open", "stable-audio-tools-vae", "stable-audio-vae"),
    ),
    NamedOption("same", "SAME"),
    NamedOption("music2latents", "Music2Latents", aliases=("music-2-latents",)),
    NamedOption("wavcube", "WavCube"),
    NamedOption("speechtokenizer", "SpeechTokenizer", aliases=("speech-tokenizer",)),
    NamedOption("x-codec", "X-Codec", aliases=("xcodec",)),
    NamedOption("mucodec", "MuCodec", aliases=("mu-codec",)),
)


def normalize_dataset(value: str) -> NamedOption:
    return _normalize_option(value, SUPPORTED_DATASETS, "dataset")


def normalize_tokenizer(value: str) -> NamedOption:
    return _normalize_option(value, SUPPORTED_TOKENIZERS, "tokenizer")


def dataset_choices_help() -> str:
    return _choices_help(SUPPORTED_DATASETS)


def tokenizer_choices_help() -> str:
    return _choices_help(SUPPORTED_TOKENIZERS)


def _normalize_option(value: str, options: tuple[NamedOption, ...], label: str) -> NamedOption:
    lookup = {}
    for option in options:
        keys = (option.key, option.display_name, *option.aliases)
        for key in keys:
            lookup[_canonicalize(key)] = option

    normalized = _canonicalize(value)
    if normalized not in lookup:
        choices = _choices_help(options)
        raise ValueError(f"Unknown {label} '{value}'. Available options: {choices}")
    return lookup[normalized]


def _canonicalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _choices_help(options: tuple[NamedOption, ...]) -> str:
    return ", ".join(option.key for option in options)
