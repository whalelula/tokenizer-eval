from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from tokenizer_evaluation.models import EXTRACTORS


def build_extractor(
    model_key: str,
    model_config: dict,
    device: str,
    dtype: str,
    max_duration_seconds: float | None,
):
    if model_key not in EXTRACTORS:
        raise ValueError(f"Unknown model '{model_key}'. Available: {', '.join(EXTRACTORS)}")
    extractor_cls = EXTRACTORS[model_key]
    kwargs = dict(model_config)
    kwargs.pop("enabled", None)
    kwargs.setdefault("device", device)
    kwargs.setdefault("dtype", dtype)
    kwargs.setdefault("max_duration_seconds", max_duration_seconds)
    return extractor_cls(**kwargs)


def extract_embeddings(
    manifest: pd.DataFrame,
    extractor,
    output_npz: str | Path,
    batch_size: int = 1,
    overwrite: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    output_npz = Path(output_npz)
    metadata_path = output_npz.with_suffix(".metadata.csv")

    if output_npz.exists() and metadata_path.exists() and not overwrite:
        loaded = np.load(output_npz)
        return loaded["embeddings"], pd.read_csv(metadata_path)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    frame = manifest.reset_index(drop=True).copy()
    audio_paths = frame["audio_path"].tolist()

    vectors = []
    for batch in tqdm(list(_batches(audio_paths, batch_size)), desc=f"extract:{extractor.name}"):
        vectors.append(extractor.extract_batch(batch))

    embeddings = np.concatenate(vectors, axis=0).astype("float32")
    np.savez_compressed(output_npz, embeddings=embeddings)
    frame.to_csv(metadata_path, index=False)

    summary = {
        "model": extractor.name,
        "num_items": int(len(frame)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_file": str(output_npz),
        "metadata_file": str(metadata_path),
    }
    output_npz.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return embeddings, frame


def load_embedding_bundle(npz_path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    npz_path = Path(npz_path)
    embeddings = np.load(npz_path)["embeddings"]
    metadata = pd.read_csv(npz_path.with_suffix(".metadata.csv"))
    return embeddings, metadata


def _batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
