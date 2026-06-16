from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tokenizer_evaluation.datasets.download import download_file, extract_archive


MEDLEY_METADATA_URL = (
    "https://zenodo.org/record/3464194/files/Medley-solos-DB_metadata.csv?download=1"
)
MEDLEY_AUDIO_URL = "https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz?download=1"


@dataclass(frozen=True)
class MedleySolosPrepareConfig:
    root: Path
    manifest_path: Path = Path("outputs/medley_solos_manifest.csv")
    subsets: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    max_per_instrument: int | None = None
    seed: int = 42
    download_metadata: bool = True
    download_audio: bool = False
    require_audio: bool = True


def prepare_medley_solos_manifest(config: MedleySolosPrepareConfig) -> pd.DataFrame:
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = _ensure_metadata(root, download=config.download_metadata)
    if config.download_audio:
        _ensure_audio(root)

    frame = load_medley_solos_metadata(metadata_path, root=root, require_audio=config.require_audio)
    frame = filter_medley_solos_manifest(
        frame,
        subsets=config.subsets,
        instruments=config.instruments,
    )
    frame = _balanced_sample(
        frame,
        label_col="instrument",
        max_per_label=config.max_per_instrument,
        seed=config.seed,
    )

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.manifest_path, index=False)
    return frame


def load_medley_solos_metadata(
    metadata_path: str | Path,
    *,
    root: str | Path,
    require_audio: bool = True,
) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    root = Path(root)
    raw = pd.read_csv(metadata_path)
    rows = []
    for _, item in raw.iterrows():
        subset = _first_present(item, ("subset", "split"))
        instrument = _first_present(item, ("instrument", "instrument_name"))
        instrument_id = _first_present(item, ("instrument_id", "instrumentID", "inst_id"))
        uuid = _first_present(item, ("uuid4", "uuid", "track_id", "id"))
        audio_path = _resolve_audio_path(root, item, subset, instrument_id, uuid)
        rows.append(
            {
                "id": str(uuid),
                "audio_path": str(audio_path.resolve()),
                "dataset": "medley-solo-db",
                "subset": _as_str(subset),
                "instrument": _as_str(instrument),
                "instrument_id": _as_str(instrument_id),
                "uuid": _as_str(uuid),
            }
        )

    frame = pd.DataFrame(rows)
    if require_audio:
        _validate_audio_paths(frame)
    return frame


def filter_medley_solos_manifest(
    frame: pd.DataFrame,
    *,
    subsets: tuple[str, ...] = (),
    instruments: tuple[str, ...] = (),
) -> pd.DataFrame:
    filtered = frame.copy()
    if subsets:
        wanted = {value.lower() for value in subsets}
        filtered = filtered[filtered["subset"].str.lower().isin(wanted)]
    if instruments:
        wanted = {value.lower() for value in instruments}
        filtered = filtered[filtered["instrument"].str.lower().isin(wanted)]
    return filtered.reset_index(drop=True)


def _ensure_metadata(root: Path, *, download: bool) -> Path:
    candidates = [
        root / "Medley-solos-DB_metadata.csv",
        root / "metadata" / "Medley-solos-DB_metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not download:
        raise FileNotFoundError(
            "Could not find Medley-solos-DB metadata. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    return download_file(MEDLEY_METADATA_URL, candidates[0])


def _ensure_audio(root: Path) -> None:
    if _looks_like_audio_root(root):
        return
    archive_path = root / "Medley-solos-DB.tar.gz"
    if not archive_path.exists():
        download_file(MEDLEY_AUDIO_URL, archive_path)
    extract_archive(archive_path, root)
    if not _looks_like_audio_root(root):
        raise FileNotFoundError(
            "Downloaded Medley-solos-DB archive did not create any .wav files under "
            f"{root}. You may need to inspect/extract {archive_path} manually."
        )


def _looks_like_audio_root(root: Path) -> bool:
    return any(root.glob("**/*.wav"))


def _resolve_audio_path(
    root: Path,
    item: pd.Series,
    subset: object,
    instrument_id: object,
    uuid: object,
) -> Path:
    for column in ("audio_path", "path", "filename", "fname"):
        if column in item and pd.notna(item[column]):
            candidate = Path(str(item[column]))
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                return candidate

    expected_names = [
        f"Medley-solos-DB_{_as_str(subset)}-{_as_str(instrument_id)}_{_as_str(uuid)}.wav",
        f"{_as_str(uuid)}.wav",
    ]
    for expected_name in expected_names:
        matches = list(root.glob(f"**/{expected_name}"))
        if matches:
            return matches[0]

    return root / "audio" / expected_names[0]


def _first_present(item: pd.Series, names: tuple[str, ...]) -> object:
    for name in names:
        if name in item and pd.notna(item[name]):
            return item[name]
    return ""


def _as_str(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _balanced_sample(
    frame: pd.DataFrame,
    *,
    label_col: str,
    max_per_label: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_per_label is None or max_per_label <= 0:
        return frame.reset_index(drop=True)
    rng = random.Random(seed)
    sampled_parts = []
    for _, group in frame.groupby(label_col, sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        sampled_parts.append(group.loc[indices[:max_per_label]])
    return (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _validate_audio_paths(frame: pd.DataFrame) -> None:
    missing = ~frame["audio_path"].map(lambda value: Path(value).exists())
    if missing.any():
        missing_count = int(missing.sum())
        example = frame.loc[missing, "audio_path"].iloc[0]
        raise FileNotFoundError(
            f"{missing_count} Medley-solos-DB audio files are missing. "
            f"First missing path: {example}. Use --download-audio, or pass --metadata-only."
        )
