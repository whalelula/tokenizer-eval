from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tokenizer_evaluation.datasets.download import download_file, extract_archive


MTG_REPO_RAW = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master"
MTG_FAST_BASE_URL = "https://cdn.freesound.org/mtg-jamendo"
MTG_BASE_URL = "https://essentia.upf.edu/documentation/datasets/mtg-jamendo"


@dataclass(frozen=True)
class MTGJamendoPrepareConfig:
    root: Path
    manifest_path: Path = Path("outputs/mtg_jamendo_manifest.csv")
    task: str = "autotagging_moodtheme"
    split: str = "test"
    split_id: int = 0
    audio_subset: str = "autotagging_moodtheme"
    audio_type: str = "audio-low"
    tags: tuple[str, ...] = ()
    max_items: int | None = None
    seed: int = 42
    download_metadata: bool = True
    download_audio: bool = False
    audio_shards: tuple[str, ...] = ()
    mirror: str = "mtg-fast"
    unpack: bool = True
    remove_tars: bool = False
    require_audio: bool = True


def prepare_mtg_jamendo_manifest(config: MTGJamendoPrepareConfig) -> pd.DataFrame:
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    split_path = _ensure_split_file(
        root,
        task=config.task,
        split=config.split,
        split_id=config.split_id,
        download=config.download_metadata,
    )
    if config.download_audio:
        download_mtg_jamendo_audio(
            root,
            dataset=config.audio_subset,
            data_type=config.audio_type,
            shards=config.audio_shards,
            mirror=config.mirror,
            unpack=config.unpack,
            remove_tars=config.remove_tars,
        )

    frame = load_mtg_jamendo_split(
        split_path,
        root=root,
        audio_subset=config.audio_subset,
        audio_type=config.audio_type,
        require_audio=config.require_audio,
    )
    frame = filter_mtg_jamendo_manifest(frame, tags=config.tags)
    frame = _sample_items(frame, max_items=config.max_items, seed=config.seed)

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.manifest_path, index=False)
    return frame


def load_mtg_jamendo_split(
    split_path: str | Path,
    *,
    root: str | Path,
    audio_subset: str,
    audio_type: str,
    require_audio: bool = True,
) -> pd.DataFrame:
    split_path = Path(split_path)
    root = Path(root)
    rows = []
    for item in _read_split_rows(split_path):
        relative_path = str(item["path"])
        audio_path = _resolve_audio_path(root, relative_path, audio_subset, audio_type)
        tags = item["tags"]
        rows.append(
            {
                "id": item["track_id"],
                "audio_path": str(audio_path.resolve()),
                "dataset": "mtg-jamendo",
                "track_id": item["track_id"],
                "artist_id": item["artist_id"],
                "album_id": item["album_id"],
                "duration": item["duration"],
                "path": relative_path,
                "tags": "|".join(tags),
                "tag_list": "|".join(tags),
            }
        )

    frame = pd.DataFrame(rows)
    if require_audio:
        _validate_audio_paths(frame)
    return frame


def filter_mtg_jamendo_manifest(
    frame: pd.DataFrame,
    *,
    tags: tuple[str, ...] = (),
) -> pd.DataFrame:
    if not tags:
        return frame.reset_index(drop=True)
    wanted = set(tags)
    keep = frame["tags"].map(lambda value: bool(wanted.intersection(_split_tags(str(value)))))
    return frame[keep].reset_index(drop=True)


def download_mtg_jamendo_audio(
    root: str | Path,
    *,
    dataset: str,
    data_type: str,
    shards: tuple[str, ...] = (),
    mirror: str = "mtg-fast",
    unpack: bool = True,
    remove_tars: bool = False,
) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tar_checksums = _download_checksum_file(root, dataset, data_type, "sha256_tars")
    entries = _read_checksum_file(tar_checksums)
    if shards:
        wanted = {_normalize_shard_name(value, dataset, data_type) for value in shards}
        entries = {name: checksum for name, checksum in entries.items() if name in wanted}
        missing = sorted(wanted - set(entries))
        if missing:
            raise ValueError(f"Unknown MTG-Jamendo shard(s): {', '.join(missing)}")

    if not entries:
        raise ValueError("No MTG-Jamendo audio shards selected.")

    downloaded = []
    for filename, checksum in entries.items():
        output_path = root / "archives" / filename
        if not output_path.exists():
            download_file(_audio_url(mirror, dataset, data_type, filename), output_path)
        _validate_sha256(output_path, checksum)
        downloaded.append(output_path)
        if unpack:
            extract_archive(output_path, root / "audio")
            if remove_tars:
                output_path.unlink()
    return downloaded


def _ensure_split_file(
    root: Path,
    *,
    task: str,
    split: str,
    split_id: int,
    download: bool,
) -> Path:
    relative = Path("metadata") / "splits" / f"split-{split_id}" / f"{task}-{split}.tsv"
    path = root / relative
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"Missing MTG-Jamendo split file: {path}")
    url = f"{MTG_REPO_RAW}/data/splits/split-{split_id}/{task}-{split}.tsv"
    return download_file(url, path)


def _download_checksum_file(root: Path, dataset: str, data_type: str, suffix: str) -> Path:
    relative = Path("metadata") / "download" / f"{dataset}_{data_type}_{suffix}.txt"
    path = root / relative
    if path.exists():
        return path
    url = f"{MTG_REPO_RAW}/data/download/{dataset}_{data_type}_{suffix}.txt"
    return download_file(url, path)


def _read_checksum_file(path: Path) -> dict[str, str]:
    entries = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            checksum, filename = line.strip().split(maxsplit=1)
            entries[filename] = checksum
    return entries


def _read_split_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:6] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]:
            raise ValueError(f"Unexpected MTG-Jamendo split header in {path}: {header}")
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                raise ValueError(f"Malformed MTG-Jamendo split row {line_number} in {path}")
            rows.append(
                {
                    "track_id": parts[0],
                    "artist_id": parts[1],
                    "album_id": parts[2],
                    "path": parts[3],
                    "duration": float(parts[4]),
                    "tags": [tag for tag in parts[5:] if tag],
                }
            )
    return rows


def _audio_url(mirror: str, dataset: str, data_type: str, filename: str) -> str:
    if mirror == "mtg":
        return f"{MTG_BASE_URL}/{dataset}/{data_type}/{filename}"
    if mirror == "mtg-fast":
        return f"{MTG_FAST_BASE_URL}/{dataset}/{data_type}/{filename}"
    raise ValueError("mirror must be 'mtg' or 'mtg-fast'.")


def _resolve_audio_path(root: Path, relative_path: str, audio_subset: str, audio_type: str) -> Path:
    candidates = [root / "audio" / relative_path, root / relative_path]
    if audio_type == "audio-low":
        low_path = _low_quality_path(relative_path)
        candidates.extend([root / "audio" / low_path, root / low_path])
    candidates.extend(root.glob(f"**/{Path(relative_path).name}"))
    if audio_type == "audio-low":
        candidates.extend(root.glob(f"**/{Path(_low_quality_path(relative_path)).name}"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if audio_type == "audio-low":
        return root / "audio" / _low_quality_path(relative_path)
    return root / "audio" / audio_subset / audio_type / relative_path


def _low_quality_path(relative_path: str) -> str:
    path = Path(relative_path)
    return str(path.with_name(path.stem + ".low" + path.suffix)).replace("\\", "/")


def _split_tags(value: str) -> list[str]:
    return [tag for tag in value.replace(",", "|").split("|") if tag]


def _sample_items(frame: pd.DataFrame, *, max_items: int | None, seed: int) -> pd.DataFrame:
    if max_items is None or max_items <= 0 or len(frame) <= max_items:
        return frame.reset_index(drop=True)
    rng = random.Random(seed)
    indices = list(frame.index)
    rng.shuffle(indices)
    return frame.loc[indices[:max_items]].reset_index(drop=True)


def _normalize_shard_name(value: str, dataset: str, data_type: str) -> str:
    value = str(value)
    if value.endswith(".tar"):
        return value
    if value.isdigit():
        return f"{dataset}_{data_type}-{int(value):02d}.tar"
    return f"{dataset}_{data_type}-{value}.tar"


def _validate_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The corrupted archive was removed; rerun the command to download it again."
        )


def _validate_audio_paths(frame: pd.DataFrame) -> None:
    missing = ~frame["audio_path"].map(lambda value: Path(value).exists())
    if missing.any():
        missing_count = int(missing.sum())
        example = frame.loc[missing, "audio_path"].iloc[0]
        raise FileNotFoundError(
            f"{missing_count} MTG-Jamendo audio files are missing. "
            f"First missing path: {example}. Use --download-audio, or pass --metadata-only."
        )
