from __future__ import annotations

import json
import random
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NSYNTH_URLS = {
    "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "validation": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
}


@dataclass(frozen=True)
class NSynthPrepareConfig:
    nsynth_root: Path
    split: str = "valid"
    manifest_path: Path = Path("outputs/nsynth_manifest.csv")
    max_per_family: int | None = None
    families: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    pitch_min: int | None = None
    pitch_max: int | None = None
    seed: int = 42
    download: bool = False


def canonical_split(split: str) -> str:
    if split == "validation":
        return "valid"
    return split


def download_nsynth_split(root: str | Path, split: str) -> Path:
    split = canonical_split(split)
    if split not in NSYNTH_URLS:
        raise ValueError(f"Unknown NSynth split '{split}'. Choose train, valid, or test.")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"nsynth-{split}.jsonwav.tar.gz"
    split_root = root / f"nsynth-{split}"

    if _is_nsynth_split_root(split_root):
        return split_root

    if not archive_path.exists():
        _download_file(NSYNTH_URLS[split], archive_path)

    try:
        _extract_archive(archive_path, root, split_root)
    except (EOFError, OSError, tarfile.TarError) as exc:
        print(f"Archive looks incomplete or corrupted: {archive_path}")
        print("Removing the bad archive and retrying the download once.")
        archive_path.unlink(missing_ok=True)
        if split_root.exists():
            shutil.rmtree(split_root)
        _download_file(NSYNTH_URLS[split], archive_path)
        try:
            _extract_archive(archive_path, root, split_root)
        except (EOFError, OSError, tarfile.TarError) as retry_exc:
            archive_path.unlink(missing_ok=True)
            if split_root.exists():
                shutil.rmtree(split_root)
            raise RuntimeError(
                "Failed to download/extract NSynth after retry. The network download was likely "
                "interrupted. Please rerun the command, or download the archive manually from "
                f"{NSYNTH_URLS[split]} and place it at {archive_path}."
            ) from retry_exc

    return split_root


def _is_nsynth_split_root(path: Path) -> bool:
    return (path / "examples.json").exists() and (path / "audio").is_dir()


def _download_file(url: str, output_path: Path, retries: int = 3, chunk_size: int = 1024 * 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {url} -> {output_path}")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_file_once(url, temp_path, chunk_size=chunk_size)
            temp_path.replace(output_path)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = min(2 * attempt, 10)
            print(
                f"Download interrupted ({exc}). Retrying in {wait_seconds}s "
                f"({attempt}/{retries})..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts. Partial file kept at {temp_path}; "
        "rerun the command to resume, or delete the .part file to restart from zero."
    ) from last_error


def _download_file_once(url: str, temp_path: Path, chunk_size: int) -> None:
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    request = urllib.request.Request(url)
    if existing_size > 0:
        request.add_header("Range", f"bytes={existing_size}-")

    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None)
        if existing_size > 0 and status != 206:
            print("Server did not resume the partial download; restarting from zero.")
            existing_size = 0
            temp_path.unlink(missing_ok=True)

        total_size = _infer_total_size(response, existing_size)
        mode = "ab" if existing_size > 0 else "wb"
        with temp_path.open(mode + "") as handle:
            with _DownloadProgressBar(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
            ) as progress:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))

    if total_size is not None and temp_path.stat().st_size < total_size:
        raise OSError(
            f"retrieval incomplete: got only {temp_path.stat().st_size} out of {total_size} bytes"
        )


def _infer_total_size(response, existing_size: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", maxsplit=1)[-1]
        if total.isdigit():
            return int(total)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return existing_size + int(content_length)
    return None
    temp_path.replace(output_path)


class _DownloadProgressBar:
    def __init__(self, **kwargs):
        from tqdm import tqdm

        self._progress = tqdm(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._progress.close()

    def update(self, amount: int) -> None:
        self._progress.update(amount)

    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self._progress.total = total_size
        downloaded = block_num * block_size
        self._progress.update(downloaded - self._progress.n)


def _extract_archive(archive_path: Path, root: Path, split_root: Path) -> None:
    if split_root.exists():
        print(f"Removing incomplete split directory: {split_root}")
        shutil.rmtree(split_root)

    print(f"Extracting {archive_path} -> {root}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(root)

    if not _is_nsynth_split_root(split_root):
        raise FileNotFoundError(
            f"Extracted archive did not create a valid NSynth split root: {split_root}"
        )


def find_split_root(root: str | Path, split: str) -> Path:
    root = Path(root)
    split = canonical_split(split)
    candidates = [
        root,
        root / f"nsynth-{split}",
        root / split,
    ]
    for candidate in candidates:
        if (candidate / "examples.json").exists() and (candidate / "audio").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find NSynth split root. Expected examples.json and audio/ under one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_nsynth_examples(split_root: str | Path) -> pd.DataFrame:
    split_root = Path(split_root)
    examples_path = split_root / "examples.json"
    audio_dir = split_root / "audio"
    with examples_path.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)

    rows: list[dict[str, object]] = []
    for key, item in examples.items():
        note_str = str(item.get("note_str", key))
        audio_path = audio_dir / f"{note_str}.wav"
        rows.append(
            {
                "id": key,
                "note_str": note_str,
                "audio_path": str(audio_path.resolve()),
                "instrument": item.get("instrument"),
                "instrument_str": item.get("instrument_str"),
                "instrument_family": item.get("instrument_family"),
                "instrument_family_str": item.get("instrument_family_str"),
                "instrument_source": item.get("instrument_source"),
                "instrument_source_str": item.get("instrument_source_str"),
                "pitch": item.get("pitch"),
                "velocity": item.get("velocity"),
                "sample_rate": item.get("sample_rate"),
                "qualities_str": "|".join(item.get("qualities_str", [])),
            }
        )

    frame = pd.DataFrame(rows)
    missing = ~frame["audio_path"].map(lambda value: Path(value).exists())
    if missing.any():
        missing_count = int(missing.sum())
        raise FileNotFoundError(f"{missing_count} audio files referenced by examples.json are missing.")
    return frame


def filter_nsynth_manifest(
    frame: pd.DataFrame,
    families: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
    pitch_min: int | None = None,
    pitch_max: int | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if families:
        filtered = filtered[filtered["instrument_family_str"].isin(families)]
    if sources:
        filtered = filtered[filtered["instrument_source_str"].isin(sources)]
    if pitch_min is not None:
        filtered = filtered[filtered["pitch"].astype(int) >= pitch_min]
    if pitch_max is not None:
        filtered = filtered[filtered["pitch"].astype(int) <= pitch_max]
    return filtered.reset_index(drop=True)


def balanced_sample(
    frame: pd.DataFrame,
    label_col: str = "instrument_family_str",
    max_per_label: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if max_per_label is None or max_per_label <= 0:
        return frame.reset_index(drop=True)

    rng = random.Random(seed)
    sampled_parts = []
    for _, group in frame.groupby(label_col, sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        sampled_parts.append(group.loc[indices[:max_per_label]])
    return pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_nsynth_manifest(config: NSynthPrepareConfig) -> pd.DataFrame:
    split = canonical_split(config.split)
    split_root = (
        download_nsynth_split(config.nsynth_root, split)
        if config.download
        else find_split_root(config.nsynth_root, split)
    )
    frame = load_nsynth_examples(split_root)
    frame = filter_nsynth_manifest(
        frame,
        families=config.families,
        sources=config.sources,
        pitch_min=config.pitch_min,
        pitch_max=config.pitch_max,
    )
    frame = balanced_sample(frame, max_per_label=config.max_per_family, seed=config.seed)

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.manifest_path, index=False)
    return frame
