from __future__ import annotations

import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def download_file(
    url: str,
    output_path: str | Path,
    *,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {url} -> {output_path}")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_file_once(url, temp_path, chunk_size=chunk_size)
            temp_path.replace(output_path)
            return output_path
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


def extract_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    clean: bool = False,
) -> Path:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        print(f"Removing incomplete extraction directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {output_dir}")

    if archive_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(output_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Archive is incomplete or corrupted: {archive_path}") from exc
        return output_dir

    try:
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_dir)
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"Archive is incomplete or corrupted: {archive_path}") from exc
    return output_dir


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
        with temp_path.open(mode) as handle:
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
