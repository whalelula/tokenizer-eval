from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer_evaluation.datasets.medley_solos_db import (
    MedleySolosPrepareConfig,
    prepare_medley_solos_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Medley-solos-DB manifest CSV.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("outputs/medley_solos_manifest.csv"), type=Path)
    parser.add_argument("--subsets", nargs="*", default=[])
    parser.add_argument("--instruments", nargs="*", default=[])
    parser.add_argument("--max-per-instrument", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        default=True,
        help="Download official metadata if it is missing. Enabled by default.",
    )
    parser.add_argument(
        "--no-download-metadata",
        dest="download_metadata",
        action="store_false",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download and extract the official audio tarball. This is large.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not require local audio files when writing the manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MedleySolosPrepareConfig(
        root=args.root,
        manifest_path=args.manifest,
        subsets=tuple(args.subsets),
        instruments=tuple(args.instruments),
        max_per_instrument=args.max_per_instrument,
        seed=args.seed,
        download_metadata=args.download_metadata,
        download_audio=args.download_audio,
        require_audio=not args.metadata_only,
    )
    frame = prepare_medley_solos_manifest(config)
    print(f"Saved {len(frame)} rows -> {args.manifest}")


if __name__ == "__main__":
    main()
