from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer_evaluation.datasets.mtg_jamendo import (
    MTGJamendoPrepareConfig,
    prepare_mtg_jamendo_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an MTG-Jamendo manifest CSV.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("outputs/mtg_jamendo_manifest.csv"), type=Path)
    parser.add_argument("--task", default="autotagging_moodtheme")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-id", default=0, type=int)
    parser.add_argument(
        "--audio-subset",
        default="autotagging_moodtheme",
        choices=["raw_30s", "autotagging_moodtheme"],
    )
    parser.add_argument(
        "--audio-type",
        default="audio-low",
        choices=["audio", "audio-low", "melspecs", "acousticbrainz"],
    )
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--max-items", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        default=True,
        help="Download official split TSV if it is missing. Enabled by default.",
    )
    parser.add_argument(
        "--no-download-metadata",
        dest="download_metadata",
        action="store_false",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download selected official audio tar shards. This can be very large.",
    )
    parser.add_argument(
        "--audio-shards",
        nargs="*",
        default=[],
        help=(
            "Optional shard IDs or filenames, e.g. 0 1 2 or "
            "autotagging_moodtheme_audio-low-00.tar. Omit to download all shards."
        ),
    )
    parser.add_argument("--mirror", default="mtg-fast", choices=["mtg", "mtg-fast"])
    parser.add_argument("--no-unpack", dest="unpack", action="store_false")
    parser.add_argument("--remove-tars", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not require local audio files when writing the manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MTGJamendoPrepareConfig(
        root=args.root,
        manifest_path=args.manifest,
        task=args.task,
        split=args.split,
        split_id=args.split_id,
        audio_subset=args.audio_subset,
        audio_type=args.audio_type,
        tags=tuple(args.tags),
        max_items=args.max_items,
        seed=args.seed,
        download_metadata=args.download_metadata,
        download_audio=args.download_audio,
        audio_shards=tuple(args.audio_shards),
        mirror=args.mirror,
        unpack=args.unpack,
        remove_tars=args.remove_tars,
        require_audio=not args.metadata_only,
    )
    frame = prepare_mtg_jamendo_manifest(config)
    print(f"Saved {len(frame)} rows -> {args.manifest}")


if __name__ == "__main__":
    main()
