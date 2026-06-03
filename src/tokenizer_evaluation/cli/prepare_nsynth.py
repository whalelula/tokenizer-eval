from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer_evaluation.datasets.nsynth import NSynthPrepareConfig, prepare_nsynth_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a balanced NSynth manifest CSV.")
    parser.add_argument("--nsynth-root", required=True, type=Path)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "validation", "test"])
    parser.add_argument("--manifest", default=Path("outputs/nsynth_manifest.csv"), type=Path)
    parser.add_argument("--max-per-family", default=None, type=int)
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--pitch-min", default=None, type=int)
    parser.add_argument("--pitch-max", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = NSynthPrepareConfig(
        nsynth_root=args.nsynth_root,
        split=args.split,
        manifest_path=args.manifest,
        max_per_family=args.max_per_family,
        families=tuple(args.families),
        sources=tuple(args.sources),
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        seed=args.seed,
        download=args.download,
    )
    frame = prepare_nsynth_manifest(config)
    print(f"Saved {len(frame)} rows -> {args.manifest}")


if __name__ == "__main__":
    main()
