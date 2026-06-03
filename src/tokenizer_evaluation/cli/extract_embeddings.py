from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tokenizer_evaluation.embeddings import build_extractor, extract_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract embeddings for one tokenizer/model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["same", "mert"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--max-duration-seconds", default=4.0, type=float)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--layer", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    model_config = {}
    if args.model_name is not None:
        model_config["model_name"] = args.model_name
    if args.pooling is not None:
        model_config["pooling"] = args.pooling
    if args.layer is not None:
        model_config["layer"] = args.layer

    extractor = build_extractor(
        args.model,
        model_config,
        device=args.device,
        dtype=args.dtype,
        max_duration_seconds=args.max_duration_seconds,
    )
    embeddings, _ = extract_embeddings(
        manifest,
        extractor,
        args.output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1]} embeddings -> {args.output}")


if __name__ == "__main__":
    main()
