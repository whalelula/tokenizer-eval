from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tokenizer_evaluation.embeddings import build_extractor, extract_embeddings
from tokenizer_evaluation.models import EXTRACTORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract embeddings for one tokenizer/model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(EXTRACTORS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--max-duration-seconds", default=4.0, type=float)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-bitrate", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--layer", default=None, type=int)
    parser.add_argument("--layers", nargs="+", default=None, type=int)
    parser.add_argument("--checkpoint-path", default=None, type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--config-path", default=None, type=Path)
    parser.add_argument("--repo-path", default=None, type=Path)
    parser.add_argument("--representation", default=None)
    parser.add_argument("--sample-rate", default=None, type=int)
    parser.add_argument("--channels", default=None, type=int)
    parser.add_argument("--load-path-inference", default=None, type=Path)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--no-extract-features", action="store_true")
    parser.add_argument("--layer-num", default=None, type=int)
    parser.add_argument("--n-q", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    model_config = {}
    if args.model_name is not None:
        model_config["model_name"] = args.model_name
    if args.model_bitrate is not None:
        model_config["model_bitrate"] = args.model_bitrate
    if args.tag is not None:
        model_config["tag"] = args.tag
    if args.pooling is not None:
        model_config["pooling"] = args.pooling
    if args.layer is not None:
        model_config["layer"] = args.layer
    if args.layers is not None:
        model_config["layers"] = args.layers
    if args.checkpoint_path is not None:
        model_config["checkpoint_path"] = args.checkpoint_path
    if args.model_path is not None:
        model_config["model_path"] = args.model_path
    if args.config_path is not None:
        model_config["config_path"] = args.config_path
    if args.repo_path is not None:
        model_config["repo_path"] = args.repo_path
    if args.representation is not None:
        model_config["representation"] = args.representation
    if args.sample_rate is not None:
        model_config["sample_rate"] = args.sample_rate
    if args.channels is not None:
        model_config["channels"] = args.channels
    if args.load_path_inference is not None:
        model_config["load_path_inference"] = args.load_path_inference
    if args.extract_features:
        model_config["extract_features"] = True
    if args.no_extract_features:
        model_config["extract_features"] = False
    if args.layer_num is not None:
        model_config["layer_num"] = args.layer_num
    if args.n_q is not None:
        model_config["n_q"] = args.n_q

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
