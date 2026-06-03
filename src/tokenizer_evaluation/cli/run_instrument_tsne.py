from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tokenizer_evaluation.config import load_yaml_config
from tokenizer_evaluation.datasets.nsynth import NSynthPrepareConfig, prepare_nsynth_manifest
from tokenizer_evaluation.embeddings import build_extractor, extract_embeddings
from tokenizer_evaluation.metrics import compute_embedding_metrics, save_metrics
from tokenizer_evaluation.reduction import run_tsne
from tokenizer_evaluation.visualization import save_comparison_plot, save_tsne_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAME/MERT NSynth instrument t-SNE evaluation.")
    parser.add_argument("--config", default=Path("configs/instrument_classification.yaml"), type=Path)
    parser.add_argument("--nsynth-root", required=True, type=Path)
    parser.add_argument("--split", default=None, choices=["train", "valid", "validation", "test"])
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--models", nargs="+", default=None, choices=["same", "mert"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=["auto", "float16", "float32"])
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-per-family", default=None, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg: dict[str, Any] = config.get("dataset", {})
    runtime_cfg: dict[str, Any] = config.get("runtime", {})
    models_cfg: dict[str, Any] = config.get("models", {})
    tsne_cfg: dict[str, Any] = config.get("tsne", {})
    metrics_cfg: dict[str, Any] = config.get("metrics", {})
    output_cfg: dict[str, Any] = config.get("output", {})

    split = args.split or dataset_cfg.get("split", "valid")
    manifest_path = args.manifest or Path(dataset_cfg.get("manifest", "outputs/nsynth_manifest.csv"))
    output_dir = args.output_dir or Path(output_cfg.get("dir", "outputs/nsynth_instrument"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_cfg = NSynthPrepareConfig(
        nsynth_root=args.nsynth_root,
        split=split,
        manifest_path=manifest_path,
        max_per_family=args.max_per_family
        if args.max_per_family is not None
        else dataset_cfg.get("max_per_family"),
        families=tuple(dataset_cfg.get("families") or ()),
        sources=tuple(dataset_cfg.get("sources") or ()),
        pitch_min=dataset_cfg.get("pitch_min"),
        pitch_max=dataset_cfg.get("pitch_max"),
        seed=int(dataset_cfg.get("seed", 42)),
        download=args.download,
    )
    manifest = prepare_nsynth_manifest(prepare_cfg)

    device = args.device or runtime_cfg.get("device", "cuda")
    dtype = args.dtype or runtime_cfg.get("dtype", "auto")
    batch_size = args.batch_size or int(runtime_cfg.get("batch_size", 1))
    overwrite = bool(args.overwrite or runtime_cfg.get("overwrite", False))
    max_duration = runtime_cfg.get("max_duration_seconds", 4.0)
    model_keys = args.models or [
        name for name, model_cfg in models_cfg.items() if model_cfg.get("enabled", True)
    ]

    panels = []
    for model_key in model_keys:
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        extractor = build_extractor(
            model_key,
            models_cfg.get(model_key, {}),
            device=device,
            dtype=dtype,
            max_duration_seconds=max_duration,
        )
        embedding_path = model_output_dir / "embeddings.npz"
        embeddings, metadata = extract_embeddings(
            manifest,
            extractor,
            embedding_path,
            batch_size=batch_size,
            overwrite=overwrite,
        )

        coords = run_tsne(
            embeddings,
            perplexity=float(tsne_cfg.get("perplexity", 30)),
            learning_rate=tsne_cfg.get("learning_rate", "auto"),
            n_iter=int(tsne_cfg.get("n_iter", 1500)),
            init=str(tsne_cfg.get("init", "pca")),
            pca_dim=tsne_cfg.get("pca_dim", 50),
            seed=int(tsne_cfg.get("seed", 42)),
        )
        coords_path = model_output_dir / "tsne.csv"
        coords_frame = metadata.copy()
        coords_frame.insert(0, "tsne_y", coords[:, 1])
        coords_frame.insert(0, "tsne_x", coords[:, 0])
        coords_frame.to_csv(coords_path, index=False)
        save_tsne_plot(
            coords,
            metadata,
            title=model_key.upper(),
            output_path=model_output_dir / "tsne.png",
            dpi=int(output_cfg.get("dpi", 220)),
        )

        if metrics_cfg.get("enabled", True):
            metrics = compute_embedding_metrics(
                embeddings,
                metadata,
                knn_neighbors=int(metrics_cfg.get("knn_neighbors", 5)),
                test_size=float(metrics_cfg.get("test_size", 0.25)),
                seed=int(metrics_cfg.get("seed", 42)),
            )
            save_metrics(metrics, model_output_dir / "metrics.json")

        panels.append((model_key.upper(), coords, metadata))

    save_comparison_plot(
        panels,
        output_dir / "same_vs_mert_tsne.png",
        dpi=int(output_cfg.get("dpi", 220)),
    )
    print(f"Finished. Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
