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
from tokenizer_evaluation.models import EXTRACTORS
from tokenizer_evaluation.reduction import run_tsne
from tokenizer_evaluation.visualization import save_comparison_plot, save_tsne_plot


def _format_model_label(model_key: str, model_cfg: dict[str, Any]) -> str:
    model_name = str(model_cfg.get("model_name", model_key))
    display_name = model_name.rsplit("/", maxsplit=1)[-1]
    if model_key == "mert" and "layer" in model_cfg:
        return f"{display_name} (layer {model_cfg['layer']})"
    return display_name


def _format_tsne_annotation(
    *,
    model_label: str,
    total_samples: int,
    max_per_family: int | None,
    pitch_stratified: bool,
    perplexity: float,
) -> str:
    max_per_family_label = "none" if max_per_family is None else str(max_per_family)
    pitch_label = "on" if pitch_stratified else "off"
    return (
        f"Model: {model_label}\n"
        f"Samples: {total_samples} | "
        f"max-per-family: {max_per_family_label} | pitch-stratified: {pitch_label} | "
        f"t-SNE perplexity: {perplexity:g}"
    )


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _build_run_specs(
    model_keys: list[str],
    models_cfg: dict[str, Any],
    mert_layers: list[int] | None,
    same_models: list[str] | None,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    specs = []
    for model_key in model_keys:
        model_cfg = dict(models_cfg.get(model_key, {}))
        model_cfg.pop("layers", None)
        model_cfg.pop("model_names", None)
        if model_key == "mert" and mert_layers:
            for layer in mert_layers:
                layer_cfg = dict(model_cfg)
                layer_cfg["layer"] = layer
                specs.append((f"mert_layer_{layer}", "mert", f"MERT L{layer}", layer_cfg))
        elif model_key == "same" and same_models:
            for same_model in same_models:
                same_cfg = dict(model_cfg)
                same_cfg["model_name"] = same_model
                specs.append((f"same_{_slug(same_model)}", "same", same_model.upper(), same_cfg))
        else:
            specs.append((model_key, model_key, model_key.upper(), model_cfg))
    return specs


def _default_comparison_filename(run_specs: list[tuple[str, str, str, dict[str, Any]]]) -> str:
    if run_specs and all(extractor_key == "mert" for _, extractor_key, _, _ in run_specs):
        layers = [
            str(model_cfg["layer"])
            for _, _, _, model_cfg in run_specs
            if "layer" in model_cfg
        ]
        if len(layers) == len(run_specs) and len(layers) > 1:
            return "mert_layers_tsne.png"
    if run_specs and {extractor_key for _, extractor_key, _, _ in run_specs} == {"same", "mert"}:
        return "same_models_mert_layers_tsne.png"
    return "same_vs_mert_tsne.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NSynth instrument t-SNE evaluation.")
    parser.add_argument(
        "--config",
        default=Path("configs/instrument_classification.yaml"),
        type=Path,
    )
    parser.add_argument("--nsynth-root", required=True, type=Path)
    parser.add_argument("--split", default=None, choices=["train", "valid", "validation", "test"])
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--models", nargs="+", default=None, choices=sorted(EXTRACTORS))
    parser.add_argument("--mert-layers", nargs="+", default=None, type=int)
    parser.add_argument("--same-models", nargs="+", default=None)
    parser.add_argument("--comparison-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=["auto", "float16", "float32"])
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-per-family", default=None, type=int)
    parser.add_argument("--pitch-stratified", action="store_true")
    parser.add_argument("--no-pitch-stratified", action="store_true")
    parser.add_argument("--pitch-bin-size", default=None, type=int)
    parser.add_argument("--max-per-pitch", default=None, type=int)
    parser.add_argument("--keep-incomplete-pitch-strata", action="store_true")
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
    manifest_path = args.manifest or Path(
        dataset_cfg.get("manifest", "outputs/nsynth_manifest.csv")
    )
    output_dir = args.output_dir or Path(output_cfg.get("dir", "outputs/nsynth_instrument"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pitch_stratified = bool(dataset_cfg.get("pitch_stratified", False))
    if args.pitch_stratified:
        pitch_stratified = True
    if args.no_pitch_stratified:
        pitch_stratified = False

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
        pitch_stratified=pitch_stratified,
        pitch_bin_size=args.pitch_bin_size
        if args.pitch_bin_size is not None
        else int(dataset_cfg.get("pitch_bin_size", 1)),
        max_per_pitch=args.max_per_pitch
        if args.max_per_pitch is not None
        else dataset_cfg.get("max_per_pitch"),
        pitch_require_all_families=not args.keep_incomplete_pitch_strata
        and bool(dataset_cfg.get("pitch_require_all_families", True)),
        seed=int(dataset_cfg.get("seed", 42)),
        download=args.download,
    )
    manifest = prepare_nsynth_manifest(prepare_cfg)
    total_samples = len(manifest)
    max_per_family = prepare_cfg.max_per_family

    device = args.device or runtime_cfg.get("device", "cuda")
    dtype = args.dtype or runtime_cfg.get("dtype", "auto")
    batch_size = args.batch_size or int(runtime_cfg.get("batch_size", 1))
    overwrite = bool(args.overwrite or runtime_cfg.get("overwrite", False))
    max_duration = runtime_cfg.get("max_duration_seconds", 4.0)
    perplexity = float(tsne_cfg.get("perplexity", 30))
    model_keys = args.models or [
        name for name, model_cfg in models_cfg.items() if model_cfg.get("enabled", True)
    ]
    mert_layers = args.mert_layers or models_cfg.get("mert", {}).get("layers")
    if mert_layers is not None:
        mert_layers = [int(layer) for layer in mert_layers]
    same_models = args.same_models or models_cfg.get("same", {}).get("model_names")
    if same_models is not None:
        same_models = [str(model_name) for model_name in same_models]
    run_specs = _build_run_specs(model_keys, models_cfg, mert_layers, same_models)

    panels = []
    for run_key, extractor_key, plot_title, model_cfg in run_specs:
        model_output_dir = output_dir / run_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        extractor = build_extractor(
            extractor_key,
            model_cfg,
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
            perplexity=perplexity,
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
        model_label = _format_model_label(extractor_key, model_cfg)
        annotation = _format_tsne_annotation(
            model_label=model_label,
            total_samples=total_samples,
            max_per_family=max_per_family,
            pitch_stratified=prepare_cfg.pitch_stratified,
            perplexity=perplexity,
        )
        save_tsne_plot(
            coords,
            metadata,
            title=plot_title,
            output_path=model_output_dir / "tsne.png",
            dpi=int(output_cfg.get("dpi", 220)),
            annotation=annotation,
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

        panels.append((plot_title, coords, metadata, annotation))

    save_comparison_plot(
        panels,
        output_dir / (args.comparison_name or _default_comparison_filename(run_specs)),
        dpi=int(output_cfg.get("dpi", 220)),
    )
    print(f"Finished. Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
