from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tokenizer_evaluation.embeddings import load_embedding_bundle
from tokenizer_evaluation.metrics import compute_pairwise_similarity_structure
from tokenizer_evaluation.options import (
    NamedOption,
    dataset_choices_help,
    normalize_dataset,
    normalize_tokenizer,
    tokenizer_choices_help,
)


ORDER_CHECK_COLUMNS = (
    "id",
    "note_str",
    "audio_path",
    "track_id",
    "path",
    "filepath",
    "filename",
)
ORDER_IDENTITY_COLUMNS = ("id", "note_str", "track_id", "filename")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pairwise cosine-similarity structure of two extracted latent bundles."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset option. Available: {dataset_choices_help()}",
    )
    parser.add_argument(
        "--tokenizer-a",
        required=True,
        help=f"Tokenizer for --embeddings-a. Available: {tokenizer_choices_help()}",
    )
    parser.add_argument(
        "--tokenizer-b",
        required=True,
        help=f"Tokenizer for --embeddings-b. Available: {tokenizer_choices_help()}",
    )
    parser.add_argument("--embeddings-a", required=True, type=Path)
    parser.add_argument("--embeddings-b", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument(
        "--join-col",
        default=None,
        help=(
            "Metadata column used to align the two bundles before evaluation. "
            "If omitted, rows must already be in the same order."
        ),
    )
    parser.add_argument(
        "--k",
        "--ks",
        dest="ks",
        nargs="+",
        default=[5, 10, 30, 50],
        type=int,
        help="One or more k values for SSS_local@k.",
    )
    parser.add_argument(
        "--clip-pooling",
        default="mean",
        choices=["mean", "mean_std", "flatten"],
        help="Pooling for sequence-shaped latents. Already pooled 2D latents are unchanged.",
    )
    parser.add_argument(
        "--clip-pool-axis",
        default=1,
        type=int,
        help="Sequence/time axis for clip pooling when latents are not 2D.",
    )
    parser.add_argument(
        "--max-items",
        default=None,
        type=int,
        help="Optionally sample a shared subset of rows before building pairwise matrices.",
    )
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        dataset = normalize_dataset(args.dataset)
        tokenizer_a = normalize_tokenizer(args.tokenizer_a)
        tokenizer_b = normalize_tokenizer(args.tokenizer_b)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        embeddings_a, embeddings_b, metadata, alignment = _load_align_and_sample(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or _default_output_dir(dataset, tokenizer_a, tokenizer_b)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_pairwise_similarity_structure(
        embeddings_a,
        embeddings_b,
        local_ks=args.ks,
        clip_pooling=args.clip_pooling,
        clip_pool_axis=args.clip_pool_axis,
    )

    payload = _json_payload(
        args=args,
        dataset=dataset,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        alignment=alignment,
        result=result,
    )
    json_path = output_dir / "pairwise_similarity_structure.json"
    summary_csv = output_dir / "pairwise_similarity_structure_summary.csv"
    long_csv = output_dir / "pairwise_similarity_structure_long.csv"
    metadata_csv = output_dir / "pairwise_similarity_structure_metadata.csv"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_summary_row(payload)]).to_csv(summary_csv, index=False)
    pd.DataFrame(_long_records(payload)).to_csv(long_csv, index=False)
    metadata.to_csv(metadata_csv, index=False)

    print(_markdown_summary(dataset, tokenizer_a, tokenizer_b, result))
    print(f"\nSaved pairwise structure outputs under {output_dir}")


def _load_align_and_sample(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    embeddings_a, metadata_a = load_embedding_bundle(args.embeddings_a)
    embeddings_b, metadata_b = load_embedding_bundle(args.embeddings_b)
    _validate_bundle_length(embeddings_a, metadata_a, "A")
    _validate_bundle_length(embeddings_b, metadata_b, "B")

    if args.join_col:
        embeddings_a, embeddings_b, metadata, alignment = _align_by_join_col(
            embeddings_a,
            metadata_a,
            embeddings_b,
            metadata_b,
            args.join_col,
        )
    else:
        embeddings_a, embeddings_b, metadata, alignment = _align_by_order(
            embeddings_a,
            metadata_a,
            embeddings_b,
            metadata_b,
        )

    embeddings_a, embeddings_b, metadata, sample_info = _sample_rows(
        embeddings_a,
        embeddings_b,
        metadata,
        max_items=args.max_items,
        seed=args.seed,
    )
    alignment["sampling"] = sample_info
    return embeddings_a, embeddings_b, metadata, alignment


def _validate_bundle_length(embeddings: np.ndarray, metadata: pd.DataFrame, name: str) -> None:
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding bundle {name} has {len(embeddings)} embedding rows "
            f"but {len(metadata)} metadata rows."
        )


def _align_by_join_col(
    embeddings_a: np.ndarray,
    metadata_a: pd.DataFrame,
    embeddings_b: np.ndarray,
    metadata_b: pd.DataFrame,
    join_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if join_col not in metadata_a.columns:
        raise ValueError(f"Missing --join-col '{join_col}' in metadata A.")
    if join_col not in metadata_b.columns:
        raise ValueError(f"Missing --join-col '{join_col}' in metadata B.")
    if metadata_a[join_col].isna().any() or metadata_b[join_col].isna().any():
        raise ValueError(f"--join-col '{join_col}' contains missing values.")
    if metadata_a[join_col].duplicated().any():
        raise ValueError(f"--join-col '{join_col}' is not unique in metadata A.")
    if metadata_b[join_col].duplicated().any():
        raise ValueError(f"--join-col '{join_col}' is not unique in metadata B.")

    left = metadata_a[[join_col]].copy()
    left["__embedding_index_a"] = np.arange(len(metadata_a))
    right = metadata_b[[join_col]].copy()
    right["__embedding_index_b"] = np.arange(len(metadata_b))
    joined = left.merge(right, on=join_col, how="inner", validate="one_to_one", sort=False)
    if joined.empty:
        raise ValueError(f"No shared rows found with --join-col '{join_col}'.")
    joined = joined.sort_values("__embedding_index_a", kind="mergesort")

    index_a = joined["__embedding_index_a"].to_numpy(dtype=int)
    index_b = joined["__embedding_index_b"].to_numpy(dtype=int)
    metadata = metadata_a.iloc[index_a].reset_index(drop=True)
    return (
        embeddings_a[index_a],
        embeddings_b[index_b],
        metadata,
        {
            "mode": "join_col",
            "join_col": join_col,
            "num_items_a_before": int(len(metadata_a)),
            "num_items_b_before": int(len(metadata_b)),
            "num_items_after_join": int(len(joined)),
        },
    )


def _align_by_order(
    embeddings_a: np.ndarray,
    metadata_a: pd.DataFrame,
    embeddings_b: np.ndarray,
    metadata_b: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if len(metadata_a) != len(metadata_b):
        raise ValueError(
            "Metadata row counts differ. Pass --join-col to align shared samples "
            f"({len(metadata_a)} rows in A, {len(metadata_b)} rows in B)."
        )

    checked_columns = [
        column
        for column in ORDER_CHECK_COLUMNS
        if column in metadata_a.columns and column in metadata_b.columns
    ]
    matching_columns = [
        column
        for column in checked_columns
        if _series_equal_for_alignment(metadata_a[column], metadata_b[column])
    ]
    mismatched_columns = [column for column in checked_columns if column not in matching_columns]
    matching_identity_columns = [
        column for column in matching_columns if column in ORDER_IDENTITY_COLUMNS
    ]
    if mismatched_columns and not matching_identity_columns:
        columns = ", ".join(mismatched_columns)
        raise ValueError(
            "The two metadata files do not appear to use the same row order. "
            f"Mismatched columns: {columns}. Pass --join-col to align shared samples."
        )

    return (
        embeddings_a,
        embeddings_b,
        metadata_a.reset_index(drop=True),
        {
            "mode": "row_order",
            "checked_columns": checked_columns,
            "matching_columns": matching_columns,
            "mismatched_columns": mismatched_columns,
            "num_items_after_join": int(len(metadata_a)),
        },
    )


def _series_equal_for_alignment(left: pd.Series, right: pd.Series) -> bool:
    left_values = left.fillna("<NA>").astype(str).reset_index(drop=True)
    right_values = right.fillna("<NA>").astype(str).reset_index(drop=True)
    return bool(left_values.equals(right_values))


def _sample_rows(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    metadata: pd.DataFrame,
    max_items: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if max_items is None or int(max_items) <= 0 or len(metadata) <= int(max_items):
        return (
            embeddings_a,
            embeddings_b,
            metadata,
            {
                "enabled": False,
                "num_items_before": int(len(metadata)),
                "num_items_after": int(len(metadata)),
            },
        )

    max_items = int(max_items)
    if max_items < 2:
        raise ValueError("--max-items must be at least 2 when sampling is enabled.")
    rng = np.random.default_rng(seed)
    row_indices = np.sort(rng.choice(len(metadata), size=max_items, replace=False))
    return (
        embeddings_a[row_indices],
        embeddings_b[row_indices],
        metadata.iloc[row_indices].reset_index(drop=True),
        {
            "enabled": True,
            "seed": int(seed),
            "num_items_before": int(len(metadata)),
            "num_items_after": int(max_items),
        },
    )


def _default_output_dir(
    dataset: NamedOption,
    tokenizer_a: NamedOption,
    tokenizer_b: NamedOption,
) -> Path:
    comparison_name = f"{tokenizer_a.key}_vs_{tokenizer_b.key}"
    return Path("outputs") / "pairwise_structure" / dataset.key / comparison_name


def _json_payload(
    *,
    args: argparse.Namespace,
    dataset: NamedOption,
    tokenizer_a: NamedOption,
    tokenizer_b: NamedOption,
    alignment: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": {"key": dataset.key, "display_name": dataset.display_name},
        "tokenizer_a": {"key": tokenizer_a.key, "display_name": tokenizer_a.display_name},
        "tokenizer_b": {"key": tokenizer_b.key, "display_name": tokenizer_b.display_name},
        "embedding_path_a": str(args.embeddings_a),
        "embedding_path_b": str(args.embeddings_b),
        "alignment": alignment,
        "ks": [int(k) for k in args.ks],
        "result": result,
    }


def _summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    row: dict[str, Any] = {
        "dataset": payload["dataset"]["key"],
        "tokenizer_a": payload["tokenizer_a"]["key"],
        "tokenizer_b": payload["tokenizer_b"]["key"],
        "num_items": result["num_items"],
        "num_pairs": result["num_pairs"],
        "embedding_dim_a": result["embedding_dim_a"],
        "embedding_dim_b": result["embedding_dim_b"],
        "SSS_global": result["global"]["rank_correlation"],
    }
    for k, values in result["local"].items():
        row[f"SSS_local@{k}"] = values["overlap"]
    return row


def _long_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload["result"]
    common = {
        "dataset": payload["dataset"]["key"],
        "tokenizer_a": payload["tokenizer_a"]["key"],
        "tokenizer_b": payload["tokenizer_b"]["key"],
        "num_items": result["num_items"],
        "num_pairs": result["num_pairs"],
    }
    records = [
        {
            **common,
            "metric": "SSS_global",
            "k": "",
            "value": result["global"]["rank_correlation"],
            "std": "",
            "min": "",
            "max": "",
        }
    ]
    for k, values in result["local"].items():
        records.append(
            {
                **common,
                "metric": "SSS_local",
                "k": int(k),
                "value": values["overlap"],
                "std": values["std"],
                "min": values["min"],
                "max": values["max"],
            }
        )
    return records


def _markdown_summary(
    dataset: NamedOption,
    tokenizer_a: NamedOption,
    tokenizer_b: NamedOption,
    result: dict[str, Any],
) -> str:
    rows = [
        ["Dataset", dataset.display_name],
        ["A", tokenizer_a.display_name],
        ["B", tokenizer_b.display_name],
        ["Items", str(result["num_items"])],
        ["Pairs", str(result["num_pairs"])],
        ["SSS_global", _format_score(result["global"]["rank_correlation"])],
    ]
    for k, values in result["local"].items():
        rows.append([f"SSS_local@{k}", _format_score(values["overlap"])])
    width = max(len(row[0]) for row in rows)
    return "\n".join(f"{name:<{width}} : {value}" for name, value in rows)


def _format_score(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
