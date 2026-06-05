from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def compute_embedding_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str = "instrument_family_str",
    knn_neighbors: int = 5,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, float | int | str | None]:
    from sklearn.metrics import accuracy_score, f1_score, silhouette_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    labels = metadata[label_col].astype(str).to_numpy()
    encoded = LabelEncoder().fit_transform(labels)
    metrics: dict[str, float | int | str | None] = {
        "num_items": int(len(labels)),
        "num_classes": int(len(set(labels))),
        "label_col": label_col,
    }

    if len(set(labels)) > 1 and min(np.bincount(encoded)) > 1:
        try:
            metrics["silhouette"] = float(
                silhouette_score(StandardScaler().fit_transform(embeddings), encoded)
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            metrics["silhouette_error"] = str(exc)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(embeddings, encoded))
        clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=min(knn_neighbors, len(train_idx))),
        )
        clf.fit(embeddings[train_idx], encoded[train_idx])
        pred = clf.predict(embeddings[test_idx])
        metrics["knn_accuracy"] = float(accuracy_score(encoded[test_idx], pred))
        metrics["knn_macro_f1"] = float(f1_score(encoded[test_idx], pred, average="macro"))
    else:
        metrics["silhouette"] = None
        metrics["knn_accuracy"] = None
        metrics["knn_macro_f1"] = None
        metrics["warning"] = "Not enough classes or examples per class for metrics."

    return metrics


def compute_knn_purity(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str = "instrument_family_str",
    ks: Sequence[int] = (5, 10, 30, 50),
    normalization: str = "standardize",
    pca_components: int | None = None,
    clip_pooling: str = "mean",
    clip_pool_axis: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute same-label kNN purity in clip-level embedding space.

    Purity@k(i) is the fraction of the k nearest neighbors of item i with the
    same label as item i. Neighbors are found with cosine distance and exclude
    the query item itself.
    """

    from sklearn.neighbors import NearestNeighbors

    if label_col not in metadata.columns:
        raise ValueError(f"Missing label column '{label_col}' in metadata.")

    labels = metadata[label_col].astype(str).to_numpy()
    features = _pool_clip_level_embeddings(
        embeddings,
        pooling=clip_pooling,
        pool_axis=clip_pool_axis,
    )
    if len(features) != len(labels):
        raise ValueError(
            f"Embedding count ({len(features)}) does not match metadata rows ({len(labels)})."
        )

    clean_ks = sorted({int(k) for k in ks})
    if not clean_ks or any(k <= 0 for k in clean_ks):
        raise ValueError("ks must contain at least one positive integer.")

    n_items = len(labels)
    if n_items < 2:
        raise ValueError("kNN purity needs at least two items.")

    max_k = max(clean_ks)
    if max_k >= n_items:
        raise ValueError(
            f"Cannot compute Purity@{max_k} with {n_items} items; k must be <= n_items - 1."
        )

    features, effective_pca = _prepare_knn_features(
        features,
        normalization=normalization,
        pca_components=pca_components,
        seed=seed,
    )

    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric="cosine",
        algorithm="brute",
    )
    indices = nn.fit(features).kneighbors(features, return_distance=False)
    neighbors = _drop_self_neighbors(indices, max_k=max_k)
    baseline = _random_purity_baseline(labels)

    by_k: dict[str, dict[str, float | int]] = {}
    for k in clean_ks:
        matches = labels[neighbors[:, :k]] == labels[:, None]
        per_item = matches.mean(axis=1)
        purity = float(per_item.mean())
        by_k[str(k)] = {
            "k": int(k),
            "purity": purity,
            "random_baseline": baseline,
            "delta_vs_random": purity - baseline,
        }

    return {
        "label_col": label_col,
        "num_items": int(n_items),
        "num_classes": int(len(set(labels))),
        "normalization": normalization,
        "pca_components": None if not pca_components else int(pca_components),
        "effective_pca_components": effective_pca,
        "clip_pooling": clip_pooling,
        "clip_pool_axis": int(clip_pool_axis),
        "ks": by_k,
    }


def _pool_clip_level_embeddings(
    embeddings: np.ndarray,
    pooling: str = "mean",
    pool_axis: int = 1,
) -> np.ndarray:
    features = np.asarray(embeddings)
    if features.ndim == 2:
        pooled = features
    elif features.ndim < 2:
        raise ValueError(f"Expected embeddings with at least 2 dimensions, got {features.shape}.")
    elif pooling == "flatten":
        pooled = features.reshape(features.shape[0], -1)
    else:
        axis = _normalize_axis(pool_axis, features.ndim)
        if axis == 0:
            raise ValueError("clip_pool_axis cannot be the batch axis (0).")
        if pooling == "mean":
            pooled = features.mean(axis=axis)
        elif pooling == "mean_std":
            mean = features.mean(axis=axis).reshape(features.shape[0], -1)
            std = features.std(axis=axis).reshape(features.shape[0], -1)
            pooled = np.concatenate([mean, std], axis=1)
        else:
            raise ValueError(f"Unknown clip pooling mode: {pooling}")

    pooled = np.asarray(pooled, dtype="float32").reshape(features.shape[0], -1)
    if not np.isfinite(pooled).all():
        raise ValueError("Embeddings contain NaN or infinite values after clip-level pooling.")
    return pooled


def _prepare_knn_features(
    embeddings: np.ndarray,
    normalization: str,
    pca_components: int | None,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer, StandardScaler

    if normalization == "standardize":
        features = StandardScaler().fit_transform(embeddings)
    elif normalization == "l2":
        features = Normalizer(norm="l2").fit_transform(embeddings)
    else:
        raise ValueError("normalization must be 'standardize' or 'l2'.")

    effective_pca = None
    if pca_components is not None and int(pca_components) > 0:
        effective_pca = min(int(pca_components), features.shape[0] - 1, features.shape[1])
        if effective_pca <= 0:
            raise ValueError("PCA needs at least two samples and one feature.")
        features = PCA(n_components=effective_pca, random_state=seed).fit_transform(features)
        if normalization == "l2":
            features = Normalizer(norm="l2").fit_transform(features)

    return np.asarray(features, dtype="float32"), effective_pca


def _drop_self_neighbors(indices: np.ndarray, max_k: int) -> np.ndarray:
    neighbors = np.empty((indices.shape[0], max_k), dtype=indices.dtype)
    for row_index, row in enumerate(indices):
        filtered = row[row != row_index]
        if len(filtered) < max_k:
            raise RuntimeError(
                f"Nearest-neighbor search returned fewer than {max_k} non-self neighbors."
            )
        neighbors[row_index] = filtered[:max_k]
    return neighbors


def _random_purity_baseline(labels: np.ndarray) -> float:
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return float(((counts[inverse] - 1) / (len(labels) - 1)).mean())


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for embeddings with {ndim} dimensions.")
    return normalized


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
