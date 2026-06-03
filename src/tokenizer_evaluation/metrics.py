from __future__ import annotations

import json
from pathlib import Path

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


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
