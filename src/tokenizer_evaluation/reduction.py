from __future__ import annotations

import numpy as np


def run_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30,
    learning_rate: str | float = "auto",
    n_iter: int = 1500,
    init: str = "pca",
    pca_dim: int | None = 50,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shaped [n, dim], got {embeddings.shape}.")
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 samples.")

    effective_perplexity = min(float(perplexity), max(1.0, (n_samples - 1) / 3))
    if pca_dim and embeddings.shape[1] > pca_dim:
        preprocessor = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(pca_dim, n_samples - 1), random_state=seed),
        )
        features = preprocessor.fit_transform(embeddings)
    else:
        features = StandardScaler().fit_transform(embeddings)

    tsne = _make_tsne(
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init=init,
        seed=seed,
    )
    coords = tsne.fit_transform(features)
    return coords.astype("float32")


def _make_tsne(
    perplexity: float,
    learning_rate: str | float,
    n_iter: int,
    init: str,
    seed: int,
):
    from sklearn.manifold import TSNE

    try:
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            init=init,
            random_state=seed,
        )
    except TypeError:
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            init=init,
            random_state=seed,
        )
