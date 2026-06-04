from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PALETTE = [
    "#7fb3d5",
    "#e74c3c",
    "#58d68d",
    "#f5b041",
    "#af7ac5",
    "#45b39d",
    "#f1948a",
    "#85929e",
    "#d4ac0d",
    "#5dade2",
    "#a9dfbf",
]


def save_tsne_plot(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    title: str,
    output_path: str | Path,
    label_col: str = "instrument_family_str",
    dpi: int = 220,
    annotation: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.16)
    _scatter(ax, coords, metadata, label_col=label_col)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if annotation:
        fig.text(0.5, 0.035, annotation, ha="center", va="bottom", fontsize=9)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(
    panels: list[tuple[str, np.ndarray, pd.DataFrame] | tuple[str, np.ndarray, pd.DataFrame, str]],
    output_path: str | Path,
    label_col: str = "instrument_family_str",
    dpi: int = 220,
) -> None:
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("At least one panel is required.")

    labels = sorted({label for panel in panels for label in panel[2][label_col].astype(str).unique()})
    palette = {label: DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)] for index, label in enumerate(labels)}
    n_panels = len(panels)
    n_cols = n_panels if n_panels <= 4 else 3
    n_rows = math.ceil(n_panels / n_cols)
    fig_width = max(6.0, 5.2 * n_cols)
    fig_height = 6.8 if n_rows == 1 else 5.4 * n_rows + 1.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.76 if n_rows == 1 else 0.86,
        bottom=0.2 if n_rows == 1 else 0.09,
        wspace=0.12,
        hspace=0.48,
    )
    flat_axes = axes.ravel()

    for index, panel in enumerate(panels):
        title, coords, metadata = panel[:3]
        annotation = panel[3] if len(panel) > 3 else None
        ax = flat_axes[index]
        _scatter(ax, coords, metadata, label_col=label_col, palette=palette)
        ax.set_title(f"({chr(97 + index)}) {title}", fontsize=18, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if annotation:
            ax.text(
                0.5,
                -0.12,
                annotation,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[label],
            markeredgecolor="#333333",
            markeredgewidth=0.4,
            markersize=8,
            label=label,
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 6),
        frameon=False,
        title="Instrument Family",
        title_fontsize=14,
        fontsize=10,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _scatter(
    ax,
    coords: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str,
    palette: dict[str, str] | None = None,
) -> None:
    labels = metadata[label_col].astype(str)
    unique_labels = sorted(labels.unique())
    if palette is None:
        palette = {
            label: DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
            for index, label in enumerate(unique_labels)
        }
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.78,
            c=palette[label],
            edgecolors="#333333",
            linewidths=0.25,
            label=label,
        )
    ax.set_facecolor("#fbfaf7")
    for spine in ax.spines.values():
        spine.set_color("#b8b2a4")
        spine.set_linewidth(0.8)
