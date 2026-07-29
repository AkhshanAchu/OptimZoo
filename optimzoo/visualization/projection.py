from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimzoo.core.result import Result


def _pca_2d(points: np.ndarray) -> np.ndarray:
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return centered @ components.T


def plot_nd_projection(result: Result, method: str = "pca", ax=None):
    """Project the best-position trajectory of a high-dimensional search to 2D for
    visualization. method='pca' uses a plain NumPy SVD-based PCA (no extra
    dependencies); pass method='last2' to just take the first two dimensions.
    """
    positions = np.array(result.history.best_position)

    if method == "pca":
        projected = _pca_2d(positions)
    elif method == "last2":
        projected = positions[:, :2]
    else:
        raise ValueError(f"Unknown projection method: {method}")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    colors = np.arange(len(projected))
    sc = ax.scatter(projected[:, 0], projected[:, 1], c=colors, cmap="viridis", s=20)
    ax.plot(projected[:, 0], projected[:, 1], color="gray", alpha=0.3, linewidth=0.8)
    ax.scatter(*projected[0], color="cyan", s=80, edgecolors="black", label="start", zorder=5)
    ax.scatter(*projected[-1], color="red", marker="*", s=150, edgecolors="black", label="best", zorder=5)
    plt.colorbar(sc, ax=ax, label="iteration")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(f"{result.algorithm_name}: search trajectory ({method} projection, dim={positions.shape[1]})")
    ax.legend()

    if created_fig:
        fig.tight_layout()
        return fig
    return ax
