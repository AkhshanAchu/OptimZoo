from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from optimzoo.core.problem import BaseProblem
from optimzoo.core.result import Result


def _grid(problem: BaseProblem, resolution: int = 200):
    if problem.dimension != 2:
        raise ValueError("Landscape plots require a 2D problem")
    x = np.linspace(problem.bounds.lower[0], problem.bounds.upper[0], resolution)
    y = np.linspace(problem.bounds.lower[1], problem.bounds.upper[1], resolution)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z = problem.evaluate(points).reshape(X.shape)
    return X, Y, Z


def plot_contour(problem: BaseProblem, resolution: int = 200, ax=None, levels: int = 40):
    """Filled contour plot of a 2D objective function."""
    X, Y, Z = _grid(problem, resolution)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.2, alpha=0.4)
    plt.colorbar(cs, ax=ax, label="fitness")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(problem.name)

    if created_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_surface_3d(problem: BaseProblem, resolution: int = 100, ax=None):
    """3D surface plot of a 2D objective function."""
    X, Y, Z = _grid(problem, resolution)
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("fitness")
    ax.set_title(problem.name)

    if created_fig:
        return ax.figure
    return ax


def plot_search_trace(problem: BaseProblem, result: Result, resolution: int = 200, ax=None):
    """Overlay the best-position trajectory on a contour plot (2D problems only)."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    plot_contour(problem, resolution=resolution, ax=ax)

    positions = np.array(result.history.best_position)
    ax.plot(positions[:, 0], positions[:, 1], color="white", linewidth=1.2, alpha=0.8, marker="o", markersize=2)
    ax.scatter(*positions[0], color="cyan", s=60, label="start", zorder=5, edgecolors="black")
    ax.scatter(*positions[-1], color="red", s=80, marker="*", label="best", zorder=5, edgecolors="black")
    ax.legend()

    if created_fig:
        fig.tight_layout()
        return fig
    return ax


def animate_population(problem: BaseProblem, result: Result, resolution: int = 150, interval: int = 100):
    """Animate the population cloud evolving over a 2D fitness landscape.

    Requires the optimizer to have been run with store_population_history=True.
    Returns a matplotlib.animation.FuncAnimation; call .save(path) to export
    (e.g. as .gif or .mp4, ffmpeg/pillow required for the latter/former).
    """
    if not result.history.population:
        raise ValueError(
            "No population history recorded; re-run the optimizer with store_population_history=True"
        )

    X, Y, Z = _grid(problem, resolution)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    title = ax.set_title(f"{result.algorithm_name} — iteration 0")

    pop0 = result.history.population[0]
    scatter = ax.scatter(pop0[:, 0], pop0[:, 1], color="white", edgecolors="black", s=30, zorder=5)
    best_marker = ax.scatter([], [], color="red", marker="*", s=120, zorder=6)

    def update(frame):
        pop = result.history.population[frame]
        scatter.set_offsets(pop)
        best_pos = result.history.best_position[frame]
        best_marker.set_offsets(best_pos.reshape(1, -1))
        title.set_text(f"{result.algorithm_name} — iteration {frame}")
        return scatter, best_marker, title

    anim = FuncAnimation(fig, update, frames=len(result.history.population), interval=interval, blit=False)
    return anim
