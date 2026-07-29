from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimzoo.core.result import Result


def plot_convergence(result: Result | list[Result], labels: list[str] | None = None, log_scale: bool = True, ax=None):
    """Plot best-fitness-vs-iteration convergence curve(s)."""
    results = result if isinstance(result, list) else [result]
    if labels is None:
        labels = [r.algorithm_name or f"run {i}" for i, r in enumerate(results)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    for r, label in zip(results, labels):
        ax.plot(r.history.best_fitness, label=label, linewidth=1.8)

    if log_scale and all(min(r.history.best_fitness) > 0 for r in results):
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_population_stats(result: Result, ax=None):
    """Plot best/mean/worst fitness bands over iterations for a single run."""
    h = result.history
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    iters = np.arange(len(h))
    ax.plot(iters, h.best_fitness, label="best", color="tab:green")
    ax.plot(iters, h.mean_fitness, label="mean", color="tab:blue")
    ax.plot(iters, h.worst_fitness, label="worst", color="tab:red", alpha=0.6)
    ax.fill_between(
        iters,
        np.array(h.mean_fitness) - np.array(h.std_fitness),
        np.array(h.mean_fitness) + np.array(h.std_fitness),
        alpha=0.15,
        color="tab:blue",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{result.algorithm_name}: population statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return ax
