from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optimzoo.core.problem import BaseProblem
from optimzoo.core.result import Result


def plot_1d_landscape(problem: BaseProblem, result: Result | None = None, resolution: int = 1000, ax=None):
    """Plot a 1D objective function curve, optionally overlaying the final population
    and best-found point from an optimization result."""
    if problem.dimension != 1:
        raise ValueError("plot_1d_landscape requires a 1D problem")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(problem.bounds.lower[0], problem.bounds.upper[0], resolution).reshape(-1, 1)
    y = problem.evaluate(x)
    ax.plot(x, y, color="tab:blue", linewidth=1.5, label=problem.name)

    if result is not None:
        if result.history.population:
            final_pop = result.history.population[-1].ravel()
            ax.scatter(
                final_pop,
                problem.evaluate(final_pop.reshape(-1, 1)),
                color="orange",
                s=25,
                alpha=0.7,
                label="final population",
                zorder=4,
            )
        ax.scatter(
            [result.best_position[0]],
            [result.best_fitness],
            color="red",
            marker="*",
            s=150,
            label="best found",
            zorder=5,
            edgecolors="black",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(problem.name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return ax
