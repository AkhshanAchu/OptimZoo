from __future__ import annotations

import time
from typing import Callable

import numpy as np

from optimzoo.core.callback import BaseCallback


def _to_list(x) -> list | None:
    if x is None:
        return None
    return np.asarray(x).tolist()


class DashboardCallback(BaseCallback):
    """Emits JSON-serializable iteration events to a sink callable for a live
    dashboard to stream and/or record for later replay. One instance per run."""

    def __init__(self, sink: Callable[[dict], None], max_population_points: int = 500):
        self.sink = sink
        self.max_population_points = max_population_points
        self._start_time = 0.0

    def _put(self, event: dict) -> None:
        self.sink(event)

    def on_start(self, optimizer) -> None:
        self._start_time = time.perf_counter()
        self._put(
            {
                "type": "start",
                "algorithm": optimizer.name,
                "problem": optimizer.problem.name,
                "dimension": optimizer.problem.dimension,
                "population_size": optimizer.population_size,
                "max_iterations": optimizer.stopping.max_iterations,
                "bounds": {
                    "lower": _to_list(optimizer.bounds.lower),
                    "upper": _to_list(optimizer.bounds.upper),
                },
            }
        )

    def on_iteration(self, optimizer, iteration: int) -> None:
        pop = getattr(optimizer, "positions", None)
        if pop is None:
            pop = getattr(optimizer, "population", None)
        pop_fit = getattr(optimizer, "fitness", None)

        population_payload = None
        if pop is not None and optimizer.problem.dimension <= 3:
            pop_arr = np.asarray(pop)
            if len(pop_arr) > self.max_population_points:
                idx = np.linspace(0, len(pop_arr) - 1, self.max_population_points).astype(int)
                pop_arr = pop_arr[idx]
                pop_fit_arr = np.asarray(pop_fit)[idx] if pop_fit is not None else None
            else:
                pop_fit_arr = np.asarray(pop_fit) if pop_fit is not None else None
            population_payload = {
                "positions": pop_arr.tolist(),
                "fitness": pop_fit_arr.tolist() if pop_fit_arr is not None else None,
            }

        self._put(
            {
                "type": "iteration",
                "iteration": iteration,
                "best_fitness": optimizer.best_fitness,
                "best_position": _to_list(optimizer.best_position),
                "mean_fitness": optimizer.history.mean_fitness[-1] if optimizer.history.mean_fitness else None,
                "worst_fitness": optimizer.history.worst_fitness[-1] if optimizer.history.worst_fitness else None,
                "std_fitness": optimizer.history.std_fitness[-1] if optimizer.history.std_fitness else None,
                "n_evaluations": optimizer.problem.n_evaluations,
                "elapsed_seconds": time.perf_counter() - self._start_time,
                "population": population_payload,
            }
        )

    def on_finish(self, optimizer) -> None:
        self._put(
            {
                "type": "finish",
                "best_fitness": optimizer.best_fitness,
                "best_position": _to_list(optimizer.best_position),
                "n_iterations": optimizer.iteration,
                "n_evaluations": optimizer.problem.n_evaluations,
                "elapsed_seconds": time.perf_counter() - self._start_time,
            }
        )

    def on_stop(self, optimizer) -> None:
        self._put({"type": "stopped"})
