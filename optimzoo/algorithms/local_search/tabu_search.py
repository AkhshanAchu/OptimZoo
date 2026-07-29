from __future__ import annotations

from collections import deque

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class TabuSearch(BaseOptimizer):
    """Continuous Tabu Search: neighborhood sampling with a short-term memory of
    recently visited regions (discretized to a grid) to avoid cycling."""

    name = "TabuSearch"

    def __init__(
        self,
        population_size: int = 20,
        tabu_tenure: int = 15,
        step_size: float = 0.1,
        grid_resolution: int = 50,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.tabu_tenure = tabu_tenure
        self.step_size = step_size
        self.grid_resolution = grid_resolution

    def _setup(self, problem: BaseProblem) -> None:
        d = self.bounds.dimension
        self.current = self.bounds.lower + self.rng.random(d) * self.bounds.range
        self.current_fitness = self.problem.evaluate(self.current)[0]
        self.best_pos = self.current.copy()
        self.best_fit = self.current_fitness
        self.tabu_list: deque = deque(maxlen=self.tabu_tenure)
        self.tabu_list.append(self._discretize(self.current))

    def _discretize(self, x: np.ndarray) -> tuple:
        normalized = (x - self.bounds.lower) / (self.bounds.range + 1e-12)
        return tuple(np.floor(normalized * self.grid_resolution).astype(int))

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        scale = self.step_size * self.bounds.range

        neighbors = self.bounds.clip(self.current + self.rng.normal(0, scale, size=(n, d)))
        neighbor_fitness = problem.evaluate(neighbors)

        best_neighbor_idx = None
        best_neighbor_fit = np.inf
        for i in range(n):
            key = self._discretize(neighbors[i])
            is_tabu = key in self.tabu_list
            aspiration = neighbor_fitness[i] < self.best_fit
            if (not is_tabu or aspiration) and neighbor_fitness[i] < best_neighbor_fit:
                best_neighbor_fit = neighbor_fitness[i]
                best_neighbor_idx = i

        if best_neighbor_idx is not None:
            self.current = neighbors[best_neighbor_idx]
            self.current_fitness = best_neighbor_fit
            self.tabu_list.append(self._discretize(self.current))

        if self.current_fitness < self.best_fit:
            self.best_fit = self.current_fitness
            self.best_pos = self.current.copy()

        return self.best_pos, self.best_fit, neighbors, neighbor_fitness
