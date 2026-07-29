from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """WOA: bubble-net feeding behavior of humpback whales."""

    name = "WhaleOptimizationAlgorithm"

    def __init__(self, population_size: int = 30, b: float = 1.0, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.b = b

    def _setup(self, problem: BaseProblem) -> None:
        self.max_iter = self.stopping.max_iterations
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)
        best_idx = np.argmin(self.fitness)
        self.best_position = self.positions[best_idx].copy()
        self.best_fit = self.fitness[best_idx]

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        a = 2 - 2 * self.iteration / max(self.max_iter, 1)

        for i in range(n):
            r1, r2 = self.rng.random(), self.rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = self.rng.random()

            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * self.best_position - self.positions[i])
                    self.positions[i] = self.best_position - A * D
                else:
                    rand_idx = self.rng.integers(0, n)
                    rand_pos = self.positions[rand_idx]
                    D = np.abs(C * rand_pos - self.positions[i])
                    self.positions[i] = rand_pos - A * D
            else:
                l = self.rng.uniform(-1, 1)
                D_prime = np.abs(self.best_position - self.positions[i])
                self.positions[i] = (
                    D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.best_position
                )

        self.positions = self.bounds.clip(self.positions)
        self.fitness = problem.evaluate(self.positions)

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fit:
            self.best_fit = self.fitness[best_idx]
            self.best_position = self.positions[best_idx].copy()

        return self.best_position, self.best_fit, self.positions, self.fitness
