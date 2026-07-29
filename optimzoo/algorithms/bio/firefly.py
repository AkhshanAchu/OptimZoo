from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class FireflyAlgorithm(BaseOptimizer):
    """Firefly Algorithm: attraction decays with distance; brighter (better) fireflies attract others."""

    name = "FireflyAlgorithm"

    def __init__(
        self,
        population_size: int = 30,
        alpha: float = 0.2,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha_decay: float = 0.97,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha_decay = alpha_decay

    def _setup(self, problem: BaseProblem) -> None:
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)
        self._alpha_current = self.alpha

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        scale = float(np.mean(self.bounds.range))

        order = np.argsort(self.fitness)
        new_positions = self.positions.copy()

        for i in range(n):
            for j in range(n):
                if self.fitness[j] < self.fitness[i]:
                    r2 = np.sum((self.positions[i] - self.positions[j]) ** 2)
                    beta = self.beta0 * np.exp(-self.gamma * r2 / (scale**2 + 1e-12))
                    noise = self._alpha_current * scale * (self.rng.random(d) - 0.5)
                    new_positions[i] = new_positions[i] + beta * (self.positions[j] - new_positions[i]) + noise

        new_positions = self.bounds.clip(new_positions)
        self.positions = new_positions
        self.fitness = problem.evaluate(self.positions)
        self._alpha_current *= self.alpha_decay

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.positions, self.fitness
