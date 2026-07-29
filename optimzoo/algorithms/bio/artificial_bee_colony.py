from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class ArtificialBeeColony(BaseOptimizer):
    """ABC: employed bees, onlooker bees, and scout bees search a food-source landscape."""

    name = "ArtificialBeeColony"

    def __init__(self, population_size: int = 40, limit: int | None = None, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.limit = limit

    def _setup(self, problem: BaseProblem) -> None:
        n, d = self.population_size, self.bounds.dimension
        self.limit = self.limit or n * d
        self.sources = self.bounds.sample(n, self.rng)
        self.fitness = problem.evaluate(self.sources)
        self.trials = np.zeros(n, dtype=int)
        best_idx = np.argmin(self.fitness)
        self.best_position = self.sources[best_idx].copy()
        self.best_fit = self.fitness[best_idx]

    @staticmethod
    def _fitness_score(f: np.ndarray) -> np.ndarray:
        return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))

    def _try_new_source(self, i: int) -> None:
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        j = self.rng.integers(0, d)
        k = self.rng.choice([x for x in range(n) if x != i])
        phi = self.rng.uniform(-1, 1)

        candidate = self.sources[i].copy()
        candidate[j] = candidate[j] + phi * (candidate[j] - self.sources[k, j])
        candidate = self.bounds.clip(candidate)
        candidate_fit = problem.evaluate(candidate)[0]

        if candidate_fit < self.fitness[i]:
            self.sources[i] = candidate
            self.fitness[i] = candidate_fit
            self.trials[i] = 0
        else:
            self.trials[i] += 1

    def _step(self):
        n = self.population_size

        for i in range(n):
            self._try_new_source(i)

        scores = self._fitness_score(self.fitness)
        probs = scores / scores.sum()
        i = 0
        t = 0
        while t < n:
            if self.rng.random() < probs[i]:
                self._try_new_source(i)
                t += 1
            i = (i + 1) % n

        scout_idx = np.argmax(self.trials)
        if self.trials[scout_idx] > self.limit:
            self.sources[scout_idx] = self.bounds.sample(1, self.rng)[0]
            self.fitness[scout_idx] = self.problem.evaluate(self.sources[scout_idx])[0]
            self.trials[scout_idx] = 0

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fit:
            self.best_fit = self.fitness[best_idx]
            self.best_position = self.sources[best_idx].copy()

        return self.best_position, self.best_fit, self.sources, self.fitness
