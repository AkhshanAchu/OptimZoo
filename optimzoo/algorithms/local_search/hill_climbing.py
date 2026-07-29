from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class HillClimbing(BaseOptimizer):
    """Stochastic hill climbing with restarts: perturb and accept only if improved."""

    name = "HillClimbing"

    def __init__(self, population_size: int = 20, step_size: float = 0.1, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.step_size = step_size

    def _setup(self, problem: BaseProblem) -> None:
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        scale = self.step_size * self.bounds.range

        candidates = self.bounds.clip(self.positions + self.rng.normal(0, scale, size=(n, d)))
        candidate_fitness = problem.evaluate(candidates)

        improved = candidate_fitness < self.fitness
        self.positions[improved] = candidates[improved]
        self.fitness[improved] = candidate_fitness[improved]

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.positions, self.fitness


class RandomSearch(BaseOptimizer):
    """Pure random search baseline: uniformly resample the domain each iteration."""

    name = "RandomSearch"

    def _setup(self, problem: BaseProblem) -> None:
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)

    def _step(self):
        problem = self.problem
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.positions, self.fitness
