from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class SimulatedAnnealing(BaseOptimizer):
    """Single-point simulated annealing with geometric cooling.

    population_size controls how many independent candidate perturbations are
    tried per iteration purely for reporting population statistics; the walk
    itself is a single accepted point per chain (population_size chains run
    independently, in parallel, each with its own temperature).
    """

    name = "SimulatedAnnealing"

    def __init__(
        self,
        population_size: int = 20,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        step_size: float = 0.1,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size

    def _setup(self, problem: BaseProblem) -> None:
        self.positions = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.positions)
        self.temperature = self.initial_temperature

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        scale = self.step_size * self.bounds.range

        candidates = self.bounds.clip(self.positions + self.rng.normal(0, scale, size=(n, d)))
        candidate_fitness = problem.evaluate(candidates)

        delta = candidate_fitness - self.fitness
        accept_prob = np.exp(np.clip(-delta / max(self.temperature, 1e-12), -700, 0))
        accept = (delta < 0) | (self.rng.random(n) < accept_prob)

        self.positions[accept] = candidates[accept]
        self.fitness[accept] = candidate_fitness[accept]
        self.temperature *= self.cooling_rate

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.positions, self.fitness
