from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


def _levy_flight(rng: np.random.Generator, shape, beta: float = 1.5) -> np.ndarray:
    from math import gamma, pi, sin

    sigma_u = (
        gamma(1 + beta) * sin(pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=shape)
    v = rng.normal(0, 1, size=shape)
    return u / np.abs(v) ** (1 / beta)


class CuckooSearch(BaseOptimizer):
    """Cuckoo Search via Levy flights, with fraction of worst nests abandoned each iteration."""

    name = "CuckooSearch"

    def __init__(self, population_size: int = 30, discovery_rate: float = 0.25, alpha: float = 0.01, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.discovery_rate = discovery_rate
        self.alpha = alpha

    def _setup(self, problem: BaseProblem) -> None:
        self.nests = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.nests)

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension

        step = _levy_flight(self.rng, (n, d))
        best_idx = np.argmin(self.fitness)
        new_nests = self.nests + self.alpha * step * (self.nests - self.nests[best_idx])
        new_nests = self.bounds.clip(new_nests)
        new_fitness = problem.evaluate(new_nests)

        improved = new_fitness < self.fitness
        self.nests[improved] = new_nests[improved]
        self.fitness[improved] = new_fitness[improved]

        # Biased random walk for abandoned nests (standard CS formulation): each
        # discovered nest is nudged toward a random other pair's difference, rather
        # than fully re-sampled, so the population keeps exploiting good regions.
        abandon_mask = self.rng.random((n, d)) < self.discovery_rate
        perm1 = self.rng.permutation(n)
        perm2 = self.rng.permutation(n)
        step = self.rng.random((n, d)) * (self.nests[perm1] - self.nests[perm2])
        candidates = self.bounds.clip(np.where(abandon_mask, self.nests + step, self.nests))
        candidate_fit = problem.evaluate(candidates)
        improved = candidate_fit < self.fitness
        self.nests[improved] = candidates[improved]
        self.fitness[improved] = candidate_fit[improved]

        best_idx = np.argmin(self.fitness)
        return self.nests[best_idx], self.fitness[best_idx], self.nests, self.fitness
