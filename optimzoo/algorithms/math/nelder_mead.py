from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class NelderMead(BaseOptimizer):
    """Nelder-Mead simplex method: derivative-free direct search via reflect/expand/contract/shrink.

    Note: population_size is unused; the simplex has dimension + 1 vertices.
    """

    name = "NelderMead"

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, rho: float = 0.5, sigma: float = 0.5, **kwargs):
        kwargs.setdefault("population_size", 0)
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def _setup(self, problem: BaseProblem) -> None:
        d = self.bounds.dimension
        x0 = self.bounds.lower + self.rng.random(d) * self.bounds.range
        step = 0.05 * self.bounds.range
        step = np.where(step == 0, 0.05, step)

        self.simplex = np.tile(x0, (d + 1, 1))
        for i in range(d):
            self.simplex[i + 1, i] += step[i]
        self.simplex = self.bounds.clip(self.simplex)
        self.simplex_fitness = self.problem.evaluate(self.simplex)

    def _step(self):
        problem = self.problem
        order = np.argsort(self.simplex_fitness)
        self.simplex = self.simplex[order]
        self.simplex_fitness = self.simplex_fitness[order]

        d = self.bounds.dimension
        best, worst, second_worst = self.simplex_fitness[0], self.simplex_fitness[-1], self.simplex_fitness[-2]
        centroid = np.mean(self.simplex[:-1], axis=0)

        xr = self.bounds.clip(centroid + self.alpha * (centroid - self.simplex[-1]))
        fr = problem.evaluate(xr)[0]

        if best <= fr < second_worst:
            self.simplex[-1], self.simplex_fitness[-1] = xr, fr
        elif fr < best:
            xe = self.bounds.clip(centroid + self.gamma * (xr - centroid))
            fe = problem.evaluate(xe)[0]
            if fe < fr:
                self.simplex[-1], self.simplex_fitness[-1] = xe, fe
            else:
                self.simplex[-1], self.simplex_fitness[-1] = xr, fr
        else:
            xc = self.bounds.clip(centroid + self.rho * (self.simplex[-1] - centroid))
            fc = problem.evaluate(xc)[0]
            if fc < worst:
                self.simplex[-1], self.simplex_fitness[-1] = xc, fc
            else:
                self.simplex[1:] = self.simplex[0] + self.sigma * (self.simplex[1:] - self.simplex[0])
                self.simplex = self.bounds.clip(self.simplex)
                self.simplex_fitness[1:] = problem.evaluate(self.simplex[1:])

        best_idx = np.argmin(self.simplex_fitness)
        return self.simplex[best_idx], self.simplex_fitness[best_idx], self.simplex, self.simplex_fitness
