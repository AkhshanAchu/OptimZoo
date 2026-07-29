from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class GreyWolfOptimizer(BaseOptimizer):
    """Grey Wolf Optimizer: alpha/beta/delta wolves guide the pack toward prey."""

    name = "GreyWolfOptimizer"

    def _setup(self, problem: BaseProblem) -> None:
        n = self.population_size
        self.max_iter = self.stopping.max_iterations
        self.positions = self.bounds.sample(n, self.rng)
        self.fitness = problem.evaluate(self.positions)

    def _update_leaders(self):
        order = np.argsort(self.fitness)
        self.alpha_pos, self.alpha_fit = self.positions[order[0]].copy(), self.fitness[order[0]]
        self.beta_pos = self.positions[order[1]].copy() if len(order) > 1 else self.alpha_pos.copy()
        self.delta_pos = self.positions[order[2]].copy() if len(order) > 2 else self.alpha_pos.copy()

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        self._update_leaders()

        a = 2 - 2 * self.iteration / max(self.max_iter, 1)

        r1 = self.rng.random((n, d))
        r2 = self.rng.random((n, d))
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = np.abs(C1 * self.alpha_pos - self.positions)
        X1 = self.alpha_pos - A1 * D_alpha

        r1 = self.rng.random((n, d))
        r2 = self.rng.random((n, d))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = np.abs(C2 * self.beta_pos - self.positions)
        X2 = self.beta_pos - A2 * D_beta

        r1 = self.rng.random((n, d))
        r2 = self.rng.random((n, d))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = np.abs(C3 * self.delta_pos - self.positions)
        X3 = self.delta_pos - A3 * D_delta

        self.positions = self.bounds.clip((X1 + X2 + X3) / 3.0)
        self.fitness = problem.evaluate(self.positions)

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.positions, self.fitness
