from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class CMAES(BaseOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy (standard (mu/mu_w, lambda)-CMA-ES)."""

    name = "CMAES"

    def __init__(self, population_size: int | None = None, sigma0: float = 0.3, **kwargs):
        super().__init__(population_size=population_size or 0, **kwargs)
        self.sigma0 = sigma0

    def _setup(self, problem: BaseProblem) -> None:
        d = self.bounds.dimension
        self.d = d
        if self.population_size <= 0:
            self.population_size = 4 + int(3 * np.log(d))
        n = self.population_size

        self.mean = self.bounds.lower + self.rng.random(d) * self.bounds.range
        self.sigma = self.sigma0 * float(np.mean(self.bounds.range))

        self.mu = n // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / d) / (d + 4 + 2 * self.mueff / d)
        self.cs = (self.mueff + 2) / (d + self.mueff + 5)
        self.c1 = 2 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((d + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (d + 1)) - 1) + self.cs

        self.pc = np.zeros(d)
        self.ps = np.zeros(d)
        self.B = np.eye(d)
        self.D = np.ones(d)
        self.C = np.eye(d)
        self.chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        self.eigeneval = 0
        self.gen = 0

        self.best_position_internal = self.mean.copy()
        self.best_fitness_internal = np.inf

    def _step(self):
        problem = self.problem
        d, n = self.d, self.population_size
        self.gen += 1

        z = self.rng.normal(size=(n, d))
        y = z @ (self.B * self.D).T
        x = self.mean + self.sigma * y
        x_clipped = self.bounds.clip(x)
        fitness = problem.evaluate(x_clipped)

        # Recompute y/z from the clipped sample so subsequent mean/covariance/step-size
        # updates stay consistent with what was actually evaluated (matters when bounds
        # clip a meaningful fraction of samples, e.g. large sigma0 relative to the domain).
        y = (x_clipped - self.mean) / self.sigma
        invD = np.where(self.D > 1e-300, 1.0 / self.D, 0.0)
        z = y @ (self.B * invD).T

        order = np.argsort(fitness)
        best_idx = order[0]
        if fitness[best_idx] < self.best_fitness_internal:
            self.best_fitness_internal = fitness[best_idx]
            self.best_position_internal = x_clipped[best_idx].copy()

        selected = order[: self.mu]
        y_sel = y[selected]
        z_sel = z[selected]

        y_mean = self.weights @ y_sel
        z_mean = self.weights @ z_sel

        self.mean = self.mean + self.sigma * y_mean

        invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (invsqrtC @ z_mean if False else (self.B @ z_mean))

        ps_norm = np.linalg.norm(self.ps)
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi_n < 1.4 + 2 / (d + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean

        artmp = y_sel
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * (artmp.T * self.weights) @ artmp
        )

        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1))

        if self.gen - self.eigeneval > 1.0 / ((self.c1 + self.cmu) * d / 10):
            self.eigeneval = self.gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-20, None)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs

        return (
            self.best_position_internal,
            self.best_fitness_internal,
            x_clipped,
            fitness,
        )
