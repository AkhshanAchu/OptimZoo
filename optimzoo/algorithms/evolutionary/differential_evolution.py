from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class DifferentialEvolution(BaseOptimizer):
    """Classic DE/rand/1/bin differential evolution."""

    name = "DifferentialEvolution"

    def __init__(self, population_size: int = 50, F: float = 0.5, CR: float = 0.9, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.F = F
        self.CR = CR

    def _setup(self, problem: BaseProblem) -> None:
        self.population = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.population)

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        for i in range(n):
            idxs = [j for j in range(n) if j != i]
            r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
            mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            mutant = self.bounds.clip(mutant)

            cross_mask = self.rng.random(d) < self.CR
            j_rand = self.rng.integers(0, d)
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, self.population[i])

            trial_fitness = problem.evaluate(trial)[0]
            if trial_fitness <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        self.population = new_population
        self.fitness = new_fitness
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], self.population, self.fitness


class SHADE(BaseOptimizer):
    """Success-History based Adaptive DE (simplified single-population SHADE)."""

    name = "SHADE"

    def __init__(self, population_size: int = 50, memory_size: int = 10, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.memory_size = memory_size

    def _setup(self, problem: BaseProblem) -> None:
        self.population = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.population)
        self.memory_f = np.full(self.memory_size, 0.5)
        self.memory_cr = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension

        mem_indices = self.rng.integers(0, self.memory_size, size=n)
        f_vals = np.clip(
            self.memory_f[mem_indices] + 0.1 * self.rng.standard_cauchy(n), 0.0, 1.0
        )
        f_vals[f_vals <= 0] = 0.1
        cr_vals = np.clip(self.rng.normal(self.memory_cr[mem_indices], 0.1), 0.0, 1.0)

        order = np.argsort(self.fitness)
        p_best_pool = max(2, int(0.1 * n))

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()
        successful_f, successful_cr, deltas = [], [], []

        for i in range(n):
            p_best_idx = order[self.rng.integers(0, p_best_pool)]
            idxs = [j for j in range(n) if j != i]
            r1, r2 = self.rng.choice(idxs, size=2, replace=False)
            mutant = (
                self.population[i]
                + f_vals[i] * (self.population[p_best_idx] - self.population[i])
                + f_vals[i] * (self.population[r1] - self.population[r2])
            )
            mutant = self.bounds.clip(mutant)

            cross_mask = self.rng.random(d) < cr_vals[i]
            j_rand = self.rng.integers(0, d)
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, self.population[i])

            trial_fitness = problem.evaluate(trial)[0]
            if trial_fitness <= self.fitness[i]:
                deltas.append(self.fitness[i] - trial_fitness)
                successful_f.append(f_vals[i])
                successful_cr.append(cr_vals[i])
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        if successful_f:
            weights = np.array(deltas)
            weights = weights / weights.sum()
            sf = np.array(successful_f)
            scr = np.array(successful_cr)
            self.memory_f[self.memory_idx] = np.sum(weights * sf**2) / np.sum(weights * sf)
            self.memory_cr[self.memory_idx] = np.sum(weights * scr)
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

        self.population = new_population
        self.fitness = new_fitness
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], self.population, self.fitness
