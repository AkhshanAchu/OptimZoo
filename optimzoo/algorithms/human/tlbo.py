from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class TeachingLearningBasedOptimization(BaseOptimizer):
    """TLBO: a teacher phase (move class mean toward the teacher) followed by a
    learner phase (peer-to-peer learning), with no algorithm-specific parameters."""

    name = "TeachingLearningBasedOptimization"

    def _setup(self, problem: BaseProblem) -> None:
        self.population = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.population)

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension

        # Teacher phase
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]
        mean = np.mean(self.population, axis=0)
        tf = self.rng.integers(1, 3)  # teaching factor in {1, 2}
        r = self.rng.random((n, d))
        new_population = self.population + r * (teacher - tf * mean)
        new_population = self.bounds.clip(new_population)
        new_fitness = problem.evaluate(new_population)

        improved = new_fitness < self.fitness
        self.population[improved] = new_population[improved]
        self.fitness[improved] = new_fitness[improved]

        # Learner phase
        partners = self.rng.permutation(n)
        for i in range(n):
            j = partners[i]
            if j == i:
                continue
            r = self.rng.random(d)
            if self.fitness[i] < self.fitness[j]:
                candidate = self.population[i] + r * (self.population[i] - self.population[j])
            else:
                candidate = self.population[i] + r * (self.population[j] - self.population[i])
            candidate = self.bounds.clip(candidate)
            candidate_fit = problem.evaluate(candidate)[0]
            if candidate_fit < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fit

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], self.population, self.fitness
