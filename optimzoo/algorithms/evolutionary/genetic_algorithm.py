from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class GeneticAlgorithm(BaseOptimizer):
    """Real-coded genetic algorithm with tournament selection, blend crossover,
    Gaussian mutation, and elitism."""

    name = "GeneticAlgorithm"

    def __init__(
        self,
        population_size: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        tournament_size: int = 3,
        elitism: int = 2,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elitism = elitism

    def _setup(self, problem: BaseProblem) -> None:
        self.population = self.bounds.sample(self.population_size, self.rng)
        self.fitness = problem.evaluate(self.population)

    def _tournament_select(self) -> np.ndarray:
        idx = self.rng.integers(0, self.population_size, size=self.tournament_size)
        best_idx = idx[np.argmin(self.fitness[idx])]
        return self.population[best_idx]

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        order = np.argsort(self.fitness)
        elites = self.population[order[: self.elitism]].copy()

        children = []
        while len(children) < n - self.elitism:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            if self.rng.random() < self.crossover_rate:
                alpha = self.rng.uniform(-0.25, 1.25, size=d)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            mutate_mask = self.rng.random(d) < self.mutation_rate
            noise = self.rng.normal(0, self.mutation_strength * self.bounds.range, size=d)
            child = np.where(mutate_mask, child + noise, child)
            child = self.bounds.clip(child)
            children.append(child)

        children = np.array(children)
        self.population = np.vstack([elites, children])
        self.fitness = problem.evaluate(self.population)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], self.population, self.fitness
