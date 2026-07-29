from __future__ import annotations

import numpy as np

from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem


class ParticleSwarmOptimization(BaseOptimizer):
    """Standard PSO with inertia weight, cognitive and social coefficients."""

    name = "ParticleSwarmOptimization"

    def __init__(
        self,
        population_size: int = 40,
        inertia: float = 0.7298,
        cognitive: float = 1.49618,
        social: float = 1.49618,
        velocity_clamp: float = 0.2,
        **kwargs,
    ):
        super().__init__(population_size=population_size, **kwargs)
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.velocity_clamp = velocity_clamp

    def _setup(self, problem: BaseProblem) -> None:
        n, d = self.population_size, self.bounds.dimension
        self.positions = self.bounds.sample(n, self.rng)
        v_max = self.velocity_clamp * self.bounds.range
        self.velocities = self.rng.uniform(-v_max, v_max, size=(n, d))
        self.fitness = self.problem.evaluate(self.positions)

        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = self.fitness.copy()

        best_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[best_idx].copy()
        self.global_best_fitness = self.fitness[best_idx]
        self.v_max = v_max

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension

        r1 = self.rng.random((n, d))
        r2 = self.rng.random((n, d))
        cognitive_term = self.cognitive * r1 * (self.personal_best_positions - self.positions)
        social_term = self.social * r2 * (self.global_best_position - self.positions)

        self.velocities = self.inertia * self.velocities + cognitive_term + social_term
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)

        self.positions = self.bounds.clip(self.positions + self.velocities)
        self.fitness = problem.evaluate(self.positions)

        improved = self.fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = self.fitness[improved]

        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()

        return self.global_best_position, self.global_best_fitness, self.positions, self.fitness


class ComprehensiveLearningPSO(BaseOptimizer):
    """CLPSO: each particle learns from a per-dimension exemplar built from personal bests
    of (possibly different) particles, promoting diversity for multimodal landscapes."""

    name = "ComprehensiveLearningPSO"

    def __init__(self, population_size: int = 40, inertia: float = 0.729, c: float = 1.49618, **kwargs):
        super().__init__(population_size=population_size, **kwargs)
        self.inertia = inertia
        self.c = c

    def _setup(self, problem: BaseProblem) -> None:
        n, d = self.population_size, self.bounds.dimension
        self.positions = self.bounds.sample(n, self.rng)
        v_max = 0.2 * self.bounds.range
        self.velocities = self.rng.uniform(-v_max, v_max, size=(n, d))
        self.v_max = v_max
        self.fitness = self.problem.evaluate(self.positions)
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = self.fitness.copy()
        self.pc = 0.05 + 0.45 * (np.exp(10 * np.arange(n) / (n - 1)) - 1) / (np.exp(10) - 1) if n > 1 else np.full(n, 0.1)

        best_idx = np.argmin(self.fitness)
        self.global_best_position = self.positions[best_idx].copy()
        self.global_best_fitness = self.fitness[best_idx]

    def _step(self):
        problem = self.problem
        n, d = self.population_size, self.bounds.dimension
        exemplars = np.zeros((n, d))

        for i in range(n):
            use_own = self.rng.random(d) >= self.pc[i]
            candidates = self.rng.integers(0, n, size=(d, 2))
            for j in range(d):
                if use_own[j]:
                    exemplars[i, j] = self.personal_best_positions[i, j]
                else:
                    a, b = candidates[j]
                    winner = a if self.personal_best_fitness[a] < self.personal_best_fitness[b] else b
                    exemplars[i, j] = self.personal_best_positions[winner, j]

        r = self.rng.random((n, d))
        self.velocities = self.inertia * self.velocities + self.c * r * (exemplars - self.positions)
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        self.positions = self.bounds.clip(self.positions + self.velocities)
        self.fitness = problem.evaluate(self.positions)

        improved = self.fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = self.fitness[improved]

        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()

        return self.global_best_position, self.global_best_fitness, self.positions, self.fitness
