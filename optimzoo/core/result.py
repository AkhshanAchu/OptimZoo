from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class History:
    """Per-iteration record of an optimization run."""

    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    worst_fitness: list[float] = field(default_factory=list)
    std_fitness: list[float] = field(default_factory=list)
    best_position: list[np.ndarray] = field(default_factory=list)
    population: list[np.ndarray] = field(default_factory=list)
    n_evaluations: list[int] = field(default_factory=list)

    def record(
        self,
        *,
        best_fitness: float,
        best_position: np.ndarray,
        population_fitness: np.ndarray | None = None,
        population: np.ndarray | None = None,
        n_evaluations: int = 0,
        store_population: bool = False,
    ) -> None:
        self.best_fitness.append(float(best_fitness))
        self.best_position.append(np.array(best_position, copy=True))
        self.n_evaluations.append(n_evaluations)

        if population_fitness is not None and len(population_fitness) > 0:
            self.mean_fitness.append(float(np.mean(population_fitness)))
            self.worst_fitness.append(float(np.max(population_fitness)))
            self.std_fitness.append(float(np.std(population_fitness)))
        else:
            self.mean_fitness.append(float(best_fitness))
            self.worst_fitness.append(float(best_fitness))
            self.std_fitness.append(0.0)

        if store_population and population is not None:
            self.population.append(np.array(population, copy=True))

    def __len__(self) -> int:
        return len(self.best_fitness)


@dataclass
class Result:
    """Final outcome of an optimization run."""

    best_position: np.ndarray
    best_fitness: float
    n_iterations: int
    n_evaluations: int
    success: bool
    message: str
    runtime_seconds: float
    history: History
    algorithm_name: str = ""

    def summary(self) -> str:
        return (
            f"{self.algorithm_name} finished: best_fitness={self.best_fitness:.6g}, "
            f"iterations={self.n_iterations}, evaluations={self.n_evaluations}, "
            f"runtime={self.runtime_seconds:.3f}s, success={self.success} ({self.message})"
        )

    def __repr__(self) -> str:
        return self.summary()
