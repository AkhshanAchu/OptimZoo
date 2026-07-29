from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from optimzoo.core.bounds import Bounds


class BaseProblem(ABC):
    """Base class for an optimization problem (objective + constraints)."""

    #: Whether this problem has a known global optimum, used for testing/benchmarking.
    known_optimum: float | None = None
    known_optimum_position: np.ndarray | None = None

    def __init__(self, dimension: int, bounds: Bounds | None = None, name: str | None = None):
        self.dimension = dimension
        self.bounds = bounds if bounds is not None else self.default_bounds(dimension)
        self.name = name or type(self).__name__
        self.n_evaluations = 0

    @classmethod
    def default_bounds(cls, dimension: int) -> Bounds:
        return Bounds(-10.0, 10.0, dimension=dimension)

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the objective for a batch of candidate solutions.

        x has shape (n_samples, dimension); returns shape (n_samples,).
        """

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        self.n_evaluations += x.shape[0]
        return self._evaluate(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def constraint_violation(self, x: np.ndarray) -> np.ndarray:
        """Total constraint violation (0 = feasible). Override for constrained problems."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.zeros(x.shape[0])

    def reset_evaluation_count(self) -> None:
        self.n_evaluations = 0

    def __repr__(self) -> str:
        return f"{self.name}(dimension={self.dimension})"
