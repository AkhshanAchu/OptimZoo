from __future__ import annotations

from typing import Callable

import numpy as np

from optimzoo.algorithms import ALL_ALGORITHMS, ParticleSwarmOptimization
from optimzoo.core.bounds import Bounds
from optimzoo.core.callback import BaseCallback
from optimzoo.core.optimizer import BaseOptimizer
from optimzoo.core.problem import BaseProblem
from optimzoo.core.result import Result


class _FunctionProblem(BaseProblem):
    """Adapts a plain Python/NumPy callable into a BaseProblem."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], dimension: int, bounds: Bounds, name: str):
        self._func = func
        super().__init__(dimension=dimension, bounds=bounds, name=name)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        result = self._func(x)
        return np.atleast_1d(np.asarray(result, dtype=float))


def minimize(
    function: Callable[[np.ndarray], np.ndarray],
    bounds: tuple | Bounds,
    dimension: int | None = None,
    algorithm: str | type[BaseOptimizer] = "ParticleSwarmOptimization",
    callbacks: list[BaseCallback] | None = None,
    **kwargs,
) -> Result:
    """One-shot convenience entry point.

    Example:
        result = optimzoo.minimize(sphere, bounds=(-5, 5), dimension=30, algorithm="DifferentialEvolution")
    """
    if isinstance(bounds, Bounds):
        problem_bounds = bounds
        dimension = dimension or bounds.dimension
    else:
        lower, upper = bounds
        if dimension is None:
            raise ValueError("dimension must be provided when bounds is a (lower, upper) tuple")
        problem_bounds = Bounds(lower, upper, dimension=dimension)

    problem = _FunctionProblem(function, dimension=dimension, bounds=problem_bounds, name=getattr(function, "__name__", "objective"))

    if isinstance(algorithm, str):
        if algorithm not in ALL_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {sorted(ALL_ALGORITHMS)}")
        algorithm_cls = ALL_ALGORITHMS[algorithm]
    else:
        algorithm_cls = algorithm

    optimizer = algorithm_cls(callbacks=callbacks, **kwargs)
    return optimizer.optimize(problem)
