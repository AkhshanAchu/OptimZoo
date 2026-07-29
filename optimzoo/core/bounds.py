from __future__ import annotations

import numpy as np


class Bounds:
    """Box constraints for a continuous search space."""

    def __init__(self, lower, upper, dimension: int | None = None):
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        if lower_arr.ndim == 0 and upper_arr.ndim == 0:
            if dimension is None:
                raise ValueError("dimension must be provided when lower/upper are scalars")
            lower_arr = np.full(dimension, float(lower_arr))
            upper_arr = np.full(dimension, float(upper_arr))

        if lower_arr.shape != upper_arr.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if np.any(lower_arr > upper_arr):
            raise ValueError("lower bound must not exceed upper bound in any dimension")

        self.lower = lower_arr
        self.upper = upper_arr
        self.dimension = lower_arr.shape[0]

    @property
    def range(self) -> np.ndarray:
        return self.upper - self.lower

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=(n, self.dimension))

    def is_feasible(self, x: np.ndarray) -> np.ndarray:
        return np.all((x >= self.lower) & (x <= self.upper), axis=-1)

    def __repr__(self) -> str:
        return f"Bounds(lower={self.lower.tolist()}, upper={self.upper.tolist()})"
