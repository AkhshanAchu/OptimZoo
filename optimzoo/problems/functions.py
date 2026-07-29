"""Classic benchmark functions for continuous optimization.

Each function is N-dimensional unless noted otherwise (a few classics like
Booth, Beale, Branin are fixed at 2D). All are minimization problems.
"""
from __future__ import annotations

import numpy as np

from optimzoo.core.bounds import Bounds
from optimzoo.core.problem import BaseProblem


class Sphere(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.12, 5.12, dimension=dimension)

    def _evaluate(self, x):
        return np.sum(x**2, axis=1)


class Ackley(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-32.768, 32.768, dimension=dimension)

    def _evaluate(self, x):
        d = x.shape[1]
        sum_sq = np.sum(x**2, axis=1)
        sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return term1 + term2 + 20 + np.e


class Rastrigin(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.12, 5.12, dimension=dimension)

    def _evaluate(self, x):
        d = x.shape[1]
        return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


class Rosenbrock(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


class Griewank(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-600.0, 600.0, dimension=dimension)

    def _evaluate(self, x):
        d = x.shape[1]
        sum_term = np.sum(x**2, axis=1) / 4000.0
        i = np.arange(1, d + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(i)), axis=1)
        return sum_term - prod_term + 1


class Schwefel(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-500.0, 500.0, dimension=dimension)

    def _evaluate(self, x):
        d = x.shape[1]
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


class Michalewicz(BaseProblem):
    def __init__(self, dimension: int, m: float = 10.0, **kwargs):
        self.m = m
        super().__init__(dimension, **kwargs)

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(0.0, np.pi, dimension=dimension)

    def _evaluate(self, x):
        i = np.arange(1, x.shape[1] + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * self.m), axis=1)


class Levy(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-10.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        w = 1 + (x - 1) / 4.0
        term1 = np.sin(np.pi * w[:, 0]) ** 2
        wi = w[:, :-1]
        term2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=1)
        wd = w[:, -1]
        term3 = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
        return term1 + term2 + term3


class Zakharov(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        i = np.arange(1, x.shape[1] + 1)
        s1 = np.sum(x**2, axis=1)
        s2 = np.sum(0.5 * i * x, axis=1)
        return s1 + s2**2 + s2**4


class StyblinskiTang(BaseProblem):
    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.0, 5.0, dimension=dimension)

    def _evaluate(self, x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=1)

    @property
    def known_optimum(self):
        return -39.16599 * self.dimension


class Alpine(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-10.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)


class BentCigar(BaseProblem):
    known_optimum = 0.0

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-100.0, 100.0, dimension=dimension)

    def _evaluate(self, x):
        return x[:, 0] ** 2 + 1e6 * np.sum(x[:, 1:] ** 2, axis=1)


class HappyCat(BaseProblem):
    known_optimum = 0.0

    def __init__(self, dimension: int, alpha: float = 0.125, **kwargs):
        self.alpha = alpha
        super().__init__(dimension, **kwargs)

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-2.0, 2.0, dimension=dimension)

    def _evaluate(self, x):
        d = x.shape[1]
        norm_sq = np.sum(x**2, axis=1)
        s = np.sum(x, axis=1)
        return (np.abs(norm_sq - d) ** (2 * self.alpha)) + (0.5 * norm_sq + s) / d + 0.5


# ---------------------------------------------------------------------------
# Fixed low-dimensional classics (2D unless noted)
# ---------------------------------------------------------------------------


class _Fixed2DProblem(BaseProblem):
    """Base for classic 2D-only test functions."""

    def __init__(self, bounds: Bounds | None = None, **kwargs):
        super().__init__(dimension=2, bounds=bounds, **kwargs)


class Booth(_Fixed2DProblem):
    known_optimum = 0.0
    known_optimum_position = np.array([1.0, 3.0])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-10.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        a = x[:, 0] + 2 * x[:, 1] - 7
        b = 2 * x[:, 0] + x[:, 1] - 5
        return a**2 + b**2


class Beale(_Fixed2DProblem):
    known_optimum = 0.0
    known_optimum_position = np.array([3.0, 0.5])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-4.5, 4.5, dimension=dimension)

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        t1 = (1.5 - x1 + x1 * x2) ** 2
        t2 = (2.25 - x1 + x1 * x2**2) ** 2
        t3 = (2.625 - x1 + x1 * x2**3) ** 2
        return t1 + t2 + t3


class GoldsteinPrice(_Fixed2DProblem):
    known_optimum = 3.0
    known_optimum_position = np.array([0.0, -1.0])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-2.0, 2.0, dimension=dimension)

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        a = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        b = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return a * b


class McCormick(_Fixed2DProblem):
    known_optimum = -1.9133
    known_optimum_position = np.array([-0.54719, -1.54719])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(np.array([-1.5, -3.0]), np.array([4.0, 4.0]))

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1


class SixHumpCamel(_Fixed2DProblem):
    known_optimum = -1.0316
    known_optimum_position = np.array([0.0898, -0.7126])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(np.array([-3.0, -2.0]), np.array([3.0, 2.0]))

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return (
            (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
            + x1 * x2
            + (-4 + 4 * x2**2) * x2**2
        )


class Branin(_Fixed2DProblem):
    known_optimum = 0.397887

    def __init__(self, a=1.0, b=5.1 / (4 * np.pi**2), c=5 / np.pi, r=6.0, s=10.0, t=1 / (8 * np.pi), **kwargs):
        self.a, self.b, self.c, self.r, self.s, self.t = a, b, c, r, s, t
        super().__init__(**kwargs)

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(np.array([-5.0, 0.0]), np.array([10.0, 15.0]))

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return (
            self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
            + self.s * (1 - self.t) * np.cos(x1)
            + self.s
        )


class Bukin6(_Fixed2DProblem):
    known_optimum = 0.0
    known_optimum_position = np.array([-10.0, 1.0])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(np.array([-15.0, -3.0]), np.array([-5.0, 3.0]))

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)


class DropWave(_Fixed2DProblem):
    known_optimum = -1.0
    known_optimum_position = np.array([0.0, 0.0])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-5.12, 5.12, dimension=dimension)

    def _evaluate(self, x):
        norm_sq = np.sum(x**2, axis=1)
        return -(1 + np.cos(12 * np.sqrt(norm_sq))) / (0.5 * norm_sq + 2)


class CrossInTray(_Fixed2DProblem):
    known_optimum = -2.06261

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-10.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        fact = np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi)
        inner = np.abs(np.sin(x1) * np.sin(x2) * np.exp(fact)) + 1
        return -0.0001 * inner**0.1


class HolderTable(_Fixed2DProblem):
    known_optimum = -19.2085

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-10.0, 10.0, dimension=dimension)

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        fact = np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(fact))


class Eggholder(_Fixed2DProblem):
    known_optimum = -959.6407
    known_optimum_position = np.array([512.0, 404.2319])

    @classmethod
    def default_bounds(cls, dimension):
        return Bounds(-512.0, 512.0, dimension=dimension)

    def _evaluate(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        t1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        t2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        return t1 + t2


ALL_PROBLEMS: dict[str, type[BaseProblem]] = {
    cls.__name__: cls
    for cls in [
        Sphere, Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Michalewicz,
        Levy, Zakharov, StyblinskiTang, Alpine, BentCigar, HappyCat,
        Booth, Beale, GoldsteinPrice, McCormick, SixHumpCamel, Branin,
        Bukin6, DropWave, CrossInTray, HolderTable, Eggholder,
    ]
}

FIXED_2D_PROBLEMS = {
    "Booth", "Beale", "GoldsteinPrice", "McCormick", "SixHumpCamel", "Branin",
    "Bukin6", "DropWave", "CrossInTray", "HolderTable", "Eggholder",
}

ND_PROBLEMS = {name: cls for name, cls in ALL_PROBLEMS.items() if name not in FIXED_2D_PROBLEMS}
