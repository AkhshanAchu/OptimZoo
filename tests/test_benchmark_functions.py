import numpy as np
import pytest

from optimzoo.problems.functions import (
    ALL_PROBLEMS,
    FIXED_2D_PROBLEMS,
    ND_PROBLEMS,
    Beale,
    Booth,
    Branin,
    Bukin6,
    CrossInTray,
    DropWave,
    Eggholder,
    GoldsteinPrice,
    HolderTable,
    McCormick,
    SixHumpCamel,
)

KNOWN_OPTIMA_2D = [
    (Booth, [1.0, 3.0], 0.0),
    (Beale, [3.0, 0.5], 0.0),
    (GoldsteinPrice, [0.0, -1.0], 3.0),
    (McCormick, [-0.54719, -1.54719], -1.9133),
    (SixHumpCamel, [0.0898, -0.7126], -1.0316),
    (Branin, [-np.pi, 12.275], 0.397887),
    (Bukin6, [-10.0, 1.0], 0.0),
    (DropWave, [0.0, 0.0], -1.0),
    (Eggholder, [512.0, 404.2319], -959.6407),
    (HolderTable, [8.05502, 9.66459], -19.2085),
]


@pytest.mark.parametrize("cls,point,expected", KNOWN_OPTIMA_2D)
def test_known_optimum_value(cls, point, expected):
    problem = cls()
    value = problem.evaluate(np.array(point))[0]
    assert value == pytest.approx(expected, abs=1e-2)


# Functions whose global optimum sits at the origin (0,...,0), independent of dimension.
ZERO_AT_ORIGIN = {"Sphere", "Ackley", "Rastrigin", "Griewank", "Zakharov", "Alpine", "BentCigar"}


@pytest.mark.parametrize("name", sorted(ZERO_AT_ORIGIN))
def test_nd_problem_zero_at_origin(name):
    cls = ND_PROBLEMS[name]
    problem = cls(dimension=4)
    value = problem.evaluate(np.zeros(4))[0]
    assert value == pytest.approx(0.0, abs=1e-6)


def test_rosenbrock_zero_at_ones():
    problem = ND_PROBLEMS["Rosenbrock"](dimension=4)
    value = problem.evaluate(np.ones(4))[0]
    assert value == pytest.approx(0.0, abs=1e-6)


def test_schwefel_known_optimum():
    problem = ND_PROBLEMS["Schwefel"](dimension=4)
    value = problem.evaluate(np.full(4, 420.9687))[0]
    assert value == pytest.approx(0.0, abs=1e-3)


def test_levy_zero_at_ones():
    problem = ND_PROBLEMS["Levy"](dimension=4)
    value = problem.evaluate(np.ones(4))[0]
    assert value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", sorted(ND_PROBLEMS))
def test_nd_problem_accepts_batch_evaluation(name):
    cls = ND_PROBLEMS[name]
    problem = cls(dimension=6)
    batch = problem.bounds.sample(15, np.random.default_rng(0))
    values = problem.evaluate(batch)
    assert values.shape == (15,)
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("name", sorted(FIXED_2D_PROBLEMS))
def test_2d_problem_is_fixed_dimension(name):
    cls = ALL_PROBLEMS[name]
    problem = cls()
    assert problem.dimension == 2


def test_all_problems_registered_and_instantiable():
    for name, cls in ALL_PROBLEMS.items():
        if name in FIXED_2D_PROBLEMS:
            problem = cls()
        else:
            problem = cls(dimension=3)
        assert problem.name == name
