import pytest

from optimzoo.algorithms import ALL_ALGORITHMS
from optimzoo.problems import Sphere

ALGORITHM_NAMES = sorted(ALL_ALGORITHMS)


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_algorithm_runs_and_improves_on_sphere(name):
    problem = Sphere(dimension=5)
    cls = ALL_ALGORITHMS[name]
    opt = cls(seed=0, max_iterations=150)
    result = opt.optimize(problem)

    assert result.best_position.shape == (5,)
    assert result.best_fitness >= 0
    # Every algorithm should beat a "do nothing" fitness of a corner point.
    corner_fitness = problem.evaluate(problem.bounds.upper)[0]
    assert result.best_fitness < corner_fitness
    assert result.history.best_fitness[0] >= result.history.best_fitness[-1]


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_algorithm_converges_reasonably_well_on_sphere(name):
    problem = Sphere(dimension=5)
    cls = ALL_ALGORITHMS[name]
    opt = cls(seed=0, max_iterations=300)
    result = opt.optimize(problem)
    # Sphere is easy; any reasonable metaheuristic should get well below 1.0
    # from the default [-5.12, 5.12]^5 box within a few hundred iterations.
    assert result.best_fitness < 1.0


def test_algorithms_respect_bounds():
    problem = Sphere(dimension=4)
    for name, cls in ALL_ALGORITHMS.items():
        opt = cls(seed=0, max_iterations=20)
        result = opt.optimize(problem)
        assert problem.bounds.is_feasible(result.best_position), f"{name} produced out-of-bounds solution"
