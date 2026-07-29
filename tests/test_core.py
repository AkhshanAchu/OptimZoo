import numpy as np
import pytest

from optimzoo.core.bounds import Bounds
from optimzoo.problems import Sphere


def test_bounds_scalar_construction():
    b = Bounds(-5, 5, dimension=3)
    assert b.dimension == 3
    assert np.allclose(b.lower, [-5, -5, -5])
    assert np.allclose(b.upper, [5, 5, 5])


def test_bounds_rejects_inverted():
    with pytest.raises(ValueError):
        Bounds(5, -5, dimension=2)


def test_bounds_clip():
    b = Bounds(-1, 1, dimension=2)
    clipped = b.clip(np.array([2.0, -3.0]))
    assert np.allclose(clipped, [1.0, -1.0])


def test_bounds_sample_within_range():
    b = Bounds(-2, 2, dimension=4)
    rng = np.random.default_rng(0)
    samples = b.sample(100, rng)
    assert samples.shape == (100, 4)
    assert np.all(samples >= -2) and np.all(samples <= 2)


def test_problem_evaluate_counts_evaluations():
    problem = Sphere(dimension=3)
    problem.evaluate(np.zeros((5, 3)))
    assert problem.n_evaluations == 5
    problem.reset_evaluation_count()
    assert problem.n_evaluations == 0


def test_problem_evaluate_single_point():
    problem = Sphere(dimension=2)
    result = problem.evaluate(np.array([1.0, 2.0]))
    assert result.shape == (1,)
    assert np.isclose(result[0], 5.0)
