import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from optimzoo.algorithms import ParticleSwarmOptimization
from optimzoo.problems import Ackley, Sphere
from optimzoo.visualization import (
    animate_population,
    plot_1d_landscape,
    plot_contour,
    plot_convergence,
    plot_nd_projection,
    plot_population_stats,
    plot_search_trace,
    plot_surface_3d,
)


@pytest.fixture
def result_2d():
    problem = Ackley(dimension=2)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=20, store_population_history=True)
    return problem, opt.optimize(problem)


def test_plot_contour_runs():
    fig = plot_contour(Ackley(dimension=2), resolution=30)
    assert fig is not None
    plt.close(fig)


def test_plot_surface_3d_runs():
    fig = plot_surface_3d(Ackley(dimension=2), resolution=20)
    assert fig is not None
    plt.close(fig)


def test_plot_search_trace_runs(result_2d):
    problem, result = result_2d
    fig = plot_search_trace(problem, result, resolution=30)
    assert fig is not None
    plt.close(fig)


def test_plot_convergence_runs(result_2d):
    _, result = result_2d
    fig = plot_convergence(result)
    assert fig is not None
    plt.close(fig)


def test_plot_population_stats_runs(result_2d):
    _, result = result_2d
    fig = plot_population_stats(result)
    assert fig is not None
    plt.close(fig)


def test_animate_population_requires_history():
    problem = Ackley(dimension=2)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=5)
    result = opt.optimize(problem)
    with pytest.raises(ValueError):
        animate_population(problem, result)


def test_animate_population_runs(result_2d):
    problem, result = result_2d
    anim = animate_population(problem, result, resolution=20)
    try:
        assert anim is not None
    finally:
        plt.close(anim._fig)


def test_plot_1d_landscape_runs():
    problem = Sphere(dimension=1)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=10, store_population_history=True)
    result = opt.optimize(problem)
    fig = plot_1d_landscape(problem, result)
    assert fig is not None
    plt.close(fig)


def test_plot_1d_landscape_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        plot_1d_landscape(Sphere(dimension=2))


def test_plot_nd_projection_runs():
    problem = Sphere(dimension=15)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=20)
    result = opt.optimize(problem)
    fig = plot_nd_projection(result)
    assert fig is not None
    plt.close(fig)
