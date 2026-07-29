import numpy as np

from optimzoo.algorithms import ParticleSwarmOptimization
from optimzoo.callbacks import EarlyStopping
from optimzoo.core.callback import BaseCallback
from optimzoo.problems import Sphere


def test_optimize_respects_max_iterations():
    problem = Sphere(dimension=3)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=25)
    result = opt.optimize(problem)
    assert result.n_iterations == 25
    assert result.message == "max_iterations reached"


def test_optimize_respects_fitness_threshold():
    problem = Sphere(dimension=3)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=1000, fitness_threshold=1.0)
    result = opt.optimize(problem)
    assert result.best_fitness <= 1.0
    assert result.message == "fitness_threshold reached"
    assert result.n_iterations < 1000


def test_optimize_respects_max_evaluations():
    problem = Sphere(dimension=3)
    opt = ParticleSwarmOptimization(population_size=10, seed=0, max_iterations=1000, max_evaluations=100)
    result = opt.optimize(problem)
    assert result.n_evaluations >= 100
    assert result.message == "max_evaluations reached"


def test_callbacks_are_invoked():
    events = []

    class RecordingCallback(BaseCallback):
        def on_start(self, optimizer):
            events.append("start")

        def on_iteration(self, optimizer, iteration):
            events.append(("iteration", iteration))

        def on_finish(self, optimizer):
            events.append("finish")

    problem = Sphere(dimension=2)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=5, callbacks=[RecordingCallback()])
    opt.optimize(problem)

    assert events[0] == "start"
    assert events[-1] == "finish"
    iteration_events = [e for e in events if isinstance(e, tuple)]
    assert len(iteration_events) == 5


def test_early_stopping_callback_halts_run():
    problem = Sphere(dimension=3)
    opt = ParticleSwarmOptimization(
        seed=0,
        max_iterations=1000,
        callbacks=[EarlyStopping(patience=5, min_delta=1e-9)],
    )
    result = opt.optimize(problem)
    assert result.n_iterations < 1000
    assert result.message == "stopped by callback"


def test_history_records_population_when_requested():
    problem = Sphere(dimension=2)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=10, store_population_history=True)
    result = opt.optimize(problem)
    assert len(result.history.population) == 10
    assert result.history.population[0].shape == (opt.population_size, 2)


def test_history_not_stored_by_default():
    problem = Sphere(dimension=2)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=10)
    result = opt.optimize(problem)
    assert result.history.population == []


def test_best_fitness_is_monotonically_non_increasing():
    problem = Sphere(dimension=5)
    opt = ParticleSwarmOptimization(seed=0, max_iterations=100)
    result = opt.optimize(problem)
    diffs = np.diff(result.history.best_fitness)
    assert np.all(diffs <= 1e-12)


def test_seed_reproducibility():
    problem = Sphere(dimension=5)
    r1 = ParticleSwarmOptimization(seed=42, max_iterations=50).optimize(problem)
    r2 = ParticleSwarmOptimization(seed=42, max_iterations=50).optimize(problem)
    assert np.isclose(r1.best_fitness, r2.best_fitness)
    assert np.allclose(r1.best_position, r2.best_position)
