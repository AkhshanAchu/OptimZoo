from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from optimzoo.core.bounds import Bounds
from optimzoo.core.callback import BaseCallback, StopOptimization
from optimzoo.core.problem import BaseProblem
from optimzoo.core.result import History, Result


@dataclass
class StoppingCriteria:
    max_iterations: int = 1000
    max_evaluations: int | None = None
    tolerance: float | None = None
    fitness_threshold: float | None = None
    max_no_improvement: int | None = None
    time_budget_seconds: float | None = None


class BaseOptimizer(ABC):
    """Base class for all optimization algorithms.

    Subclasses implement ``_setup`` (initialize internal state/population) and
    ``_step`` (perform one iteration, returning the current best position and
    fitness plus optional population fitness for statistics/visualization).
    """

    name: str = "BaseOptimizer"

    def __init__(
        self,
        population_size: int = 30,
        seed: int | None = None,
        callbacks: list[BaseCallback] | None = None,
        store_population_history: bool = False,
        **stopping_kwargs,
    ):
        self.population_size = population_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.callbacks = callbacks or []
        self.store_population_history = store_population_history
        self.stopping = StoppingCriteria(**stopping_kwargs)

        # Populated during optimize()
        self.problem: BaseProblem | None = None
        self.bounds: Bounds | None = None
        self.history = History()
        self.iteration: int = 0
        self.best_position: np.ndarray | None = None
        self.best_fitness: float = np.inf
        self._no_improvement_count = 0
        self._stop_requested = False
        self._start_time = 0.0

    # -- Callback dispatch -------------------------------------------------

    def _dispatch(self, event: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, event)(self, *args)

    def request_stop(self) -> None:
        """Callbacks (or user code) may call this to halt the loop after the current iteration."""
        self._stop_requested = True

    # -- Subclass interface --------------------------------------------------

    @abstractmethod
    def _setup(self, problem: BaseProblem) -> None:
        """Initialize algorithm-specific state (e.g. population) for a fresh run."""

    @abstractmethod
    def _step(self) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
        """Perform one iteration.

        Returns (best_position, best_fitness, population, population_fitness).
        population/population_fitness may be None for single-point methods.
        """

    # -- Main loop -----------------------------------------------------------

    def optimize(self, problem: BaseProblem) -> Result:
        self.problem = problem
        self.bounds = problem.bounds
        self.history = History()
        self.iteration = 0
        self.best_fitness = np.inf
        self.best_position = None
        self._no_improvement_count = 0
        self._stop_requested = False
        problem.reset_evaluation_count()

        self._setup(problem)
        self._start_time = time.perf_counter()
        self._dispatch("on_start")

        message = "max_iterations reached"
        success = True

        try:
            for iteration in range(1, self.stopping.max_iterations + 1):
                self.iteration = iteration

                best_pos, best_fit, pop, pop_fit = self._step()

                improved = best_fit < self.best_fitness
                if improved:
                    self.best_fitness = float(best_fit)
                    self.best_position = np.array(best_pos, copy=True)
                    self._no_improvement_count = 0
                else:
                    self._no_improvement_count += 1

                self.history.record(
                    best_fitness=self.best_fitness,
                    best_position=self.best_position,
                    population_fitness=pop_fit,
                    population=pop,
                    n_evaluations=problem.n_evaluations,
                    store_population=self.store_population_history,
                )

                self._dispatch("on_iteration", iteration)
                if improved:
                    self._dispatch("on_improvement", iteration)

                stop = self.stopping
                if stop.fitness_threshold is not None and self.best_fitness <= stop.fitness_threshold:
                    message = "fitness_threshold reached"
                    break
                if stop.tolerance is not None and len(self.history) > 1:
                    delta = abs(self.history.best_fitness[-2] - self.history.best_fitness[-1])
                    if delta < stop.tolerance:
                        message = "tolerance reached"
                        self._dispatch("on_convergence")
                        break
                if stop.max_no_improvement is not None and self._no_improvement_count >= stop.max_no_improvement:
                    message = "no_improvement limit reached"
                    break
                if stop.max_evaluations is not None and problem.n_evaluations >= stop.max_evaluations:
                    message = "max_evaluations reached"
                    break
                if stop.time_budget_seconds is not None:
                    if time.perf_counter() - self._start_time >= stop.time_budget_seconds:
                        message = "time_budget reached"
                        break
                if self._stop_requested:
                    message = "stopped by callback"
                    self._dispatch("on_stop")
                    break
        except KeyboardInterrupt:
            message = "interrupted by user"
            success = False
        except StopOptimization as exc:
            message = str(exc) or "stopped by callback"

        runtime = time.perf_counter() - self._start_time
        result = Result(
            best_position=self.best_position,
            best_fitness=self.best_fitness,
            n_iterations=self.iteration,
            n_evaluations=problem.n_evaluations,
            success=success,
            message=message,
            runtime_seconds=runtime,
            history=self.history,
            algorithm_name=self.name,
        )
        self._dispatch("on_finish")
        return result

    def __repr__(self) -> str:
        return f"{self.name}(population_size={self.population_size})"
