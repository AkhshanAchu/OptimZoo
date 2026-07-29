from __future__ import annotations

import datetime
import queue
import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from optimzoo.algorithms import ALL_ALGORITHMS
from optimzoo.core.callback import BaseCallback
from optimzoo.dashboard.callback import DashboardCallback
from optimzoo.problems import ALL_PROBLEMS, FIXED_2D_PROBLEMS

# Cap how many completed/in-flight runs are kept in memory for history/replay.
MAX_RETAINED_RUNS = 30


class _StopRequested(BaseCallback):
    """Lets the manager cooperatively cancel a running optimizer from another thread."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def on_iteration(self, optimizer, iteration: int) -> None:
        if self.stop_event.is_set():
            optimizer.request_stop()


@dataclass
class RunHandle:
    run_id: str
    queue: "queue.Queue"
    algorithm: str
    problem: str
    dimension: int
    population_size: int
    max_iterations: int
    seed: int | None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    done: bool = False
    # Full ordered event log, accumulated as the run streams, so a finished
    # (or in-progress) run can be replayed/scrubbed after the fact.
    events: list[dict] = field(default_factory=list)
    best_fitness: float | None = None
    n_iterations_completed: int = 0

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "problem": self.problem,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
            "created_at": self.created_at,
            "done": self.done,
            "best_fitness": self.best_fitness,
            "n_iterations_completed": self.n_iterations_completed,
        }


class DashboardManager:
    """Owns all in-flight and completed dashboard runs. One process-wide instance."""

    def __init__(self):
        self._runs: dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    def list_algorithms(self) -> list[str]:
        return sorted(ALL_ALGORITHMS)

    def list_problems(self) -> list[dict]:
        return [
            {"name": name, "fixed_2d": name in FIXED_2D_PROBLEMS}
            for name in sorted(ALL_PROBLEMS)
        ]

    def landscape_grid(self, problem_name: str, dimension: int, resolution: int = 150) -> dict:
        cls = ALL_PROBLEMS[problem_name]
        problem = cls() if problem_name in FIXED_2D_PROBLEMS else cls(dimension=2)
        if problem.dimension != 2:
            raise ValueError("Landscape grid is only available for 2D problems")

        x = np.linspace(problem.bounds.lower[0], problem.bounds.upper[0], resolution)
        y = np.linspace(problem.bounds.lower[1], problem.bounds.upper[1], resolution)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        Z = problem.evaluate(points).reshape(X.shape)
        return {"x": x.tolist(), "y": y.tolist(), "z": Z.tolist()}

    def start_run(
        self,
        algorithm: str,
        problem: str,
        dimension: int,
        population_size: int,
        max_iterations: int,
        seed: int | None,
    ) -> str:
        if algorithm not in ALL_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'")
        if problem not in ALL_PROBLEMS:
            raise ValueError(f"Unknown problem '{problem}'")

        run_id = uuid.uuid4().hex[:12]
        event_queue: queue.Queue = queue.Queue()
        handle = RunHandle(
            run_id=run_id,
            queue=event_queue,
            algorithm=algorithm,
            problem=problem,
            dimension=dimension,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=seed,
        )

        problem_cls = ALL_PROBLEMS[problem]
        problem_instance = problem_cls() if problem in FIXED_2D_PROBLEMS else problem_cls(dimension=dimension)

        def _record(event: dict) -> None:
            # Every event is retained on the handle (for replay/scrubbing after
            # the fact) as well as pushed to the live queue (for streaming).
            handle.events.append(event)
            if event.get("type") == "iteration":
                handle.best_fitness = event.get("best_fitness")
                handle.n_iterations_completed = event.get("iteration", handle.n_iterations_completed)
            event_queue.put(event)

        algo_cls = ALL_ALGORITHMS[algorithm]
        optimizer = algo_cls(
            population_size=population_size,
            seed=seed,
            max_iterations=max_iterations,
            callbacks=[DashboardCallback(_record), _StopRequested(handle.stop_event)],
            store_population_history=False,
        )

        def _run():
            try:
                optimizer.optimize(problem_instance)
            except Exception as exc:  # surface errors to the frontend instead of dying silently
                _record({"type": "error", "message": str(exc)})
            finally:
                _record({"type": "closed"})
                handle.done = True

        thread = threading.Thread(target=_run, daemon=True)
        handle.thread = thread

        with self._lock:
            self._runs[run_id] = handle
            self._evict_old_runs()
        thread.start()
        return run_id

    def _evict_old_runs(self) -> None:
        """Caller must hold self._lock. Drops oldest finished runs past the retention cap."""
        if len(self._runs) <= MAX_RETAINED_RUNS:
            return
        finished = [rid for rid, h in self._runs.items() if h.done]
        finished.sort(key=lambda rid: self._runs[rid].created_at)
        excess = len(self._runs) - MAX_RETAINED_RUNS
        for rid in finished[:excess]:
            del self._runs[rid]

    def get_run(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            handles = list(self._runs.values())
        handles.sort(key=lambda h: h.created_at, reverse=True)
        return [h.summary() for h in handles]

    def stop_run(self, run_id: str) -> bool:
        handle = self.get_run(run_id)
        if handle is None:
            return False
        handle.stop_event.set()
        return True


manager = DashboardManager()
