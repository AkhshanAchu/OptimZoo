from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from optimzoo.algorithms import ALL_ALGORITHMS
from optimzoo.core.callback import BaseCallback
from optimzoo.dashboard.callback import DashboardCallback
from optimzoo.problems import ALL_PROBLEMS, FIXED_2D_PROBLEMS


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
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    done: bool = False


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
        handle = RunHandle(run_id=run_id, queue=event_queue)

        problem_cls = ALL_PROBLEMS[problem]
        problem_instance = problem_cls() if problem in FIXED_2D_PROBLEMS else problem_cls(dimension=dimension)

        algo_cls = ALL_ALGORITHMS[algorithm]
        optimizer = algo_cls(
            population_size=population_size,
            seed=seed,
            max_iterations=max_iterations,
            callbacks=[DashboardCallback(event_queue), _StopRequested(handle.stop_event)],
            store_population_history=False,
        )

        def _run():
            try:
                optimizer.optimize(problem_instance)
            except Exception as exc:  # surface errors to the frontend instead of dying silently
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put({"type": "closed"})
                handle.done = True

        thread = threading.Thread(target=_run, daemon=True)
        handle.thread = thread

        with self._lock:
            self._runs[run_id] = handle
        thread.start()
        return run_id

    def get_run(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

    def stop_run(self, run_id: str) -> bool:
        handle = self.get_run(run_id)
        if handle is None:
            return False
        handle.stop_event.set()
        return True


manager = DashboardManager()
