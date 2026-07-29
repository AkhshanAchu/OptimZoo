from __future__ import annotations

import time

from optimzoo.core.callback import BaseCallback

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


class ProgressBarCallback(BaseCallback):
    """Rich-powered live progress bar; falls back to plain stdout if rich is unavailable."""

    def __init__(self, description: str | None = None):
        self.description = description
        self._progress = None
        self._task_id = None
        self._start_time = 0.0

    def on_start(self, optimizer) -> None:
        self._start_time = time.perf_counter()
        desc = self.description or optimizer.name
        total = optimizer.stopping.max_iterations

        if _HAS_RICH:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("best={task.fields[best]:.4g}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task(desc, total=total, best=float("inf"))
        else:
            print(f"[{desc}] starting, max_iterations={total}")

    def on_iteration(self, optimizer, iteration: int) -> None:
        if _HAS_RICH and self._progress is not None:
            self._progress.update(self._task_id, completed=iteration, best=optimizer.best_fitness)
        elif iteration % max(1, optimizer.stopping.max_iterations // 20) == 0:
            elapsed = time.perf_counter() - self._start_time
            print(
                f"  iter {iteration}/{optimizer.stopping.max_iterations} "
                f"best={optimizer.best_fitness:.6g} elapsed={elapsed:.2f}s"
            )

    def on_finish(self, optimizer) -> None:
        if _HAS_RICH and self._progress is not None:
            self._progress.stop()
        else:
            print(f"finished: best={optimizer.best_fitness:.6g}")
