from __future__ import annotations

from optimzoo.core.callback import BaseCallback


class EarlyStopping(BaseCallback):
    """Stop the optimizer if best fitness hasn't improved by more than `min_delta`
    for `patience` consecutive iterations."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-12):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._wait = 0

    def on_start(self, optimizer) -> None:
        self._best = float("inf")
        self._wait = 0

    def on_iteration(self, optimizer, iteration: int) -> None:
        if optimizer.best_fitness < self._best - self.min_delta:
            self._best = optimizer.best_fitness
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                optimizer.request_stop()


class VerboseLogger(BaseCallback):
    """Print a one-line status update every `every` iterations."""

    def __init__(self, every: int = 10):
        self.every = every

    def on_start(self, optimizer) -> None:
        print(f"[{optimizer.name}] start: population_size={optimizer.population_size}")

    def on_iteration(self, optimizer, iteration: int) -> None:
        if iteration % self.every == 0:
            print(f"[{optimizer.name}] iter {iteration}: best={optimizer.best_fitness:.6g}")

    def on_finish(self, optimizer) -> None:
        print(f"[{optimizer.name}] done: best={optimizer.best_fitness:.6g}")
