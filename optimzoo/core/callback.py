from __future__ import annotations

from abc import ABC


class BaseCallback(ABC):
    """Hook points into an optimizer's run loop. Override any subset of methods."""

    def on_start(self, optimizer: "BaseOptimizer") -> None:  # noqa: F821
        pass

    def on_iteration(self, optimizer: "BaseOptimizer", iteration: int) -> None:  # noqa: F821
        pass

    def on_improvement(self, optimizer: "BaseOptimizer", iteration: int) -> None:  # noqa: F821
        pass

    def on_convergence(self, optimizer: "BaseOptimizer") -> None:  # noqa: F821
        pass

    def on_finish(self, optimizer: "BaseOptimizer") -> None:  # noqa: F821
        pass

    def on_stop(self, optimizer: "BaseOptimizer") -> None:  # noqa: F821
        pass


class StopOptimization(Exception):
    """Raised by a callback (e.g. manual stop) to halt the optimization loop early."""
