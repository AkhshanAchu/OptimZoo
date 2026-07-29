"""Compare several algorithms on the same problem and plot convergence curves."""
import matplotlib.pyplot as plt

from optimzoo.algorithms import (
    CMAES,
    DifferentialEvolution,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    ParticleSwarmOptimization,
)
from optimzoo.problems import Rastrigin
from optimzoo.visualization import plot_convergence


def main():
    problem = Rastrigin(dimension=20)
    algorithms = {
        "PSO": ParticleSwarmOptimization,
        "DE": DifferentialEvolution,
        "GA": GeneticAlgorithm,
        "CMA-ES": CMAES,
        "GWO": GreyWolfOptimizer,
    }

    results = []
    labels = []
    for label, cls in algorithms.items():
        optimizer = cls(seed=0, max_iterations=300)
        result = optimizer.optimize(problem)
        print(f"{label:10s} best={result.best_fitness:.4e}")
        results.append(result)
        labels.append(label)

    fig = plot_convergence(results, labels=labels)
    fig.savefig("convergence_comparison.png", dpi=150)
    print("Saved convergence_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
