"""Minimal usage of the optimzoo API."""
import optimzoo
from optimzoo.problems import Rastrigin


def main():
    problem = Rastrigin(dimension=10)
    result = optimzoo.minimize(
        lambda x: (x**2).sum(axis=1),
        bounds=(-5.12, 5.12),
        dimension=10,
        algorithm="DifferentialEvolution",
        seed=0,
        max_iterations=300,
    )
    print(result.summary())

    # Or work directly with an optimizer + problem for more control.
    from optimzoo.algorithms import ParticleSwarmOptimization

    optimizer = ParticleSwarmOptimization(population_size=40, seed=0, max_iterations=300)
    result = optimizer.optimize(problem)
    print(result.summary())
    print("best position:", result.best_position)


if __name__ == "__main__":
    main()
