"""Visualize a 2D optimization run: contour + search trace, and an animated GIF."""
import matplotlib.pyplot as plt

from optimzoo.algorithms import ParticleSwarmOptimization
from optimzoo.problems import Eggholder
from optimzoo.visualization import animate_population, plot_search_trace


def main():
    problem = Eggholder()  # fixed 2D benchmark, highly multimodal
    optimizer = ParticleSwarmOptimization(
        population_size=50, seed=0, max_iterations=150, store_population_history=True
    )
    result = optimizer.optimize(problem)
    print(result.summary())

    fig = plot_search_trace(problem, result)
    fig.savefig("eggholder_search_trace.png", dpi=150)
    plt.close(fig)
    print("Saved eggholder_search_trace.png")

    anim = animate_population(problem, result, interval=60)
    anim.save("eggholder_swarm.gif", writer="pillow", fps=15)
    print("Saved eggholder_swarm.gif")


if __name__ == "__main__":
    main()
