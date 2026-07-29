# OptimZoo

A modular Python optimization and evolutionary computation library with a clean
`BaseOptimizer` / `BaseProblem` / `BaseCallback` architecture, a broad set of
metaheuristic and classical algorithms, a rich benchmark function suite, and
matplotlib-based visualization for 1D/2D/3D landscapes and N-dimensional search
trajectories (via PCA projection).

## Install

```bash
pip install -e .
# optional extras
pip install -e ".[rich,plotly,dev]"
```

## Quickstart

```python
import optimzoo

result = optimzoo.minimize(
    lambda x: (x**2).sum(axis=1),
    bounds=(-5, 5),
    dimension=30,
    algorithm="DifferentialEvolution",
    seed=0,
    max_iterations=500,
)
print(result.summary())
```

Or work with the object API directly for full control (custom stopping
criteria, callbacks, population history for visualization):

```python
from optimzoo.algorithms import ParticleSwarmOptimization
from optimzoo.problems import Rastrigin
from optimzoo.callbacks import ProgressBarCallback, EarlyStopping

problem = Rastrigin(dimension=10)
optimizer = ParticleSwarmOptimization(
    population_size=40,
    seed=0,
    max_iterations=500,
    tolerance=1e-10,
    callbacks=[ProgressBarCallback(), EarlyStopping(patience=50)],
)
result = optimizer.optimize(problem)
```

## Algorithms

| Family | Algorithms |
|---|---|
| Evolutionary | `GeneticAlgorithm`, `DifferentialEvolution`, `SHADE`, `CMAES` |
| Swarm | `ParticleSwarmOptimization`, `ComprehensiveLearningPSO`, `GreyWolfOptimizer` |
| Bio-inspired | `WhaleOptimizationAlgorithm`, `ArtificialBeeColony`, `CuckooSearch`, `FireflyAlgorithm` |
| Human-based | `TeachingLearningBasedOptimization` |
| Direct search | `NelderMead` |
| Local/stochastic search | `SimulatedAnnealing`, `TabuSearch`, `HillClimbing`, `RandomSearch` |

All are registered in `optimzoo.algorithms.ALL_ALGORITHMS`.

## Benchmark problems

`optimzoo.problems` includes N-dimensional classics (Sphere, Ackley,
Rastrigin, Rosenbrock, Griewank, Schwefel, Michalewicz, Levy, Zakharov,
Styblinski-Tang, Alpine, BentCigar, HappyCat) and fixed 2D classics (Booth,
Beale, Goldstein-Price, McCormick, Six-Hump-Camel, Branin, Bukin6, DropWave,
CrossInTray, HolderTable, Eggholder), each with known global optima where
applicable. See `optimzoo.problems.ALL_PROBLEMS`.

## Visualization

```python
from optimzoo.visualization import plot_contour, plot_search_trace, animate_population, plot_convergence

fig = plot_search_trace(problem, result)         # 2D contour + best-position trajectory
fig = plot_convergence([result1, result2])        # compare convergence curves
anim = animate_population(problem, result)        # requires store_population_history=True
anim.save("search.gif", writer="pillow", fps=15)
```

For problems with more than 3 dimensions, `plot_nd_projection(result)`
projects the best-position trajectory to 2D via PCA (implemented directly on
top of NumPy, no extra dependency).

See `examples/` for complete runnable scripts.

## CLI

```bash
optimzoo list
optimzoo run --algorithm ParticleSwarmOptimization --problem Ackley --dimension 20 --max-iterations 300
optimzoo benchmark --algorithms PSO DifferentialEvolution --problems Sphere Rastrigin --repeats 10
```

## Live dashboard

For 2D problems, watch an optimization run live in the browser, then go back
and inspect it: population scatter over the fitness landscape, a convergence
chart with hover tooltips, and live statistics (iteration, best/mean/worst
fitness, evaluations, iterations/sec). Backend is FastAPI + WebSocket; the
frontend is plain HTML/CSS/JS with no build step.

```bash
pip install -e ".[dashboard]"
optimzoo dashboard
```

This starts a local server (default `http://127.0.0.1:8000`) and opens it in
your browser. Pick an algorithm and problem, set the population/iterations,
and click "Start run" to stream the search live. Runs execute in a background
thread; "Stop run" cancels cooperatively at the next iteration boundary.

Once data exists (live or finished), the dashboard supports:

- **Timeline scrubber** — drag the slider (or use the prev/next/play buttons,
  or arrow keys / spacebar) to replay any past iteration of the current run,
  seeing the population and best point exactly as they were then.
- **Zoom & pan on the landscape** — scroll to zoom into the fitness heatmap
  around the cursor, drag to pan, double-click to reset.
- **Zoom on the convergence chart** — drag-select an iteration range to zoom
  the fitness-vs-iteration chart into that slice; double-click to reset.
- **Run history** — every run started this session is listed in the sidebar;
  click one to reload it (read-only replay, scrubbable) even after the page
  has moved on to a different run.
- **Compare two runs** — check "Compare two runs," pick a run from history to
  load as "run B," and see both runs' best-fitness convergence overlaid.

## Extending

- New algorithm: subclass `optimzoo.core.BaseOptimizer`, implement `_setup(problem)`
  and `_step()` (return `(best_position, best_fitness, population, population_fitness)`).
- New problem: subclass `optimzoo.core.BaseProblem`, implement `_evaluate(x)`
  for a batch `(n, dimension)` array.
- New callback: subclass `optimzoo.core.BaseCallback` and override any of
  `on_start`, `on_iteration`, `on_improvement`, `on_convergence`, `on_finish`, `on_stop`.
---
Made with ❤️ from NiceGuy
