from __future__ import annotations

import argparse
import json
import sys

from optimzoo.algorithms import ALL_ALGORITHMS
from optimzoo.callbacks import ProgressBarCallback
from optimzoo.problems import ALL_PROBLEMS, ND_PROBLEMS


def _build_problem(name: str, dimension: int):
    if name not in ALL_PROBLEMS:
        raise SystemExit(f"Unknown problem '{name}'. Available: {sorted(ALL_PROBLEMS)}")
    cls = ALL_PROBLEMS[name]
    if name in ND_PROBLEMS:
        return cls(dimension=dimension)
    return cls()


def _build_algorithm(name: str, **kwargs):
    if name not in ALL_ALGORITHMS:
        raise SystemExit(f"Unknown algorithm '{name}'. Available: {sorted(ALL_ALGORITHMS)}")
    return ALL_ALGORITHMS[name](**kwargs)


def cmd_run(args: argparse.Namespace) -> None:
    problem = _build_problem(args.problem, args.dimension)
    callbacks = [] if args.quiet else [ProgressBarCallback(description=f"{args.algorithm} on {args.problem}")]
    optimizer = _build_algorithm(
        args.algorithm,
        population_size=args.population_size,
        seed=args.seed,
        max_iterations=args.max_iterations,
        callbacks=callbacks,
    )
    result = optimizer.optimize(problem)
    print(result.summary())
    print(f"best position: {result.best_position}")

    if args.output:
        payload = {
            "algorithm": result.algorithm_name,
            "problem": args.problem,
            "dimension": args.dimension,
            "best_fitness": result.best_fitness,
            "best_position": result.best_position.tolist(),
            "n_iterations": result.n_iterations,
            "n_evaluations": result.n_evaluations,
            "runtime_seconds": result.runtime_seconds,
            "success": result.success,
            "message": result.message,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved result to {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    problems = args.problems or sorted(ND_PROBLEMS)
    algorithms = args.algorithms or sorted(ALL_ALGORITHMS)

    rows = []
    for problem_name in problems:
        problem = _build_problem(problem_name, args.dimension)
        for algo_name in algorithms:
            best_fitnesses = []
            for seed in range(args.repeats):
                optimizer = _build_algorithm(
                    algo_name,
                    population_size=args.population_size,
                    seed=seed,
                    max_iterations=args.max_iterations,
                )
                result = optimizer.optimize(problem)
                best_fitnesses.append(result.best_fitness)
            import numpy as np

            rows.append(
                {
                    "problem": problem_name,
                    "algorithm": algo_name,
                    "mean_best": float(np.mean(best_fitnesses)),
                    "std_best": float(np.std(best_fitnesses)),
                }
            )
            print(f"{problem_name:15s} {algo_name:30s} mean={rows[-1]['mean_best']:.4e} std={rows[-1]['std_best']:.4e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved benchmark results to {args.output}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    from optimzoo.dashboard import run_server

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_list(args: argparse.Namespace) -> None:
    print("Algorithms:")
    for name in sorted(ALL_ALGORITHMS):
        print(f"  - {name}")
    print("\nProblems:")
    for name in sorted(ALL_PROBLEMS):
        print(f"  - {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optimzoo", description="OptimZoo optimization toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run a single algorithm on a single problem")
    run_p.add_argument("--algorithm", required=True, help="Algorithm class name, e.g. ParticleSwarmOptimization")
    run_p.add_argument("--problem", required=True, help="Problem class name, e.g. Ackley")
    run_p.add_argument("--dimension", type=int, default=10)
    run_p.add_argument("--population-size", type=int, default=30)
    run_p.add_argument("--max-iterations", type=int, default=200)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--quiet", action="store_true")
    run_p.add_argument("--output", default=None, help="Path to save JSON result")
    run_p.set_defaults(func=cmd_run)

    bench_p = subparsers.add_parser("benchmark", help="Compare algorithms across problems")
    bench_p.add_argument("--algorithms", nargs="*", default=None)
    bench_p.add_argument("--problems", nargs="*", default=None)
    bench_p.add_argument("--dimension", type=int, default=10)
    bench_p.add_argument("--population-size", type=int, default=30)
    bench_p.add_argument("--max-iterations", type=int, default=200)
    bench_p.add_argument("--repeats", type=int, default=5)
    bench_p.add_argument("--output", default=None)
    bench_p.set_defaults(func=cmd_benchmark)

    list_p = subparsers.add_parser("list", help="List available algorithms and problems")
    list_p.set_defaults(func=cmd_list)

    dash_p = subparsers.add_parser("dashboard", help="Launch the live web dashboard")
    dash_p.add_argument("--host", default="127.0.0.1")
    dash_p.add_argument("--port", type=int, default=8000)
    dash_p.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    dash_p.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
