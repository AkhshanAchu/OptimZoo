from optimzoo.algorithms.bio import (
    ArtificialBeeColony,
    CuckooSearch,
    FireflyAlgorithm,
    WhaleOptimizationAlgorithm,
)
from optimzoo.algorithms.evolutionary import (
    CMAES,
    SHADE,
    DifferentialEvolution,
    GeneticAlgorithm,
)
from optimzoo.algorithms.human import TeachingLearningBasedOptimization
from optimzoo.algorithms.local_search import (
    HillClimbing,
    RandomSearch,
    SimulatedAnnealing,
    TabuSearch,
)
from optimzoo.algorithms.math import NelderMead
from optimzoo.algorithms.swarm import (
    ComprehensiveLearningPSO,
    GreyWolfOptimizer,
    ParticleSwarmOptimization,
)

ALL_ALGORITHMS: dict[str, type] = {
    cls.__name__: cls
    for cls in [
        GeneticAlgorithm,
        DifferentialEvolution,
        SHADE,
        CMAES,
        ParticleSwarmOptimization,
        ComprehensiveLearningPSO,
        GreyWolfOptimizer,
        WhaleOptimizationAlgorithm,
        ArtificialBeeColony,
        CuckooSearch,
        FireflyAlgorithm,
        TeachingLearningBasedOptimization,
        NelderMead,
        SimulatedAnnealing,
        TabuSearch,
        HillClimbing,
        RandomSearch,
    ]
}

__all__ = [
    "ALL_ALGORITHMS",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "SHADE",
    "CMAES",
    "ParticleSwarmOptimization",
    "ComprehensiveLearningPSO",
    "GreyWolfOptimizer",
    "WhaleOptimizationAlgorithm",
    "ArtificialBeeColony",
    "CuckooSearch",
    "FireflyAlgorithm",
    "TeachingLearningBasedOptimization",
    "NelderMead",
    "SimulatedAnnealing",
    "TabuSearch",
    "HillClimbing",
    "RandomSearch",
]
