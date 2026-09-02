"""Learning parametric monotone games: cost parameterizations, monotonicity
enforcement, training losses, and equilibrium computation.

[1] A. Bemporad and T. Tatarenko, "Learning Parametric Monotone Games,"
    arXiv preprint, 2026.

(C) 2026 A. Bemporad
"""

from .cost_models import CostModel, StructuredMonotoneCost, PenalizedConvexCost
from .penalties import (MonotonicityPenalty, ViolationPenalty, JacobianEigPenalty,
                         AuxiliaryPotentialPenalty)
from .losses import (DataLoss, CostSampleLoss, BestResponseLoss, StationarityLoss,
                      NESampleRegularizer)
from .game_learner import GameLearner
from .inverse_quadratic import QuadraticInverseSolver
from .equilibrium import EquilibriumSolver

__all__ = [
    "CostModel", "StructuredMonotoneCost", "PenalizedConvexCost",
    "MonotonicityPenalty", "ViolationPenalty", "JacobianEigPenalty", "AuxiliaryPotentialPenalty",
    "DataLoss", "CostSampleLoss", "BestResponseLoss", "StationarityLoss", "NESampleRegularizer",
    "GameLearner",
    "QuadraticInverseSolver",
    "EquilibriumSolver",
    "plot_contour", "plot_scalar",
]
