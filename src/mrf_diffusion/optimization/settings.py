"""Solver and constraint policy, separate from tissue/sequence choices."""

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class ConstraintSettings:
    minimum_flip_angle: float = pi / 18
    maximum_flip_angle: float = pi / 3
    maximum_angle_step: float = 0.02
    coefficient_minimum: float = -1000.0
    coefficient_maximum: float = 1000.0
    minimum_split_fraction: float = 0.2
    maximum_split_fraction: float = 0.8
    minimum_positive_curvature: float = 0.01
    maximum_negative_curvature: float = -0.01


@dataclass(frozen=True)
class SolverSettings:
    method: str = "SLSQP"
    max_iterations: int = 1000
    callback_change_tolerance: float = 0.01
    # Legacy callback checks >1000 while maxiter=1000, normally never triggered.
    minimum_callback_iterations: int = 1000
