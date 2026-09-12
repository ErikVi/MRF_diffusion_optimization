"""Tissue values are dynamic inputs, independent of sequence/solver settings."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TissueParameters:
    t1_ms: float
    t2_ms: float
    equilibrium_magnetization: float
    diffusion: Any  # scalar diffusivity or (3,3) tensor, in mm^2/s
