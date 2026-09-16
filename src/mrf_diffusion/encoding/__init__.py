"""CPU spatial encoding, independent of EPG and sequence simulation.

Coordinates are (ky, kx) in cycles/pixel. See docs/acquisition.md.
SigPy is loaded only when an operation requiring it is called.
"""

from .trajectory import (
    cartesian_trajectory,
    generate_rotated_spiral_trajectory,
    generate_variable_density_spiral,
    load_spiral_coordinates,
    rotate_trajectory,
    to_sigpy_coordinates,
)
from .nufft import NufftOperator
from .density_compensation import (
    radial_increment_density_compensation,
    pipe_menon_density_compensation,
)
from .psf import calculate_point_spread_function, direct_sampling_psf

__all__ = [
    "cartesian_trajectory",
    "generate_rotated_spiral_trajectory",
    "generate_variable_density_spiral",
    "load_spiral_coordinates",
    "rotate_trajectory",
    "to_sigpy_coordinates",
    "NufftOperator",
    "radial_increment_density_compensation",
    "pipe_menon_density_compensation",
    "calculate_point_spread_function",
    "direct_sampling_psf",
]
